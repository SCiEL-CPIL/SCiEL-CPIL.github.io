"""Improved program: maximize pairwise disjoint residue classes with distinct moduli 2..128."""
import math
import random
import time
from itertools import combinations

MAX_MOD = 128
DEADLINE_FRACTION = 0.85  # safety margin

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _compatible(r1, m1, r2, m2):
    """Two residue classes {x: x ≡ r1 (mod m1)} and {x: x ≡ r2 (mod m2)} are disjoint
    iff gcd(m1,m2) does NOT divide (r1 - r2)."""
    g = _gcd(m1, m2)
    return (r1 - r2) % g != 0

def _is_pairwise_disjoint(pairs):
    for i in range(len(pairs)):
        r1, m1 = pairs[i]
        for j in range(i+1, len(pairs)):
            r2, m2 = pairs[j]
            if not _compatible(r1, m1, r2, m2):
                return False
    return True

def _greedy_build(moduli_order, residue_strategy, rng):
    """Greedy: iterate moduli in given order, pick a residue that is compatible with all chosen."""
    chosen = []  # list of (r, m)
    used_moduli = set()
    for m in moduli_order:
        if m in used_moduli:
            continue
        # Try residues in some order
        if residue_strategy == 'first':
            residues = range(m)
        elif residue_strategy == 'last':
            residues = range(m-1, -1, -1)
        else:  # random
            residues = list(range(m))
            rng.shuffle(residues)
        
        found = False
        for r in residues:
            ok = True
            for (r2, m2) in chosen:
                if not _compatible(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                chosen.append((r, m))
                used_moduli.add(m)
                found = True
                break
        if not found:
            continue
    return chosen

def _local_search_swap(pairs, rng, max_iters=5000, deadline=None):
    """Try to improve by swapping residues or removing/adding moduli."""
    best = list(pairs)
    best_len = len(best)
    all_moduli = list(range(2, MAX_MOD+1))
    
    iters = 0
    while iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        iters += 1
        
        current = list(best)
        used_moduli = {m for _, m in current}
        
        # Try to add a new modulus
        candidates = [m for m in all_moduli if m not in used_moduli]
        if candidates:
            rng.shuffle(candidates)
            for m_new in candidates[:20]:  # limit candidates per iter
                # Try random residues
                for _ in range(30):
                    r_new = rng.randint(0, m_new - 1)
                    ok = True
                    for (r2, m2) in current:
                        if not _compatible(r_new, m_new, r2, m2):
                            ok = False
                            break
                    if ok:
                        new_pairs = current + [(r_new, m_new)]
                        if len(new_pairs) > best_len:
                            best = new_pairs
                            best_len = len(best)
                        break
        
        # Try swapping a residue in an existing pair
        if current:
            idx = rng.randint(0, len(current)-1)
            r_old, m_old = current[idx]
            for _ in range(20):
                r_new = rng.randint(0, m_old - 1)
                if r_new == r_old:
                    continue
                # Check compatibility with all others
                ok = True
                for i, (r2, m2) in enumerate(current):
                    if i == idx:
                        continue
                    if not _compatible(r_new, m_old, r2, m2):
                        ok = False
                        break
                if ok:
                    # Try adding a new modulus after swap
                    current2 = list(current)
                    current2[idx] = (r_new, m_old)
                    used_moduli2 = {m for _, m in current2}
                    candidates2 = [m for m in all_moduli if m not in used_moduli2]
                    if candidates2:
                        rng.shuffle(candidates2)
                        for m_new in candidates2[:10]:
                            for _ in range(10):
                                r_add = rng.randint(0, m_new - 1)
                                ok_add = True
                                for (r2, m2) in current2:
                                    if not _compatible(r_add, m_new, r2, m2):
                                        ok_add = False
                                        break
                                if ok_add:
                                    new_pairs = current2 + [(r_add, m_new)]
                                    if len(new_pairs) > best_len:
                                        best = new_pairs
                                        best_len = len(best)
                                    break
        
        # Occasionally try removing a pair to free up space
        if rng.random() < 0.1 and len(current) > 3:
            idx = rng.randint(0, len(current)-1)
            current2 = current[:idx] + current[idx+1:]
            used_moduli2 = {m for _, m in current2}
            candidates2 = [m for m in all_moduli if m not in used_moduli2]
            if candidates2:
                rng.shuffle(candidates2)
                added = 0
                for m_new in candidates2[:15]:
                    for _ in range(20):
                        r_add = rng.randint(0, m_new - 1)
                        ok_add = True
                        for (r2, m2) in current2:
                            if not _compatible(r_add, m_new, r2, m2):
                                ok_add = False
                                break
                        if ok_add:
                            current2.append((r_add, m_new))
                            added += 1
                            break
                if len(current2) > best_len:
                    best = current2
                    best_len = len(best)
    
    return best

def _backtrack_search(moduli_sorted, deadline, max_depth=40):
    """Backtracking with pruning to find a large set of compatible pairs."""
    best = []
    
    def _compat_with_all(r, m, chosen):
        for (r2, m2) in chosen:
            if not _compatible(r, m, r2, m2):
                return False
        return True
    
    def _bt(index, chosen):
        nonlocal best
        if time.time() > deadline:
            return
        if len(chosen) > len(best):
            best = list(chosen)
        if index >= len(moduli_sorted):
            return
        m = moduli_sorted[index]
        # Try a few residues
        residues_to_try = list(range(m))
        if len(residues_to_try) > 20:
            # Sample residues
            residues_to_try = [0, m-1] + [random.randint(0, m-1) for _ in range(18)]
        for r in residues_to_try:
            if _compat_with_all(r, m, chosen):
                chosen.append((r, m))
                _bt(index + 1, chosen)
                chosen.pop()
                if time.time() > deadline:
                    return
        # Also try skipping this modulus
        _bt(index + 1, chosen)
    
    _bt(0, [])
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION
    
    rng = random.Random(42)
    
    best = []
    best_len = 0
    
    all_moduli = list(range(2, MAX_MOD + 1))
    
    # Phase 1: Multi-strategy greedy with many restarts
    strategies = ['first', 'last', 'random']
    orders = [
        'ascending', 'descending', 'prime_first', 'composite_first', 'random'
    ]
    
    primes = [m for m in all_moduli if all(m % i != 0 for i in range(2, int(m**0.5)+1))]
    composites = [m for m in all_moduli if m not in primes]
    
    restarts = 0
    max_restarts = 5000
    
    while time.time() < deadline and restarts < max_restarts:
        restarts += 1
        
        # Choose order
        order_type = orders[restarts % len(orders)]
        if order_type == 'ascending':
            moduli_order = list(all_moduli)
        elif order_type == 'descending':
            moduli_order = list(reversed(all_moduli))
        elif order_type == 'prime_first':
            moduli_order = primes + composites
        elif order_type == 'composite_first':
            moduli_order = composites + primes
        else:
            moduli_order = list(all_moduli)
            rng.shuffle(moduli_order)
        
        strategy = strategies[restarts % len(strategies)]
        
        pairs = _greedy_build(moduli_order, strategy, rng)
        
        if len(pairs) > best_len:
            best = pairs
            best_len = len(pairs)
        
        # Phase 2: Local search on best found so far
        if len(pairs) >= best_len - 1:  # only do local search if competitive
            improved = _local_search_swap(pairs, rng, max_iters=1000, deadline=deadline)
            if len(improved) > best_len:
                best = improved
                best_len = len(best)
        
        if time.time() > deadline - 15:
            break
    
    # Phase 3: Backtracking with time budget
    if time.time() < deadline - 10:
        bt_deadline = min(deadline, time.time() + 10)
        # Use a subset of moduli for backtracking to keep it tractable
        # Try with all moduli but limited depth
        bt_result = _backtrack_search(all_moduli, bt_deadline)
        if len(bt_result) > best_len:
            best = bt_result
            best_len = len(best)
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 5 and best:
        final = _local_search_swap(best, rng, max_iters=3000, deadline=deadline)
        if len(final) > best_len:
            best = final
            best_len = len(best)
    
    # Convert to required format: list of [residue, modulus]
    result = [[r, m] for (r, m) in best]
    
    # Ensure nonempty
    if not result:
        result = [[0, 2]]
    
    # Verify pairwise disjointness
    if not _is_pairwise_disjoint([(r, m) for r, m in result]):
        # Fallback to a known valid small solution
        result = [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    return result

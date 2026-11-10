import time
import math
import random

MAX_MOD = 128
DEADLINE_MARGIN = 5.0  # seconds before the 90s limit

def _gcd_table():
    g = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for a in range(2, MAX_MOD + 1):
        for b in range(2, MAX_MOD + 1):
            g[a][b] = math.gcd(a, b)
    return g

def _is_disjoint(r1, m1, r2, m2, gt):
    return (r1 - r2) % gt[m1][m2] != 0

def _can_add(r, m, selected, gt):
    for (r2, m2) in selected:
        if not _is_disjoint(r, m, r2, m2, gt):
            return False
    return True

def _greedy(order, gt, time_limit):
    selected = []
    used = set()
    t0 = time.time()
    for m in order:
        if time.time() - t0 > time_limit:
            break
        if m in used:
            continue
        for r in range(m):
            if _can_add(r, m, selected, gt):
                selected.append((r, m))
                used.add(m)
                break
    return selected, used

def _local_search(selected, gt, time_limit):
    """1-for-2 local search: drop one, try to add two."""
    t0 = time.time()
    improved = True
    while improved:
        improved = False
        if time.time() - t0 > time_limit:
            break
        n = len(selected)
        for drop_i in range(n):
            if time.time() - t0 > time_limit:
                break
            r_d, m_d = selected[drop_i]
            base = [x for x in selected if x is not selected[drop_i]]
            base_used = set(m2 for (_, m2) in base)
            cand_mods = [m for m in range(2, MAX_MOD + 1) if m not in base_used and m != m_d]
            found = None
            for m1 in cand_mods:
                if found is not None and len(found) == 2:
                    break
                # find first valid residue for m1
                r1_found = None
                for r1 in range(m1):
                    if _can_add(r1, m1, base, gt):
                        r1_found = r1
                        break
                if r1_found is None:
                    continue
                base.append((r1_found, m1))
                base_used.add(m1)
                for m2 in cand_mods:
                    if m2 == m1 or m2 in base_used:
                        continue
                    r2_found = None
                    for r2 in range(m2):
                        if _can_add(r2, m2, base, gt):
                            r2_found = r2
                            break
                    if r2_found is not None:
                        found = [(r1_found, m1), (r2_found, m2)]
                        break
                base.pop()
                base_used.discard(m1)
                if found is not None:
                    break
            if found is not None:
                new_sel = base + found
                if len(new_sel) > n:
                    selected = new_sel
                    improved = True
                    break
    return selected

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def _construct_with_prime_groups(gt, deadline):
    """
    Key insight: if we pick multiple moduli that share a common prime factor p,
    we can assign them distinct residues mod p to guarantee disjointness.
    This allows us to include many more moduli than a naive greedy approach.
    """
    start = time.time()
    
    # Group moduli by their prime factors
    # For each prime p, find all moduli in 2..128 that are divisible by p
    # We can pick multiple such moduli if we assign distinct residues mod p
    
    # Strategy: For each prime p, we can pick up to p moduli from the set of multiples of p
    # (since residues mod p must be distinct)
    
    # Let's build a solution by processing primes in order of decreasing potential
    # (i.e., primes with more multiples in range)
    
    selected = []
    used_moduli = set()
    
    # Process primes in order of decreasing count of multiples
    primes = [p for p in range(2, MAX_MOD + 1) if _is_prime(p)]
    primes_sorted = sorted(primes, key=lambda p: -sum(1 for m in range(2, MAX_MOD + 1) if m % p == 0))
    
    for p in primes_sorted:
        if time.time() > deadline:
            break
        # Find all unused moduli that are multiples of p
        candidates = [m for m in range(2, MAX_MOD + 1) if m % p == 0 and m not in used_moduli]
        
        # Sort candidates by some heuristic (e.g., by number of prime factors, or just ascending)
        # We'll try to pick up to p of them
        candidates.sort()
        
        # For each candidate, try to find a residue that is:
        # 1. Disjoint from all already selected classes
        # 2. Distinct mod p from other classes in this group
        
        group_residues = set()
        added_in_group = 0
        
        for m in candidates:
            if added_in_group >= p:
                break
            if time.time() > deadline:
                break
            
            # Try residues 0..m-1
            for r in range(m):
                # Check if r mod p is already used in this group
                if r % p in group_residues:
                    continue
                
                # Check if (r, m) is disjoint from all selected
                if _can_add(r, m, selected, gt):
                    selected.append((r, m))
                    used_moduli.add(m)
                    group_residues.add(r % p)
                    added_in_group += 1
                    break
    
    return selected

def construct():
    start = time.time()
    deadline = start + 90.0 - DEADLINE_MARGIN  # 85 seconds
    gt = _gcd_table()
    
    all_mods = list(range(2, MAX_MOD + 1))
    primes = [m for m in all_mods if _is_prime(m)]
    composites = [m for m in all_mods if not _is_prime(m)]
    
    # Build diverse orderings
    orders = []
    orders.append(list(all_mods))
    orders.append(list(reversed(all_mods)))
    orders.append(primes + composites)
    orders.append(composites + primes)
    orders.append(sorted(all_mods, key=lambda m: m.bit_count(), reverse=True))
    orders.append(sorted(all_mods, key=lambda m: m.bit_count()))
    orders.append(sorted(all_mods, key=lambda m: -len([p for p in range(2, m+1) if m % p == 0])))
    
    rnd = random.Random(42)
    for _ in range(10):
        o = list(all_mods)
        rnd.shuffle(o)
        orders.append(o)
    
    best = []
    best_size = 0
    
    # Seed with power-of-2 family
    seed = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    best = seed
    best_size = len(seed)
    
    # Try prime-group construction first
    if time.time() < deadline:
        sel_pg = _construct_with_prime_groups(gt, deadline - 1.0)
        if len(sel_pg) > best_size:
            best_size = len(sel_pg)
            best = [list(x) for x in sel_pg]
    
    # Try greedy + local search with various orderings
    for order in orders:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        # Greedy
        sel, used = _greedy(order, gt, remaining * 0.3)
        if len(sel) > best_size:
            best_size = len(sel)
            best = [list(x) for x in sel]
        
        # Local search
        remaining2 = deadline - time.time()
        if remaining2 > 1.0:
            sel2 = _local_search(sel, gt, remaining2 * 0.5)
            if len(sel2) > best_size:
                best_size = len(sel2)
                best = [list(x) for x in sel2]
    
    # Final polish on best
    if time.time() < deadline and best:
        sel = [tuple(x) for x in best]
        remaining3 = deadline - time.time()
        if remaining3 > 1.0:
            sel2 = _local_search(sel, gt, remaining3 * 0.7)
            if len(sel2) > best_size:
                best_size = len(sel2)
                best = [list(x) for x in sel2]
    
    if not best:
        best = [[1, 2]]
    return best


if __name__ == "__main__":
    sol = construct()
    print("size:", len(sol))
    print(sol)
    for i in range(len(sol)):
        for j in range(i + 1, len(sol)):
            r1, m1 = sol[i]
            r2, m2 = sol[j]
            g = math.gcd(m1, m2)
            assert (r1 - r2) % g != 0, (i, j, r1, m1, r2, m2)
    mods = [m for (_, m) in sol]
    assert len(set(mods)) == len(mods)
    for r, m in sol:
        assert 0 <= r < m and 2 <= m <= 128
    print("OK")

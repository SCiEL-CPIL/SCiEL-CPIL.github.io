import math
import random
import time

def _gcd_table(max_n):
    """Precompute GCDs for all pairs up to max_n."""
    table = [[0] * (max_n + 1) for _ in range(max_n + 1)]
    for i in range(2, max_n + 1):
        for j in range(i, max_n + 1):
            table[i][j] = math.gcd(i, j)
            table[j][i] = table[i][j]
    return table

# Precompute GCD table for moduli 2..128
MAX_MOD = 128
_GCD_TABLE = _gcd_table(MAX_MOD)

def _is_disjoint(r1, m1, r2, m2):
    g = _GCD_TABLE[m1][m2]
    return (r1 - r2) % g != 0

def _can_add(r, m, current_set, idx_skip=-1):
    """Check if (r, m) can be added to current_set, optionally skipping index idx_skip."""
    for i, (r_i, m_i) in enumerate(current_set):
        if i == idx_skip:
            continue
        if not _is_disjoint(r, m, r_i, m_i):
            return False
    return True

def _greedy_fill(moduli_order, current_set):
    """Fill current_set with moduli from moduli_order using greedy residue assignment."""
    in_set = set()
    for m in moduli_order:
        if m in in_set:
            continue
        # Try residues in random order to break symmetries
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            if _can_add(r, m, current_set):
                current_set.append([r, m])
                in_set.add(m)
                break

def _local_search(current_set, all_moduli, time_limit):
    """
    Local search to improve the set.
    Strategies:
    1. Try to add missing moduli by reassigning residues of existing ones (1-for-1 residue change).
    2. Try to add missing moduli by reassigning residues of multiple existing ones (k-swap).
    3. Try to swap out a 'bad' modulus for a missing one.
    """
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        in_set = {m for _, m in current_set}
        missing = [m for m in all_moduli if m not in in_set]
        
        if not missing:
            break
            
        random.shuffle(missing)
        
        for m_new in missing:
            # Strategy 1: Direct add (check all residues)
            added = False
            for r_new in range(m_new):
                if _can_add(r_new, m_new, current_set):
                    current_set.append([r_new, m_new])
                    in_set.add(m_new)
                    improved = True
                    added = True
                    break
            
            if added:
                continue # Restart loop with new set
            
            # Strategy 2: 1-for-1 residue reassignment
            # Try to find a pair (existing_idx, new_r_existing) such that
            # changing existing residue allows m_new to be added with some r_new.
            # This is expensive, so we sample.
            if time.time() > time_limit:
                break
                
            # Pick a random r_new for the new modulus
            r_new = random.randint(0, m_new - 1)
            
            # Identify conflicts
            conflicts = []
            for i, (r_i, m_i) in enumerate(current_set):
                if not _is_disjoint(r_new, m_new, r_i, m_i):
                    conflicts.append(i)
            
            if not conflicts:
                # Should have been caught by Strategy 1, but safety check
                current_set.append([r_new, m_new])
                in_set.add(m_new)
                improved = True
                break
            
            # Try to fix one conflict
            random.shuffle(conflicts)
            for idx in conflicts:
                m_conf = current_set[idx][1]
                # Try all possible new residues for m_conf
                for r_conf_new in range(m_conf):
                    if r_conf_new == current_set[idx][0]:
                        continue
                    
                    # Check if r_conf_new is valid against all other existing moduli (except idx)
                    valid_against_others = True
                    for j, (r_j, m_j) in enumerate(current_set):
                        if j == idx:
                            continue
                        if not _is_disjoint(r_conf_new, m_conf, r_j, m_j):
                            valid_against_others = False
                            break
                    
                    if not valid_against_others:
                        continue
                    
                    # Check if this resolves the conflict with m_new
                    if _is_disjoint(r_new, m_new, r_conf_new, m_conf):
                        # Apply change
                        current_set[idx][0] = r_conf_new
                        # Now try to add m_new
                        if _can_add(r_new, m_new, current_set):
                            current_set.append([r_new, m_new])
                            in_set.add(m_new)
                            improved = True
                            break # Restart while loop
                            
                if improved:
                    break
                    
            if improved:
                break
                
            # Strategy 3: 1-for-2 swap (remove one, add two? No, distinct moduli constraint is on the set size)
            # Actually, let's try swapping one existing modulus for a missing one.
            # Remove m_old, add m_new.
            # This is implicitly handled if we can reassign residues, but sometimes we need to *remove* a modulus.
            # Let's try to remove a random existing modulus and see if we can add m_new.
            # This is a "move" that keeps the set size same but changes composition.
            # But we want to *increase* size.
            
            # Let's try a 2-conflict resolution (k=2)
            if len(conflicts) >= 2:
                # Try to change residues of two conflicting moduli
                # Sample a few pairs
                for _ in range(50):
                    i1, i2 = random.sample(conflicts, 2)
                    m1, m2 = current_set[i1][1], current_set[i2][1]
                    r1_old, r2_old = current_set[i1][0], current_set[i2][0]
                    
                    # Try random new residues for both
                    for _ in range(20):
                        r1_new = random.randint(0, m1 - 1)
                        r2_new = random.randint(0, m2 - 1)
                        
                        # Check if new residues are valid against all others (excluding i1, i2)
                        valid1 = True
                        valid2 = True
                        for j, (r_j, m_j) in enumerate(current_set):
                            if j == i1 or j == i2:
                                continue
                            if not _is_disjoint(r1_new, m1, r_j, m_j):
                                valid1 = False
                                break
                            if not _is_disjoint(r2_new, m2, r_j, m_j):
                                valid2 = False
                                break
                        
                        if valid1 and valid2:
                            # Check if they are disjoint from each other
                            if _is_disjoint(r1_new, m1, r2_new, m2):
                                # Check if they are disjoint from m_new
                                if _is_disjoint(r_new, m_new, r1_new, m1) and _is_disjoint(r_new, m_new, r2_new, m2):
                                    # Apply changes
                                    current_set[i1][0] = r1_new
                                    current_set[i2][0] = r2_new
                                    if _can_add(r_new, m_new, current_set):
                                        current_set.append([r_new, m_new])
                                        in_set.add(m_new)
                                        improved = True
                                        break
                    if improved:
                        break
                if improved:
                    break

    return current_set

def _difficulty(m):
    """Heuristic for modulus difficulty. Higher is harder."""
    # Count number of divisors or sum of prime factors
    # More small prime factors -> more conflicts
    count = 0
    n = m
    d = 2
    while d * d <= n:
        if n % d == 0:
            while n % d == 0:
                count += 1
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 80 # Safety margin for 90s limit
    
    all_moduli = list(range(2, MAX_MOD + 1))
    
    # Sort by difficulty: hardest first (most prime factors)
    # This forces the search to handle complex moduli early
    order_hard = sorted(all_moduli, key=lambda m: _difficulty(m), reverse=True)
    
    best_set = []
    
    # Run multiple strategies
    strategies = [
        order_hard,
        sorted(all_moduli), # Easy first
        sorted(all_moduli, reverse=True), # Large first
        list(all_moduli), # Randomized later
    ]
    
    for strategy_idx, order in enumerate(strategies):
        if time.time() > deadline:
            break
        
        current_set = []
        _greedy_fill(order, current_set)
        
        # Local search
        current_set = _local_search(current_set, all_moduli, deadline)
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        # If we found a perfect set (127 moduli), stop
        if len(best_set) == 127:
            break
        
    # Randomized restarts with local search
    while time.time() < deadline:
        random.shuffle(all_moduli)
        current_set = []
        _greedy_fill(all_moduli, current_set)
        current_set = _local_search(current_set, all_moduli, deadline)
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        if len(best_set) == 127:
            break

    if not best_set:
        return [[0, 2]]
        
    return [[r, m] for r, m in best_set]

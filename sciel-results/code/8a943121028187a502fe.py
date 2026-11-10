import time
import math
import random

def _is_disjoint(r1, m1, r2, m2, gcd_table=None):
    """Check if two residue classes are disjoint using exact gcd criterion."""
    if gcd_table is not None:
        g = gcd_table[m1][m2]
    else:
        g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    MAX_MOD = 128
    
    moduli_list = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for i in range(2, MAX_MOD + 1):
        for j in range(i, MAX_MOD + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute number of distinct prime factors for each modulus
    def num_distinct_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    pf_table = [0] * (MAX_MOD + 1)
    for i in range(2, MAX_MOD + 1):
        pf_table[i] = num_distinct_prime_factors(i)
    
    best_count = 0
    best_result = None
    
    def get_valid_residues(chosen, m, gcd_table):
        """Get all valid residues for modulus m given current assignments."""
        valid = []
        for r in range(m):
            ok = True
            for (r2, m2) in chosen:
                g = gcd_table[m][m2]
                if (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                valid.append(r)
        return valid
    
    def csp_solve(chosen, remaining_moduli, gcd_table, deadline, max_depth=50):
        """CSP solver with forward checking and MRV heuristic."""
        if time.time() > deadline:
            return chosen
        
        if not remaining_moduli:
            return chosen
        
        # MRV: pick modulus with fewest valid residues
        best_m = None
        best_valid = None
        min_count = float('inf')
        
        for m in remaining_moduli:
            valid = get_valid_residues(chosen, m, gcd_table)
            if len(valid) < min_count:
                min_count = len(valid)
                best_m = m
                best_valid = valid
                if min_count == 0:
                    return chosen  # Dead end
                if min_count == 1:
                    break
        
        if best_m is None or not best_valid:
            return chosen
        
        # Try each valid residue for best_m
        # Shuffle for variety
        residues = best_valid[:]
        random.shuffle(residues)
        
        best_solution = None
        best_len = len(chosen)
        
        for r in residues:
            if time.time() > deadline:
                break
            
            new_chosen = chosen + [(r, best_m)]
            new_remaining = [m for m in remaining_moduli if m != best_m]
            
            # Forward check: ensure all remaining moduli still have at least one valid residue
            feasible = True
            for m2 in new_remaining:
                if not get_valid_residues(new_chosen, m2, gcd_table):
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            # Recursively solve
            result = csp_solve(new_chosen, new_remaining, gcd_table, deadline, max_depth)
            
            if len(result) > best_len:
                best_len = len(result)
                best_solution = result
            
            # If we've used up most of the deadline, stop
            if time.time() > deadline - 2.0:
                break
        
        if best_solution is not None:
            return best_solution
        return chosen
    
    def greedy_construct(order, gcd_table):
        """Greedy: add moduli in given order, picking smart residue."""
        chosen = []
        used_set = set()
        for m in order:
            valid = get_valid_residues(chosen, m, gcd_table)
            if valid:
                # Pick residue that leaves most options for future moduli
                best_r = valid[0]
                best_score = -1
                for r in valid:
                    score = 0
                    for m2 in range(2, MAX_MOD + 1):
                        if m2 in used_set or m2 == m:
                            continue
                        g = gcd_table[m][m2]
                        if g > 1:
                            # Check how many residues of m2 are blocked
                            blocked = sum(1 for r2 in range(m2) if (r2 - r) % g == 0)
                            score += blocked
                    if score > best_score:
                        best_score = score
                        best_r = r
                chosen.append((best_r, m))
                used_set.add(m)
        return chosen
    
    def local_search(chosen, gcd_table, deadline):
        """Try to improve by removing one and adding two, or swapping."""
        chosen = list(chosen)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(chosen)
            used_moduli = set(m for _, m in chosen)
            unused = [m for m in range(2, MAX_MOD + 1) if m not in used_moduli]
            
            # Try removing one element and adding two new ones
            for i in range(n):
                if time.time() > deadline:
                    break
                remaining = [c for j, c in enumerate(chosen) if j != i]
                remaining_used = set(m for _, m in remaining)
                unused_after = [m for m in unused if m not in remaining_used]
                
                added = 0
                new_chosen = list(remaining)
                for m in unused_after:
                    valid = get_valid_residues(new_chosen, m, gcd_table)
                    if valid:
                        new_chosen.append((valid[0], m))
                        added += 1
                    if added >= 2:
                        break
                if added >= 2:
                    chosen = new_chosen
                    improved = True
                    break
        
        return chosen
    
    # Multiple random restarts with different strategies
    num_restarts = 200
    
    for it in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create a random ordering with various strategies
        if it % 5 == 0:
            # Sort by descending number of prime factors (harder to place first)
            order = sorted(moduli_list, key=lambda m: (-pf_table[m], random.random()))
        elif it % 5 == 1:
            # Sort by ascending number of prime factors (easier to place first)
            order = sorted(moduli_list, key=lambda m: (pf_table[m], random.random()))
        elif it % 5 == 2:
            # Random
            order = moduli_list[:]
            random.shuffle(order)
        elif it % 5 == 3:
            # Sort by descending modulus
            order = sorted(moduli_list, reverse=True)
        else:
            # Sort by ascending modulus
            order = sorted(moduli_list)
        
        # Use CSP solver for some restarts, greedy for others
        if it % 3 == 0:
            # Start with a small seed and use CSP
            seed_size = min(5, len(order))
            seed_order = order[:seed_size]
            seed_chosen = greedy_construct(seed_order, gcd_table)
            remaining = [m for m in order if m not in set(m2 for _, m2 in seed_chosen)]
            chosen = csp_solve(seed_chosen, remaining, gcd_table, deadline)
        else:
            chosen = greedy_construct(order, gcd_table)
        
        chosen = local_search(chosen, gcd_table, deadline)
        
        count = len(chosen)
        if count > best_count:
            best_count = count
            best_result = [[r, m] for r, m in chosen]
    
    # Fallback
    fallback = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    if best_result is None or len(best_result) < len(fallback):
        best_result = fallback
        
    return best_result

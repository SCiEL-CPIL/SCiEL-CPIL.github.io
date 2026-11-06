import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    MAX_M = 128
    MODULI = list(range(2, MAX_M + 1))
    
    # Precompute gcd table
    gcd_table = {}
    for i, m1 in enumerate(MODULI):
        for m2 in MODULI[i:]:
            g = gcd(m1, m2)
            gcd_table[(m1, m2)] = g
            if m1 != m2:
                gcd_table[(m2, m1)] = g
    
    # Key insight: all chosen moduli must be pairwise non-coprime
    # The largest such sets are multiples of a single prime
    # Try each prime p, find all moduli that are multiples of p
    # Within that set, solve the residue CSP
    
    def solve_csp(moduli_list, time_limit):
        """Solve residue assignment for a given set of moduli using backtracking."""
        n = len(moduli_list)
        if n == 0:
            return []
        
        # Sort moduli by size (smaller first for tighter constraints)
        sorted_mods = sorted(moduli_list)
        
        # Precompute gcd for each pair
        pair_gcd = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                g = gcd_table[(sorted_mods[i], sorted_mods[j])]
                pair_gcd[i][j] = g
                pair_gcd[j][i] = g
        
        assignment = [None] * n
        
        def backtrack(idx, remaining_time):
            if time.time() > remaining_time:
                return False
            if idx == n:
                return True
            
            m = sorted_mods[idx]
            # Try residues in order 0, 1, 2, ...
            for r in range(m):
                # Check compatibility with all previously assigned
                ok = True
                for j in range(idx):
                    g = pair_gcd[idx][j]
                    if g == 1:
                        # Shouldn't happen if we filtered, but safety
                        ok = False
                        break
                    if (r - assignment[j]) % g == 0:
                        ok = False
                        break
                if ok:
                    assignment[idx] = r
                    if backtrack(idx + 1, remaining_time):
                        return True
                    assignment[idx] = None
            return False
        
        # Try with a time limit
        if backtrack(0, time_limit):
            return [(assignment[i], sorted_mods[i]) for i in range(n)]
        return None
    
    def solve_csp_random(moduli_list, num_trials, time_limit):
        """Try multiple random orderings of moduli for backtracking."""
        n = len(moduli_list)
        best = None
        
        for trial in range(num_trials):
            if time.time() > time_limit:
                break
            
            # Random permutation of moduli
            import random
            shuffled = list(moduli_list)
            random.shuffle(shuffled)
            
            # Precompute gcd for this ordering
            pair_gcd = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    g = gcd_table[(shuffled[i], shuffled[j])]
                    pair_gcd[i][j] = g
                    pair_gcd[j][i] = g
            
            assignment = [None] * n
            
            def backtrack(idx, remaining_time):
                if time.time() > remaining_time:
                    return False
                if idx == n:
                    return True
                
                m = shuffled[idx]
                for r in range(m):
                    ok = True
                    for j in range(idx):
                        g = pair_gcd[idx][j]
                        if g == 1:
                            ok = False
                            break
                        if (r - assignment[j]) % g == 0:
                            ok = False
                            break
                    if ok:
                        assignment[idx] = r
                        if backtrack(idx + 1, remaining_time):
                            return True
                        assignment[idx] = None
                return False
            
            if backtrack(0, time_limit):
                result = [(assignment[i], shuffled[i]) for i in range(n)]
                if best is None or len(result) > len(best):
                    best = result
        
        return best
    
    # Strategy: For each prime p, find all moduli divisible by p
    # These form a clique (all pairwise gcd >= p > 1)
    # Try to solve the CSP for each such clique
    
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 
              113, 127]
    
    best_result = []
    
    for p in primes:
        if time.time() > deadline:
            break
        
        # All moduli in [2, 128] that are multiples of p
        multiples = [m for m in MODULI if m % p == 0]
        n = len(multiples)
        
        if n <= len(best_result):
            continue
        
        # Try to solve CSP for this set
        # Use limited time per attempt
        time_per_attempt = min(5.0, (deadline - time.time()) / max(1, len(primes) - primes.index(p)))
        
        # First try deterministic (sorted)
        result = solve_csp(multiples, time.time() + time_per_attempt)
        if result and len(result) > len(best_result):
            best_result = result
        
        # If didn't get full solution, try random orderings
        if time.time() < deadline:
            remaining = deadline - time.time()
            if remaining > 2.0:
                num_trials = min(20, int(remaining / 0.5))
                result2 = solve_csp_random(multiples, num_trials, time.time() + min(10.0, remaining))
                if result2 and len(result2) > len(best_result):
                    best_result = result2
    
    # Also try: all even moduli specifically (p=2 case is most important)
    # And try to add odd moduli that are multiples of some other prime
    # but compatible with the even set
    
    if time.time() < deadline:
        # Try to expand: for the best result found, try adding more moduli
        used_moduli = {m for r, m in best_result}
        
        # Try each unused modulus
        for m in MODULI:
            if m in used_moduli:
                continue
            if time.time() > deadline:
                break
            
            # Check gcd > 1 with all used
            possible = True
            for r2, m2 in best_result:
                if gcd(m, m2) == 1:
                    possible = False
                    break
            if not possible:
                continue
            
            # Try to find a compatible residue
            found = False
            for r in range(m):
                ok = True
                for r2, m2 in best_result:
                    g = gcd(m, m2)
                    if (r - r2) % g == 0:
                        ok = False
                        break
                if ok:
                    best_result.append((r, m))
                    used_moduli.add(m)
                    found = True
                    break
    
    # Convert to required format
    if not best_result:
        # Fallback: just use multiples of 2 with simple assignment
        best_result = [(0, 2), (1, 4)]
    
    return [[r, m] for r, m in best_result]

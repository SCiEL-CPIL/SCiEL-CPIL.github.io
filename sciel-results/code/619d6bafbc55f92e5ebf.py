import sys
import time
import numpy as np

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    primes_asc = sorted(primes)
    n_primes = len(primes_asc)
    
    L_target = 1000
    
    # Initial greedy solution
    cov = np.zeros(L_target + 1, dtype=np.int8)
    init_residues = [0] * n_primes
    for i, p in enumerate(primes_asc):
        best_r = 0
        best_gain = -1
        for r in range(p):
            xs = np.arange(1, L_target + 1)
            mask = (xs % p == r)
            gain = np.sum(cov[mask] < 2)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        init_residues[i] = best_r
        xs = np.arange(1, L_target + 1)
        cov[xs % p == best_r] += 1
        
    # Calculate score of init
    prefix_ok = True
    for x in range(1, L_target + 1):
        if cov[x] < 2:
            init_score = x - 1
            break
    else:
        init_score = L_target
        
    best_score = init_score
    best_residues = init_residues[:]
    
    if best_score >= L_target:
        return best_residues

    start_time = time.time()
    time_limit = 110.0
    
    def solve(i, cov, residues):
        nonlocal best_score, best_residues
        
        if time.time() - start_time > time_limit:
            return False
            
        if i == n_primes:
            if np.min(cov[1:]) >= 2:
                # Found solution
                # Calculate exact L
                L_exact = L_target
                # Check extension
                for x in range(L_target + 1, L_target + 1000):
                    count = 0
                    for j, p in enumerate(primes_asc):
                        if x % p == residues[j]:
                            count += 1
                            if count >= 2:
                                break
                    if count >= 2:
                        L_exact = x
                    else:
                        break
                        
                if L_exact > best_score:
                    best_score = L_exact
                    best_residues = residues[:]
                return True
            return False

        p = primes_asc[i]
        rem = n_primes - i - 1
        
        threshold = 2 - rem
        if threshold <= 0:
            valid_residues = list(range(p))
        else:
            bad_xs = np.where(cov < threshold)[0]
            bad_xs = bad_xs[bad_xs >= 1]
            
            if len(bad_xs) == 0:
                valid_residues = list(range(p))
            else:
                r0 = bad_xs[0] % p
                is_consistent = True
                for x in bad_xs[1:]:
                    if x % p != r0:
                        is_consistent = False
                        break
                if is_consistent:
                    valid_residues = [r0]
                else:
                    valid_residues = []
                    
        if not valid_residues:
            return False
            
        threshold2 = 1 - rem
        final_candidates = []
        for r in valid_residues:
            if threshold2 > 0:
                valid = True
                start_x = r
                if start_x == 0:
                    start_x = p
                if start_x <= L_target:
                    for x in range(start_x, L_target + 1, p):
                        if cov[x] < threshold2:
                            valid = False
                            break
                if valid:
                    final_candidates.append(r)
            else:
                final_candidates.append(r)
                
        if not final_candidates:
            return False
            
        # Heuristic: Try r that covers more low-coverage points?
        # For now, just iterate.
        for r in final_candidates:
            new_cov = cov.copy()
            start_x = r
            if start_x == 0:
                start_x = p
            if start_x <= L_target:
                for x in range(start_x, L_target + 1, p):
                    new_cov[x] += 1
                    
            new_residues = residues + [r]
            if solve(i + 1, new_cov, new_residues):
                return True
                
        return False

    init_cov = np.zeros(L_target + 1, dtype=np.int8)
    solve(0, init_cov, [])
    
    if best_residues is None:
        return init_residues
    else:
        return best_residues

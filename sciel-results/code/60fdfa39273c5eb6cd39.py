import numpy as np

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    MAX_L = 3000
    x = np.arange(1, MAX_L + 1)
    
    coverage = np.zeros(MAX_L, dtype=np.int32)
    residues = np.zeros(n_primes, dtype=np.int64)
    
    def compute_L(cov):
        # Find largest L such that all x in [1,L] have coverage >= 2
        # We can use np.argmax on the inverted boolean array?
        # Or just loop.
        # Since we call this often, let's make it fast.
        # cov is numpy array.
        # We want the first index where cov < 2.
        # idx = np.argmax(cov < 2)
        # if cov[idx] < 2: return idx
        # else return MAX_L
        
        # But we only care about prefix.
        # Actually, if cov[0] < 2, L=0? No, x starts at 1.
        # cov[0] corresponds to x=1.
        
        # Let's just iterate. It's MAX_L steps max.
        # But we can optimize:
        # The bottleneck is likely near the current L.
        # For now, simple loop.
        for i in range(MAX_L):
            if cov[i] < 2:
                return i
        return MAX_L

    # Phase 1: Greedy Assignment
    for i in range(n_primes):
        p = primes[i]
        
        # Determine candidates
        current_L = compute_L(coverage)
        
        if p < 100:
            # Evaluate all residues
            best_r = 0
            best_L = -1
            best_score = -1 # Tie breaker
            for r in range(p):
                # Simulate
                # We don't want to copy coverage every time.
                # We can calculate new L by checking prefix.
                # new_cov[x] = coverage[x] + (1 if x % p == r else 0)
                
                # We know L_new >= current_L.
                # Start checking from current_L
                L_new = current_L
                # Check if we can extend
                # We need to check x from current_L to MAX_L-1
                # Condition: coverage[x] + (1 if x % p == r else 0) >= 2
                
                # Optimization:
                # If coverage[x] >= 2, it's fine regardless of r.
                # If coverage[x] == 1, we need x % p == r.
                # If coverage[x] == 0, we need x % p == r AND ... wait, 0+1=1 < 2.
                # So if coverage[x] == 0, it's a bottleneck regardless of this prime (unless we add 2 primes? No, we add 1).
                # So if coverage[x] == 0, L cannot pass x.
                
                # So L_new is the first x >= current_L where:
                # 1. coverage[x] == 0 (fail)
                # 2. coverage[x] == 1 and x % p != r (fail)
                
                # Let's find the first failure.
                # We can iterate x from current_L to MAX_L.
                # But we can jump?
                
                # For small p, we can just loop.
                # For large p, we only check candidates.
                
                fail_idx = MAX_L
                for x_idx in range(current_L, MAX_L):
                    c = coverage[x_idx]
                    if c == 0:
                        fail_idx = x_idx
                        break
                    elif c == 1:
                        if (x_idx + 1) % p != r: # x is 1-based, x_idx is 0-based
                            fail_idx = x_idx
                            break
                    else: # c >= 2
                        continue
                
                # If no fail, L_new = MAX_L
                # Else L_new = fail_idx (since x_idx is 0-based, L is count, so L = fail_idx)
                # Wait.
                # If fail at x_idx, then x = x_idx + 1 is the first bad one.
                # So valid integers are 1..x_idx.
                # So L = x_idx.
                
                if fail_idx < MAX_L:
                    L_candidate = fail_idx
                else:
                    L_candidate = MAX_L
                
                # Tie breaker: Sum of coverage in [L_candidate, L_candidate+10]
                score = 0
                for x_idx in range(L_candidate, min(L_candidate + 10, MAX_L)):
                    c = coverage[x_idx]
                    if (x_idx + 1) % p == r:
                        c += 1
                    score += c
                    
                if L_candidate > best_L or (L_candidate == best_L and score > best_score):
                    best_L = L_candidate
                    best_score = score
                    best_r = r
            
            residues[i] = best_r
            # Update coverage
            mask = (x % p) == best_r
            coverage[mask] += 1
            
        else:
            # Large prime: Evaluate specific candidates
            x_fail = current_L + 1
            if current_L >= MAX_L:
                # No bottleneck, pick arbitrary
                best_r = 0
            else:
                candidates = set()
                candidates.add(x_fail % p)
                candidates.add((x_fail + 1) % p)
                candidates.add((x_fail + 2) % p)
                # Add a random one?
                # candidates.add(np.random.randint(p))
                
                best_r = -1
                best_L = -1
                best_score = -1
                
                for r in candidates:
                    # Same evaluation logic as above
                    L_new = current_L
                    fail_idx = MAX_L
                    for x_idx in range(current_L, MAX_L):
                        c = coverage[x_idx]
                        if c == 0:
                            fail_idx = x_idx
                            break
                        elif c == 1:
                            if (x_idx + 1) % p != r:
                                fail_idx = x_idx
                                break
                        else:
                            continue
                    
                    if fail_idx < MAX_L:
                        L_candidate = fail_idx
                    else:
                        L_candidate = MAX_L
                    
                    score = 0
                    for x_idx in range(L_candidate, min(L_candidate + 10, MAX_L)):
                        c = coverage[x_idx]
                        if (x_idx + 1) % p == r:
                            c += 1
                        score += c
                        
                    if L_candidate > best_L or (L_candidate == best_L and score > best_score):
                        best_L = L_candidate
                        best_score = score
                        best_r = r
                        
            residues[i] = best_r
            mask = (x % p) == best_r
            coverage[mask] += 1

    # Phase 2: Local Search
    L = compute_L(coverage)
    
    for iteration in range(100):
        x_fail = L + 1
        if x_fail > MAX_L:
            break
            
        improved = False
        
        # Try to fix x_fail by changing a prime
        # We look for a prime p where changing r_p to x_fail % p
        # does not break the prefix [1, L]
        
        for i in range(n_primes):
            p = primes[i]
            r_new = x_fail % p
            r_old = residues[i]
            
            if r_new == r_old:
                continue
                
            # Check safety:
            # For all x in [1, L], if x % p == r_old, then coverage[x] must be >= 3
            # So that after removing p's contribution, it stays >= 2
            
            # Find all x in [1, L] such that x % p == r_old
            # x = r_old + k*p
            # k starts from ceil(1/p) ?
            # x >= 1
            
            # We can just iterate x from 1 to L?
            # L can be up to 3000. 168 primes * 3000 = 500k ops per iteration.
            # 100 iterations = 50M ops. Might be slow in Python.
            # Let's optimize.
            
            # We only care about x where coverage[x] == 2.
            # If coverage[x] >= 3, it's safe.
            # If coverage[x] == 2 and x % p == r_old, it's UNSAFE.
            
            # So we need to check if there exists x in [1, L] such that:
            # 1. x % p == r_old
            # 2. coverage[x] == 2
            
            # We can precompute for each p? No, coverage changes.
            
            # Let's just loop x in [1, L] but break early if found unsafe.
            safe = True
            for x_val in range(1, L + 1):
                if x_val % p == r_old and coverage[x_val - 1] == 2:
                    safe = False
                    break
            
            if not safe:
                continue
            
            # If safe, apply the change
            residues[i] = r_new
            old_mask = (x % p) == r_old
            new_mask = (x % p) == r_new
            
            # Update coverage
            # We subtract old, add new
            # But we must be careful with the order if they overlap?
            # They don't overlap because r_old != r_new.
            
            coverage[old_mask] -= 1
            coverage[new_mask] += 1
            
            # Check new L
            new_L = compute_L(coverage)
            
            if new_L > L:
                L = new_L
                improved = True
                break # Restart search with new L
            else:
                # Revert
                coverage[old_mask] += 1
                coverage[new_mask] -= 1
                residues[i] = r_old
        
        if not improved:
            break
            
    return [int(r) for r in residues]

import time
import random
import math

def construct():
            # 1. Primes
            primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
            n_primes = len(primes)
            
            # 2. Init
            # Try a few seeds
            best_res = None
            best_L = 0
            
            # Helper to calc L
            def calc_L(res):
                # Max L to check, say 10000 initially, expand if needed
                max_check = 5000 
                counts = [0] * (max_check + 1)
                for i, p in enumerate(primes):
                    r = res[i]
                    for x in range(r, max_check + 1, p):
                        if x > 0: # x starts at 1
                            counts[x] += 1
                        else:
                            # r=0 means x=p, 2p...
                            # range(0, max+1, p) includes 0.
                            # We only care about 1..max
                            pass
                # Correct loop for r=0:
                # if r == 0: start at p
                # else: start at r
                # Let's rewrite the count update efficiently.
                
                L = 0
                for x in range(1, max_check + 1):
                    if counts[x] >= 2:
                        L = x
                    else:
                        break
                return L, counts, max_check

            # Better count init:
            def init_counts(res, max_check):
                counts = [0] * (max_check + 1)
                for i, p in enumerate(primes):
                    r = res[i]
                    start = r
                    if start == 0:
                        start = p
                    if start > max_check:
                        continue
                    for x in range(start, max_check + 1, p):
                        counts[x] += 1
                return counts

            # Try initializations
            inits = [
                [0] * n_primes,
                [1] * n_primes,
                [random.randint(0, p-1) for p in primes]
            ]
            
            for res in inits:
                L, counts, max_check = calc_L(res) # This logic needs to be integrated
                # Actually, let's just write the main loop with a robust calc.
            
            # Main Optimization Loop
            # State
            res = [random.randint(0, p-1) for p in primes]
            # Initialize counts
            MAX_L_INIT = 2000
            counts = [0] * (MAX_L_INIT + 1)
            for i, p in enumerate(primes):
                r = res[i]
                start = r if r > 0 else p
                if start <= MAX_L_INIT:
                    for x in range(start, MAX_L_INIT + 1, p):
                        counts[x] += 1
            
            L = 0
            for x in range(1, MAX_L_INIT + 1):
                if counts[x] >= 2:
                    L = x
                else:
                    break
            
            best_L = L
            best_res = res[:]
            
            start_time = time.time()
            deadline = 110.0 # 10s margin
            
            no_improve = 0
            
            while time.time() < deadline:
                # Extend L if possible
                extended = False
                while L < MAX_L_INIT and counts[L+1] >= 2:
                    L += 1
                    extended = True
                
                if extended:
                    if L > best_L:
                        best_L = L
                        best_res = res[:]
                    no_improve = 0
                    continue # Re-enter loop to try extending further or fix next
                
                # If we can't extend, L is stuck at MAX_L_INIT or counts[L+1] < 2
                if L == MAX_L_INIT:
                    # Expand array? Or just assume we are doing well.
                    # For now, if we hit max, we can try to verify higher.
                    # But let's assume 2000 is enough for "good" score, or we can expand.
                    # Let's expand if we hit the limit.
                    MAX_L_INIT *= 2
                    # Re-init counts? No, we can just append.
                    # This is complex. Let's just keep MAX_L_INIT high enough, e.g., 10000.
                    # If L reaches 10000, we are likely optimal or near it.
                    break
                
                # Fix L+1
                x_star = L + 1
                current_count = counts[x_star]
                
                if current_count < 2:
                    # Try to increase count of x_star
                    # Identify primes NOT covering x_star
                    # If count == 1, find the one covering it, avoid it? No, we can change it too, but we need ANOTHER one.
                    # So we want to change a prime p_j such that x_star % p_j != r_j (currently not covering)
                    # to r_j = x_star % p_j.
                    
                    candidates = []
                    for i, p in enumerate(primes):
                        if x_star % p != res[i]:
                            candidates.append(i)
                    
                    if not candidates:
                        # All primes cover x_star? Impossible if count < 168 and count < 2.
                        # Actually if count < 2, at most 1 prime covers it.
                        # So there are many candidates.
                        pass
                    
                    # Shuffle candidates to randomize
                    random.shuffle(candidates)
                    
                    found_fix = False
                    for i in candidates:
                        p = primes[i]
                        old_r = res[i]
                        new_r = x_star % p
                        
                        # Apply change
                        # Decrement old
                        start_old = old_r if old_r > 0 else p
                        if start_old <= MAX_L_INIT:
                            for x in range(start_old, MAX_L_INIT + 1, p):
                                counts[x] -= 1
                        # Increment new
                        start_new = new_r if new_r > 0 else p
                        if start_new <= MAX_L_INIT:
                            for x in range(start_new, MAX_L_INIT + 1, p):
                                counts[x] += 1
                        
                        # Check if we broke [1, L]
                        # We need min(counts[1..L]) >= 2
                        # Optimization: Only check if the change affected [1, L]
                        # The change affects x = new_r + k*p.
                        # If any affected x in [1, L] had count 2 and is now 1, we broke it.
                        # If any affected x in [1, L] had count 1 (and was the only cover? No, min was 2)
                        # Wait, if min was 2, and we decrement a spot that was 2, it becomes 1. Break.
                        # If we increment a spot that was 1, it becomes 2. Good.
                        # If we decrement a spot that was 3, it becomes 2. Good.
                        
                        # So we just need to check if any x in [1, L] that was decremented had count == 2.
                        # Or simpler: just check min(counts[1:L+1]) >= 2.
                        # Since L is small, this is fast.
                        
                        valid = True
                        # Check only the affected points in [1, L] for the decrement?
                        # No, increment can't hurt. Decrement can hurt.
                        # So check decremented points.
                        if start_old <= L:
                            for x in range(start_old, L + 1, p):
                                if counts[x] < 2:
                                    valid = False
                                    break
                        
                        if valid:
                            res[i] = new_r
                            found_fix = True
                            no_improve = 0
                            break
                        else:
                            # Revert
                            if start_old <= MAX_L_INIT:
                                for x in range(start_old, MAX_L_INIT + 1, p):
                                    counts[x] += 1
                            if start_new <= MAX_L_INIT:
                                for x in range(start_new, MAX_L_INIT + 1, p):
                                    counts[x] -= 1
                            # Note: counts are back to original state for this p
                            
                    if found_fix:
                        continue # Try to extend L again
                    
                    # If no single change fixed it, try perturbation
                    no_improve += 1
                    if no_improve > 100:
                        # Restore best and perturb
                        res = best_res[:]
                        # Re-init counts from best_res
                        counts = [0] * (MAX_L_INIT + 1)
                        for i, p in enumerate(primes):
                            r = res[i]
                            start = r if r > 0 else p
                            if start <= MAX_L_INIT:
                                for x in range(start, MAX_L_INIT + 1, p):
                                    counts[x] += 1
                        L = 0
                        for x in range(1, MAX_L_INIT + 1):
                            if counts[x] >= 2:
                                L = x
                            else:
                                break
                        
                        # Perturb
                        for _ in range(5):
                            idx = random.randint(0, n_primes - 1)
                            p = primes[idx]
                            res[idx] = random.randint(0, p - 1)
                            # Update counts for this change
                            # (Simpler to just re-init counts after perturbation? 
                            # 5 changes * (L/p) updates is fast. Or just re-init all.)
                            # Re-init is O(N * L). 168 * 2000 = 336k ops. Fast.
                            pass
                        # Re-init counts fully
                        counts = [0] * (MAX_L_INIT + 1)
                        for i, p in enumerate(primes):
                            r = res[i]
                            start = r if r > 0 else p
                            if start <= MAX_L_INIT:
                                for x in range(start, MAX_L_INIT + 1, p):
                                    counts[x] += 1
                        L = 0
                        for x in range(1, MAX_L_INIT + 1):
                            if counts[x] >= 2:
                                L = x
                            else:
                                break
                        no_improve = 0

            return best_res

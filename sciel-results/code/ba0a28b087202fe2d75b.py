import time
import math
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (150s limit, 5s buffer)

    best_score = -1
    best_set = [0, 1, 3]
    score_cache = {}

    def next_prime(n):
        if n < 2:
            return 2
        if n == 2:
            return 2
        if n % 2 == 0:
            n += 1
        while not all(n % i for i in range(3, int(math.isqrt(n)) + 1, 2)):
            n += 2
        return n

    def compute_score(U):
        key = tuple(sorted(set(U)))
        if key in score_cache:
            return score_cache[key]
        
        U = list(key)
        if not U or 0 not in U:
            score_cache[key] = -1
            return -1
        if len(U) < 2:
            score_cache[key] = 0
            return 0
        max_u = max(U)
        if max_u == 0:
            score_cache[key] = 0
            return 0
        
        # Find a prime N > 2 * max_u to ensure cyclic wrap-around doesn't affect support size
        # For sumset U+U, max element is 2*max_u.
        # For diffset U-U, max element is max_u (if we consider absolute diff) or we look at the whole ring.
        # Actually, |U-U| in integers is the same as support of cyclic diff if N > 2*max_u.
        # The range of U-U is [-max_u, max_u]. In Z_N, this is a contiguous block of length 2*max_u+1.
        # If N > 2*max_u, there is no wrap-around collision for the supports.
        
        min_N = 2 * max_u + 1
        N = next_prime(min_N)
        
        # Create indicator vector
        f = np.zeros(N)
        for u in U:
            f[u] = 1.0
            
        # Sumset: Convolution f * f
        # Diffset: Convolution f * f_rev where f_rev[i] = f[N-1-i]
        # Note: In Z_N, a-b corresponds to a + (-b). -b mod N is N-b (for b!=0).
        # So diffset indicator is f * f_neg where f_neg[x] = f[-x mod N].
        
        f_neg = np.zeros(N)
        for u in U:
            idx = (-u) % N
            f_neg[idx] = 1.0
            
        # Compute convolutions via FFT
        F = np.fft.fft(f)
        F_neg = np.fft.fft(f_neg)
        
        # Inverse FFT to get convolution coefficients
        # The result of IFFT(F * F) is the circular convolution.
        # We need the number of non-zero entries.
        # Due to floating point errors, we threshold.
        
        conv_sum = np.fft.ifft(F * F)
        conv_diff = np.fft.ifft(F * F_neg)
        
        # The real part should be integer counts.
        # Threshold: 0.5 is safe for small N.
        n_sum = np.sum(np.abs(conv_sum.real) > 0.5)
        n_diff = np.sum(np.abs(conv_diff.real) > 0.5)
        
        if n_sum == 0:
            score_cache[key] = -1
            return -1
            
        base = 2 * max_u + 1
        if base <= 1:
            score_cache[key] = 0
            return 0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            score_cache[key] = 1.0
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        score_cache[key] = score
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if len(cand) < 2:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Structured Initializations
    # Try sets that are "base-B" like or have large gaps
    # 1. Powers of 2 (maximally sparse, unique sums)
    # 2. Arithmetic Progressions (maximally dense, many diff collisions)
    # 3. Mixed: Large jumps followed by small steps
    
    # Small exhaustive search to establish a good baseline
    max_m_exhaustive = 30
    max_n_exhaustive = 6
    
    for n in range(2, max_n_exhaustive + 1):
        # Sample a subset of combinations to save time
        # We can't do full combinations(30, 5) which is 142506, that's fine.
        # combinations(30, 5) is manageable.
        for combo in combinations(range(1, max_m_exhaustive + 1), n - 1):
            cand = [0] + list(combo)
            if time.time() > deadline - 10:
                break
            try_set(cand)
        if time.time() > deadline - 10:
            break

    # Phase 2: Local Search / Genetic Algorithm on the best set
    # We will mutate the elements of the set.
    
    def mutate_set(U, mutation_rate=0.2):
        """
        Mutate a set U by changing some elements to nearby values.
        Keeps 0 fixed.
        """
        if time.time() > deadline - 5:
            return U
        
        U = list(U)
        n = len(U)
        if n < 2:
            return U
        
        # Identify non-zero elements
        non_zero_indices = [i for i in range(n) if U[i] != 0]
        
        new_U = U.copy()
        
        # Mutate a few elements
        num_mutations = max(1, int(n * mutation_rate))
        for _ in range(num_mutations):
            if not non_zero_indices:
                break
            idx = random.choice(non_zero_indices)
            old_val = U[idx]
            
            # Choose a new value
            # Strategy: Try values in a range around old_val, or random in [1, max_val]
            max_val = max(U)
            # Bias towards keeping diameter similar or slightly expanding
            lower = max(1, int(old_val * 0.5))
            upper = min(max_val + 10, int(old_val * 1.5) + 5)
            
            if random.random() < 0.1:
                # Large jump: pick random in [1, max_val + 20]
                new_val = random.randint(1, max_val + 20)
            else:
                # Local move
                new_val = random.randint(lower, upper)
            
            # Avoid duplicates
            if new_val in new_U and new_val != old_val:
                # Try a few more times or skip
                continue
                
            new_U[idx] = new_val
        
        # Ensure 0 is in the set and sort
        if 0 not in new_U:
            new_U.append(0)
        return sorted(set(new_U))

    def hill_climb(start_set, max_iters=5000):
        nonlocal best_score, best_set
        curr = sorted(set(start_set))
        curr_score = compute_score(curr)
        
        if curr_score > best_score:
            best_score = curr_score
            best_set = list(curr)
            
        no_improve_count = 0
        max_no_improve = 100
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            
            # Generate a mutated neighbor
            # We can try different mutation strengths
            if random.random() < 0.5:
                neighbor = mutate_set(curr, mutation_rate=0.1)
            else:
                neighbor = mutate_set(curr, mutation_rate=0.3)
            
            if neighbor == curr:
                continue
                
            n_score = compute_score(neighbor)
            
            if n_score > curr_score:
                curr = neighbor
                curr_score = n_score
                improved = True
                no_improve_count = 0
                
                if curr_score > best_score:
                    best_score = curr_score
                    best_set = list(curr)
            else:
                no_improve_count += 1
                
                # Simulated Annealing style acceptance?
                # For simplicity, just restart if stuck too long
                if no_improve_count > max_no_improve:
                    break
            
            # Also try adding/removing an element occasionally
            if random.random() < 0.1 and len(curr) < 15:
                # Add element
                max_val = max(curr)
                new_el = random.randint(1, max_val + 10)
                if new_el not in curr:
                    cand = sorted(curr + [new_el])
                    s = compute_score(cand)
                    if s > curr_score:
                        curr = cand
                        curr_score = s
                        improved = True
                        no_improve_count = 0
                        if curr_score > best_score:
                            best_score = curr_score
                            best_set = list(curr)
            elif random.random() < 0.1 and len(curr) > 3:
                # Remove element
                # Remove a random non-zero element
                non_zero = [x for x in curr if x != 0]
                if non_zero:
                    rem = random.choice(non_zero)
                    cand = [x for x in curr if x != rem]
                    s = compute_score(cand)
                    if s > curr_score:
                        curr = cand
                        curr_score = s
                        improved = True
                        no_improve_count = 0
                        if curr_score > best_score:
                            best_score = curr_score
                            best_set = list(curr)
        
        return curr

    # Run Hill Climbing from the best set found in Phase 1
    if time.time() < deadline - 10:
        hill_climb(best_set, max_iters=10000)
        
    # Run Hill Climbing from some random structured starts
    starts = 5
    for i in range(starts):
        if time.time() > deadline - 10:
            break
            
        # Generate a random structured start
        # Base-B like: pick random base, expand
        n = random.randint(3, 8)
        base = random.randint(2, 5)
        U = [0]
        current_val = 1
        for _ in range(n - 1):
            U.append(current_val)
            current_val *= base
            # Add some noise
            if random.random() < 0.2:
                U.append(current_val + random.randint(1, 5))
        
        # Shuffle and sort
        U = sorted(set(U))
        
        if len(U) >= 2:
            hill_climb(U, max_iters=5000)
            
    # Final Polish: Try single element replacements on best_set
    if time.time() < deadline - 2:
        max_cur = max(best_set)
        # Limit search space for final polish to ensure speed
        search_limit = max_cur + 5
        
        for _ in range(2): # 2 passes
            if time.time() > deadline - 2:
                break
            changed = False
            
            # Try adding
            for new_el in range(1, search_limit + 1):
                if new_el in best_set:
                    continue
                cand = sorted(best_set + [new_el])
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    changed = True
                    break
            
            if changed:
                continue

            # Try replacing
            for i in range(len(best_set)):
                if best_set[i] == 0:
                    continue
                for new_el in range(1, search_limit + 1):
                    if new_el in best_set:
                        continue
                    cand = list(best_set)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand or len(cand) < 2:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        changed = True
                        break
                if changed:
                    break
                    
            if not changed:
                break

    return list(best_set)

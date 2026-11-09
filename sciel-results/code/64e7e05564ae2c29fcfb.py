import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    # Parameters from task
    # q=3, n=6, radius=1
    N = q ** n
    deadline = time.time() + 110.0  # 10s margin
    
    # Precompute balls for all words in {0..q-1}^n
    # Represent words as indices 0..N-1
    # Convert index to base-q digits
    def idx_to_digits(idx):
        return [int(idx // (q ** i) % q) for i in range(n)]
    
    def digits_to_idx(digits):
        idx = 0
        for i, d in enumerate(digits):
            idx += d * (q ** i)
        return idx

    # Precompute ball membership as a list of lists of indices
    # balls[c] contains all indices within distance 1 from c
    balls = []
    for idx in range(N):
        digits = idx_to_digits(idx)
        ball = [idx]
        for pos in range(n):
            for delta in range(1, q):
                new_digits = digits[:]
                new_digits[pos] = (digits[pos] + delta) % q
                ball.append(digits_to_idx(new_digits))
        balls.append(ball)
    
    # Precompute ball sets for fast membership testing
    ball_sets = [set(b) for b in balls]
    
    # Precompute a boolean membership matrix: mem[c][x] = True if x in ball(c)
    # This is N x N = 729 x 729, which is fine
    mem = np.zeros((N, N), dtype=bool)
    for c in range(N):
        mem[c, balls[c]] = True
    
    # Precompute ball sizes (all same for radius 1: 1 + n*(q-1))
    ball_size = len(balls[0])
    
    # Initial code: Tetracode direct sum
    # Tetracode words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            initial_codewords.append(digits_to_idx(word))
    
    def greedy_reduce(codewords_list, cov_array):
        """Greedy pass to remove redundant codewords."""
        changed = True
        while changed:
            changed = False
            for i in range(len(codewords_list)):
                c = codewords_list[i]
                if np.min(cov_array[balls[c]]) > 1:
                    cov_array[balls[c]] -= 1
                    codewords_list.pop(i)
                    changed = True
                    break
        return codewords_list, cov_array
    
    def simulated_annealing(codewords_list, cov_array, T_start, cooling_rate, min_T, time_budget):
        """Run simulated annealing on the given state with targeted swap moves."""
        T = T_start
        best_size = len(codewords_list)
        best_codewords = codewords_list[:]
        best_cov = cov_array.copy()
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < time_budget and T > min_T:
            if no_improve_count > max_no_improve:
                # Restart temperature to escape local optimum
                T = T_start * 0.5
                no_improve_count = 0
            
            # 1. Try to Delete a random codeword
            if len(codewords_list) > 0:
                i = random.randint(0, len(codewords_list) - 1)
                c = codewords_list[i]
                # Check if c is redundant
                if np.min(cov_array[balls[c]]) > 1:
                    # Remove c
                    cov_array[balls[c]] -= 1
                    codewords_list.pop(i)
                    if len(codewords_list) < best_size:
                        best_size = len(codewords_list)
                        best_codewords = codewords_list[:]
                        best_cov = cov_array.copy()
                        no_improve_count = 0
                    T *= cooling_rate
                    continue
                    
            # 2. Try a Swap
            if len(codewords_list) == 0:
                break
                
            i = random.randint(0, len(codewords_list) - 1)
            c_old = codewords_list[i]
            
            # Tentative removal of c_old
            cov_array[balls[c_old]] -= 1
            
            # Find uncovered words
            missing = np.where(cov_array < 1)[0]
            
            if len(missing) == 0:
                # c_old was redundant - remove it
                codewords_list.pop(i)
                if len(codewords_list) < best_size:
                    best_size = len(codewords_list)
                    best_codewords = codewords_list[:]
                    best_cov = cov_array.copy()
                    no_improve_count = 0
                T *= cooling_rate
                continue
            else:
                # c_old is essential. Need to find c_new that covers all missing words.
                # Strategy: pick a random missing word, find candidates that cover it,
                # then check if they cover ALL missing words.
                
                # Find all words that cover at least one missing word
                # Use a more targeted approach: sample from candidates
                found = False
                # Try a few random candidates first
                for attempt in range(5):
                    c_new = random.randint(0, N - 1)
                    if c_new == c_old:
                        continue
                    if mem[c_new, missing].all():
                        # Accept swap
                        cov_array[balls[c_new]] += 1
                        codewords_list[i] = c_new
                        no_improve_count = 0
                        found = True
                        break
                
                if not found:
                    # Try targeted: pick a missing word, find all candidates covering it
                    if len(missing) > 0:
                        # Pick the missing word with fewest candidates (hardest to cover)
                        # For efficiency, just pick the first one
                        target_missing = missing[0]
                        # Find all codewords that cover this missing word
                        # mem[:, target_missing] gives all words covering target_missing
                        candidates = np.where(mem[:, target_missing])[0]
                        candidates = candidates[candidates != c_old]
                        
                        if len(candidates) > 0:
                            # Try a few random candidates from this set
                            for attempt in range(min(10, len(candidates))):
                                idx_c = random.randint(0, len(candidates) - 1)
                                c_new = int(candidates[idx_c])
                                if c_new == c_old:
                                    continue
                                if mem[c_new, missing].all():
                                    # Accept swap
                                    cov_array[balls[c_new]] += 1
                                    codewords_list[i] = c_new
                                    no_improve_count = 0
                                    found = True
                                    break
                
                if not found:
                    # Reject swap, revert
                    cov_array[balls[c_old]] += 1
                    no_improve_count += 1
                
                T *= cooling_rate
        
        return best_codewords, best_cov, best_size
    
    def local_search_polish(codewords_list, cov_array, time_budget):
        """Final local search: try all possible single swaps to find improvements."""
        best_size = len(codewords_list)
        best_codewords = codewords_list[:]
        best_cov = cov_array.copy()
        
        # First, greedy reduce
        codewords_list, cov_array = greedy_reduce(codewords_list, cov_array)
        best_size = len(codewords_list)
        best_codewords = codewords_list[:]
        best_cov = cov_array.copy()
        
        # Try systematic swaps
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for i in range(len(codewords_list)):
                if time.time() >= time_budget:
                    break
                c_old = codewords_list[i]
                
                # Try removing c_old
                cov_array[balls[c_old]] -= 1
                missing = np.where(cov_array < 1)[0]
                
                if len(missing) == 0:
                    # c_old is redundant
                    codewords_list.pop(i)
                    best_size = len(codewords_list)
                    best_codewords = codewords_list[:]
                    best_cov = cov_array.copy()
                    improved = True
                    break
                else:
                    # Find if any single replacement covers all missing
                    # Only check if missing set is small enough to be practical
                    if len(missing) <= 20:
                        # Find candidates that cover all missing words
                        # Start with candidates for the first missing word
                        target = missing[0]
                        candidates = np.where(mem[:, target])[0]
                        candidates = candidates[candidates != c_old]
                        
                        for c_new in candidates:
                            if mem[c_new, missing].all():
                                # Accept swap
                                cov_array[balls[c_new]] += 1
                                codewords_list[i] = int(c_new)
                                best_size = len(codewords_list)
                                best_codewords = codewords_list[:]
                                best_cov = cov_array.copy()
                                improved = True
                                break
                        else:
                            cov_array[balls[c_old]] += 1
                        if improved:
                            break
                    else:
                        cov_array[balls[c_old]] += 1
        
        return best_codewords, best_cov, best_size
    
    # Run multiple restarts with different initial conditions and parameters
    all_best_codewords = []
    all_best_sizes = []
    
    # First, do a greedy reduction on the initial code
    codewords = initial_codewords[:]
    cov = np.zeros(N, dtype=np.int32)
    for c in codewords:
        cov[balls[c]] += 1
    
    codewords, cov = greedy_reduce(codewords, cov)
    
    # Run simulated annealing with different parameters
    sa_configs = [
        (2.0, 0.997, 1e-4, 25),   # Medium temp, slow cooling
        (1.0, 0.999, 1e-5, 25),   # Low temp, very slow cooling
        (5.0, 0.995, 1e-3, 20),   # High temp, moderate cooling
        (0.5, 0.9995, 1e-6, 20),  # Very low temp, very slow cooling
    ]
    
    for T_start, cooling_rate, min_T, time_budget in sa_configs:
        if time.time() >= deadline - 10:
            break
        
        # Start from the greedy-reduced code
        cw = codewords[:]
        cv = cov.copy()
        
        best_cw, best_cv, best_sz = simulated_annealing(cw, cv, T_start, cooling_rate, min_T, time.time() + time_budget)
        
        all_best_codewords.append(best_cw)
        all_best_sizes.append(best_sz)
    
    # Also try starting from scratch with random initialization
    for restart in range(3):
        if time.time() >= deadline - 10:
            break
        
        # Random initial code: pick random words until all are covered
        random_cw = []
        random_cov = np.zeros(N, dtype=np.int32)
        
        while np.any(random_cov < 1) and len(random_cw) < 100:
            c = random.randint(0, N - 1)
            if c not in random_cw:
                random_cw.append(c)
                random_cov[balls[c]] += 1
        
        random_cw, random_cov = greedy_reduce(random_cw, random_cov)
        
        best_cw, best_cv, best_sz = simulated_annealing(random_cw, random_cov, 2.0, 0.997, 1e-4, time.time() + 15)
        
        all_best_codewords.append(best_cw)
        all_best_sizes.append(best_sz)
    
    # Find the best result
    if not all_best_sizes:
        # Fallback: return the greedy-reduced initial code
        best_result = codewords
        best_cov = cov
    else:
        best_idx = all_best_sizes.index(min(all_best_sizes))
        best_result = all_best_codewords[best_idx]
        best_cov = np.zeros(N, dtype=np.int32)
        for c in best_result:
            best_cov[balls[c]] += 1
    
    # Final local search polish
    if time.time() < deadline - 5:
        best_result, best_cov, _ = local_search_polish(best_result, best_cov, time.time() + min(20, deadline - time.time() - 5))
    
    # Final greedy reduction
    final_cov = np.zeros(N, dtype=np.int32)
    for c in best_result:
        final_cov[balls[c]] += 1
    best_result, final_cov = greedy_reduce(best_result, final_cov)
    
    # Return the best code found
    result = [idx_to_digits(c) for c in best_result]
    return result

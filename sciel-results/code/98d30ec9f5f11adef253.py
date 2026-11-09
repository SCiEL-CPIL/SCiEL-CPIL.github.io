import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    N = q ** n
    deadline = time.time() + 145.0
    
    # Precompute balls
    def idx_to_digits(idx):
        return [int(idx // (q ** i) % q) for i in range(n)]
    
    def digits_to_idx(digits):
        idx = 0
        for i, d in enumerate(digits):
            idx += d * (q ** i)
        return idx

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
    
    # Membership matrix
    mem = np.zeros((N, N), dtype=bool)
    for c in range(N):
        mem[c, balls[c]] = True

    def greedy_set_cover():
        uncovered = np.ones(N, dtype=bool)
        code = []
        
        while np.any(uncovered):
            if time.time() > deadline - 15:
                break
            new_covered = mem.dot(uncovered.astype(int))
            max_cov = np.max(new_covered)
            if max_cov == 0:
                break
            candidates = np.where(new_covered == max_cov)[0]
            c = random.choice(candidates)
            code.append(c)
            uncovered[balls[c]] = False
        return code

    def greedy_reduce(codewords_list, cov_array):
        changed = True
        while changed:
            changed = False
            for i in range(len(codewords_list)):
                if time.time() > deadline - 10:
                    return codewords_list, cov_array
                c = codewords_list[i]
                if np.min(cov_array[balls[c]]) > 1:
                    cov_array[balls[c]] -= 1
                    codewords_list.pop(i)
                    changed = True
                    break
        return codewords_list, cov_array

    def simulated_annealing(codewords_list, cov_array, T_start, cooling_rate, min_T, time_budget):
        T = T_start
        best_size = len(codewords_list)
        best_codewords = codewords_list[:]
        best_cov = cov_array.copy()
        
        while time.time() < time_budget and T > min_T:
            if len(codewords_list) == 0:
                break
                
            # Try deletion first
            i = random.randint(0, len(codewords_list) - 1)
            c = codewords_list[i]
            if np.min(cov_array[balls[c]]) > 1:
                cov_array[balls[c]] -= 1
                codewords_list.pop(i)
                if len(codewords_list) < best_size:
                    best_size = len(codewords_list)
                    best_codewords = codewords_list[:]
                    best_cov = cov_array.copy()
                T *= cooling_rate
                continue
                    
            # Try swap
            i = random.randint(0, len(codewords_list) - 1)
            c_old = codewords_list[i]
            c_new = random.randint(0, N - 1)
            
            if c_new == c_old:
                T *= cooling_rate
                continue

            cov_array[balls[c_old]] -= 1
            missing = np.where(cov_array < 1)[0]
            
            if len(missing) == 0:
                # Deletion was valid
                codewords_list.pop(i)
                if len(codewords_list) < best_size:
                    best_size = len(codewords_list)
                    best_codewords = codewords_list[:]
                    best_cov = cov_array.copy()
                T *= cooling_rate
            else:
                if mem[c_new, missing].all():
                    cov_array[balls[c_new]] += 1
                    codewords_list[i] = c_new
                    T *= cooling_rate
                else:
                    cov_array[balls[c_old]] += 1
                    T *= cooling_rate
        
        return best_codewords, best_cov, best_size

    def final_polish(codewords_list, cov_array):
        """Aggressive final reduction pass."""
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(codewords_list)):
                if time.time() > deadline - 5:
                    break
                c = codewords_list[i]
                if np.min(cov_array[balls[c]]) > 1:
                    cov_array[balls[c]] -= 1
                    codewords_list.pop(i)
                    changed = True
                    break
        return codewords_list, cov_array

    # Start from Tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    tet_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            tet_code.append(digits_to_idx(word))
            
    tet_cov = np.zeros(N, dtype=np.int32)
    for c in tet_code:
        tet_cov[balls[c]] += 1
    tet_code, tet_cov = greedy_reduce(tet_code, tet_cov)
    
    # Also try greedy from scratch as alternative start
    greedy_code = greedy_set_cover()
    greedy_cov = np.zeros(N, dtype=np.int32)
    for c in greedy_code:
        greedy_cov[balls[c]] += 1
    greedy_code, greedy_cov = greedy_reduce(greedy_code, greedy_cov)
    
    # Pick better starting point
    if len(greedy_code) < len(tet_code):
        start_code, start_cov = greedy_code, greedy_cov
    else:
        start_code, start_cov = tet_code, tet_cov
    
    all_best_codewords = [start_code]
    all_best_sizes = [len(start_code)]
    
    # Run multiple SA passes with different parameters
    sa_configs = [
        (3.0, 0.997, 1e-4, 20),
        (2.0, 0.998, 1e-5, 20),
        (5.0, 0.995, 1e-3, 20),
        (1.0, 0.999, 1e-6, 20),
        (4.0, 0.996, 1e-4, 15),
        (1.5, 0.9985, 1e-5, 15),
    ]
    
    for T_start, cooling_rate, min_T, time_budget in sa_configs:
        if time.time() >= deadline - 8:
            break
        
        cw = start_code[:]
        cv = start_cov.copy()
        
        best_cw, best_cv, best_sz = simulated_annealing(cw, cv, T_start, cooling_rate, min_T, time.time() + time_budget)
        
        all_best_codewords.append(best_cw)
        all_best_sizes.append(best_sz)
        
        # Also try SA from the best found so far if it improved
        if best_sz < min(all_best_sizes[:len(all_best_sizes)-1]) and time.time() < deadline - 12:
            cw2 = best_cw[:]
            cv2 = best_cv.copy()
            best_cw2, best_cv2, best_sz2 = simulated_annealing(cw2, cv2, T_start * 0.7, cooling_rate, min_T, time.time() + 12)
            all_best_codewords.append(best_cw2)
            all_best_sizes.append(best_sz2)
    
    # Find best
    best_idx = all_best_sizes.index(min(all_best_sizes))
    best_result = all_best_codewords[best_idx]
    
    # Final greedy reduce
    final_cov = np.zeros(N, dtype=np.int32)
    for c in best_result:
        final_cov[balls[c]] += 1
    best_result, final_cov = greedy_reduce(best_result, final_cov)
    
    # Final polish pass
    if time.time() < deadline - 3:
        best_result, final_cov = final_polish(best_result, final_cov)
    
    result = [idx_to_digits(c) for c in best_result]
    return result

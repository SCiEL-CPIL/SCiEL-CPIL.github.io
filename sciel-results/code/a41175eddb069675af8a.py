import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    def compute_score(U):
        if not U or len(U) <= 1:
            return 1.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        u_arr = np.array(U, dtype=np.int64)
        n = len(u_arr)
        
        # FFT size
        # Max sum is 2*max_u. Max diff is max_u - 0 = max_u (and negative).
        # We need to resolve indices from -max_u to 2*max_u.
        # Length of signal for sum: 2*max_u + 1.
        # Length of signal for diff: 2*max_u + 1 (lags -max_u to max_u).
        # Let's use a power of 2 >= 2 * (2*max_u + 1) to be safe for convolution.
        L = 1
        while L < 4 * max_u + 4:
            L *= 2
        
        x = np.zeros(L, dtype=np.float64)
        for u in U:
            x[u] = 1.0
            
        X = np.fft.fft(x)
        
        # Sumset: x * x
        C_sum = np.fft.ifft(X * X)
        # Count non-zero coefficients
        # Due to floating point errors, use threshold
        sum_counts = np.round(np.real(C_sum)).astype(np.int64)
        size_sum = np.sum(sum_counts > 0)
        
        # Diffset: x * reverse(x) is equivalent to autocorrelation
        # Autocorrelation R[k] = sum x[i] x[i+k]
        # np.fft.ifft(X * conj(X)) gives circular autocorrelation.
        C_diff = np.fft.ifft(X * np.conj(X))
        diff_counts = np.round(np.real(C_diff)).astype(np.int64)
        size_diff = np.sum(diff_counts > 0)
        
        if size_sum == 0:
            return 1.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
            
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        return 1.0 + math.log(ratio) / denom

    # Initial candidates: Known Golomb Rulers
    golomb_rulers = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 25],
        8: [0, 1, 4, 10, 12, 17, 25, 29],
        9: [0, 1, 4, 10, 12, 17, 25, 29, 34],
        10: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41],
    }
    
    initial_sets = []
    for n in range(1, 11):
        if n in golomb_rulers:
            initial_sets.append(golomb_rulers[n])
            
    # Add some random sparse sets
    for _ in range(20):
        n = random.randint(3, 8)
        M = random.randint(10, 50)
        U = {0}
        for _ in range(n-1):
            U.add(random.randint(1, M))
        initial_sets.append(list(U))
        
    best_score = 0.0
    best_set = [0]
    
    for s in initial_sets:
        score = compute_score(s)
        if score > best_score:
            best_score = score
            best_set = list(s)
            
    best_overall_score = best_score
    best_overall_set = best_set
    
    # Simulated Annealing / Local Search
    # We use the fast FFT scorer to allow many iterations
    
    current_set = list(best_overall_set)
    current_score = best_overall_score
    
    T = 0.1  # Initial temperature
    cooling_rate = 0.999
    min_T = 0.001
    
    while time.time() < deadline - 5:
        if T < min_T:
            T = min_T
            
        # Generate neighbor
        neighbor = list(current_set)
        op = random.randint(0, 4)
        
        if op == 0 and len(neighbor) > 1:
            # Remove
            idx = random.randint(1, len(neighbor) - 1)
            neighbor.pop(idx)
            if len(neighbor) == 0:
                continue
        elif op == 1:
            # Add
            max_val = max(neighbor) if neighbor else 0
            # Bias towards extending the range slightly or filling gaps
            # Try values in [1, max_val + 10]
            new_val = random.randint(1, max_val + 10)
            if new_val not in neighbor:
                neighbor.append(new_val)
                neighbor.sort()
            else:
                continue
        elif op == 2:
            # Modify
            if len(neighbor) > 1:
                idx = random.randint(1, len(neighbor) - 1)
                old_val = neighbor[idx]
                # Small shift
                shift = random.randint(-5, 5)
                new_val = old_val + shift
                if new_val >= 0 and new_val not in neighbor:
                    neighbor[idx] = new_val
                    neighbor.sort()
                else:
                    continue
            else:
                continue
        elif op == 3:
            # Swap
            if len(neighbor) > 1:
                idx = random.randint(1, len(neighbor) - 1)
                old_val = neighbor.pop(idx)
                max_val = max(neighbor) if neighbor else 0
                new_val = random.randint(1, max_val + 10)
                if new_val not in neighbor:
                    neighbor.append(new_val)
                    neighbor.sort()
                else:
                    neighbor.append(old_val)
                    neighbor.sort()
                    continue
            else:
                continue
        else:
            # Randomize a bit
            if len(neighbor) > 2:
                # Swap two random non-zero elements? Or just add/remove
                idx1 = random.randint(1, len(neighbor) - 1)
                idx2 = random.randint(1, len(neighbor) - 1)
                if idx1 != idx2:
                    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                else:
                    continue
            else:
                continue
                
        if not neighbor or 0 not in neighbor:
            continue
            
        new_score = compute_score(neighbor)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_set = neighbor
            current_score = new_score
            if current_score > best_overall_score:
                best_overall_score = current_score
                best_overall_set = list(current_set)
                
        T *= cooling_rate
        
    # Final polish: Exhaustive check of small neighborhoods
    if time.time() < deadline - 2:
        polish_set = list(best_overall_set)
        polish_score = compute_score(polish_set)
        
        # Try removing each element
        for i in range(1, len(polish_set)):
            candidate = list(polish_set)
            candidate.pop(i)
            if len(candidate) > 0 and 0 in candidate:
                candidate.sort()
                s = compute_score(candidate)
                if s > polish_score:
                    polish_set = candidate
                    polish_score = s
        
        # Try adding small values
        max_val = max(polish_set)
        for v in range(1, max_val + 5):
            if v not in polish_set:
                candidate = list(polish_set)
                candidate.append(v)
                candidate.sort()
                s = compute_score(candidate)
                if s > polish_score:
                    polish_set = candidate
                    polish_score = s
        
        if polish_score > best_overall_score:
            best_overall_score = polish_score
            best_overall_set = polish_set
            
    return best_overall_set

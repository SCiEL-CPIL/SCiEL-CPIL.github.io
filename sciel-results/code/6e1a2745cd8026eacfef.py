import numpy as np
import time

N = 100

def calculate_score(h):
    # h is a numpy array of length N
    # Calculate overlap for all shifts
    # Overlap k = sum(h[i] * (1 - h[(i-k) % N]))
    # We want to minimize the max overlap.
    
    # Efficient calculation using FFT?
    # corr = ifft(fft(h) * fft(1-h)) ...
    # Let's just do it directly for N=100, it's fast.
    
    ones_minus_h = 1.0 - h
    max_overlap = 0.0
    for k in range(N):
        # Shift h by k
        # overlap = sum(h * ones_minus_h_shifted)
        # ones_minus_h_shifted[i] = ones_minus_h[(i-k) % N]
        # This is equivalent to sum(h[i] * ones_minus_h[(i-k) % N])
        # Let's use np.roll
        shifted = np.roll(ones_minus_h, k)
        overlap = np.dot(h, shifted)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initial guess: constant 0.5
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = calculate_score(best_h)
    
    # Local Search / Simulated Annealing
    # We want to minimize max_overlap
    # Score = max_overlap
    
    T = 0.1  # Temperature
    T_min = 0.001
    cooling = 0.99
    
    # Random initial perturbation to escape local optima if any
    # But 0.5 is likely the global optimum.
    # Let's try a few random restarts.
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        if restart == 0:
            h = best_h.copy()
        else:
            h = np.random.rand(N) * 0.5 + 0.25 # Random start in [0.25, 0.75]
            # Enforce sum constraint roughly
            s = np.sum(h)
            h = h * (N/2.0) / s
            h = np.clip(h, 0, 1)
            # Re-adjust sum exactly?
            diff = N/2.0 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0, 1)
            
        current_score = calculate_score(h)
        
        # Simple Hill Climbing with restarts
        steps = 1000
        for i in range(steps):
            if time.time() > deadline:
                break
                
            # Pick a random index to perturb
            idx = np.random.randint(N)
            delta = np.random.randn() * 0.1
            old_val = h[idx]
            new_val = np.clip(old_val + delta, 0, 1)
            
            # Enforce sum constraint by adjusting another index or all?
            # Adjusting one index breaks the sum.
            # We can adjust a random other index to compensate.
            if new_val == old_val:
                continue
                
            diff = old_val - new_val
            if abs(diff) < 1e-9:
                continue
                
            # Pick a second index to adjust
            idx2 = np.random.randint(N)
            if idx2 == idx:
                continue
                
            old_val2 = h[idx2]
            # We need h[idx2] to change by -diff to keep sum constant
            new_val2 = old_val2 - diff
            if new_val2 < 0 or new_val2 > 1:
                # Try to find a valid idx2
                # Just skip this move or use a small delta
                continue
                
            h[idx] = new_val
            h[idx2] = new_val2
            
            new_score = calculate_score(h)
            
            # Accept if better or with probability exp(-(new_score - current_score)/T)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                prob = np.exp(-(new_score - current_score) / T)
                if np.random.rand() < prob:
                    current_score = new_score
            T *= cooling
            if T < T_min:
                T = T_min

    return best_h

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 100

def _calculate_max_overlap(h):
    """
    Calculate the maximum overlap of h with shifted 1-h.
    Overlap k = sum(h[i] * (1 - h[(i-k) % N]))
    """
    ones_minus_h = 1.0 - h
    max_overlap = 0.0
    # For N=100, N shifts is 100.
    # Each shift involves a dot product of length 100.
    # Total 10,000 operations. Very fast.
    for k in range(N):
        shifted = np.roll(ones_minus_h, k)
        overlap = np.dot(h, shifted)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial guess: constant 0.5
    # This is the theoretical optimum for the continuous case.
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _calculate_max_overlap(best_h)
    
    # Local Search
    # We try to see if any discrete structure beats the constant.
    # Given the averaging argument, it's unlikely, but we try.
    
    T = 0.05
    T_min = 0.001
    cooling = 0.995
    max_iterations = 2000
    
    for restart in range(3):
        if time.time() > deadline:
            break
            
        if restart == 0:
            h = best_h.copy()
        else:
            # Random initialization
            h = np.random.rand(N)
            # Normalize to sum N/2
            s = np.sum(h)
            h = h * (N/2.0) / s
            h = np.clip(h, 0, 1)
            # Correct sum
            diff = N/2.0 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0, 1)
            
        current_score = _calculate_max_overlap(h)
        
        for i in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Perturb two indices to maintain sum
            idx1 = np.random.randint(N)
            idx2 = np.random.randint(N)
            if idx1 == idx2:
                continue
                
            delta = np.random.randn() * 0.2
            v1 = h[idx1] + delta
            v2 = h[idx2] - delta
            
            if v1 < 0 or v1 > 1 or v2 < 0 or v2 > 1:
                continue
                
            h[idx1] = v1
            h[idx2] = v2
            
            new_score = _calculate_max_overlap(h)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                prob = np.exp(-(new_score - current_score) / T)
                if np.random.rand() < prob:
                    current_score = new_score
            T *= cooling
            if T < T_min:
                T = T_min

    return best_h
# EVOLVE-BLOCK-END

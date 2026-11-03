import numpy as np
import time
from scipy.fft import fft, ifft

N = 100
M = N // 2

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin, evaluator kills at 150s
    
    # Initialize x (binary vector of length M)
    # Start with a random pattern or a known good pattern like all 0s
    x = np.random.randint(0, 2, M)
    
    best_h = None
    best_score = float('inf')
    
    # Helper to compute h from x
    def get_h(x):
        h = np.empty(N)
        h[0::2] = x
        h[1::2] = 1 - x
        return h
    
    # Helper to compute score (max overlap)
    def compute_score(h):
        # Overlap V(k) = sum h_i (1 - h_{i+k}) = N/2 - A(k)
        # We want to minimize max(V(k)), which is N/2 - min(A(k))
        # So minimizing max(V(k)) is equivalent to maximizing min(A(k))
        # Let's calculate A(k) using FFT
        # A(k) = sum h_i h_{i+k}
        # Using FFT: A = ifft(fft(h) * conj(fft(h)))
        H = fft(h)
        A = np.real(ifft(H * np.conj(H)))
        # A[0] is energy. We care about k != 0 usually, but "maximum overlap" 
        # likely includes all shifts or just non-zero. 
        # "overlap of h with shifted 1-h"
        # If k=0, V(0) = sum h_i (1-h_i) = sum h_i - sum h_i^2.
        # For binary h, sum h_i^2 = sum h_i = N/2. So V(0) = 0.
        # So k=0 is always 0. We look at k=1..N-1.
        # We want to minimize max_{k=1..N-1} V(k).
        # V(k) = N/2 - A(k).
        # min max V(k) <=> max min A(k).
        
        # Exclude k=0
        A_sidelobes = A[1:]
        min_A = np.min(A_sidelobes)
        max_overlap = N/2 - min_A
        return max_overlap, min_A

    # Initial evaluation
    h = get_h(x)
    score, _ = compute_score(h)
    best_score = score
    best_h = h.copy()
    
    # Local Search
    # We want to maximize min(A(k)).
    # Objective: maximize f(x) = min_{k=1..N-1} A(k)
    
    while time.time() < deadline - 5.0:
        h = get_h(x)
        H = fft(h)
        A = np.real(ifft(H * np.conj(H)))
        
        # Find the "worst" shift k (min A)
        # Ignore k=0
        A_sidelobes = A[1:]
        k_worst = np.argmin(A_sidelobes) + 1
        
        # Try flipping each bit in x to see if it improves min A
        # Or just try to improve A(k_worst)
        
        improved = False
        # To be efficient, we can just pick a random bit or the bit that most improves A(k_worst)
        # Let's try all M bits
        best_bit = -1
        best_new_min_A = -np.inf
        current_min_A = np.min(A_sidelobes)
        
        # Pre-compute deltas for speed? 
        # Flipping x_j changes h_{2j} and h_{2j+1}.
        # Let's just recompute for each candidate bit. M=50, so 50 FFTs per iteration.
        # 50 * 100log100 is fast.
        
        for j in range(M):
            x[j] = 1 - x[j]
            h_trial = get_h(x)
            H_trial = fft(h_trial)
            A_trial = np.real(ifft(H_trial * np.conj(H_trial)))
            min_A_trial = np.min(A_trial[1:])
            
            # We want to maximize min_A
            # If min_A_trial > current_min_A, it's an improvement in the objective
            # But we might also care about the max overlap directly?
            # Minimizing max overlap is exactly maximizing min A.
            
            if min_A_trial > best_new_min_A:
                best_new_min_A = min_A_trial
                best_bit = j
            
            # Revert
            x[j] = 1 - x[j]
            
        # If we found a bit that improves the global min A
        if best_bit != -1 and best_new_min_A > current_min_A:
            x[best_bit] = 1 - x[best_bit]
            h = get_h(x)
            score, _ = compute_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
            improved = True
        else:
            # If no single bit flip improves the min A, try random moves or accept worse to escape local optima?
            # Simple hill climbing might get stuck. 
            # Let's add a simple simulated annealing step or just random restarts if stuck.
            # For now, let's try a random flip if stuck for a while? 
            # Or just pick a random bit to flip to explore.
            # Let's try to reduce the max V(k) specifically if min A doesn't improve?
            # Actually, if min A doesn't improve, the score (max V) doesn't improve.
            # We can try to improve the second worst A?
            
            # Simple strategy: Random walk with occasional backtracking?
            # Let's just do a random flip to keep exploring, but only keep if score improves or is equal?
            # No, that's just random search.
            
            # Better: Try to improve A(k_worst) specifically, even if it lowers other A's, 
            # but only if the new min A is not too much worse?
            # Let's stick to strict hill climbing on min A first.
            # If stuck, do a random perturbation.
            
            # Random perturbation
            if np.random.rand() < 0.1:
                j = np.random.randint(M)
                x[j] = 1 - x[j]
                h = get_h(x)
                score, _ = compute_score(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
            else:
                # Try flipping the bit that most improves A(k_worst)
                # even if it doesn't improve global min A?
                # This might lead to a better min A later.
                best_delta_A_worst = -np.inf
                best_bit_worst = -1
                A_worst_val = A[k_worst]
                
                for j in range(M):
                    x[j] = 1 - x[j]
                    h_trial = get_h(x)
                    H_trial = fft(h_trial)
                    A_trial = np.real(ifft(H_trial * np.conj(H_trial)))
                    # Check improvement in A(k_worst)
                    delta = A_trial[k_worst] - A_worst_val
                    if delta > best_delta_A_worst:
                        best_delta_A_worst = delta
                        best_bit_worst = j
                    x[j] = 1 - x[j]
                    
                if best_bit_worst != -1 and best_delta_A_worst > 0:
                    x[best_bit_worst] = 1 - x[best_bit_worst]
                    h = get_h(x)
                    score, _ = compute_score(h)
                    if score < best_score:
                        best_score = score
                        best_h = h.copy()

    return best_h

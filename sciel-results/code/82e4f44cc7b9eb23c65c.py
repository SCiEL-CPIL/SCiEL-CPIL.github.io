import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 128
    min_sum = 128
    max_sum = 256
    
    # State
    # V: indicator vector of A
    # C: convolution vector (ordered pair counts)
    
    def init_state(A):
        V = np.zeros(n + 1, dtype=np.int32)
        V[A] = 1
        C = np.convolve(V, V, mode='full')
        return V, C, list(A)

    def calc_peak(C):
        # C contains ordered counts
        # We need unordered counts for k in [128, 256]
        # R_k = ceil(C[k] / 2)
        # But we must check coverage: C[k] > 0
        relevant_C = C[min_sum:max_sum+1]
        
        # Uncovered if C[k] == 0
        uncovered = np.sum(relevant_C == 0)
        
        # Peak unordered
        # R_k = (C[k] + 1) // 2
        peak = np.max((relevant_C + 1) // 2)
        
        return peak, uncovered

    def try_swap(V, C, A, a_out, a_in):
        """
        Returns (new_V, new_C, new_A, new_peak, new_uncovered)
        """
        # Check if a_in is already in A
        if V[a_in] == 1:
            return None, None, None, None, None
            
        # Calculate new C using the formula
        # C'_k = C_k - 2 V_{k-a_out} + 2 V_{k-a_in} + delta(2*a_out, k) - 2 delta(a_out+a_in, k) + delta(2*a_in, k)
        
        new_C = C.copy()
        
        # We only need to update k where terms are non-zero.
        # But for simplicity and safety, let's update the whole relevant range or use vectorized ops.
        # Vectorized update:
        
        # Term 1: C_k (already in new_C)
        
        # Term 2: - 2 V_{k-a_out}
        # V is length n+1.
        # For k in 0..2n, k-a_out is valid if 0 <= k-a_out <= n.
        # So k in [a_out, a_out+n].
        # We can create a shifted V.
        
        # Let's just compute the delta for the relevant range [0, 256]
        # Actually, C is length 2n+1 = 257.
        
        # Shifted V for a_out
        # V_shifted_a_out[k] = V[k - a_out] if 0 <= k - a_out <= n else 0
        # This is V shifted right by a_out.
        
        # It might be faster to just do the convolution update in Python loops for the specific indices?
        # No, vectorized is better.
        
        # Let's create the full update vector
        # u = -2 * shift(V, a_out) + 2 * shift(V, a_in)
        # v = delta(2*a_out) - 2*delta(a_out+a_in) + delta(2*a_in)
        
        def shift_vec(vec, shift):
            new_vec = np.zeros_like(vec)
            # vec is length n+1
            # We want new_vec[k] = vec[k-shift]
            # Valid k: shift <= k <= shift + n
            start = shift
            end = shift + n + 1
            # Clip to new_vec size (2n+1)
            end = min(end, len(new_vec))
            start = max(0, start)
            if start < end:
                new_vec[start:end] = vec[0:end-start]
            return new_vec

        shift_out = shift_vec(V, a_out)
        shift_in = shift_vec(V, a_in)
        
        delta_vec = np.zeros_like(C)
        delta_vec[2*a_out] += 1
        delta_vec[2*a_in] += 1
        delta_vec[a_out+a_in] -= 2
        
        new_C = C - 2 * shift_out + 2 * shift_in + delta_vec
        
        # Update V
        new_V = V.copy()
        new_V[a_out] = 0
        new_V[a_in] = 1
        
        # Update A
        new_A = list(A)
        new_A.remove(a_out)
        new_A.append(a_in)
        
        peak, uncovered = calc_peak(new_C)
        
        return new_V, new_C, new_A, peak, uncovered

    # Initial Solution
    random.seed(42)
    
    best_A = None
    best_peak = float('inf')
    
    # Try to find initial covering set
    for size in [20, 25, 30, 40]:
        if time.time() > deadline - 10:
            break
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            A = random.sample(range(0, n + 1), size)
            A = sorted(A)
            V, C, _ = init_state(A)
            peak, uncovered = calc_peak(C)
            if uncovered == 0 and peak < best_peak:
                best_peak = peak
                best_A = A.copy()
                best_V = V
                best_C = C
                break
        if best_A is not None:
            break
            
    if best_A is None:
        best_A = list(range(64, 129))
        best_V, best_C, _ = init_state(best_A)
        best_peak, _ = calc_peak(best_C)

    # Simulated Annealing
    current_A = best_A
    current_V = best_V
    current_C = best_C
    current_peak, current_uncovered = calc_peak(current_C)
    
    T = 5.0
    cooling = 0.999
    
    while time.time() < deadline - 1:
        if T < 0.1:
            T = 1.0
            
        # Pick swap
        if len(current_A) > 0:
            a_out = random.choice(current_A)
            a_in = random.randint(0, n)
            
            if a_in == a_out or current_V[a_in] == 1:
                continue
                
            new_V, new_C, new_A, new_peak, new_uncovered = try_swap(current_V, current_C, current_A, a_out, a_in)
            
            if new_V is None:
                continue
                
            if new_uncovered == 0:
                if new_peak < current_peak:
                    current_V, current_C, current_A = new_V, new_C, new_A
                    current_peak, current_uncovered = new_peak, 0
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = current_A.copy()
                else:
                    # Accept with probability
                    # Delta = new_peak - current_peak
                    if new_peak == current_peak:
                        if random.random() < 0.1: # Low prob to accept equal
                            current_V, current_C, current_A = new_V, new_C, new_A
                    else:
                        # Reject worse
                        pass
            else:
                # Reject uncovered
                pass
                
        T *= cooling

    return best_A

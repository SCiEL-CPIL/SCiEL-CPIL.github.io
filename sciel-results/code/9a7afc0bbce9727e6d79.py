"""
Third autocorrelation inequality: minimize the upper bound on C3.
The score is seed/value where value is the raw upper bound (lower is better).
We search over signed step functions with 400 intervals on [-1/4, 1/4].
"""
import numpy as np
import time

N = 400
# Safety margin: 150s kill, aim to finish by 145s to be safe.
# The prompt says 120s in one place and 150s in another.
# "Each evaluation must finish within 120 s."
# "the evaluator KILLS the program after 150s wall-clock."
# Let's use 115s as a hard stop for computation to ensure we return well within 120s.
TIME_LIMIT = 115.0

def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    Uses a vectorized approach for speed: computes all lag sums simultaneously.
    Metric: dx * max_{k!=0} |sum_i h_i h_{i+k}|
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    if n == 0:
        return 0.0
    dx = 0.5 / n
    
    # We need max_{k=1..n-1} |sum_{i=0}^{n-1-k} h[i] h[i+k]|
    # Construct a matrix M where M[k, i] = h[i] * h[i+k] for valid i.
    # This is a bit memory heavy for large N, but N=400 is small (400x400 = 160k floats).
    # However, we can do it more efficiently by iterating k and using vectorized dot products,
    # or just using a loop which is fast enough in Python if the inner op is C-level (numpy).
    
    # Let's use a loop over lags k, which is O(N^2) operations but with low overhead per op.
    # For N=400, 400 iterations of a vectorized sum is very fast.
    max_sidelobe = 0.0
    for k in range(1, n):
        # sum h[i] * h[i+k] for i=0 to n-1-k
        # h[:n-k] * h[k:]
        val = np.dot(h[:n-k], h[k:])
        abs_val = abs(val)
        if abs_val > max_sidelobe:
            max_sidelobe = abs_val
            
    return dx * max_sidelobe

def _local_search(heights, time_limit=10.0):
    """
    Local search with simple annealing (allowing flat/worse moves occasionally).
    Uses vectorized bound calculation for speed.
    """
    h = np.array(heights, dtype=np.float64)
    n = len(h)
    current_bound = _compute_c3_bound(h)
    best_h = h.copy()
    best_bound = current_bound
    
    start_time = time.time()
    # Temperature for "annealing" - probability of accepting non-improving moves
    # Start high, decay
    temp = 0.5 
    
    while (time.time() - start_time) < time_limit:
        # Decay temperature
        temp *= 0.99
        
        best_gain = -np.inf
        best_idx = -1
        best_new_bound = current_bound
        best_is_improvement = False
        
        # Find the best flip
        for j in range(n):
            old_val = h[j]
            h[j] = -old_val
            new_bound = _compute_c3_bound(h)
            h[j] = old_val
            
            gain = current_bound - new_bound
            # Accept if gain is best, or with some probability if not improving (annealing)
            # For simplicity in this loop, we just find the absolute best move first.
            # Then decide if we accept it.
            if gain > best_gain:
                best_gain = gain
                best_idx = j
                best_new_bound = new_bound
                best_is_improvement = (gain > 0)
        
        if best_idx == -1:
            break
            
        # Acceptance criterion
        accept = False
        if best_is_improvement:
            accept = True
        else:
            # Accept worse move with probability based on 'temp' and 'gain'
            # gain is negative here
            # P = exp(gain / temp) -- but gain is in bound units, temp is abstract.
            # Let's scale: if gain > -0.1, accept with prob 0.5 * temp
            # Simple heuristic:
            if temp > 0.01 and np.random.rand() < temp * 0.5:
                accept = True
                # To prevent getting stuck in very bad local minima, only accept small worsenings
                if best_gain < -1.0: 
                    accept = False

        if accept:
            h[best_idx] = -h[best_idx]
            current_bound = best_new_bound
            
            if current_bound < best_bound:
                best_bound = current_bound
                best_h = h.copy()
                
    return best_h, best_bound

def _generate_chirp_candidates(rng, count=10):
    """
    Generate candidates based on chirp signals (linear and quadratic phase).
    h[i] = sign( sum_k A_k * exp(1j * (a_k * i^2 + b_k * i)) )
    """
    candidates = []
    t = np.arange(N)
    
    for _ in range(count):
        # Random quadratic phase (chirp)
        # f(t) = exp(1j * (alpha * t^2 + beta * t))
        alpha = rng.uniform(0.1, 2.0)
        beta = rng.uniform(0.0, 2.0 * np.pi)
        
        # Add a bit of noise or a second chirp to break symmetry?
        # Or just use the sign of the real part of a complex chirp.
        signal_val = np.exp(1j * (alpha * t**2 + beta * t))
        
        # Option 1: Sign of real part
        h1 = np.sign(signal_val.real)
        # Ensure no zeros
        h1[h1 == 0] = 1.0
        candidates.append(h1)
        
        # Option 2: Sum of two chirps
        alpha2 = rng.uniform(0.1, 2.0)
        beta2 = rng.uniform(0.0, 2.0 * np.pi)
        signal_val2 = np.exp(1j * (alpha2 * t**2 + beta2 * t))
        h2 = np.sign(signal_val.real + signal_val2.real)
        h2[h2 == 0] = 1.0
        candidates.append(h2)
        
        # Option 3: Zadoff-Chu like (quadratic phase only, specific alpha)
        # For N=400, a standard ZC sequence has alpha = 1/N or similar.
        # Let's try alpha = 2*pi*k/N for small k.
        for k in [1, 2, 3]:
            alpha_zc = 2.0 * np.pi * k / N
            h_zc = np.sign(np.exp(1j * alpha_zc * t**2).real)
            h_zc[h_zc == 0] = 1.0
            candidates.append(h_zc)
            
    return candidates

def construct():
    start_time = time.time()
    
    # Initialize with random signs
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_bound = _compute_c3_bound(best_h)
    
    # 1. Try Thue-Morse initialization
    # Thue-Morse sequence has good low autocorrelation properties
    tm = np.zeros(N)
    # Generate Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    for i in range(N):
        tm[i] = -1.0 if (bin(i).count('1') % 2 == 0) else 1.0
    bound = _compute_c3_bound(tm)
    if bound < best_bound:
        best_bound = bound
        best_h = tm.copy()

    # 2. Try a few random initializations
    for trial in range(10):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        h = rng.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
            
    # 3. Try chirp-based initializations
    chirp_candidates = _generate_chirp_candidates(rng, count=5)
    for h in chirp_candidates:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()

    # 4. Local search from the best candidate found so far
    remaining_time = TIME_LIMIT - (time.time() - start_time)
    if remaining_time > 5:
        # Allocate most of the time to the best candidate
        alloc_time = remaining_time - 2
        best_h, best_bound = _local_search(best_h, time_limit=alloc_time)
        
    # 5. Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 5:
        # Try a few more random restarts with short local search
        while time.time() - start_time < TIME_LIMIT - 2:
            h = rng.choice([-1.0, 1.0], size=N)
            bound = _compute_c3_bound(h)
            
            # Quick check: if it's already better than best, do a short polish
            if bound < best_bound:
                short_time = min(1.0, TIME_LIMIT - (time.time() - start_time) - 1)
                if short_time > 0.2:
                    h, bound = _local_search(h, time_limit=short_time)
                    if bound < best_bound:
                        best_bound = bound
                        best_h = h.copy()
            elif time.time() - start_time > TIME_LIMIT - 5:
                break
            # If not better, just move to next random start quickly
            
    # Final check
    final_bound = _compute_c3_bound(best_h)
    if final_bound < best_bound:
        best_bound = final_bound
        
    return best_h

# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"C3 bound: {_compute_c3_bound(h)}")
    print(f"Score: {1.0 / _compute_c3_bound(h)}")

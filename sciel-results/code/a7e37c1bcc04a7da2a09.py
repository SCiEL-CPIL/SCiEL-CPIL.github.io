import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Finds the 24 coefficients in {+1, -1} that minimize the sup-norm of P(z) on |z|=1.
    Uses Simulated Annealing with a high-fidelity objective function.
    """
    N = 24
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    # 1. Objective Function: Accurate estimation of max |P(z)| / sqrt(N)
    # We use dense sampling to find candidate peaks, then refine them.
    def objective(coeffs):
        # coeffs: np.array of shape (N,) with values +/-1
        # Evaluate P(z) on the unit circle.
        # Using FFT for speed: P(z) = sum c_k z^k. 
        # Note: numpy.fft.ifft computes sum x_k exp(2j pi k n / L).
        # We want sum c_k exp(-i k theta) or similar.
        # Let z = exp(-i theta). P(z) = sum c_k exp(-i k theta).
        # This is the Fourier transform of the sequence c_k.
        # np.fft.fft(coeffs) gives sum c_k exp(-2j pi k n / L).
        # So if we set theta = 2 pi n / L, fft gives us the values.
        
        L = 4096  # Dense grid size
        vals = np.fft.fft(coeffs)
        
        # Find the top few peaks to refine
        # We care about magnitude
        mags = np.abs(vals)
        
        # Get indices of the largest magnitudes
        # We need to check if the peak is at a grid point or between points.
        # Golden section search on the interval surrounding the peak.
        
        # To speed up, we can just take the max on the grid first.
        # But the true max might be higher.
        # Let's refine the top 5 peaks.
        
        top_k = 5
        # Get top k indices
        # np.argsort is O(L log L). L=4096 is fast.
        # But we only need top k.
        # For simplicity and speed, let's just take the max index first.
        # If we have time, we can do more.
        
        # Actually, let's do a quick refinement for the single highest peak on the grid.
        # This is usually sufficient for convergence in SA.
        
        idx_max = np.argmax(mags)
        
        # Refine around idx_max
        # The grid points are theta_n = 2 * pi * n / L
        # The peak is near theta_max = 2 * pi * idx_max / L
        
        # We will perform a few steps of golden section search 
        # on the function f(theta) = |P(exp(-i theta))|
        # in the interval [theta_max - delta, theta_max + delta]
        
        # To evaluate P at arbitrary theta:
        # P(theta) = sum c_k exp(-i k theta)
        # This is a dot product: coeffs @ exp(-i * k * theta)
        
        k = np.arange(N)
        
        def val_at_theta(theta):
            # Vectorized evaluation
            # exp(-i * k * theta)
            # coeffs is (N,), k is (N,)
            return np.abs(np.sum(coeffs * np.exp(-1j * k * theta)))
            
        # Interval width
        # The grid spacing is 2*pi/L. Let's search +/- 2 grid spacings.
        delta_theta = 2 * (2 * np.pi / L)
        a = (2 * np.pi * idx_max / L) - delta_theta
        b = (2 * np.pi * idx_max / L) + delta_theta
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2  # 0.618...
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = val_at_theta(c)
        fd = val_at_theta(d)
        
        # We want to maximize the magnitude
        for _ in range(20):  # 20 iterations gives ~1e-5 accuracy on theta, sufficient for 1e-9 score?
            # Check time
            if time.time() - start_time > time_limit - 5:
                break
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = val_at_theta(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = val_at_theta(c)
                
        max_val = max(fc, fd)
        # Also check the endpoints just in case, though unlikely to be max
        # max_val = max(max_val, val_at_theta(a), val_at_theta(b))
        
        return max_val / np.sqrt(N)

    # 2. Initialization: Rudin-Shapiro
    # Generate RS coefficients
    def get_rs_coeffs(n):
        c = np.array([1])
        while len(c) < n:
            # Rudin-Shapiro recursion: P -> P + Q, Q -> P - Q
            # But we need to maintain length powers of 2?
            # Standard RS: 
            # P_0 = 1
            # P_{k+1} = P_k + Q_k
            # Q_{k+1} = P_k - Q_k
            # Let's just use the known property or generate directly.
            # Actually, simpler:
            # The RS polynomial of degree 2^n - 1.
            # For N=24, we can take the first 24 of a longer RS sequence.
            pass
            
        # Let's just hardcode a good RS-like seed or use random.
        # RS seed is known to be good.
        # Let's generate RS of length 32 and truncate.
        p = np.array([1])
        q = np.array([1])
        while len(p) < 32:
            p_new = np.concatenate([p, q])
            q_new = np.concatenate([p, -q])
            p, q = p_new, q_new
        return p[:n]

    coeffs = get_rs_coeffs(N).astype(np.float64)
    
    best_coeffs = coeffs.copy()
    best_score = objective(best_coeffs)
    
    # 3. Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute k for speed in val_at_theta?
    # val_at_theta is called frequently.
    # We can optimize by using complex exponentials.
    
    k_vec = np.arange(N)
    
    def fast_val(coeffs, theta):
        # coeffs is float array +/-1
        # theta is float
        # return |sum coeffs * exp(-i k theta)|
        # Use np.dot
        # exp(-i * k_vec * theta)
        return np.abs(np.dot(coeffs, np.exp(-1j * k_vec * theta)))
        
    # Rewrite objective to use fast_val
    def objective_fast(coeffs):
        L = 4096
        vals = np.fft.fft(coeffs)
        mags = np.abs(vals)
        idx_max = np.argmax(mags)
        
        delta_theta = 2 * (2 * np.pi / L)
        center = 2 * np.pi * idx_max / L
        a = center - delta_theta
        b = center + delta_theta
        
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = fast_val(coeffs, c)
        fd = fast_val(coeffs, d)
        
        for _ in range(15): # Reduced iterations for speed
            if time.time() - start_time > time_limit - 5:
                break
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = fast_val(coeffs, d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = fast_val(coeffs, c)
                
        max_val = max(fc, fd)
        return max_val / np.sqrt(N)

    # Re-evaluate best with faster method
    best_score = objective_fast(best_coeffs)
    
    # SA Loop
    # Number of steps depends on time.
    # Each step: pick random index, flip, evaluate.
    # Evaluation is ~15 fast_val calls + FFT.
    # FFT is 4096 points. Fast.
    # fast_val is N=24 dot products. Very fast.
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick a random coefficient to flip
        idx = np.random.randint(N)
        old_val = coeffs[idx]
        new_val = -old_val
        
        coeffs[idx] = new_val
        
        new_score = objective_fast(coeffs)
        
        delta = new_score - best_score
        
        if delta < 0 or np.random.random() < math.exp(-delta / T):
            # Accept
            if delta < 0:
                best_score = new_score
                best_coeffs = coeffs.copy()
            # else: accept but not best
            
        else:
            # Reject
            coeffs[idx] = old_val
            
        T *= cooling_rate
        
        # Occasionally, if stuck, we might want to do a restart or different move
        # But simple SA usually works for N=24.
        
    return best_coeffs.astype(int)
# EVOLVE-BLOCK-END

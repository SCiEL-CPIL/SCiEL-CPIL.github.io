import numpy as np
import time

N = 400
DEADLINE = 140.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound (max absolute autoconvolution) for a signed step function.
    Uses exact convolution for speed and accuracy.
    """
    # signed_step is an array of N values in {-1, 1}
    # We want the autoconvolution of the step function.
    # The step function is constant on intervals of width h = 1/N over [-1/4, 1/4].
    # Total width is 1/2.
    # Autoconvolution at lag k (where k is an integer index difference) is proportional to the sum of products.
    
    # Let's use np.convolve with 'full' mode.
    # The length of the result is 2*N - 1.
    # The indices of the result correspond to lags from -(N-1) to N-1.
    # The value at index i in the conv result corresponds to the sum over j of f[j] * f[i-j].
    # Wait, standard convolution (f * g)[n] = sum_k f[k] g[n-k].
    # For autoconvolution, g = f.
    # The "center" of the autoconvolution (lag 0) is at index N-1 in the 'full' output.
    
    conv = np.convolve(signed_step, signed_step, mode='full')
    
    # The width of each interval is h = 1/N.
    # The integral of the product over the overlapping interval of width w is w * (value1 * value2).
    # For a lag of k intervals, the overlap width is (N - |k|) * h.
    # The convolution sum gives the sum of products.
    # So the autoconvolution value at lag k is conv[N-1 + k] * (N - |k|) * h^2?
    # Let's check units.
    # f(x) is the step function.
    # (f * f)(t) = integral f(t-u) f(u) du.
    # If t = k * h, the overlap is (N - |k|) * h.
    # The integral is sum over the overlapping intervals of (f_i * f_j) * h.
    # conv[N-1+k] is sum of f_i * f_{i+k}.
    # So the integral is conv[N-1+k] * h.
    
    # Wait, h = 1/N.
    # So autoconv[k] = conv[N-1+k] * (1/N).
    
    # Let's verify for k=0.
    # conv[N-1] = sum f_i^2 = N (since f_i in {-1,1}).
    # autoconv[0] = N * (1/N) = 1.
    # The integral of f^2 over the domain is 1 (since width is 1/2 and height is 1, wait. Width is 1/2. Area is 1/2? No.)
    # Domain is [-1/4, 1/4], width 1/2.
    # f is +/- 1.
    # integral f^2 = 1/2.
    # My formula gave 1. Why?
    # h = (1/4 - (-1/4)) / N = 0.5 / N.
    # Ah, the interval width is 0.5/N, not 1/N.
    
    h = 0.5 / N
    autoconv = conv * h
    
    # The maximum absolute value is the C3 bound.
    return np.max(np.abs(autoconv))

def construct():
    start_time = time.time()
    
    # Initialize with all ones
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Simulated Annealing
    # We will run multiple restarts if time permits, but usually one long SA run with good cooling is better.
    # Or, a hybrid: quick SA to get close, then local search?
    # Given the "multi-flip" insight, SA with large perturbations is the key.
    
    np.random.seed(42)
    
    # Parameters
    T_start = 1.0
    T_end = 0.01
    # Number of iterations
    # We need to estimate speed.
    # _compute_c3 is very fast (~10-20us).
    # Let's aim for as many iterations as possible.
    
    # We will use a geometric cooling schedule.
    # We need to choose alpha (cooling rate).
    # Let's try to do ~100,000 iterations.
    n_iter = 100000
    alpha = (T_end / T_start) ** (1.0 / n_iter)
    
    current_step = best_step.copy()
    current_c3 = best_c3
    
    for i in range(n_iter):
        if time.time() - start_time > DEADLINE - 5.0:
            break
            
        T = T_start * (alpha ** i)
        
        # Perturbation: flip a random subset of indices
        # Size of subset can vary. Small subsets for fine tuning, larger for exploration.
        # Let's flip a random number of bits, say between 1 and 10.
        # Or just a fixed number, e.g., 5.
        # Let's try a random number of flips to mix exploration and exploitation.
        n_flips = np.random.randint(1, 11)
        
        # Select indices to flip
        if n_flips < N:
            flip_indices = np.random.choice(N, n_flips, replace=False)
        else:
            flip_indices = np.random.permutation(N)
            
        new_step = current_step.copy()
        new_step[flip_indices] *= -1
        
        new_c3 = _compute_c3(new_step)
        
        # Acceptance probability
        if new_c3 < current_c3:
            accept = True
        else:
            delta = new_c3 - current_c3
            if T > 0:
                # Avoid overflow in exp
                if delta < 50 * T: 
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
            else:
                prob = 0.0
            accept = (np.random.random() < prob)
        
        if accept:
            current_step = new_step
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_step = current_step.copy()
        
        # Adaptive cooling? Or fixed? Fixed is simpler and robust.
        
    return best_step

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Best C3: {_compute_c3(res):.6f}")
    print(f"Score: {1.0 / _compute_c3(res):.4f}")

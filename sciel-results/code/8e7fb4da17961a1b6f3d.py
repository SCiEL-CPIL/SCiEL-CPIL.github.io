import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Objective: Minimize sum of squares of sidelobes of autocorrelation
    # This is equivalent to minimizing L4 norm of FFT of x
    def objective(x):
        # x is real vector
        fx = np.fft.fft(x)
        # Sum of |fx|^4
        return np.sum(np.abs(fx)**4)
        
    def gradient(x):
        fx = np.fft.fft(x)
        # grad_fx = 2 * fx * |fx|^2
        # Note: for real x, fft is conjugate symmetric. 
        # The gradient of sum |fx|^4 wrt x is IFFT(2 * fx * |fx|^2)
        grad_fx = 2 * fx * np.abs(fx)**2
        return np.real(np.fft.ifft(grad_fx))

    # Initial guess: Random signs
    # Using multiple restarts if time permits
    best_x = None
    best_val = np.inf
    
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() - start_time > deadline:
            break
            
        # Random initialization
        x0 = np.random.choice([-1, 1], N).astype(float)
        
        # Optimization
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception as e:
            pass

    if best_x is None:
        # Fallback to random
        best_x = np.random.choice([-1, 1], N).astype(float)
        
    # Discretize
    s = np.sign(best_x)
    # Ensure no zeros (sign(0)=0, map to 1)
    s[s == 0] = 1
    
    # Local Search Refinement
    # We want to minimize the peak sidelobe, not just energy.
    # Let's compute the actual autocorrelation peak for the discrete sequence.
    
    def get_peak_sidelobe(seq):
        # Circular autocorrelation
        # We want min of max |C(k)| for k != 0
        ac = np.correlate(seq, seq, mode='full')
        # ac has length 2N-1. Center is at N-1.
        # C(k) corresponds to ac[N-1+k]
        # Sidelobes are k=1 to N-1
        # Due to symmetry, we can just check half
        # But let's just take the max of the whole thing excluding the center
        center = N - 1
        # Exclude center
        ac_sidelobes = np.concatenate([ac[:center], ac[center+1:]])
        return np.max(np.abs(ac_sidelobes))
    
    current_peak = get_peak_sidelobe(s)
    best_s = s.copy()
    best_peak = current_peak
    
    # Local search: flip bits that reduce peak
    # To be efficient, we can precompute the autocorrelation
    # Flipping bit i changes C(k) by -2 * s_i * s_{i+k} (if we consider linear)
    # For circular, it's more complex. 
    # Given N=400, O(N^2) per flip check is 160,000 ops. 
    # If we do 1000 flips, that's 160M ops. Might be slow in Python.
    # Let's use a smarter update or limit iterations.
    
    # Alternative: Use the energy minimization result as the primary driver.
    # The L4 minimizer is a good proxy for low peak.
    # Let's do a few rounds of bit-flips guided by the gradient of the peak?
    # No, let's just do simple hill climbing on the peak using FFT updates if possible,
    # or just accept the L4 minimizer if time is tight.
    
    # Let's try a few local improvements
    improved = True
    max_iter_local = 200
    iter_count = 0
    
    # Precompute FFT of s for faster updates? 
    # Actually, recalculating peak is O(N log N) using FFT.
    # 200 * 400 log 400 is cheap.
    
    while improved and iter_count < max_iter_local:
        improved = False
        iter_count += 1
        if time.time() - start_time > deadline - 10:
            break
            
        # Try flipping each bit
        # To speed up, we can sample a subset of bits if N is large, but N=400 is small enough.
        # However, checking all 400 bits every iteration is 400 * O(N log N) = 400 * 4000 = 1.6M ops.
        # 200 iterations * 1.6M = 320M ops. This might take ~10-30 seconds in Python.
        # Let's optimize the peak calculation.
        
        for i in range(N):
            if time.time() - start_time > deadline - 5:
                return best_s
                
            # Flip bit i
            s[i] *= -1
            
            # Calculate peak
            # Using FFT for circular autocorrelation
            fx = np.fft.fft(s)
            ac = np.real(np.fft.ifft(fx * fx))
            # Normalize? No, raw values.
            # Peak sidelobe
            # ac[0] is the sum of squares = N.
            # We want max |ac[k]| for k != 0
            # ac is circular.
            peak = np.max(np.abs(np.concatenate([ac[1:], ac[:0]]))) # Wait, ac[0] is lag 0.
            # Actually ac[k] is lag k.
            # We want max of |ac[k]| for k=1..N-1.
            # ac is length N.
            # lag 0 is ac[0].
            # lag 1 is ac[1].
            # lag N-1 is ac[N-1].
            # So we want max of ac[1:]
            # But wait, circular autocorrelation of real sequence is symmetric.
            # ac[k] = ac[N-k].
            # So we can just check ac[1:N//2+1]
            
            # Let's just take max of all except index 0
            sidelobes = np.abs(ac[1:])
            # Also ac[0] is the big one.
            # Is there a lag N? No, indices 0 to N-1.
            # So max of ac[1:]
            
            # Wait, the linear autocorrelation peak might be different from circular.
            # The task says "step function on [-1/4, 1/4]". This implies a continuous signal, 
            # but "400 equal intervals" and "signed step heights" implies a discrete sequence 
            # approximating a function. The "autoconvolution" of a step function is a piecewise 
            # linear function. Its peak is likely determined by the discrete autocorrelation 
            # sum.
            # For a step function $f(t) = s_i$ on interval $i$, the autoconvolution at lag $k \Delta t$ 
            # is proportional to $\sum s_i s_{i+k}$.
            # So minimizing the max of these sums is the goal.
            
            current_peak_trial = np.max(sidelobes)
            
            if current_peak_trial < best_peak:
                best_peak = current_peak_trial
                best_s = s.copy()
                improved = True
            else:
                # Revert
                s[i] *= -1
            # If we found an improvement, we break the loop over i? 
            # No, standard hill climbing tries all, picks best, or first improvement.
            # First improvement is faster.
            if improved:
                break # Restart the scan from 0? Or continue?
                # Restarting is safer to avoid local minima traps caused by order.
                # But for speed, let's just continue to the next i in the same pass?
                # No, if we break, we restart the for loop.
                break

    return best_s

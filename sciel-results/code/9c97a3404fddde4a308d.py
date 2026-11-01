import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 400

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Objective function for continuous relaxation
    # Minimize sum of squared sidelobes + penalty for deviating from +/- 1
    def objective(x):
        # Compute autocorrelation via FFT
        # x is shape (N,)
        # Autocorrelation c = conv(x, x, 'full')
        # We can use fftconvolve for speed, but N=400 is small.
        # np.convolve is fine.
        c = np.convolve(x, x, mode='full')
        
        # Main lobe is at index N-1
        # Sidelobes are all other indices
        # We want to minimize the energy of sidelobes
        # c[N-1] is the main lobe.
        
        # Create a mask for sidelobes
        # Indices 0 to 2N-2. Main lobe is N-1.
        c_sidelobes = np.delete(c, N-1)
        
        # Sum of squares
        energy = np.sum(c_sidelobes ** 2)
        
        # Penalty to keep x close to +/- 1
        # (x^2 - 1)^2 is 0 when x=1 or x=-1
        penalty = np.sum((x ** 2 - 1.0) ** 2)
        
        return energy + 1.0 * penalty

    best_f = None
    best_score = float('inf')
    
    # We will try a few random initializations
    # The hint suggests L-BFGS-B is good.
    
    for seed in range(5):
        if time.time() > deadline - 10.0:
            break
            
        np.random.seed(seed)
        # Initialize with random +/- 1
        x0 = np.random.choice([-1.0, 1.0], size=N)
        
        # Run L-BFGS-B
        # Limit iterations to ensure speed
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 20, 'ftol': 1e-4, 'gtol': 1e-4})
            
            # Get the optimized continuous solution
            x_opt = res.x
            
            # Round to discrete +/- 1
            f_discrete = np.sign(x_opt)
            # Handle zeros (shouldn't happen with penalty, but just in case)
            f_discrete[f_discrete == 0] = 1.0
            
            # Evaluate the discrete result
            c = np.convolve(f_discrete, f_discrete, mode='full')
            main_lobe = c[N-1]
            
            # Calculate the metric: "upper bound on C3"
            # Assuming this is Peak Sidelobe Level (PSL) or similar.
            # The prompt says "score is the upper bound on C3".
            # In many contexts for these problems, the score is the max sidelobe magnitude.
            # Let's assume score = max(|c_k| / |c_{N-1}|) for k != N-1.
            # Or just max(|c_k|) for k != N-1?
            # "raw value (lower is better)".
            # Let's compute the max sidelobe absolute value.
            abs_c = np.abs(c)
            abs_c[N-1] = 0
            max_sidelobe = np.max(abs_c)
            
            # If main lobe is 0, it's bad. But main lobe is sum of squares, so >= 0.
            # For signed steps, main lobe = N.
            
            if max_sidelobe < best_score:
                best_score = max_sidelobe
                best_f = f_discrete.copy()
                
        except Exception:
            pass
            
        # Quick Discrete Refinement on best_f
        if best_f is not None and time.time() < deadline - 5.0:
            f = best_f.copy()
            c = np.convolve(f, f, mode='full')
            
            # Greedy polish with strict time limit
            # Try flipping single bits if it reduces max sidelobe
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                abs_c = np.abs(c)
                abs_c[N-1] = 0
                current_max = np.max(abs_c)
                
                # Find the index of the peak sidelobe
                # We want to reduce the max.
                # Try flipping each i? That's O(N^2) per step.
                # N=400, N^2=160k.
                # If we do this for 100 steps, 16M ops. Fast.
                
                best_i = -1
                best_new_max = current_max
                
                # To speed up, we can just check a subset of flips or use the previous logic
                # But the previous logic timed out because it did this for many seeds/steps.
                # Here we do it only on the best candidate found by L-BFGS-B.
                
                for i in range(N):
                    # Fast update of c
                    # c_trial = c - 2*f[i]*f[i:i+N] + correction at 2i
                    # We can just compute the new max for the affected range?
                    # No, the max might be elsewhere.
                    # But if we only change c in range [i, i+N-1], the max outside that range is unchanged.
                    # So new_max = max( max(c_trial[affected]), max(c[unaffected]) )
                    
                    # Pre-compute max of unaffected?
                    # This is complex. Given N=400, just recomputing max on the whole array
                    # is O(N). Doing this N times is O(N^2) = 160k ops per step.
                    # This is very fast in Python/NumPy.
                    
                    c_trial = c.copy()
                    fi = f[i]
                    k_start = i
                    k_end = i + N
                    
                    delta = -2.0 * fi * f
                    c_trial[k_start:k_end] += delta
                    if 2*i < len(c_trial):
                        c_trial[2*i] += 2.0 * fi * fi
                    
                    abs_c_trial = np.abs(c_trial)
                    abs_c_trial[N-1] = 0
                    new_max = np.max(abs_c_trial)
                    
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_i = i
                        
                if best_i != -1:
                    f[best_i] = -f[best_i]
                    c = np.convolve(f, f, mode='full') # Recompute for next iteration
                    improved = True
                else:
                    break
                    
            # Update best if improved
            abs_c_final = np.abs(c)
            abs_c_final[N-1] = 0
            final_max = np.max(abs_c_final)
            if final_max < best_score:
                best_score = final_max
                best_f = f.copy()

    if best_f is not None:
        return best_f
    else:
        # Fallback: Random
        np.random.seed(0)
        return np.random.choice([-1.0, 1.0], size=N)

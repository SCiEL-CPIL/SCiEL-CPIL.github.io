import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 120.0
    
    best_f = None
    best_psl = float('inf')
    
    for seed in range(10):
        np.random.seed(seed)
        f = np.random.choice([-1, 1], size=N).astype(float)
        
        # Compute initial convolution
        c = np.convolve(f, f, mode='full')
        main_lobe = c[N-1]
        
        # Helper to get PSL
        def get_psl(c, main_lobe):
            if main_lobe == 0: return 0
            abs_c = np.abs(c)
            abs_c[N-1] = 0
            return np.max(abs_c) / abs(main_lobe)
        
        current_psl = get_psl(c, main_lobe)
        
        improved = True
        steps = 0
        while improved and (time.time() - start_time) < deadline:
            improved = False
            steps += 1
            
            # Find current peak sidelobe
            abs_c = np.abs(c)
            abs_c[N-1] = 0
            peak_val = np.max(abs_c)
            
            if peak_val == 0:
                break
            
            best_i = -1
            best_new_peak = peak_val
            
            # Try flipping each i
            for i in range(N):
                # Calculate change in c if we flip f[i]
                # delta_c[k] = -2 * f[i] * f[k-i] for k != 2i
                # delta_c[2i] = 0
                
                # We need to find the new max sidelobe.
                # Instead of updating the whole array and finding max, 
                # we can just compute the new peak value for the specific 
                # indices that might change, or just update the array.
                
                # Efficient update:
                # Create a copy of c? No, that's O(N).
                # We can compute the new values for the affected k.
                # But the "max" might be at an index that doesn't change.
                # So we need to know the max of the unchanged parts and the changed parts.
                
                # Unchanged part max:
                # This is hard to compute quickly without a data structure.
                # Given N=400, just updating the array is fine.
                
                c_trial = c.copy()
                
                # Apply change
                # f_i is the value before flip
                fi = f[i]
                
                # Indices k where c[k] changes
                # k ranges from 0 to 2N-2
                # f[k-i] is valid if 0 <= k-i < N
                # So k ranges from i to i+N-1
                
                # Vectorized update
                k_start = i
                k_end = i + N
                
                # Extract the slice of f that multiplies with f[i]
                # c_k involves f_i * f_{k-i}
                # So we need f[0:N] shifted?
                # f_{k-i} for k in [i, i+N-1] is f[0:N]
                
                # delta = -2 * f[i] * f[0:N]
                delta = -2.0 * fi * f
                
                c_trial[k_start:k_end] += delta
                
                # Note: at k=2i, the term f_i^2 is unchanged.
                # Our formula added -2 * f_i * f_i = -2 f_i^2.
                # But the true change is 0.
                # So we must add back 2 f_i^2 at index 2i.
                if 2*i < len(c_trial):
                    c_trial[2*i] += 2.0 * fi * fi
                    
                # Calculate new peak
                abs_c_trial = np.abs(c_trial)
                abs_c_trial[N-1] = 0
                new_peak = np.max(abs_c_trial)
                
                if new_peak < best_new_peak:
                    best_new_peak = new_peak
                    best_i = i
            
            if best_i != -1:
                # Apply the flip
                f[best_i] = -f[best_i]
                
                # Update c globally for the next iteration
                # We already computed c_trial for best_i? 
                # No, we discarded it. We should store the best c_trial.
                # Let's modify the loop to store best_c_trial.
                
                # Actually, let's just recompute c from f. It's O(N^2) = 160k.
                # The loop above is N * (N copy + N update) = N^2.
                # So recomputing is same cost as one iteration of the inner loop.
                # But we did N iterations.
                # So total cost per step is ~2 * N^2.
                # 2 * 160k = 320k ops.
                # 1000 steps = 320M ops. ~1-2 seconds.
                # This is very fast.
                
                c = np.convolve(f, f, mode='full')
                main_lobe = c[N-1]
                current_psl = get_psl(c, main_lobe)
                
                if current_psl < best_psl:
                    best_psl = current_psl
                    best_f = f.copy()
                improved = True
            else:
                # Try a random kick to escape local minimum
                # Flip a random block of 5 elements
                if steps < 50: # Limit kicks
                    idx = np.random.randint(0, N-5)
                    f[idx:idx+5] *= -1
                    c = np.convolve(f, f, mode='full')
                    main_lobe = c[N-1]
                    current_psl = get_psl(c, main_lobe)
                    if current_psl < best_psl:
                        best_psl = current_psl
                        best_f = f.copy()
                    improved = True
                else:
                    break

        if (time.time() - start_time) > deadline - 5:
            break

    if best_f is not None:
        return best_f
    else:
        return np.ones(N)

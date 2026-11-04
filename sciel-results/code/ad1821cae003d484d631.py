"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Deterministic Greedy Local Search with Exact Delta Updates
    # Objective: Minimize ISL = sum_{k != 0} R[k]^2
    
    def compute_autocorr(f):
        F = np.fft.fft(f)
        power = F * np.conj(F)
        R = np.real(np.fft.ifft(power))
        return R

    def compute_isl(R):
        # R[0] is the peak. We sum squares of all other lags.
        # Due to symmetry, R[k] = R[N-k].
        # We just sum R[1:]**2. Note R[N-1] is included in R[1:].
        return np.sum(R[1:]**2)

    best_f = None
    best_isl = float('inf')
    
    # Run multiple random restarts
    for _ in range(20):
        f = np.random.choice([-1, 1], size=N)
        R = compute_autocorr(f)
        current_isl = compute_isl(R)
        
        improved = True
        # Greedy descent
        while improved:
            improved = False
            for i in range(N):
                # Calculate delta in ISL if we flip f[i]
                # Delta R[k] = -2 * f[i] * (f[(i+k)%N] + f[(i-k)%N]) for k != 0
                # Note: f[(i-k)%N] is f[i-k] in circular sense.
                
                # We need sum_{k!=0} R[k] * DeltaR[k] and sum_{k!=0} DeltaR[k]^2
                # Let A[k] = f[(i+k)%N] + f[(i-k)%N]
                # DeltaR[k] = -2 * f[i] * A[k]
                
                # Efficient calculation:
                # Sum1 = sum_{k!=0} R[k] * (-2 * f[i] * A[k]) = -2 * f[i] * sum_{k!=0} R[k] * A[k]
                # Sum2 = sum_{k!=0} (-2 * f[i] * A[k])^2 = 4 * sum_{k!=0} A[k]^2 (since f[i]^2=1)
                
                # Construct A vector
                # A[k] = f[i+k] + f[i-k]
                # We can use rolling or slicing.
                # f_i_plus = np.roll(f, -i) -> f[i], f[i+1], ...
                # f_i_minus = np.roll(f, i)   -> f[i], f[i-1], ...
                # A[k] = f_i_plus[k] + f_i_minus[k]
                
                f_plus = np.roll(f, -i)
                f_minus = np.roll(f, i)
                A = f_plus + f_minus
                
                # Exclude k=0 from sums
                # R[0] is peak, A[0] = 2*f[i]
                # We need k in 1..N-1
                R_nonzero = R[1:]
                A_nonzero = A[1:]
                
                # Sum1 term
                term1 = -2 * f[i] * np.dot(R_nonzero, A_nonzero)
                
                # Sum2 term
                term2 = 4 * np.sum(A_nonzero**2)
                
                delta_isl = term1 + term2
                
                if delta_isl < 0:
                    # Perform flip
                    f[i] = -f[i]
                    # Update R analytically or recompute?
                    # Recomputing R is O(N log N). Updating R is O(N).
                    # Since N=400, O(N) update is better.
                    # R_new[k] = R[k] + DeltaR[k]
                    # DeltaR[0] = 4 (since R[0] = sum f^2 = N, flipping one bit doesn't change sum f^2? 
                    # Wait. R[0] = sum f[n]^2 = N. It is constant.
                    # My formula: R_new[k] = R[k] - 2 f_i (f_{i+k} + f_{i-k}) + 4 delta_{k,0}.
                    # For k=0: R_new[0] = R[0] - 2 f_i (2 f_i) + 4 = R[0] - 4 f_i^2 + 4 = R[0]. Correct.
                    
                    # Update R for k != 0
                    # We have A_nonzero which corresponds to k=1..N-1.
                    # DeltaR[k] = -2 * f[i] * A[k]
                    # But we flipped f[i], so f[i] changed sign.
                    # The formula used the OLD f[i].
                    # So DeltaR is correct for the transition.
                    
                    R[1:] += -2 * f[i] * A_nonzero
                    # Note: f[i] was the OLD value. We used it in delta calculation.
                    # Now we flip f[i].
                    
                    current_isl += delta_isl
                    improved = True
                    # After a flip, the delta for other indices changes.
                    # We must restart the loop or continue?
                    # Greedy: usually we continue scanning, but deltas are stale.
                    # However, for strict greedy descent, it's safer to restart the scan 
                    # or just continue and rely on the next pass to fix inconsistencies.
                    # Given N=400, restarting is cheap.
                    break
            
            # If we broke out of for loop due to improvement, we restart the while loop
            # If we finished the for loop without improvement, improved is False, and we exit while.
        
        if current_isl < best_isl:
            best_isl = current_isl
            best_f = f.copy()

    return best_f
# EVOLVE-BLOCK-END

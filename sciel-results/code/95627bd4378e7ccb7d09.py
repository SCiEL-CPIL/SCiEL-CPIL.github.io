import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    """
    Construct the optimal sequence using Spectral Phase Optimization.
    
    Strategy:
    1. Represent h via its DFT: H[k] = M[k] * exp(i * phi[k]).
    2. Fix M[k] to be constant (M[k] = N/2) to satisfy the sum constraint sum(h) = N/2 
       (since h[0] = sum H[k]/N = N/2).
       Note: For real h, H[k] = conj(H[N-k]), so we only optimize phi for k=1..N/2-1 
       and fix phi[0]=0, phi[N/2]=0 (or pi).
    3. Objective: Minimize the maximum overlap.
       Overlap[k] = N/2 - A[k]. Minimizing max(Overlap) is equivalent to maximizing min(A).
       Since maximizing a minimum is non-smooth, we use a smooth approximation:
       Maximize the "soft min" of A[k] (k!=0).
       max_overlap = N/2 - min(A).
       We want to minimize max_overlap, so we want to maximize min(A).
       Let S = -min(A). We want to minimize S.
       Approximate min(A) with -logsumexp(-A) / N (up to a constant).
       Actually, min(A) ≈ - (1/tau) * log(sum exp(-tau * A)).
       So we minimize F(phi) = (1/tau) * log(sum_{k!=0} exp(-tau * A[k])).
       This is a convex function of A, but A is quadratic in h, and h is trigonometric in phi.
       The resulting objective is non-convex in phi, allowing us to find non-uniform solutions.
    4. Use L-BFGS-B to optimize the phase angles phi.
    5. Multiple restarts with different initial phases to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Indices for real signal:
    # H[0] is real, H[N/2] is real (for even N).
    # H[k] and H[N-k] are conjugates.
    # We optimize phi[1], ..., phi[N/2 - 1].
    # Let K = N // 2 - 1.
    K = N // 2 - 1
    if K <= 0:
        return np.full(N, 0.5)
    
    # Fixed magnitude spectrum
    # To satisfy sum(h) = N/2, we need H[0] = N/2.
    # If we set all |H[k]| = N/2, then sum(h) = (1/N) * sum H[k] = (1/N) * (N/2 + sum_{k=1}^{N-1} H[k]).
    # This doesn't automatically guarantee sum(h)=N/2 unless the other terms sum to 0.
    # However, if we fix M[k] = N/2 for all k, then h is the inverse FFT of N/2 * exp(i phi).
    # sum(h) = h_mean * N = (1/N) * H[0] * N = H[0] = N/2.
    # So fixing M[0] = N/2 and M[k] = 0 for k>0 gives h=0.5.
    # But we want non-trivial h.
    # Wait, the constraint is sum(h) = N/2.
    # H[0] = sum(h). So H[0] MUST be N/2.
    # We can vary the other H[k].
    # Let's set M[0] = N/2.
    # For k > 0, we can choose M[k]. To maximize degrees of freedom, let's start with M[k] = N/2 for all k?
    # If M[k] = N/2 for all k, then H[0] = N/2 is fixed.
    # The other components H[k] for k>0 have magnitude N/2.
    # This allows h to vary significantly.
    # Does this violate any bounds? h[i] must be in [0,1].
    # We will check bounds and project if necessary, or just accept if valid.
    # Given N=100 and M[k]=N/2, the max possible h[i] is sum M[k]/N = N/2 = 50? No.
    # h[i] = (1/N) sum M[k] exp(i(k phi[k] - i 2pi k / N)).
    # Max h[i] <= (1/N) sum M[k] = (1/N) * N * (N/2) = N/2 = 50.
    # This is way above 1. So we need to scale down M[k] for k>0 or ensure the phases cancel out.
    # Better strategy: Fix M[0] = N/2. Set M[k] = c for k>0.
    # Choose c such that typical h[i] is within [0,1].
    # If phases are random, h[i] ~ N(0, sqrt(sum M[k]^2 / N^2)).
    # Variance = (1/N^2) sum M[k]^2 = (1/N^2) (M[0]^2 + (N-1) c^2).
    # We want std(h) to be small enough that h stays in [0,1] with high probability, 
    # or we can just project.
    # Let's try c = N/2 * 0.5 = N/4?
    # Or simpler: Start with M[k] = N/2 for all k, but scale the "ac" part.
    # Actually, let's just use M[0] = N/2 and M[k] = N/2 for k>0, and rely on the optimizer to find phases that keep h in [0,1].
    # If it violates bounds, we can add a penalty term to the objective.
    
    M = np.full(N, N / 2.0)
    
    def compute_h(phi):
        """
        phi: array of length K (phases for k=1..K)
        Returns h: array of length N
        """
        H = np.zeros(N, dtype=complex)
        H[0] = M[0]
        for k in range(1, K + 1):
            H[k] = M[k] * np.exp(1j * phi[k-1])
            H[N-k] = np.conj(H[k])
        if N % 2 == 0:
            H[N//2] = M[N//2] # Real, phase 0 or pi. Let's fix phase 0 for simplicity, or include it?
            # For even N, H[N/2] is real. We can optimize its sign (0 or pi).
            # Let's just fix it to M[N/2] for now.
        h = np.fft.ifft(H).real
        return h
    
    def objective_and_grad(phi):
        h = compute_h(phi)
        
        # Check bounds violation penalty
        # If h goes out of [0,1], penalize heavily
        penalty = 0.0
        grad_penalty = np.zeros_like(phi)
        
        if np.any(h < 0) or np.any(h > 1):
            # Soft penalty
            viol_low = np.maximum(0, -h)
            viol_high = np.maximum(0, h - 1)
            penalty = 1e6 * (np.sum(viol_low**2) + np.sum(viol_high**2))
            
            # Gradient of penalty w.r.t phi is complex to compute analytically.
            # Use finite differences for the gradient of the whole objective?
            # Or just use numerical gradient in scipy.optimize?
            # L-BFGS-B can do finite difference if jac is None.
            # Given N=100 and K=49, finite diff might be slow but acceptable for 110s?
            # 49 * 2 evaluations per step. 100 steps -> 10k evals.
            # Each eval is FFT + logsumexp. Very fast.
            # Let's return only the value and let scipy compute grad.
            
            # However, for better performance, let's compute analytical grad of the smooth part
            # and ignore penalty grad (assuming we stay in bounds or penalty dominates).
            # Actually, if we use finite diff for the whole thing, it's safer.
            
            # Let's try to compute the gradient of the smooth objective analytically.
            # F = (1/tau) log(sum_{k!=0} exp(-tau A[k]))
            # dF/dh = - (sum_{k!=0} exp(-tau A[k]) * (-tau) * dA[k]/dh) / (tau * sum_{k!=0} exp(-tau A[k]))
            # dF/dh = (sum_{k!=0} w_k * dA[k]/dh) where w_k = exp(-tau A[k]) / sum(exp(-tau A)))
            # dA[k]/dh_i = h[i-k] + h[i+k]
            
            # Let's compute this.
            H_fft = np.fft.fft(h)
            power = np.abs(H_fft)**2
            A = np.fft.ifft(power).real
            A[0] = 0 # Exclude DC component from min/max overlap calculation?
            # The overlap with shift 0 is sum h[i](1-h[i]). This is not the "shifted 1-h" in the same sense?
            # "Overlap of h with shifted 1-h".
            # Overlap[k] = sum h[i] (1-h[i-k]).
            # For k=0, Overlap[0] = sum h[i] - sum h[i]^2.
            # The task says "max overlap". Usually this includes all shifts.
            # Let's include k=0.
            
            # Recalculate A including k=0
            A_full = np.fft.ifft(power).real
            
            tau = 50.0 # Temperature for logsumexp
            # We want to minimize max(Overlap) = N/2 - min(A_full).
            # So we want to maximize min(A_full).
            # Max min(A) ~ - (1/tau) log(sum exp(-tau A)).
            # Let G = sum_{k} exp(-tau A[k]).
            # obj = (1/tau) log(G) + penalty
            # d obj / d h = (1/tau) * (1/G) * sum_k (-tau exp(-tau A[k]) * dA[k]/dh)
            #             = - (1/G) * sum_k exp(-tau A[k]) * dA[k]/dh
            
            exp_term = np.exp(-tau * A_full)
            G = np.sum(exp_term)
            w = exp_term / G # weights
            
            # dA[k]/dh_i = h[i-k] + h[i+k]
            # Sum_k w[k] (h[i-k] + h[i+k])
            # This is conv(w, h) + conv(w_rev, h)
            # w is real and symmetric? A[k] = A[N-k]. So w[k] = w[N-k].
            # So w is symmetric.
            # Sum_k w[k] h[i-k] = (w * h)[i]
            # Sum_k w[k] h[i+k] = (w * h)[i] (since w is symmetric)
            # So sum = 2 * (w * h)[i]
            
            w_fft = np.fft.fft(w)
            h_fft = np.fft.fft(h)
            conv = np.fft.ifft(w_fft * h_fft).real
            grad_h = -2.0 * conv
            
            # Now we need d h / d phi.
            # h = ifft(H). H[k] = M[k] exp(i phi[k]).
            # This derivative is tricky to compute analytically in a vectorized way for all phi.
            # Given the complexity, let's just use numerical gradients via scipy.
            
            return obj + penalty, None
        
        # If no penalty, we still need to return value.
        # Let's just use numerical gradients for the whole thing to be safe and simple.
        H_fft = np.fft.fft(h)
        power = np.abs(H_fft)**2
        A_full = np.fft.ifft(power).real
        
        tau = 50.0
        exp_term = np.exp(-tau * A_full)
        G = np.sum(exp_term)
        obj = (1.0 / tau) * np.log(G)
        
        return obj, None

    # We will use scipy.optimize.minimize with method='L-BFGS-B'
    # Since we didn't provide jac, it will use finite differences.
    
    best_h = np.full(N, 0.5)
    best_val = np.inf
    
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize phi
        if r == 0:
            phi0 = np.zeros(K)
        else:
            phi0 = np.random.uniform(-np.pi, np.pi, K)
        
        # Define objective function for scipy
        def obj_func(phi):
            h = compute_h(phi)
            H_fft = np.fft.fft(h)
            power = np.abs(H_fft)**2
            A_full = np.fft.ifft(power).real
            
            # Check bounds
            if np.any(h < -1e-9) or np.any(h > 1.0 + 1e-9):
                # Heavy penalty
                viol = np.sum(np.maximum(0, -h)**2) + np.sum(np.maximum(0, h-1)**2)
                return 1e10 + viol
            
            tau = 50.0
            exp_term = np.exp(-tau * A_full)
            G = np.sum(exp_term)
            obj = (1.0 / tau) * np.log(G)
            return obj
        
        # Run optimization
        try:
            res = minimize(obj_func, phi0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if res.success or res.fun < best_val:
                phi_opt = res.x
                h_opt = compute_h(phi_opt)
                
                # Validate h_opt
                if np.all(h_opt >= -1e-9) and np.all(h_opt <= 1.0 + 1e-9):
                    # Clip for safety
                    h_opt = np.clip(h_opt, 0.0, 1.0)
                    
                    # Check sum constraint
                    # sum(h) should be N/2 because H[0] = N/2.
                    # Numerical errors might creep in.
                    sum_h = np.sum(h_opt)
                    target_sum = N / 2.0
                    if abs(sum_h - target_sum) > 1e-6:
                        # Adjust
                        # Simple correction: add (target_sum - sum_h) / N to all elements?
                        # This might break bounds.
                        # Better: find the element with most room and adjust it.
                        diff = target_sum - sum_h
                        # Find index with max room (if diff > 0, look for min h; if diff < 0, look for max h)
                        if diff > 0:
                            idx = np.argmin(h_opt)
                        else:
                            idx = np.argmax(h_opt)
                        
                        # Check if we can apply the full diff
                        if diff > 0:
                            if h_opt[idx] + diff <= 1.0:
                                h_opt[idx] += diff
                            else:
                                # Distribute? For now, just accept slight error if small
                                pass
                        else:
                            if h_opt[idx] + diff >= 0.0:
                                h_opt[idx] += diff
                            else:
                                pass
                    
                    # Evaluate score
                    # Score is max overlap = N/2 - min(A)
                    H_fft = np.fft.fft(h_opt)
                    power = np.abs(H_fft)**2
                    A_full = np.fft.ifft(power).real
                    max_overlap = N / 2.0 - np.min(A_full)
                    
                    if max_overlap < best_val:
                        best_val = max_overlap
                        best_h = h_opt.copy()
                        
        except Exception:
            continue

    # If optimization failed or didn't improve, return uniform
    if best_val >= N/4.0 - 1e-6: # Uniform gives N/4 overlap
        return np.full(N, 0.5)
        
    return best_h

def _max_overlap(h):
    """
    Calculate the maximum overlap of h with shifted 1-h.
    Overlap[k] = sum_i h[i] * (1 - h[(i-k)%N])
    """
    H = np.fft.fft(h)
    power = np.abs(H)**2
    A = np.fft.ifft(power).real
    min_A = np.min(A)
    max_overlap = N / 2.0 - min_A
    return max_overlap

if __name__ == "__main__":
    h = construct()
    print(_max_overlap(h))

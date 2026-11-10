# EVOLVE-BLOCK-START
import numpy as np
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    # Precompute indices for slopes
    # x, y in 0..9
    # P[y, x]
    
    # Slopes: -1, 0, 1, 2
    # Values:
    # r=-1: x - y
    # r=0: x
    # r=1: x + y
    # r=2: x + 2y
    
    # Create coordinate grids
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    X = X.T # X[x, y] -> but we want P[y, x], so let's just compute values for each cell (y, x)
    # Actually, P is indexed [y, x].
    # Let's create arrays of values for each slope for each cell (y, x)
    # cell (y, x) corresponds to x=x, y=y
    
    slopes = [-1, 0, 1, 2]
    # vals[r_index, y, x] = x + r * y
    vals = np.zeros((4, n, n))
    for i, r in enumerate(slopes):
        vals[i] = X + r * Y
        
    # Map values to indices for binning
    # r=-1: x-y ranges from -9 to 9. Offset by 9 -> 0..18
    # r=0: x ranges from 0 to 9.
    # r=1: x+y ranges from 0 to 18.
    # r=2: x+2y ranges from 0 to 27.
    
    # We can just use np.bincount or unique
    
    start_time = time.time()
    timeout = 600.0
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = 1.0 # Uniform score is 1.0
    
    def compute_entropies(P):
        """
        P: (n, n) array
        Returns H for slopes -1, 0, 1, 2
        """
        Hs = []
        for i in range(4):
            v = vals[i]
            # Flatten P and v
            p_flat = P.flatten()
            v_flat = v.flatten()
            
            # Get unique values and sums
            # Using np.unique with return_inverse or just binning
            # Since ranges are small, we can use np.bincount if we shift to positive integers
            # r=-1: min -9, max 9. Shift 9.
            # r=0: min 0, max 9.
            # r=1: min 0, max 18.
            # r=2: min 0, max 27.
            
            # Simpler: use np.unique
            unique_vals, inverse = np.unique(v_flat, return_inverse=True)
            probs = np.bincount(inverse, weights=p_flat)
            probs = probs / probs.sum() # Just in case of numerical drift, though P sums to 1
            
            # Shannon entropy in bits
            # Ignore zero probs
            nonzero = probs > 1e-15
            if not np.any(nonzero):
                H = 0.0
            else:
                H = -np.sum(probs[nonzero] * np.log2(probs[nonzero]))
            Hs.append(H)
        return Hs

    def compute_score(P):
        Hs = compute_entropies(P)
        H_neg1 = Hs[0]
        H_0 = Hs[1]
        H_1 = Hs[2]
        H_2 = Hs[3]
        
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return 0.0, Hs
        return H_neg1 / M, Hs

    def gradient_ascent_restart(v0, iterations=2000):
        """
        Perform gradient ascent on log-probs v.
        """
        v = v0.copy()
        # Annealing schedule for temperature tau in LSE
        # Start with small tau (smooth), increase to large tau (sharp)
        # Also anneal step size
        
        best_local_P = None
        best_local_score = -np.inf
        
        for it in range(iterations):
            # Check time
            if time.time() - start_time > timeout - 30:
                break
                
            # Normalize v to get P
            # Subtract max for numerical stability
            v_max = np.max(v)
            P_unnorm = np.exp(v - v_max)
            P = P_unnorm / np.sum(P_unnorm)
            
            Hs = compute_entropies(P)
            H_neg1 = Hs[0]
            H_0, H_1, H_2 = Hs[1], Hs[2], Hs[3]
            
            M_exact = max(H_0, H_1, H_2)
            if M_exact < 1e-9:
                score = 0.0
            else:
                score = H_neg1 / M_exact
                
            if score > best_local_score:
                best_local_score = score
                best_local_P = P.copy()
                
            # If we have hit a good score, we can try to refine
            # Compute gradient of log ratio using LSE approximation
            
            # Temperature tau
            # Start low, increase over iterations
            tau = 1.0 + 10.0 * (it / iterations)
            
            # LSE of H_0, H_1, H_2
            # M_tau = (1/tau) * log(sum(exp(tau * H_r)))
            exp_Hs = np.exp(tau * np.array([H_0, H_1, H_2]))
            sum_exp = np.sum(exp_Hs)
            M_tau = (1.0 / tau) * np.log(sum_exp)
            
            if H_neg1 < 1e-9:
                # Degenerate
                grad_log_H_neg1 = np.zeros((n, n))
            else:
                # Gradient of log H_neg1
                # d log H / dv = (1/H) * dH/dv
                # dH_r/dv_{a,b} = P_{a,b} * (g_{a,b}^{(r)} - H_r - 1/ln2)
                
                def grad_H(r_idx, H_r):
                    v_flat = vals[r_idx].flatten()
                    p_flat = P.flatten()
                    
                    # Calculate g_{i,j}^{(r)} for all i,j
                    # g_{i,j} = -log2(p_k) - 1/ln2 where k is the bin index for (i,j)
                    # We need p_k for each cell (i,j)
                    
                    # Efficiently get p_k for each cell
                    # Use unique/inverse again
                    unique_vals, inverse = np.unique(v_flat, return_inverse=True)
                    probs = np.bincount(inverse, weights=p_flat)
                    probs = probs / probs.sum()
                    
                    # g for each cell is based on the prob of its bin
                    g_flat = -np.log2(probs[inverse]) - (1.0 / np.log(2))
                    g = g_flat.reshape(n, n)
                    
                    # grad = P * (g - H_r - 1/ln2)
                    grad = P * (g - H_r - (1.0 / np.log(2)))
                    return grad

                grad_H_neg1 = grad_H(0, H_neg1)
                grad_log_H_neg1 = grad_H_neg1 / H_neg1
                
            # Gradient of log M_tau
            # d log M_tau / dv = (1/M_tau) * d M_tau / dv
            # d M_tau / dv = (1/tau) * (1/sum_exp) * sum_r (tau * exp(tau H_r) * dH_r/dv)
            #             = (1/sum_exp) * sum_r exp(tau H_r) * dH_r/dv
            #             = sum_r w_r * dH_r/dv where w_r = exp(tau H_r) / sum_exp
            
            weights = exp_Hs / sum_exp
            
            grad_M_tau = np.zeros((n, n))
            for r_idx, w in zip([1, 2, 3], weights):
                H_r = [H_0, H_1, H_2][r_idx - 1]
                grad_H_r = grad_H(r_idx, H_r)
                grad_M_tau += w * grad_H_r
                
            grad_log_M_tau = grad_M_tau / M_tau
            
            # Total gradient for maximizing log(H_neg1) - log(M_tau)
            grad_total = grad_log_H_neg1 - grad_log_M_tau
            
            # Step size
            # Simple decay
            eta = 0.1 * (1.0 / (1.0 + 0.01 * it))
            
            v += eta * grad_total
            
            # Keep v bounded to prevent overflow/underflow issues in exp
            # Center v
            v -= np.mean(v)
            
        return best_local_P, best_local_score

    # Run multiple restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() - start_time > timeout - 60:
            break
            
        # Initialize v
        # Random Dirichlet-like start
        # v = np.random.randn(n, n)
        # Or start near uniform
        v0 = np.zeros((n, n)) + np.random.randn(n, n) * 0.5
        
        P_local, score_local = gradient_ascent_restart(v0, iterations=1000)
        
        if P_local is not None and score_local > best_score:
            best_score = score_local
            best_P = P_local.copy()
            
    # Final check and cleanup
    # Ensure P is valid
    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s > 1e-9:
        best_P = best_P / s
    else:
        best_P = np.ones((n, n)) / (n * n)
        
    return best_P

# EVOLVE-BLOCK-END

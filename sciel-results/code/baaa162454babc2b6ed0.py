# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600 - 30  # 30s safety margin
    
    # Max K allowed
    K = 200000
    
    # Compute Mobius function up to K
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    mu[0] = 0
    mu[1] = 1
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
                
    # Filter domain to non-zero mu
    domain = np.nonzero(mu[1:])[0] + 1
    values = mu[domain].astype(np.float64)
    
    # Calculate S = sum mu(k)/k
    # Note: evaluator uses clipped values, but mu is in [-1,1] so no clipping
    S = np.sum(values / domain)
    
    # Calculate M_g(x) = sum mu(k) floor(x/k) for x in [1, 10K]
    # M_g(x) is used to check constraint M_eval(x) = alpha * M_g(x) - S * x <= 1.0001
    max_x = 10 * K
    M_g = np.zeros(max_x + 1, dtype=np.float64)
    
    # Efficient computation of M_g
    # Loop over k with mu(k) != 0
    for k in domain:
        m = mu[k]
        if m == 0:
            continue
        # Add m * floor(x/k) to M_g[x]
        # floor(x/k) = j for x in [j*k, (j+1)*k - 1]
        # j ranges from 1 to max_x // k
        max_j = max_x // k
        for j in range(1, max_j + 1):
            start = j * k
            end = min((j + 1) * k, max_x + 1)
            M_g[start:end] += m * j
            
    if time.time() > deadline:
        # If too slow, return a simple valid solution
        # Just return mu for small K?
        # For now, assume it finishes.
        pass

    # Check constraint with alpha = 1
    # M_eval(x) = 1 * M_g(x) - S * x
    # We need M_eval(x) <= 1.0001 for all x in [1, 10K]
    x_vals = np.arange(1, max_x + 1)
    M_eval = M_g[1:] - S * x_vals
    
    max_M_eval = np.max(M_eval)
    
    if max_M_eval > 1.0001:
        # Scale down
        # We need alpha * M_g(x) - S * x <= 1.0001
        # alpha <= (1.0001 + S * x) / M_g(x) for M_g(x) > 0
        # If M_g(x) <= 0, constraint is alpha * M_g(x) <= 1.0001 + S * x
        # If 1.0001 + S * x >= 0, always true for alpha >= 0
        # If 1.0001 + S * x < 0, we need alpha >= (1.0001 + S * x) / M_g(x) (flipped)
        
        # Find upper bounds on alpha
        mask_pos = M_g[1:] > 0
        if np.any(mask_pos):
            bounds = (1.0001 + S * x_vals[mask_pos]) / M_g[1:][mask_pos]
            alpha_max = np.min(bounds)
        else:
            alpha_max = 1.0
            
        # Find lower bounds on alpha
        mask_neg = M_g[1:] < 0
        if np.any(mask_neg):
            rhs = 1.0001 + S * x_vals[mask_neg]
            # If rhs < 0, we have a lower bound
            mask_rhs_neg = rhs < 0
            if np.any(mask_rhs_neg):
                bounds_lower = rhs[mask_rhs_neg] / M_g[1:][mask_neg][mask_rhs_neg]
                alpha_min = np.max(bounds_lower)
            else:
                alpha_min = 0.0
        else:
            alpha_min = 0.0
            
        if alpha_min > alpha_max:
            # Infeasible? Fallback to small K
            # Try K=1000
            K = 1000
            # Recompute mu, S, M_g for K=1000
            mu_small = np.ones(K + 1, dtype=np.int64)
            is_prime_small = np.ones(K + 1, dtype=bool)
            primes_small = []
            mu_small[0] = 0
            mu_small[1] = 1
            for i in range(2, K + 1):
                if is_prime_small[i]:
                    primes_small.append(i)
                    mu_small[i] = -1
                for p in primes_small:
                    if i * p > K:
                        break
                    is_prime_small[i * p] = False
                    if i % p == 0:
                        mu_small[i * p] = 0
                        break
                    else:
                        mu_small[i * p] = -mu_small[i]
                        
            domain_small = np.nonzero(mu_small[1:])[0] + 1
            values_small = mu_small[domain_small].astype(np.float64)
            S_small = np.sum(values_small / domain_small)
            
            max_x_small = 10 * K
            M_g_small = np.zeros(max_x_small + 1, dtype=np.float64)
            for k in domain_small:
                m = mu_small[k]
                if m == 0: continue
                max_j = max_x_small // k
                for j in range(1, max_j + 1):
                    start = j * k
                    end = min((j + 1) * k, max_x_small + 1)
                    M_g_small[start:end] += m * j
                    
            x_vals_small = np.arange(1, max_x_small + 1)
            M_eval_small = M_g_small[1:] - S_small * x_vals_small
            max_M_eval_small = np.max(M_eval_small)
            
            if max_M_eval_small > 1.0001:
                # Scale
                mask_pos_small = M_g_small[1:] > 0
                if np.any(mask_pos_small):
                    bounds_small = (1.0001 + S_small * x_vals_small[mask_pos_small]) / M_g_small[1:][mask_pos_small]
                    alpha_small = np.min(bounds_small)
                else:
                    alpha_small = 1.0
                    
                values_small *= alpha_small
                return domain_small, values_small
            else:
                return domain_small, values_small
        
        # Use the feasible alpha
        alpha = alpha_max
        # Ensure alpha is not too small
        if alpha < 0.001:
             alpha = 0.001
        values *= alpha
    else:
        # Constraint satisfied with alpha=1
        pass

    return domain, values

# EVOLVE-BLOCK-END

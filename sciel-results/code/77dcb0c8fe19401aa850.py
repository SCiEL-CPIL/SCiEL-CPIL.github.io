# EVOLVE-BLOCK-START
import numpy as np

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    # We determined that K={2} with c_2=-2, c_1=-10 gives a score of ~0.69.
    # Let's try to find a better set.
    
    # Strategy: Search for a small set of k's and optimal scaling alpha.
    # We assume c_k = -10 * alpha for k in K, c_1 = -10.
    # Constraints:
    # 1. f_final(1) = 10 * alpha * sum(1/k) <= 1
    # 2. M(x) <= 1 for all x. M(x) = f_final(1)*x - 10*alpha*sum(floor(x/k))
    
    best_score = 0
    best_domain = None
    best_values = None
    
    # Try small sets of k from 2 to 10
    candidates = list(range(2, 11))
    
    # We can't do full subset search easily, but we can try a few heuristics.
    # Heuristic 1: Single k
    for k in candidates:
        S1 = 1.0 / k
        # Max sum of fractions for this k
        # floor(x/k) = x/k - {x/k}
        # M(x) = x - 10*alpha*{x/k} (since 10*alpha*S1*x = x)
        # Max {x/k} is 1 - 1/k
        # So M(x) max = 10*alpha*(1 - 1/k)
        # We need 10*alpha*(1 - 1/k) <= 1
        # Also 10*alpha*S1 <= 1 => alpha <= 1/(10*S1)
        
        alpha_max1 = 1.0 / (10.0 * S1)
        alpha_max2 = 1.0 / (10.0 * (1.0 - 1.0/k))
        
        alpha = min(alpha_max1, alpha_max2)
        
        if alpha <= 0:
            continue
            
        c_k = -10.0 * alpha
        f1 = 10.0 * alpha * S1
        
        # Verify M(x) for x in 1..100
        valid = True
        for x in range(1, 101):
            m_val = f1 * x + c_k * (x // k)
            if m_val > 1.0001:
                valid = False
                break
        
        if valid:
            score = -c_k * (np.log(k) / k)
            if score > best_score:
                best_score = score
                best_domain = np.array([1, k], dtype=np.int64)
                best_values = np.array([-10.0, c_k], dtype=np.float64)

    # Heuristic 2: Pairs of k
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            k1, k2 = candidates[i], candidates[j]
            S1 = 1.0/k1 + 1.0/k2
            alpha_max1 = 1.0 / (10.0 * S1)
            
            # Check M(x) for a range of x to find max
            # M(x) = f1*x - 10*alpha*(floor(x/k1) + floor(x/k2))
            # f1 = 10*alpha*S1
            # M(x) = 10*alpha*S1*x - 10*alpha*(x/k1 - {x/k1} + x/k2 - {x/k2})
            # M(x) = 10*alpha*(S1*x - x + {x/k1} + {x/k2})
            # M(x) = 10*alpha*({x/k1} + {x/k2})
            
            max_m = 0
            for x in range(1, 200):
                m_val = 10.0 * alpha_max1 * ((x % k1) / k1 + (x % k2) / k2)
                if m_val > max_m:
                    max_m = m_val
            
            if max_m <= 1.0001:
                c_k1 = -10.0 * alpha_max1
                c_k2 = -10.0 * alpha_max1
                score = -c_k1 * (np.log(k1)/k1) - c_k2 * (np.log(k2)/k2)
                if score > best_score:
                    best_score = score
                    best_domain = np.array([1, k1, k2], dtype=np.int64)
                    best_values = np.array([-10.0, c_k1, c_k2], dtype=np.float64)

    if best_domain is None:
        # Fallback to seed
        return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        
    return best_domain, best_values

# EVOLVE-BLOCK-END

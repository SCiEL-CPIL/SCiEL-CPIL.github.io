import numpy as np
from scipy.optimize import minimize
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    timeout = 600.0
    margin = 10.0
    
    best_score = -1.0
    best_support = None
    best_weights = None

    # Candidate supports
    # Small integer supports with gaps
    candidates = [
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 3.0]),
        np.array([0.0, 1.0, 4.0]),
        np.array([0.0, 1.0, 5.0]),
        np.array([0.0, 2.0, 5.0]),
        np.array([0.0, 1.0, 3.0, 6.0]),
        np.array([0.0, 1.0, 2.0, 5.0]),
        np.array([0.0, 1.0, 4.0, 9.0]),
        np.array([0.0, 1.0, 3.0, 7.0, 15.0]),
        np.array([0.0, 1.0, 2.0, 4.0, 8.0]),
    ]
    
    # Add some random supports
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = rng.integers(3, 7)
        s = np.sort(rng.uniform(0, 10, n))
        s[0] = 0.0
        candidates.append(s)

    def compute_probability(support, weights):
        """
        Compute P[X1 + X2 + X3 < 2*X4] exactly.
        """
        n = len(support)
        # Normalize weights
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        w = weights / w_sum
        
        # Precompute pairwise sums s_i + s_j
        # We need sum over i,j of w_i w_j for s_i + s_j < 2*s_l - s_k
        # This is O(n^4) if done naively.
        # For n <= 10, n^4 = 10000, which is fast.
        prob = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[l] - support[k]
                # Sum of w_i w_j where s_i + s_j < threshold
                # We can precompute sorted sums and use prefix sums of weights?
                # Or just loop since n is small.
                inner_sum = 0.0
                for i in range(n):
                    for j in range(n):
                        if support[i] + support[j] < threshold:
                            inner_sum += w[i] * w[j]
                prob += w[k] * w[l] * inner_sum
        return prob

    def objective(params):
        """
        Objective function for optimizer.
        params: unconstrained parameters.
        We use softmax to ensure positive weights summing to 1.
        """
        n = len(params)
        # Softmax
        exp_params = np.exp(params - np.max(params))
        w = exp_params / np.sum(exp_params)
        # We need to pass support to this function, so we use closure or global
        # But we are inside a loop, so we use a nested function or lambda.
        # To avoid global state issues, we define the function inside the loop below.
        return -compute_probability(current_support, w)

    for support in candidates:
        if time.time() > start_time + timeout - margin:
            break
            
        n = len(support)
        current_support = support
        
        # Initial guess: uniform weights
        initial_params = np.zeros(n)
        
        # Optimize
        try:
            # Nelder-Mead is robust for small dimensions
            res = minimize(
                lambda p: -compute_probability(support, np.exp(p - np.max(p)) / np.sum(np.exp(p - np.max(p)))),
                initial_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            # Get best weights from result
            best_params = res.x
            exp_best = np.exp(best_params - np.max(best_params))
            best_w = exp_best / np.sum(exp_best)
            
            score = compute_probability(support, best_w)
            
            if score > best_score:
                best_score = score
                best_support = support.copy()
                best_weights = best_w.copy()
                
        except Exception as e:
            pass

    # If no optimization worked or score is bad, fallback to uniform on {0, 1, 3}
    if best_support is None or best_score < 0.5:
        best_support = np.array([0.0, 1.0, 3.0])
        # Simple heuristic weights for {0, 1, 3}
        # Try to balance
        best_weights = np.array([0.5, 0.3, 0.2])
        # Normalize
        best_weights = best_weights / np.sum(best_weights)
        # Re-evaluate to be sure
        s = compute_probability(best_support, best_weights)
        if s > best_score:
            best_score = s

    return best_support, best_weights

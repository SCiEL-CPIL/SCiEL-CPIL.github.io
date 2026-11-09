# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a 3-point support {0, 1, M} with large M.
    This allows for a "two-tier" structure where:
    1. Small values (0, 1) interact to sometimes satisfy the inequality.
    2. Large value M creates a clear separation.
    
    We optimize the weights (p0, p1, pM) using a grid search followed by local refinement.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    def calc_prob(support, weights):
        """
        Compute P[X1 + X2 + X3 < 2*X4] exactly in float64.
        P = sum_{k,l} w_k w_l * (sum_{i,j} w_i w_j * 1[s_i + s_j < 2*s_l - s_k])
        """
        n = len(support)
        # Build pairwise sums S[i,j] = s_i + s_j
        S = support[:, None] + support[None, :]
        
        total_prob = 0.0
        # Iterate over k (X4's index in the denominator term? No, X4 is the 4th variable)
        # Let's re-read the formula carefully.
        # P = Prob[X1 + X2 + X3 < 2*X4]
        # X1, X2, X3, X4 are i.i.d.
        # The formula given: P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)
        
        # Let's map variables to indices:
        # Let X4 take value s_k with prob w_k.
        # Let X1 take value s_l with prob w_l.
        # Then we need X2 + X3 < 2*X4 - X1 = 2*s_k - s_l.
        # Let X2 take value s_i and X3 take value s_j.
        # Condition: s_i + s_j < 2*s_k - s_l.
        
        # So the sum is over k (for X4), l (for X1), i (for X2), j (for X3).
        # P = sum_{k,l,i,j} w_k w_l w_i w_j * 1[s_i + s_j < 2*s_k - s_l]
        
        # This can be rearranged. For fixed k, l:
        # We need sum_{i,j} w_i w_j * 1[s_i + s_j < 2*s_k - s_l].
        # Let T = 2*s_k - s_l.
        # We need the probability mass of (X2+X3) < T.
        
        # Since n is small (3), we can just do the quadruple sum or optimize slightly.
        # With n=3, n^4 = 81 operations. Very fast.
        
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[k] - support[l]
                # Sum w_i * w_j for s_i + s_j < threshold
                inner_sum = 0.0
                for i in range(n):
                    for j in range(n):
                        if S[i, j] < threshold:
                            inner_sum += weights[i] * weights[j]
                total_prob += weights[k] * weights[l] * inner_sum
        return total_prob

    # Initial best
    best_prob = -1.0
    best_weights = None
    best_support = None
    
    # Support points: 0, 1, M
    # M needs to be large enough so that 0+0 < 2*M - 1 is always true, etc.
    # Actually, if M is very large, 2*M is huge.
    # Let's analyze the condition s_i + s_j < 2*s_k - s_l.
    
    # Points: 0, 1, M.
    # Weights: p0, p1, pM. p0+p1+pM = 1.
    
    # Let's do a grid search on p0, p1 (pM = 1 - p0 - p1) and M.
    # M should be large to separate the scales.
    # Let's try M = 10, 100, 1000, etc.
    
    # Since the inequality is strict, and values are discrete,
    # if M is large, terms involving M on the RHS (2*s_k) will dominate.
    
    # Let's just optimize p0, p1, pM for a fixed large M, say M=1000.
    # And maybe check a few M values.
    
    M_values = [10.0, 100.0, 1000.0, 10000.0]
    
    for M in M_values:
        support = np.array([0.0, 1.0, M], dtype=np.float64)
        
        # Grid search for p0, p1
        # p0, p1 in [0, 1], pM = 1 - p0 - p1 >= 0.
        
        # Coarse grid
        grid_size = 50
        for i in range(grid_size + 1):
            p0 = i / grid_size
            for j in range(grid_size + 1 - i):
                p1 = j / grid_size
                pM = 1.0 - p0 - p1
                if pM < 0:
                    continue
                weights = np.array([p0, p1, pM], dtype=np.float64)
                prob = calc_prob(support, weights)
                if prob > best_prob:
                    best_prob = prob
                    best_weights = weights.copy()
                    best_support = support.copy()
                    
        if time.time() > deadline:
            break

    # Local refinement around the best point found
    if best_weights is not None:
        # Use Nelder-Mead or simple coordinate descent?
        # Let's do a simple local search with shrinking step size.
        
        current_weights = best_weights.copy()
        current_support = best_support.copy()
        current_prob = best_prob
        
        step_size = 0.01
        for _ in range(1000):
            if time.time() > deadline:
                break
                
            improved = False
            for dim in range(3):
                for direction in [-1, 1]:
                    new_weights = current_weights.copy()
                    new_weights[dim] += direction * step_size
                    # Normalize and check validity
                    if np.any(new_weights < -1e-9):
                        continue
                    new_weights = np.maximum(new_weights, 0)
                    s = new_weights.sum()
                    if s <= 1e-9:
                        continue
                    new_weights = new_weights / s
                    
                    new_prob = calc_prob(current_support, new_weights)
                    if new_prob > current_prob:
                        current_weights = new_weights
                        current_prob = new_prob
                        improved = True
                        break
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
            
            if current_prob > best_prob:
                best_prob = current_prob
                best_weights = current_weights.copy()
                best_support = current_support.copy()

    # Also try optimizing M locally if time permits?
    # For now, let's stick to optimizing weights for the best M found.
    # Actually, the grid search over M was coarse. Let's refine M too.
    
    # Let's do a second pass: refine M for the best weights found so far.
    if best_weights is not None:
        w0, w1, wM = best_weights
        # Try different M values around the best M
        # We don't track which M was best explicitly, so let's just re-evaluate for a few M values
        # and pick the best.
        
        M_candidates = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
        for M in M_candidates:
            support = np.array([0.0, 1.0, M], dtype=np.float64)
            prob = calc_prob(support, best_weights)
            if prob > best_prob:
                best_prob = prob
                best_support = support.copy()
                best_weights = best_weights.copy()
                
        # Refine M locally
        M_current = best_support[2]
        step_M = 1.0
        for _ in range(100):
            if time.time() > deadline:
                break
            improved = False
            for direction in [-1, 1]:
                new_M = M_current + direction * step_M
                if new_M <= 1.0:  # Must be > 1 to be distinct from 1 and > 0
                    continue
                support = np.array([0.0, 1.0, new_M], dtype=np.float64)
                prob = calc_prob(support, best_weights)
                if prob > best_prob:
                    best_prob = prob
                    best_support = support.copy()
                    M_current = new_M
                    improved = True
                    break
            if not improved:
                step_M *= 0.5
                if step_M < 0.01:
                    break

    # Ensure weights are valid and normalized
    if best_weights is None:
        # Fallback to uniform
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights

    # Final sanity check
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9) or best_weights.sum() <= 1e-9:
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights

    return best_support, best_weights

# EVOLVE-BLOCK-END

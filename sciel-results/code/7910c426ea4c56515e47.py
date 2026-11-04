"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p  # 40
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    # Build schoolbook decomposition
    U_school = np.zeros((n * m, target_rank))
    V_school = np.zeros((m * p, target_rank))
    W_school = np.zeros((p * n, target_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    # Try to find a lower-rank decomposition
    # For rank r, we need to solve:
    # T[a,b,c] = sum_{t=0}^{r-1} U[a,t] * V[b,t] * W[c,t]
    # This is a nonlinear system. We'll try a practical approach.
    
    def verify_decomposition(U, V, W, r):
        """Verify that the decomposition exactly reproduces the target tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for t in range(r):
                        val += U[idx_u, t] * V[idx_v, t] * W[idx_w, t]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check zero entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Check if this is a "one" position
                    is_one = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if a == i * m + j and b == j * p + k and c == k * n + i:
                                    is_one = True
                    if not is_one:
                        val = 0.0
                        for t in range(r):
                            val += U[a, t] * V[b, t] * W[c, t]
                        if abs(val) > 1e-9:
                            return False
        return True
    
    # Try rank 39: use a simple heuristic
    # Start with schoolbook and try to eliminate one term by adjusting others
    # This is a nonlinear problem, so we'll try a few strategies
    
    best_rank = target_rank
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Strategy: For each pair of schoolbook terms, try to see if one can be 
    # expressed as a combination that allows merging
    # For now, let's try a direct approach: check if rank 39 is possible
    # by solving a simplified version
    
    # For 2x4x5, let's try to see if we can find a rank-39 decomposition
    # by exploiting the structure
    
    # The key insight: if we can find vectors u, v, w such that
    # u ⊗ v ⊗ w is in the span of the other 39 rank-1 tensors,
    # then we might be able to adjust to get a rank-39 decomposition.
    
    # Let's try a different approach: use the fact that for small matrices,
    # the tensor rank can sometimes be computed exactly.
    
    # For now, return the schoolbook decomposition if we can't find better quickly
    if time.time() > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to find rank 39 by fixing 38 of the 40 terms and solving for adjustments
    # This is still nonlinear, so let's try a simple perturbation approach
    
    # For each term to eliminate, try to find a small perturbation of the other terms
    # that compensates for the removed term
    
    # This is complex. Let's just return schoolbook for now.
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

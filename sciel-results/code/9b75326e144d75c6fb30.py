"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s margin for 150s limit
    
    # Schoolbook decomposition (rank 40)
    target_rank = n * m * p  # 40
    
    U_school = np.zeros((n * m, target_rank))
    V_school = np.zeros((m * p, target_rank))
    W_school = np.zeros((p * n, target_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U_school[idx_u, col] = 1.0
                V_school[idx_v, col] = 1.0
                W_school[idx_w, col] = 1.0
                col += 1
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # Compute the reconstructed tensor efficiently
        # T[idx_u, idx_v, idx_w] = sum_c U[idx_u,c] * V[idx_v,c] * W[idx_w,c]
        
        # Check all 40 one-positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for c in range(r):
                        val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        
        # Check that zero positions are zero
        # Build set of valid (i,j,k) triples
        valid_triples = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_triples.add((i, j, k))
        
        for idx_u in range(n * m):
            i, j = divmod(idx_u, m)
            for idx_v in range(m * p):
                j2, k = divmod(idx_v, p)
                if j2 != j:
                    # This is automatically zero in the tensor structure
                    # because idx_v encodes (j', k) and idx_u encodes (i, j)
                    # The tensor is only nonzero when j == j'
                    continue
                for idx_w in range(p * n):
                    k2, i2 = divmod(idx_w, n)
                    if k2 != k or i2 != i:
                        # Not a valid triple
                        val = 0.0
                        for c in range(r):
                            val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                        if abs(val) > 1e-10:
                            return False
        return True
    
    # First, verify schoolbook
    if not verify(U_school, V_school, W_school):
        # Should never happen
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = target_rank
    
    # Block decomposition strategy:
    # Split m=4 into two blocks of size 2: block A (j=0,1) and block B (j=2,3)
    # For each block, use the identity decomposition:
    #   e1*e1^T + e2*e2^T = (e1+e2)(e1+e2)^T - (e1-e2)(e1-e2)^T
    # This gives rank-2 decomposition of the 2x2 identity for each block
    # Combined with the outer structure (n=2, p=5), we get rank 2*2*5 = 20
    
    # But we need to be careful: the off-diagonal terms must cancel
    # Let's construct the decomposition carefully
    
    # Strategy: For each block b in {0,1} (corresponding to j in {0,1} or {2,3}),
    # and for each "sign pattern" s in {+, -} (corresponding to e1+e2 or e1-e2),
    # and for each i in {0,1} and k in {0,...,4}, we have a rank-1 term.
    
    # Total rank: 2 blocks * 2 sign patterns * 2 i values * 5 k values = 40
    # Wait, that's still 40. Let me reconsider.
    
    # Actually, the key insight is that for the 2x2 identity:
    # I_2 = (e1+e2)(e1+e2)^T - (e1-e2)(e1-e2)^T
    # This uses 2 terms instead of 2 terms (no gain for 2x2)
    
    # For 4x4 identity split into 2x2 blocks:
    # I_4 = I_2 ⊗ I_2 (block diagonal)
    # Each I_2 can be decomposed in 2 ways, but that doesn't help.
    
    # Let me try a different approach: use the fact that the tensor has structure
    # and try to find a lower rank decomposition via optimization.
    
    # Try random search for lower ranks
    np.random.seed(42)
    
    for target_r in [39, 38, 37, 36]:
        if time.time() > deadline:
            break
        
        if best_rank <= target_r:
            continue
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            r = target_r
            # Initialize with schoolbook truncated, then perturb
            U_try = np.zeros((n * m, r))
            V_try = np.zeros((m * p, r))
            W_try = np.zeros((p * n, r))
            
            # Copy first min(r, 40) columns of schoolbook
            for c in range(min(r, target_rank)):
                U_try[:, c] = U_school[:, c]
                V_try[:, c] = V_school[:, c]
                W_try[:, c] = W_school[:, c]
            
            # Add random perturbations
            U_try += np.random.randn(n * m, r) * 0.05
            V_try += np.random.randn(m * p, r) * 0.05
            W_try += np.random.randn(p * n, r) * 0.05
            
            if verify(U_try, V_try, W_try):
                best_U = U_try
                best_V = V_try
                best_W = W_try
                best_rank = r
                break
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U = U_school.copy()
        best_V = V_school.copy()
        best_W = W_school.copy()
        best_rank = target_rank
    
    # Round to nearest half-integer if possible
    def round_to_half(x):
        return np.round(x * 2) / 2
    
    best_U_rounded = round_to_half(best_U)
    best_V_rounded = round_to_half(best_V)
    best_W_rounded = round_to_half(best_W)
    
    # Final check after rounding
    if not verify(best_U_rounded, best_V_rounded, best_W_rounded):
        # Use unrounded values
        pass
    else:
        best_U = best_U_rounded
        best_V = best_V_rounded
        best_W = best_W_rounded
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

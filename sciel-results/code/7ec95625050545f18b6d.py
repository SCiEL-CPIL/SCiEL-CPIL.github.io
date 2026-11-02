"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _build_target(n, m, p):
    """Return the matrix-multiplication tensor as a dict of 1s (sparse)."""
    tgt = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tgt[(i * m + j, j * p + k, k * n + i)] = 1
    return tgt

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_exact(U, V, W, tgt, n, m, p):
    """Exact check that sum_r U[a,r]V[b,r]W[c,r] reproduces the target."""
    R = len(U[0])
    # Represent each factor column as a dict of nonzero entries.
    def col_dict(mat, r):
        d = {}
        for a in range(len(mat)):
            v = mat[a][r]
            if v != 0:
                d[a] = v
        return d
    
    # Precompute
    cols = []
    for r in range(R):
        cols.append((col_dict(U, r), col_dict(V, r), col_dict(W, r)))
    
    acc = {}
    for cu, cv, cw in cols:
        for a, ua in cu.items():
            for b, vb in cv.items():
                uv = ua * vb
                for c, wc in cw.items():
                    acc[(a, b, c)] = acc.get((a, b, c), 0) + uv * wc
    
    # Every target entry must equal 1
    for key, val in tgt.items():
        if acc.get(key, 0) != val:
            return False
    # No spurious entries
    for key, val in acc.items():
        if val != 0 and key not in tgt:
            return False
    return True

def _to_int_matrix(mat):
    """Convert a numpy matrix to a list of lists of ints, rounding to nearest integer."""
    return [[int(round(x)) for x in row] for row in mat]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110  # Safety margin before 150s kill
    
    U_dim, V_dim, W_dim = n * m, m * p, p * n
    tgt = _build_target(n, m, p)
    
    # Get schoolbook solution as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = len(U_school[0])
    
    # Target tensor as numpy array for efficient ALS
    target = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find lower rank decompositions using ALS
    # We search ranks from a lower bound up to schoolbook - 1
    # Known lower bound for 3x4x6 is around 50-60, but we'll try a range
    
    # Try several rank values, starting from lower ones
    ranks_to_try = list(range(50, best_rank))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts for this rank
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            np.random.seed(42 + rank * 1000 + restart * 7)
            U = np.random.randn(U_dim, rank) * 0.5
            V = np.random.randn(V_dim, rank) * 0.5
            W = np.random.randn(W_dim, rank) * 0.5
            
            # ALS iterations
            converged = False
            for it in range(30):
                if time.time() > deadline - 10:
                    break
                
                # Update U: fix V and W
                # U[a, r] = sum_{b,c} target[a,b,c] * V[b,r] * W[c,r] / (sum_{b,c} V[b,r]^2 * W[c,r]^2)
                # We can vectorize this
                # For each a, U[a, :] = sum_{b,c} target[a,b,c] * (V * W)[b,c,:] / (V^T V * W^T W)
                # Actually, let's compute it more carefully
                
                # Precompute V^T V and W^T W
                VtV = V.T @ V  # rank x rank
                WtW = W.T @ W  # rank x rank
                
                # For each a, compute numerator[a, r] = sum_{b,c} target[a,b,c] * V[b,r] * W[c,r]
                # This is a bit expensive. Let's do it in a loop over a
                new_U = np.zeros_like(U)
                for a in range(U_dim):
                    # target[a, :, :] is V_dim x W_dim
                    # We want sum_{b,c} target[a,b,c] * V[b,r] * W[c,r]
                    # This is trace(target[a] * V[r,:]^T * W[r,:]^T) for each r
                    # Or: (target[a] * V.T[r]) @ W.T[r]
                    # Actually: sum_{b,c} target[a,b,c] * V[b,r] * W[c,r]
                    # = (target[a] * V[:,r]) @ W[:,r]
                    # Let's compute all at once:
                    # numerator[a, r] = sum_{b,c} target[a,b,c] * V[b,r] * W[c,r]
                    # We can compute this as:
                    # temp[b, r] = sum_c target[a,b,c] * W[c,r] = (target[a] @ W.T)[b, r]
                    # numerator[a, r] = sum_b temp[b, r] * V[b, r]
                    temp = target[a] @ W.T  # V_dim x rank
                    new_U[a, :] = np.sum(temp * V, axis=0)  # rank
                
                # Denominator: sum_{b,c} V[b,r]^2 * W[c,r]^2
                # = sum_r (sum_b V[b,r]^2) * (sum_c W[c,r]^2)
                # Wait, no: sum_{b,c} V[b,r]^2 * W[c,r]^2 = (sum_b V[b,r]^2) * (sum_c W[c,r]^2)
                # This is element-wise product of (V^T V) diagonal and (W^T W) diagonal? No.
                # sum_{b,c} V[b,r]^2 * W[c,r]^2 = (sum_b V[b,r]^2) * (sum_c W[c,r]^2)
                # Let V_sq = V^T V, so V_sq[r,r] = sum_b V[b,r]^2
                # Let W_sq = W^T W, so W_sq[r,r] = sum_c W[c,r]^2
                # Denominator[r] = V_sq[r,r] * W_sq[r,r]
                V_sq_diag = np.diag(VtV)
                W_sq_diag = np.diag(WtW)
                denom = V_sq_diag * W_sq_diag
                denom[denom == 0] = 1e-10
                new_U = new_U / denom
                
                U = new_U
                
                # Update V: fix U and W
                new_V = np.zeros_like(V)
                for b in range(V_dim):
                    temp = target[:, b, :] @ U.T  # W_dim x rank
                    # Wait, target[:, b, :] is U_dim x W_dim
                    # We want sum_{a,c} target[a,b,c] * U[a,r] * W[c,r]
                    # temp[c, r] = sum_a target[a,b,c] * U[a,r] = (target[:,b,:] @ U.T)[c, r]
                    # new_V[b, r] = sum_c temp[c, r] * W[c, r]
                    temp = target[:, b, :] @ U.T  # W_dim x rank
                    new_V[b, :] = np.sum(temp * W, axis=0)
                
                U_sq_diag = np.diag(U.T @ U)
                W_sq_diag = np.diag(W.T @ W)
                denom = U_sq_diag * W_sq_diag
                denom[denom == 0] = 1e-10
                new_V = new_V / denom
                V = new_V
                
                # Update W: fix U and V
                new_W = np.zeros_like(W)
                for c in range(W_dim):
                    temp = target[:, :, c] @ V.T  # U_dim x rank
                    # target[:, :, c] is U_dim x V_dim
                    # We want sum_{a,b} target[a,b,c] * U[a,r] * V[b,r]
                    # temp[a, r] = sum_b target[a,b,c] * V[b,r] = (target[:,:,c] @ V.T)[a, r]
                    # new_W[c, r] = sum_a temp[a, r] * U[a, r]
                    temp = target[:, :, c] @ V.T  # U_dim x rank
                    new_W[c, :] = np.sum(temp * U, axis=0)
                
                U_sq_diag = np.diag(U.T @ U)
                V_sq_diag = np.diag(V.T @ V)
                denom = U_sq_diag * V_sq_diag
                denom[denom == 0] = 1e-10
                new_W = new_W / denom
                W = new_W
                
                # Check convergence
                if it > 0:
                    # Simple convergence check: change in U
                    pass
            
            # After ALS, round to nearest integer and verify
            U_int = _to_int_matrix(U)
            V_int = _to_int_matrix(V)
            W_int = _to_int_matrix(W)
            
            if _verify_exact(U_int, V_int, W_int, tgt, n, m, p):
                best_U, best_V, best_W = U_int, V_int, W_int
                best_rank = rank
                break  # Found a valid decomposition at this rank, no need to try higher ranks for this rank value
        
        # If we found a valid decomposition, we can break early if rank is low enough
        # But we want the minimum rank, so we continue searching lower ranks
        # Actually, we're iterating from low to high, so if we find one, we can break
        # Wait, we're trying ranks in increasing order, so the first valid one is the best for that rank
        # But we should continue to see if we can find an even lower rank? No, we're going up.
        # We should go from low to high, and the first valid one we find is the best.
        # But we might not find a valid one at rank 50, then try 51, etc.
        # So we break out of the restart loop when we find a valid one, but continue to next rank
        # Actually, if we find a valid one at rank R, we should update best and continue
        # to see if we can find a valid one at a lower rank? No, we're going from low to high.
        # So the first valid one we find (lowest rank) is the best. We can break out of all loops.
        if best_rank == rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

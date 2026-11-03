"""Matrix multiplication 2,3,5 — rank-28 with linearized rank-27 search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]
START_TIME = time.time()
DEADLINE = START_TIME + 140.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    U28, V28, W28 = _build_rank28(n, m, p)
    
    if time.time() > DEADLINE - 30.0:
        return _to_dict(U28, V28, W28)
    
    # Attempt rank-27 via linearized search
    result = _try_rank27_linear(U28, V28, W28, n, m, p)
    if result is not None:
        return _to_dict(*result)
    
    return _to_dict(U28, V28, W28)

def _build_rank28(n, m, p):
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        U[0*m + j_off][c] = 1; U[1*m + j_off+1][c] = 1
        V[j_off*p + k_off][c] = 1; V[j_off+1*p + k_off+1][c] = 1
        W[k_off*n + 0][c] = 1; W[k_off+1*n + 1][c] = 1
        c += 1
        
        U[1*m + j_off][c] = 1; U[1*m + j_off+1][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + 1][c] = 1; W[k_off+1*n + 1][c] = -1
        c += 1
        
        U[0*m + j_off][c] = 1
        V[j_off*p + k_off+1][c] = 1; V[j_off+1*p + k_off+1][c] = -1
        W[k_off+1*n + 0][c] = 1; W[k_off+1*n + 1][c] = 1
        c += 1
        
        U[1*m + j_off+1][c] = 1
        V[j_off+1*p + k_off][c] = 1; V[j_off*p + k_off][c] = -1
        W[k_off*n + 0][c] = 1; W[k_off*n + 1][c] = 1
        c += 1
        
        U[0*m + j_off][c] = 1; U[0*m + j_off+1][c] = 1
        V[j_off+1*p + k_off+1][c] = 1
        W[k_off*n + 0][c] = -1; W[k_off+1*n + 0][c] = 1
        c += 1
        
        U[1*m + j_off][c] = 1; U[0*m + j_off][c] = -1
        V[j_off*p + k_off][c] = 1; V[j_off*p + k_off+1][c] = 1
        W[k_off+1*n + 1][c] = 1
        c += 1
        
        U[0*m + j_off+1][c] = 1; U[1*m + j_off+1][c] = -1
        V[j_off+1*p + k_off][c] = 1; V[j_off+1*p + k_off+1][c] = 1
        W[k_off*n + 0][c] = 1
        c += 1

    add_strassen_block(0, 0)
    add_strassen_block(0, 2)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    
    assert c == rank
    return U, V, W

def _to_dict(U, V, W):
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank27_linear(U28, V28, W28, n, m, p):
    """
    Linearized rank-27 search:
    For each candidate column r to eliminate, try to find a vector alpha (length 27)
    such that the tensor equation holds with column r absorbed into others.
    
    Key insight: The tensor equation sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 is linear
    in each factor's column when the other two are fixed. So if we fix U and V,
    the system is linear in W.
    
    Strategy: For each pair of columns (r, s) to merge, try to find a scalar t
    such that modifying column s by t*column_r and removing column r still works.
    This is a linear system in W entries.
    """
    rank28 = U28.shape[1]
    
    # First verify rank-28 is correct
    if not _verify(U28, V28, W28, n, m, p):
        return None
    
    # Strategy 1: Try removing one column and solving for modified W
    # For column r to remove, we need:
    # sum_{s != r} U[i,j,s] * V[j,k,s] * W_new[k,i,s] = 1 for all (i,j,k)
    # where W_new is the new W matrix (27 columns)
    # But this requires the 27-column system to be solvable.
    # The system has 30 equations (n*m*p) and 27*10=270 unknowns (W entries),
    # so it's underdetermined. We need to check if a solution exists.
    
    # More practically: check if the tensor T = sum_{s!=r} U_s ⊗ V_s ⊗ W_s
    # can equal the target. This means the target tensor minus the contribution
    # of column r must be in the span of {U_s ⊗ V_s ⊗ e for s != r, e in W-space}.
    
    # Equivalent: For each (i,j,k), we need:
    # U[i,j,r]*V[j,k,r]*W[k,i,r] = sum_{s!=r} U[i,j,s]*V[j,k,s]*(W_new[k,i,s] - W28[k,i,s])
    # + sum_{s!=r} U[i,j,s]*V[j,k,s]*W28[k,i,s]
    
    # Let D[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r] - sum_{s!=r} U[i,j,s]*V[j,k,s]*W28[k,i,s]
    # We need D[i,j,k] = sum_{s!=r} U[i,j,s]*V[j,k,s]*delta_W[k,i,s]
    
    # This is a linear system: for each (i,j,k), D[i,j,k] = sum_{s!=r, k',i'} 
    #   U[i,j,s]*V[j,k,s]*delta_W[k',i',s] where (k',i') = (k,i)
    
    # Actually simpler: fix U,V, solve for W. The system is:
    # For all (i,j,k): sum_{s=0}^{26} U[i,j,s]*V[j,k,s]*W[k,i,s] = 1
    # This is 30 equations in 27*10 = 270 unknowns (W entries).
    # Under-determined, so we use least squares and check residual.
    
    for r in range(rank28):
        if time.time() > DEADLINE - 5.0:
            return None
        
        # Build U, V without column r
        idx = [s for s in range(rank28) if s != r]
        U_r = U28[:, idx]
        V_r = V28[:, idx]
        
        # Build linear system: A @ W_flat = b
        # where A has one row per (i,j,k), with entries A[(i,j,k), (s, k', i')] = 
        #   U[i,j,s]*V[j,k,s] if (k',i') == (k,i) else 0
        # b[(i,j,k)] = 1
        
        n_eq = n * m * p  # 30
        n_unk = 27 * p * n  # 27 * 10 = 270
        
        A = np.zeros((n_eq, n_unk))
        b = np.ones(n_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for s in range(27):
                        # W[k*n+i, s] is at flat index s*(p*n) + k*n + i
                        w_idx = s * (p * n) + k * n + i
                        A[eq_idx, w_idx] = U_r[i*m+j, s] * V_r[j*p+k, s]
                    eq_idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=1e-12)
        except:
            continue
        
        # Check if solution is exact
        if residuals.size > 0 and residuals[0] > 1e-8:
            continue
        
        # Reshape W_flat back to W matrix
        W_new = W_flat.reshape(27, p * n).T  # shape (p*n, 27)
        
        # Verify exactness
        if _verify(U_r, V_r, W_new, n, m, p):
            return (U_r, V_r, W_new)
    
    # Strategy 2: Try modifying two columns (merge two Strassen terms)
    # For pairs of columns that might be mergeable, try to find a 2D subspace
    # reduction. This is more complex; skip for now if time is tight.
    
    if time.time() > DEADLINE - 10.0:
        return None
    
    # Strategy 3: Try a different base decomposition structure
    # Use one Strassen block (7 terms) + schoolbook for remaining 23 = 30,
    # then try to reduce. Or try overlapping blocks.
    
    # Try: Strassen on j={0,1},k={0,1} (7 terms) + Strassen on j={1,2},k={3,4} (7 terms)
    # Overlap at j=1. This might allow cross-block cancellation.
    U_alt, V_alt, W_alt = _build_alt_rank28(n, m, p)
    if U_alt is not None:
        if _verify(U_alt, V_alt, W_alt, n, m, p):
            # Try rank-27 on this alternative
            for r in range(28):
                if time.time() > DEADLINE - 3.0:
                    break
                idx = [s for s in range(28) if s != r]
                U_r = U_alt[:, idx]
                V_r = V_alt[:, idx]
                
                n_eq = n * m * p
                n_unk = 27 * p * n
                A = np.zeros((n_eq, n_unk))
                b = np.ones(n_eq)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for s in range(27):
                                w_idx = s * (p * n) + k * n + i
                                A[eq_idx, w_idx] = U_r[i*m+j, s] * V_r[j*p+k, s]
                            eq_idx += 1
                
                try:
                    W_flat, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=1e-12)
                except:
                    continue
                
                if residuals.size > 0 and residuals[0] > 1e-8:
                    continue
                
                W_new = W_flat.reshape(27, p * n).T
                if _verify(U_r, V_r, W_new, n, m, p):
                    return (U_r, V_r, W_new)
    
    return None

def _build_alt_rank28(n, m, p):
    """Alternative rank-28 with overlapping Strassen blocks."""
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        U[0*m + j_off][c] = 1; U[1*m + j_off+1][c] = 1
        V[j_off*p + k_off][c] = 1; V[j_off+1*p + k_off+1][c] = 1
        W[k_off*n + 0][c] = 1; W[k_off+1*n + 1][c] = 1
        c += 1
        U[1*m + j_off][c] = 1; U[1*m + j_off+1][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + 1][c] = 1; W[k_off+1*n + 1][c] = -1
        c += 1
        U[0*m + j_off][c] = 1
        V[j_off*p + k_off+1][c] = 1; V[j_off+1*p + k_off+1][c] = -1
        W[k_off+1*n + 0][c] = 1; W[k_off+1*n + 1][c] = 1
        c += 1
        U[1*m + j_off+1][c] = 1
        V[j_off+1*p + k_off][c] = 1; V[j_off*p + k_off][c] = -1
        W[k_off*n + 0][c] = 1; W[k_off*n + 1][c] = 1
        c += 1
        U[0*m + j_off][c] = 1; U[0*m + j_off+1][c] = 1
        V[j_off+1*p + k_off+1][c] = 1
        W[k_off*n + 0][c] = -1; W[k_off+1*n + 0][c] = 1
        c += 1
        U[1*m + j_off][c] = 1; U[0*m + j_off][c] = -1
        V[j_off*p + k_off][c] = 1; V[j_off*p + k_off+1][c] = 1
        W[k_off+1*n + 1][c] = 1
        c += 1
        U[0*m + j_off+1][c] = 1; U[1*m + j_off+1][c] = -1
        V[j_off+1*p + k_off][c] = 1; V[j_off+1*p + k_off+1][c] = 1
        W[k_off*n + 0][c] = 1
        c += 1

    # Overlapping blocks: j={0,1},k={0,1} and j={1,2},k={3,4}
    add_strassen_block(0, 0)
    add_strassen_block(1, 3)
    
    # Schoolbook for remaining
    covered = set()
    for j in (0, 1):
        for k in (0, 1):
            covered.add((j, k))
    for j in (1, 2):
        for k in (3, 4):
            covered.add((j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (j, k) in covered:
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    
    if c != rank:
        return None, None, None
    return U, V, W
# EVOLVE-BLOCK-END

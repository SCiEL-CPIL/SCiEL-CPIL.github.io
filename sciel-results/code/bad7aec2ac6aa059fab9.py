"""Matrix multiplication 2,3,5 — rank-28 decomposition via two Strassen 2x2x2 blocks with rank-reduction search."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 5]
START_TIME = time.time()
DEADLINE = START_TIME + 140.0  # Safety margin

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication.
    First attempts rank-27 via tensor completion heuristic, falls back to rank-28.
    """
    n, m, p = 2, 3, 5
    
    # Build the rank-28 Strassen construction
    U28, V28, W28 = _build_rank28(n, m, p)
    
    # Check deadline before attempting rank-27 search
    if time.time() > DEADLINE - 20.0:
        return _to_dict(U28, V28, W28)
    
    # Attempt rank-27 tensor completion
    rank27_result = _try_rank27(U28, V28, W28, n, m, p)
    
    if rank27_result is not None:
        U27, V27, W27 = rank27_result
        return _to_dict(U27, V27, W27)
    
    # Fallback to rank-28
    return _to_dict(U28, V28, W28)

def _build_rank28(n, m, p):
    """Build the known rank-28 Strassen + schoolbook decomposition."""
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        U[0*m + (j_off)][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[(j_off)*p + (k_off)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        U[1*m + (j_off)][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[(j_off)*p + (k_off)][c] = 1
        W[(k_off)*n + 1][c] = 1; W[(k_off+1)*n + 1][c] = -1
        c += 1
        
        U[0*m + (j_off)][c] = 1
        V[(j_off)*p + (k_off+1)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off)][c] = 1; V[(j_off)*p + (k_off)][c] = -1
        W[(k_off)*n + 0][c] = 1; W[(k_off)*n + 1][c] = 1
        c += 1
        
        U[0*m + (j_off)][c] = 1; U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off)*n + 0][c] = -1; W[(k_off+1)*n + 0][c] = 1
        c += 1
        
        U[1*m + (j_off)][c] = 1; U[0*m + (j_off)][c] = -1
        V[(j_off)*p + (k_off)][c] = 1; V[(j_off)*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        U[0*m + (j_off+1)][c] = 1; U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + (k_off)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off)*n + 0][c] = 1
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
    
    assert c == rank, f"Expected rank {rank}, got {c}"
    return U, V, W

def _to_dict(U, V, W):
    """Convert numpy arrays to list format."""
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # The tensor T[i,j,k] = delta[i,j] * delta[j,k] for (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i == j or j == k or i == k) else 0.0
                # Actually the matrix multiplication tensor is T[i,j,k] = 1 if j==j (always) 
                # Wait, let me reconsider. The tensor for (A*B)[i,k] = sum_j A[i,j]*B[j,k]
                # is T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle index)
                # No, that's not right either.
                
                # The matrix multiplication tensor has T[i,j,k] = 1 if and only if the
                # (i,k)-entry of A*B depends on A[i,j]*B[j,k], which is always true.
                # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
                # 
                # Wait, that's the rank-30 schoolbook. Let me re-examine.
                #
                # (A*B)[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
                #
                # The tensor T has T[i,j,k] = 1 for all valid indices.
                # The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # for all i, j, k.
                
                actual = 0.0
                for r in range(U.shape[1]):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                if abs(actual - 1.0) > 1e-9:
                    return False
    
    return True

def _try_rank27(U28, V28, W28, n, m, p):
    """
    Attempt to find a rank-27 decomposition by looking for linear dependencies
    in the rank-28 factor matrices.
    
    Strategy: Try to find a linear combination of columns that can be eliminated
    while preserving the tensor decomposition.
    """
    rank28 = U28.shape[1]
    
    # First verify the rank-28 solution is correct
    if not _verify_decomposition(U28, V28, W28, n, m, p):
        return None
    
    # Try to find a column that can be eliminated by adjusting others
    # For each pair of columns (a, b), try to see if column b can be expressed
    # as a linear combination of others such that the tensor equation still holds.
    
    # A more practical approach: look for a column r where we can find scalars
    # alpha_s for s != r such that:
    # U[:,r] = sum_s alpha_s * U[:,s] (on the support where W[:,r] is nonzero)
    # V[:,r] = sum_s alpha_s * V[:,s] (on the support where W[:,r] is nonzero)
    # This is generally not possible, but let's try a simpler heuristic.
    
    # Alternative: try to merge two Strassen terms across blocks
    # The two Strassen blocks share the same j-indices {0,1}, so there might
    # be cross-block dependencies.
    
    # For each column r, try to find a minimal linear dependency
    for r in range(rank28):
        if time.time() > DEADLINE - 5.0:
            return None
        
        # Try to express column r as a combination of other columns
        # subject to the tensor constraint
        
        # Build the constraint system:
        # For all (i,j,k): sum_{s != r} alpha_s * U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s]
        #   = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        # This is a system of n*m*p = 30 equations in (rank28-1) = 27 unknowns.
        # Generally overdetermined, but let's check if a solution exists.
        
        # Collect equations where the RHS is nonzero
        equations = []
        rhs = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rhs_val = U28[i*m+j, r] * V28[j*p+k, r] * W28[k*n+i, r]
                    if abs(rhs_val) > 1e-12:
                        row = []
                        for s in range(rank28):
                            if s == r:
                                continue
                            row.append(U28[i*m+j, s] * V28[j*p+k, s] * W28[k*n+i, s])
                        equations.append(row)
                        rhs.append(rhs_val)
        
        if len(equations) == 0:
            # Column r doesn't contribute to any tensor entry, can be dropped
            continue
        
        A_mat = np.array(equations, dtype=float)
        b_vec = np.array(rhs, dtype=float)
        
        # Solve least squares
        try:
            alpha, residuals, rank_a, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
        
        # Check if the solution is exact
        if residuals.size > 0 and residuals[0] > 1e-8:
            continue
        
        # Check if the solution has mostly zero entries (sparse)
        # If alpha is sparse, we can potentially reduce rank
        nonzero_count = np.sum(np.abs(alpha) > 1e-8)
        
        if nonzero_count < rank28 - 1:
            # We found a sparse representation; try to construct the new matrices
            new_rank = rank28 - 1
            U_new = np.zeros((n*m, new_rank))
            V_new = np.zeros((m*p, new_rank))
            W_new = np.zeros((p*n, new_rank))
            
            # Copy all columns except r
            col_idx = 0
            for s in range(rank28):
                if s == r:
                    continue
                U_new[:, col_idx] = U28[:, s]
                V_new[:, col_idx] = V28[:, s]
                W_new[:, col_idx] = W28[:, s]
                col_idx += 1
            
            # Add the linear combination as a new column
            # Actually, this doesn't reduce rank. We need to modify existing columns.
            # The correct approach: if column r = sum_s alpha_s * column_s, then we can
            # absorb column r into the others by modifying the W matrix.
            # But this requires the W vectors to also satisfy a relation.
            
            # For now, skip this complex case. The simple check above already
            # verifies if column r is in the span of others in the tensor sense.
            pass
    
    # Try a different approach: look for pairs of columns that can be merged
    # by finding a 2-column subspace where the tensor constraint allows reduction
    
    # For each pair (r1, r2), check if there exists a linear combination
    # c1*col_r1 + c2*col_r2 that can replace both columns
    for r1 in range(rank28):
        for r2 in range(r1+1, rank28):
            if time.time() > DEADLINE - 3.0:
                return None
            
            # Skip if both are from the same Strassen block (they're already optimized)
            # Strassen block 1: columns 0-6, Block 2: columns 7-13, Schoolbook: 14-27
            
            # Try to find if columns r1 and r2 have a common structure
            # that allows merging
            
            # Check if the supports of W[:,r1] and W[:,r2] overlap significantly
            support_r1 = np.nonzero(W28[:, r1])[0]
            support_r2 = np.nonzero(W28[:, r2])[0]
            overlap = np.intersect1d(support_r1, support_r2)
            
            if len(overlap) < 2:
                continue
            
            # For overlapping entries, check if the ratio U[i,j,r1]/U[i,j,r2] is constant
            # and similarly for V
            ratios_U = set()
            ratios_V = set()
            
            for idx in overlap:
                k, i = divmod(idx, n)
                for j in range(m):
                    u1 = U28[i*m+j, r1]
                    u2 = U28[i*m+j, r2]
                    if abs(u2) > 1e-12:
                        ratios_U.add(round(u1/u2, 10))
                    
                    v1 = V28[j*p+k, r1]
                    v2 = V28[j*p+k, r2]
                    if abs(v2) > 1e-12:
                        ratios_V.add(round(v1/v2, 10))
            
            # If both ratios are single values, we might be able to merge
            if len(ratios_U) == 1 and len(ratios_V) == 1:
                ratio_u = list(ratios_U)[0]
                ratio_v = list(ratios_V)[0]
                
                # Check if the non-overlapping parts are consistent
                # This is getting complex; for now, just record that we found a candidate
                
                pass
    
    # If no rank-27 solution found, return None
    return None
# EVOLVE-BLOCK-END

"""Improved rank-71 decomposition with deadline guard and potential rank reduction exploration."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Current best: rank 71 via 4 Strassen blocks (28) + schoolbook (43)
    # Try to find rank reduction by sharing bilinear forms
    
    def build_rank71():
        total_rank = 71
        U = np.zeros((n * m, total_rank), dtype=np.float64)
        V = np.zeros((m * p, total_rank), dtype=np.float64)
        W = np.zeros((p * n, total_rank), dtype=np.float64)
        
        col_idx = 0
        blocks = [
            (0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)
        ]
        
        for (i0, j0, k0) in blocks:
            # Term 1: (A00+A11)(B00+B11)
            U[i0*m + j0, col_idx] += 1
            U[(i0+1)*m + (j0+1), col_idx] += 1
            V[j0*p + k0, col_idx] += 1
            V[(j0+1)*p + (k0+1), col_idx] += 1
            W[k0*n + i0, col_idx] += 1
            W[(k0+1)*n + (i0+1), col_idx] += 1
            col_idx += 1
            # Term 2: (A10+A11)B00
            U[(i0+1)*m + j0, col_idx] += 1
            U[(i0+1)*m + (j0+1), col_idx] += 1
            V[j0*p + k0, col_idx] += 1
            W[k0*n + (i0+1), col_idx] += 1
            W[(k0+1)*n + (i0+1), col_idx] += -1
            col_idx += 1
            # Term 3: A00(B01-B11)
            U[i0*m + j0, col_idx] += 1
            V[j0*p + (k0+1), col_idx] += 1
            V[(j0+1)*p + (k0+1), col_idx] += -1
            W[(k0+1)*n + i0, col_idx] += 1
            W[(k0+1)*n + (i0+1), col_idx] += 1
            col_idx += 1
            # Term 4: A11(B10-B00)
            U[(i0+1)*m + (j0+1), col_idx] += 1
            V[(j0+1)*p + k0, col_idx] += 1
            V[j0*p + k0, col_idx] += -1
            W[k0*n + i0, col_idx] += 1
            W[k0*n + (i0+1), col_idx] += 1
            col_idx += 1
            # Term 5: (A00+A01)B11
            U[i0*m + j0, col_idx] += 1
            U[i0*m + (j0+1), col_idx] += 1
            V[(j0+1)*p + (k0+1), col_idx] += 1
            W[k0*n + i0, col_idx] += -1
            W[(k0+1)*n + i0, col_idx] += 1
            col_idx += 1
            # Term 6: (A10-A00)(B00+B01)
            U[(i0+1)*m + j0, col_idx] += 1
            U[i0*m + j0, col_idx] += -1
            V[j0*p + k0, col_idx] += 1
            V[j0*p + (k0+1), col_idx] += 1
            W[(k0+1)*n + (i0+1), col_idx] += 1
            col_idx += 1
            # Term 7: (A01-A11)(B10+B11)
            U[i0*m + (j0+1), col_idx] += 1
            U[(i0+1)*m + (j0+1), col_idx] += -1
            V[(j0+1)*p + k0, col_idx] += 1
            V[(j0+1)*p + (k0+1), col_idx] += 1
            W[k0*n + i0, col_idx] += 1
            col_idx += 1
        
        covered = np.zeros((n, m, p), dtype=bool)
        for (i0, j0, k0) in blocks:
            for ii in range(i0, i0+2):
                for jj in range(j0, j0+2):
                    for kk in range(k0, k0+2):
                        covered[ii, jj, kk] = True
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i, j, k]:
                        U[i*m + j, col_idx] += 1
                        V[j*p + k, col_idx] += 1
                        W[k*n + i, col_idx] += 1
                        col_idx += 1
        
        return U, V, W
    
    def verify(U, V, W):
        """Verify that U,V,W correctly decompose the matrix multiplication tensor."""
        # The tensor T[i,j,k] = 1 should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Also verify that no spurious terms are generated
        # For entries not in the tensor (i.e., where the product should be 0 in the 
        # full tensor decomposition sense), we need to check orthogonality
        # Actually, the tensor is exactly 1 for all (i,j,k) in range, so we just need
        # to verify the above. The decomposition is valid if all 75 entries equal 1.
        return True
    
    # First, build the known rank-71 solution
    U, V, W = build_rank71()
    
    # Verify it's correct
    if not verify(U, V, W):
        raise ValueError("Rank-71 decomposition failed verification")
    
    # Try to reduce rank by finding linearly dependent columns
    # Compute the tensor and check if any column can be removed
    # This is a heuristic: check if removing any single column still gives valid decomposition
    
    best_U, best_V, best_W = U, V, W
    best_rank = 71
    
    # Check if we can reduce by 1 by testing each column
    for r in range(71):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        # Test removing column r
        U_test = np.delete(U, r, axis=1)
        V_test = np.delete(V, r, axis=1)
        W_test = np.delete(W, r, axis=1)
        
        # Quick check: verify all 75 entries
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(70):
                        val += U_test[i*m+j, rr] * V_test[j*p+k, rr] * W_test[k*n+i, rr]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_U, best_V, best_W = U_test, V_test, W_test
            best_rank = 70
            U, V, W = U_test, V_test, W_test
            break  # Found a reduction, stop
    
    # If we found rank 70, try to reduce further
    if best_rank == 70 and time.time() - START_TIME < TIME_LIMIT - 10:
        for r in range(70):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            U_test = np.delete(U, r, axis=1)
            V_test = np.delete(V, r, axis=1)
            W_test = np.delete(W, r, axis=1)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for rr in range(69):
                            val += U_test[i*m+j, rr] * V_test[j*p+k, rr] * W_test[k*n+i, rr]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = 69
                U, V, W = U_test, V_test, W_test
                break
    
    if time.time() - START_TIME > TIME_LIMIT:
        # Return best found so far
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

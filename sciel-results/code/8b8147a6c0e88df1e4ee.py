"""Fresh design: search for rank-minimal tensor decomposition of 3x3x6 matrix multiplication."""
import numpy as np
from itertools import product
import time

def _build_tensor_3x3x6():
    """Build the 3x3x6 matrix multiplication tensor T[i,j,k] = 1 for all i,j,k."""
    n, m, p = 3, 3, 6
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _verify_decomposition(U, V, W, T):
    """Verify that U @ V @ W reproduces tensor T."""
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    n, m, p = T.shape
    r = U.shape[1]
    computed = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        computed += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
    # Check exact match (with tolerance for floating point)
    return np.allclose(computed, T, atol=1e-10)

def _verify_exact(U, V, W, T):
    """Verify exact decomposition using rational-friendly check."""
    n, m, p = T.shape
    r = U.shape[1]
    computed = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        computed += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
    return np.max(np.abs(computed - T)) < 1e-9

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 3, 3, 6
    T = _build_tensor_3x3x6()
    
    # Schoolbook: rank 54
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_schoolbook}
    
    # Try to find better decomposition using structured search
    # For 3x3x6, we can exploit the structure: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is equivalent to multiplying three 3x3 matrices (by splitting B into two 3x3 blocks)
    # B = [B1 | B2] where B1, B2 are 3x3
    # C = [C1 | C2] where C1 = A*B1, C2 = A*B2
    # So we need rank(A*B1) + rank(A*B2) <= total rank
    # If we find rank r decomposition for 3x3x3, total rank = 2r
    
    # Known result: rank of 3x3x3 matrix multiplication is 23 (Bini et al.)
    # Let's try to construct a rank-23 decomposition for 3x3x3
    
    # Strategy: Use the fact that for 3x3x3, we can use the "23-multiplication algorithm"
    # The tensor T3[i,j,k] = 1 for all i,j,k in [0,3)
    # We need U3 (9x23), V3 (9x23), W3 (9x23) such that T3[i,j,k] = sum_c U3[i,c]*V3[j,c]*W3[k,c]
    
    # Let's try a parameterized search for small ranks
    # Use a simple gradient-free approach: try random initializations and local search
    
    def try_rank(r):
        """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
        if time.time() > deadline - 10:
            return None
        
        # For 3x3x3 subproblem
        T3 = np.ones((3, 3, 3), dtype=np.float64)
        
        # Try various approaches
        # Approach 1: Start from schoolbook and try to reduce rank via merging
        # Approach 2: Random initialization + iterative refinement
        
        # Use a simple iterative approach
        np.random.seed(42)
        
        for attempt in range(20):
            if time.time() > deadline - 15:
                return None
            
            # Initialize with random values
            U3 = np.random.randn(9, r) * 0.5
            V3 = np.random.randn(9, r) * 0.5
            W3 = np.random.randn(9, r) * 0.5
            
            # Iterative refinement: alternate projections
            for it in range(50):
                if time.time() > deadline - 15:
                    break
                
                # Update U: for each i, U[i,:] should satisfy
                # T3[i,j,k] = sum_c U[i,c]*V3[j,c]*W3[k,c]
                # This is a linear system in U[i,:]
                # Fix V3, W3, solve for U
                for i in range(3):
                    # T3[i,:,:] = sum_c U[i,c] * V3[j,c]*W3[k,c]
                    # Let M_jk = V3[j,:]*W3[k,:] (element-wise, shape 3x3)
                    # T3[i,:,:] = sum_c U[i,c] * M_c where M_c[j,k] = V3[j,c]*W3[k,c]
                    # This is a 9-variable linear system
                    A_mat = np.zeros((9, r))
                    for c in range(r):
                        A_mat[:, c] = (V3[:, c] * W3[:, c]).flatten()
                    b_vec = T3[i, :, :].flatten()
                    # Solve least squares
                    U3[i, :], residuals, rank_a, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Update V
                for j in range(3):
                    A_mat = np.zeros((9, r))
                    for c in range(r):
                        A_mat[:, c] = (U3[:, c] * W3[:, c]).flatten()
                    b_vec = T3[:, j, :].flatten()
                    V3[j, :], residuals, rank_a, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Update W
                for k in range(3):
                    A_mat = np.zeros((9, r))
                    for c in range(r):
                        A_mat[:, c] = (U3[:, c] * V3[:, c]).flatten()
                    b_vec = T3[:, :, k].flatten()
                    W3[k, :], residuals, rank_a, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check convergence
                err = 0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            val = sum(U3[i,c]*V3[j,c]*W3[k,c] for c in range(r))
                            err += (val - T3[i,j,k])**2
                if err < 1e-15:
                    return U3, V3, W3
            else:
                # Check if converged
                err = 0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            val = sum(U3[i,c]*V3[j,c]*W3[k,c] for c in range(r))
                            err += (val - T3[i,j,k])**2
                if err < 1e-15:
                    return U3, V3, W3
        
        return None
    
    # Try ranks from 23 upward for the 3x3x3 subproblem
    found_333 = None
    for r in range(23, 40):
        if time.time() > deadline - 30:
            break
        result = try_rank(r)
        if result is not None:
            U3, V3, W3 = result
            # Verify
            err = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        val = sum(U3[i,c]*V3[j,c]*W3[k,c] for c in range(r))
                        err += (val - 1.0)**2
            if err < 1e-15:
                found_333 = (U3, V3, W3, r)
                break
    
    if found_333 is not None:
        U3, V3, W3, r3 = found_333
        # Build 3x3x6 decomposition by duplicating
        # B = [B1 | B2], C = [C1 | C2]
        # C1 = A*B1, C2 = A*B2
        # Use same U3, V3 for both blocks, but different W3 for each 3x3 block
        
        rank_total = 2 * r3
        U_full = np.zeros((9, rank_total))
        V_full = np.zeros((18, rank_total))
        W_full = np.zeros((18, rank_total))
        
        # First block: columns 0..r3-1
        U_full[:, 0:r3] = U3
        V_full[0:9, 0:r3] = V3  # B1 rows
        W_full[0:9, 0:r3] = W3   # C1 rows (k=0,1,2)
        
        # Second block: columns r3..2*r3-1
        U_full[:, r3:2*r3] = U3
        V_full[9:18, r3:2*r3] = V3  # B2 rows
        W_full[9:18, r3:2*r3] = W3   # C2 rows (k=3,4,5)
        
        # Verify full decomposition
        if _verify_decomposition(U_full, V_full, W_full, T):
            if rank_total < best["rank"]:
                best = {"U": U_full, "V": V_full, "W": W_full, "rank": rank_total}
    
    # Try direct search for 3x3x6 with various ranks
    def try_rank_direct_336(r):
        """Try rank-r decomposition directly for 3x3x6."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(123)
        for attempt in range(10):
            if time.time() > deadline - 10:
                return None
            
            U = np.random.randn(9, r) * 0.3
            V = np.random.randn(18, r) * 0.3
            W = np.random.randn(18, r) * 0.3
            
            for it in range(30):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for i in range(3):
                    A_mat = np.zeros((18, r))
                    for c in range(r):
                        A_mat[:, c] = (V[:, c] * W[:, c]).flatten()
                    b_vec = T[i, :, :].flatten()
                    U[i, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Update V
                for j in range(3):
                    A_mat = np.zeros((18, r))
                    for c in range(r):
                        A_mat[:, c] = (U[:, c] * W[:, c]).flatten()
                    b_vec = T[:, j, :].flatten()
                    V[j, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Update W
                for k in range(6):
                    A_mat = np.zeros((18, r))
                    for c in range(r):
                        A_mat[:, c] = (U[:, c] * V[:, c]).flatten()
                    b_vec = T[:, :, k].flatten()
                    W[k, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check error
                err = 0
                for i in range(3):
                    for j in range(3):
                        for k in range(6):
                            val = sum(U[i,c]*V[j,c]*W[k,c] for c in range(r))
                            err += (val - T[i,j,k])**2
                if err < 1e-15:
                    return U, V, W
            else:
                err = 0
                for i in range(3):
                    for j in range(3):
                        for k in range(6):
                            val = sum(U[i,c]*V[j,c]*W[k,c] for c in range(r))
                            err += (val - T[i,j,k])**2
                if err < 1e-15:
                    return U, V, W
        return None
    
    # Try direct ranks
    for r in range(24, 55):
        if time.time() > deadline - 20:
            break
        result = try_rank_direct_336(r)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, T):
                if r < best["rank"]:
                    best = {"U": U, "V": V, "W": W, "rank": r}
                break  # Found smallest rank, stop
    
    # Return best found
    return {"U": best["U"].tolist(), "V": best["V"].tolist(), "W": best["W"].tolist()}

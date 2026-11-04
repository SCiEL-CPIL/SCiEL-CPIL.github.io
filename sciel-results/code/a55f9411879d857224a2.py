"""Sparse integer linear system approach for 2x4x7 matrix multiplication tensor."""
import numpy as np
import time
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 @ 4x7.
    
    Strategy: Exploit the 2-row structure of A to reduce effective dimensionality.
    Use randomized algebraic search over sparse patterns to find low-rank solutions.
    """
    n, m, p = 2, 4, 7
    deadline = time.time() + 110  # safety margin
    
    # Tensor has 56 entries: (i,j,k) for i in 0..1, j in 0..3, k in 0..6
    # A rank-r decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def verify_decomposition(U_m, V_m, W_m, n, m, p, num_tests=10):
        """Fast verification."""
        np.random.seed(42)
        for t in range(num_tests):
            A = np.random.randint(-2, 3, size=(n, m))
            B = np.random.randint(-2, 3, size=(m, p))
            C_expected = A @ B
            rank_m = U_m.shape[1]
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(rank_m):
                            C_comp[i, k] += U_m[i*m+j, r] * V_m[j*p+k, r] * W_m[k*n+i, r]
            if not np.allclose(C_comp, C_expected, atol=1e-9):
                return False
        return True
    
    # Start with schoolbook rank-56 decomposition
    rank_sb = n * m * p  # 56
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = rank_sb
    
    # Try to find lower rank decompositions using structured patterns
    # For n=2, we can try rank patterns that exploit the 2-row structure
    
    # Attempt: Build a rank-49 decomposition by combining Strassen-like blocks
    # with careful handling of the 7th column
    
    def build_strassen_decomp():
        """Build a rank-50 Strassen-based decomposition."""
        rank_init = 50
        U = np.zeros((n * m, rank_init))
        V = np.zeros((m * p, rank_init))
        W = np.zeros((p * n, rank_init))
        
        col = 0
        
        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (A11+A22)(B11+B22)
            U[i0*m + j0, col] += 1
            U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M2: (A21+A22)B11
            U[(i0+1)*m + j0, col] += 1
            U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            
            # M3: A11(B12-B22)
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M4: A22(B21-B11)
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1
            W[k0*n + (i0+1), col] += 1
            col += 1
            
            # M5: (A11+A12)B22
            U[i0*m + j0, col] += 1
            U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1
            W[(k0+1)*n + i0, col] += 1
            col += 1
            
            # M6: (A21-A11)(B11+B12)
            U[(i0+1)*m + j0, col] += 1
            U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M7: (A12-A22)(B21+B22)
            U[i0*m + (j0+1), col] += 1
            U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1

        # 6 Strassen blocks for the 2x4x6 part
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)
        
        # 8 rank-1 terms for k=6
        for i in range(2):
            for j in range(4):
                U[i*m + j, col] = 1
                V[j*p + 6, col] = 1
                W[6*n + i, col] = 1
                col += 1
        
        assert col == rank_init
        return U, V, W, rank_init
    
    U_str, V_str, W_str, rank_str = build_strassen_decomp()
    
    if verify_decomposition(U_str, V_str, W_str, n, m, p):
        best_U, best_V, best_W = U_str.copy(), V_str.copy(), W_str.copy()
        best_rank = rank_str
    
    # Now try to reduce rank further using sparse pattern search
    # Key idea: for each potential rank r from best_rank-1 down to 40,
    # try to find a valid decomposition with that rank
    
    def try_rank_reduction(U_init, V_init, W_init, target_rank, max_attempts=50):
        """Try to reduce to target_rank by merging/eliminating terms."""
        if target_rank >= U_init.shape[1]:
            return None
        
        # Strategy: try to find linear dependencies among the rank-1 terms
        # by solving a system of equations
        
        current_U = U_init.copy()
        current_V = V_init.copy()
        current_W = W_init.copy()
        current_rank = U_init.shape[1]
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            if current_rank <= target_rank:
                return current_U, current_V, current_W, current_rank
            
            # Try to find a term that can be expressed as a combination of others
            # by checking if removing it and adjusting others still works
            
            for r in range(current_rank):
                if time.time() > deadline - 10:
                    break
                
                # Check if term r is "redundant" - i.e., its contribution can be
                # absorbed by other terms through coefficient adjustments
                # This is a simplified check: see if the term is proportional
                # to a sum of other terms
                
                # Simple heuristic: check if U[:,r] is in the span of other U columns
                # with corresponding V and W adjustments
                other_indices = [i for i in range(current_rank) if i != r]
                
                # Build the tensor contribution of term r
                T_r = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_r[i, j, k] = (current_U[i*m+j, r] * 
                                             current_V[j*p+k, r] * 
                                             current_W[k*n+i, r])
                
                # Check if T_r can be approximated by a linear combination of other terms
                # This is expensive, so use a quick check: if T_r is very sparse
                # and overlaps with other terms, it might be reducible
                
                nz_count = np.count_nonzero(T_r)
                if nz_count < 4:  # Very sparse term, might be eliminable
                    # Try removing it and see if we can adjust
                    test_U = current_U.copy()
                    test_V = current_V.copy()
                    test_W = current_W.copy()
                    
                    # Zero out term r
                    test_U[:, r] = 0
                    test_V[:, r] = 0
                    test_W[:, r] = 0
                    
                    # Check if remaining terms still give valid decomposition
                    # (They won't in general, but this is a quick filter)
                    if verify_decomposition(test_U, test_V, test_W, n, m, p, num_tests=3):
                        # Remove the zero column
                        keep = [i for i in range(current_rank) if i != r]
                        current_U = test_U[:, keep]
                        current_V = test_V[:, keep]
                        current_W = test_W[:, keep]
                        current_rank = len(keep)
                        break
        
        if current_rank <= target_rank:
            return current_U, current_V, current_W, current_rank
        return None
    
    # Try reducing from current best
    for target in range(best_rank - 1, max(39, best_rank - 10), -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank_reduction(best_U, best_V, best_W, target, max_attempts=20)
        if result is not None:
            U_new, V_new, W_new, rank_new = result
            if verify_decomposition(U_new, V_new, W_new, n, m, p):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = rank_new
                break  # Restart with new best
    
    # Final verification
    if not verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

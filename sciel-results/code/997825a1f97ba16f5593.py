"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin before 150s kill
    
    # Build the rank-55 Strassen-based decomposition
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part A: j=0 (first column of A, first row of B)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # A2 is 2x2, B2 is 2x10
    # Split B2 into 5 blocks of 2x2
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2
        
        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1
        
        # M1 = (a11 + a22)(b11 + b22)
        U[u_11, col] += 1
        U[u_22, col] += 1
        V[v_11, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[u_21, col] += 1
        U[u_22, col] += 1
        V[v_11, col] += 1
        W[w_10, col] += 1
        W[w_11, col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[u_11, col] += 1
        V[v_12, col] += 1
        V[v_22, col] += -1
        W[w_01, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[u_22, col] += 1
        V[v_21, col] += 1
        V[v_11, col] += -1
        W[w_00, col] += 1
        W[w_10, col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[u_11, col] += 1
        U[u_12, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += -1
        W[w_01, col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[u_21, col] += 1
        U[u_11, col] += -1
        V[v_11, col] += 1
        V[v_12, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[u_12, col] += 1
        U[u_22, col] += -1
        V[v_21, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += 1
        col += 1
    
    # Verify the decomposition is exact
    def verify(U, V, W):
        """Check if U,V,W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # For each rank component r, check that:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{ik} (for the multiplication)
        # Actually, the tensor is T[i,j,k,l] = delta_{i,l} where C[i,l] = sum_j A[i,j]*B[j,l]
        # The decomposition says: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r[i*m+j] = U[i*m+j, r], v_r[j*p+k] = V[j*p+k, r], w_r[k*n+i] = W[k*n+i, r]
        # And T[i,j,k,l] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+l,r] should equal delta_{i,l}
        
        # Efficient check: For each r, the outer product U[:,r] ⊗ V[:,r] gives a (n*m) x (m*p) matrix
        # which corresponds to a 4-tensor. We need to contract with W.
        # Let's use the identity: sum_j U[i*m+j,r] * V[j*p+k,r] * W[k*n+l,r] should be 0 unless i==l, 
        # and when i==l it should be 1.
        
        # More efficiently: For each r, define M_r[i,k] = sum_j U[i*m+j,r] * V[j*p+k,r]
        # Then sum_r M_r[i,k] * W[k*n+l,r] should equal delta_{i,l}
        
        # Compute M_r for all r at once:
        # U is (n*m, rank), V is (m*p, rank)
        # Reshape U to (n, m, rank), V to (m, p, rank)
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        # M_r[i,k] = sum_j U3[i,j,r] * V3[j,k,r]
        # This is a batched matrix multiply over the last dim
        # M = einsum('ijr,jkr->ikr', U3, V3)
        M = np.einsum('ijr,jkr->ikr', U3, V3)
        # Now result[i,l] = sum_r M[i,k,r] * W[k*n+l,r]
        # W is (p*n, rank), reshape to (p, n, rank)
        W3 = W.reshape(p, n, -1)
        # result[i,l] = sum_k sum_r M[i,k,r] * W3[k,l,r]
        result = np.einsum('ikr,klr->il', M, W3)
        expected = np.eye(n)
        return np.allclose(result, expected, atol=1e-10)
    
    if not verify(U, V, W):
        # Fallback: return the schoolbook decomposition
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Local search: try to find linear dependencies to reduce rank
    # Strategy: try random linear combinations of columns and check if the decomposition still holds
    # with fewer effective columns
    
    np.random.seed(42)
    
    # First, check if the current columns are linearly independent
    # If some columns are linearly dependent, we can potentially reduce rank
    
    # Try to find a basis for the column space of U
    # If rank(U) < rank, we might be able to reduce
    U_rank = np.linalg.matrix_rank(U, tol=1e-10)
    V_rank = np.linalg.matrix_rank(V, tol=1e-10)
    W_rank = np.linalg.matrix_rank(W, tol=1e-10)
    
    # Try random linear transformations to find sparser equivalent decompositions
    # and check if any columns become zero (which would allow rank reduction)
    
    max_iter = 200
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        # Random Gaussian transformation
        T = np.random.randn(rank, rank)
        
        # Apply transformation: U' = U @ T, V' = V @ T^{-1}, W' = W (unchanged)
        # This preserves the decomposition since sum_r U[:,r]*V[:,r]*W[:,r] is invariant under
        # U -> U@T, V -> V@T^{-1}
        
        try:
            T_inv = np.linalg.inv(T)
        except np.linalg.LinAlgError:
            continue
        
        U_new = U @ T
        V_new = V @ T_inv
        W_new = W.copy()
        
        # Check if this is valid
        if verify(U_new, V_new, W_new):
            # Check if any columns are now (nearly) zero
            # If column r of U_new is zero, or column r of V_new is zero, or column r of W_new is zero,
            # then that rank component is unused and we can drop it
            zero_cols = []
            for r in range(rank):
                if np.allclose(U_new[:, r], 0, atol=1e-10) or \
                   np.allclose(V_new[:, r], 0, atol=1e-10) or \
                   np.allclose(W_new[:, r], 0, atol=1e-10):
                    zero_cols.append(r)
            
            if len(zero_cols) > 0:
                # Remove zero columns
                keep = [r for r in range(rank) if r not in zero_cols]
                U_new = U_new[:, keep]
                V_new = V_new[:, keep]
                W_new = W_new[:, keep]
                new_rank = len(keep)
                
                if new_rank < best_rank and verify(U_new, V_new, W_new):
                    best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                    best_rank = new_rank
                    rank = new_rank  # Update for next iterations
                    # Reset U, V, W to the new best
                    U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
        
        # Also try: random linear combinations that might create zeros
        # Try mixing pairs of columns
        if it % 5 == 0:
            # Try a structured transformation: mix two columns
            i1, i2 = np.random.randint(0, rank, 2)
            if i1 == i2:
                continue
            T = np.eye(rank)
            T[i1, i1] = np.random.randn()
            T[i1, i2] = np.random.randn()
            T[i2, i1] = np.random.randn()
            T[i2, i2] = np.random.randn()
            
            try:
                T_inv = np.linalg.inv(T)
            except np.linalg.LinAlgError:
                continue
            
            U_new = U @ T
            V_new = V @ T_inv
            W_new = W.copy()
            
            if verify(U_new, V_new, W_new):
                zero_cols = []
                for r in range(rank):
                    if np.allclose(U_new[:, r], 0, atol=1e-10) or \
                       np.allclose(V_new[:, r], 0, atol=1e-10) or \
                       np.allclose(W_new[:, r], 0, atol=1e-10):
                        zero_cols.append(r)
                
                if len(zero_cols) > 0:
                    keep = [r for r in range(rank) if r not in zero_cols]
                    U_new = U_new[:, keep]
                    V_new = V_new[:, keep]
                    W_new = W_new[:, keep]
                    new_rank = len(keep)
                    
                    if new_rank < best_rank and verify(U_new, V_new, W_new):
                        best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                        best_rank = new_rank
                        rank = new_rank
                        U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
    
    # Final check
    if not verify(best_U, best_V, best_W):
        # Something went wrong, use original
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

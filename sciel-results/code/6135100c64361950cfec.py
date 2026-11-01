import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit

    # Build the multiplication tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if i, j, k are all valid indices (which they always are in standard matmul tensor definition 
    # where T is the structure tensor of multiplication).
    # Actually, the structure tensor T for C = A * B is defined such that C_k = sum_{i,j} T[i,j,k] A_i B_j.
    # For standard matrix multiplication, T[i,j,k] = 1 if i,j,k are valid, but specifically:
    # C[k*n + i] = sum_j A[i*m + j] B[j*p + k]
    # So the tensor T has indices (i, j, k) and value 1.
    T = np.ones((n, m, p), dtype=float)

    def verify_factors(U, V, W):
        """
        Checks if U, V, W exactly decompose T.
        U: (n*m, r), V: (m*p, r), W: (p*n, r)
        T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        """
        r = U.shape[1]
        # Flatten T to (n, m, p)
        # We need to check if sum_c U[i, j, c] V[j, k, c] W[k, i, c] == T[i, j, k]
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # Compute the reconstruction
        # Rec[i, j, k] = sum_c U_3d[i, j, c] * V_3d[j, k, c] * W_3d[k, i, c]
        # This is a tensor contraction.
        # Efficient way:
        # Temp[j, k, c] = sum_i U[i, j, c] * W[k, i, c] ? No, indices don't match directly for simple dot.
        # Let's use einsum or loops for small dimensions.
        
        # Since n=2, m=3, p=6, and r is likely small (<36), loops are fine and fast enough for verification.
        # But let's try vectorization.
        # Rec[i, j, k] = sum_c U[i, j, c] * (V[j, k, c] * W[k, i, c])
        
        # For each c, we have rank-1 tensor U_c (n,m) outer V_c (m,p) outer W_c (p,n)
        # Sum over c.
        
        # Let's just do the check via matrix multiplication logic on random inputs?
        # No, "exact decomposition" means the tensor identity holds.
        # Let's compute the reconstructed tensor.
        
        Rec = np.zeros((n, m, p), dtype=float)
        for c in range(r):
            u_c = U_3d[:, :, c]  # (n, m)
            v_c = V_3d[:, :, c]  # (m, p)
            w_c = W_3d[:, :, c]  # (p, n)
            
            # Product is u_c[i,j] * v_c[j,k] * w_c[k,i]
            # This is a trace-like operation.
            # Rec[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
            
            # We can compute this by looping over i, j, k or using einsum.
            # np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
            Rec += np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
            
        # Check if Rec is close to T
        # The problem requires integer/half-integer entries. 
        # If the decomposition is exact, Rec should be exactly T (which is 1s).
        # Due to floating point, we check closeness.
        # If it's close, we check if we can snap to half-integers.
        
        diff = np.abs(Rec - T)
        if np.max(diff) < 1e-8:
            # Check if U, V, W entries are close to half-integers
            # Multiply by 2 and check if close to integers
            U2 = U * 2
            V2 = V * 2
            W2 = W * 2
            
            U_int = np.round(U2)
            V_int = np.round(V2)
            W_int = np.round(W2)
            
            if np.allclose(U, U_int / 2.0, atol=1e-8) and \
               np.allclose(V, V_int / 2.0, atol=1e-8) and \
               np.allclose(W, W_int / 2.0, atol=1e-8):
                return True, U_int / 2.0, V_int / 2.0, W_int / 2.0
        return False, None, None, None

    def hadamard_iteration(U, V, W, iterations=50):
        """
        Performs Hadamard product iteration to minimize rank.
        """
        n, m, p = 2, 3, 6
        r = U.shape[1]
        
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        
        # Precompute some structures for speed?
        # U shape (nm, r), V (mp, r), W (pn, r)
        
        for _ in range(iterations):
            # Update U
            # U_new[i, j, c] = sum_{k} T[i, j, k] * V[j, k, c] * W[k, i, c]
            # Note: T is 1 everywhere.
            # So U_new[i, j, c] = sum_k V[j, k, c] * W[k, i, c]
            
            V_3d = V_curr.reshape(m, p, r)
            W_3d = W_curr.reshape(p, n, r)
            
            # For each c, compute U_new[:, :, c]
            # U_new[i, j, c] = sum_k V_curr[j, k, c] * W_curr[k, i, c]
            
            U_new = np.zeros((n, m, r), dtype=float)
            for c in range(r):
                # v_c is (m, p), w_c is (p, n)
                # We want (n, m) matrix where U_new[i, j] = sum_k V[j, k] * W[k, i]
                # This is (W^T * V)^T ?
                # (W^T) is (n, p). V is (m, p).
                # W_c^T @ V_c^T ? No.
                # Let's check indices:
                # sum_k V[j, k] W[k, i]
                # This is element (j, i) of V @ W^T.
                # V is (m, p), W^T is (p, n). Product is (m, n).
                # Element (j, i) is V[j, :] dot W[:, i].
                # So U_new[i, j] = (V @ W^T)[j, i] = (V @ W^T).T[i, j]
                prod = V_curr.reshape(m, p, r)[:, :, c] @ W_curr.reshape(p, n, r)[:, :, c].T
                # prod is (m, n). We need U_new[i, j] = prod[j, i].
                U_new[:, :, c] = prod.T
                
            U_curr = U_new.reshape(n * m, r)
            
            # Update V
            # V_new[j, k, c] = sum_i T[i, j, k] * U[i, j, c] * W[k, i, c]
            # V_new[j, k, c] = sum_i U[i, j, c] * W[k, i, c]
            # U is (n, m, r). W is (p, n, r).
            # For fixed c:
            # U_c is (n, m). W_c is (p, n).
            # V_new[j, k] = sum_i U[i, j] * W[k, i]
            # This is element (j, k) of U^T @ W.
            # U^T is (m, n). W is (p, n). Wait.
            # sum_i U[i, j] W[k, i]
            # Let A = U_c^T (m, n). B = W_c (p, n).
            # We want sum_i A[j, i] B[k, i].
            # This is element (j, k) of A @ B^T.
            # A @ B^T is (m, p).
            
            U_3d = U_curr.reshape(n, m, r)
            W_3d = W_curr.reshape(p, n, r)
            
            V_new = np.zeros((m, p, r), dtype=float)
            for c in range(r):
                u_c = U_3d[:, :, c] # (n, m)
                w_c = W_3d[:, :, c] # (p, n)
                # u_c.T is (m, n). w_c.T is (n, p).
                # u_c.T @ w_c.T is (m, p).
                # Element (j, k) is sum_i u_c.T[j, i] * w_c.T[i, k] = sum_i u_c[i, j] * w_c[k, i].
                # This matches V_new[j, k].
                prod = u_c.T @ w_c.T
                V_new[:, :, c] = prod
                
            V_curr = V_new.reshape(m * p, r)
            
            # Update W
            # W_new[k, i, c] = sum_j T[i, j, k] * U[i, j, c] * V[j, k, c]
            # W_new[k, i, c] = sum_j U[i, j, c] * V[j, k, c]
            # U is (n, m, r). V is (m, p, r).
            # For fixed c:
            # U_c is (n, m). V_c is (m, p).
            # W_new[k, i] = sum_j U[i, j] * V[j, k]
            # This is element (i, k) of U @ V.
            # U is (n, m), V is (m, p). Product is (n, p).
            # Element (i, k) is sum_j U[i, j] V[j, k].
            # So W_new[k, i] = (U @ V)[i, k].
            
            U_3d = U_curr.reshape(n, m, r)
            V_3d = V_curr.reshape(m, p, r)
            
            W_new = np.zeros((p, n, r), dtype=float)
            for c in range(r):
                u_c = U_3d[:, :, c]
                v_c = V_3d[:, :, c]
                prod = u_c @ v_c # (n, p)
                # W_new[k, i] = prod[i, k]
                W_new[:, :, c] = prod.T
                
            W_curr = W_new.reshape(p * n, r)
            
        return U_curr, V_curr, W_curr

    # Try ranks from high to low
    # Known bound: 22 (from inspiration). Schoolbook is 36.
    # Let's try 22, 23, 24... down to maybe 18?
    # If we find a valid one, return it.
    
    best_result = None
    best_rank = 36 # Fallback to schoolbook if search fails
    
    # Generate schoolbook fallback
    U_sb = np.zeros((n*m, n*m*p))
    V_sb = np.zeros((m*p, n*m*p))
    W_sb = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    # Search loop
    # We will try a few random restarts for each rank.
    # Time budget: 145s.
    
    ranks_to_try = [22, 23, 24, 25, 26, 27, 28, 29, 30]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10: # Stop if less than 10s left
            break
            
        # Try a few restarts for this rank
        max_restarts = 20
        if time.time() > deadline - 5:
            max_restarts = 1
            
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            # Initialize with random values
            # Scale matters. Let's start with small random values.
            U_init = np.random.randn(n*m, r) * 0.1
            V_init = np.random.randn(m*p, r) * 0.1
            W_init = np.random.randn(p*n, r) * 0.1
            
            # Run iteration
            # Number of iterations depends on rank. More rank = more stable?
            # Let's do 100 iterations.
            U_opt, V_opt, W_opt = hadamard_iteration(U_init, V_init, W_init, iterations=100)
            
            # Verify
            is_valid, U_val, V_val, W_val = verify_factors(U_opt, V_opt, W_opt)
            if is_valid:
                # We found a valid decomposition of rank r
                # Convert to lists of lists as required by interface?
                # Interface: {'U': U, 'V': V, 'W': W} with integer/half-integer entries.
                # The evaluator likely accepts numpy arrays or lists. 
                # "factor matrices of shapes ... with integer/half-integer entries"
                # Let's return numpy arrays for efficiency, or lists?
                # Seed program returned lists. Let's stick to lists to be safe, 
                # but numpy arrays are usually accepted if convertible.
                # The prompt says "construct() -> {'U', 'V', 'W'}: factor matrices of shapes ... with integer/half-integer entries"
                # It doesn't explicitly forbid numpy. But "entries" suggests values.
                # Let's return numpy arrays, as it's standard in such tasks.
                # If the evaluator expects lists, it might fail. 
                # Looking at seed: `U = [[0] * rank ...]` returns lists.
                # Let's convert to lists to be safe.
                
                U_list = U_val.tolist()
                V_list = V_val.tolist()
                W_list = W_val.tolist()
                
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # If no better rank found, return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END

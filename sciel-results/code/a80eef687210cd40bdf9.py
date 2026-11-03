import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_decomp = (U_school, V_school, W_school)
    
    # The tensor T is sparse. We represent it implicitly.
    # T[i, j, k] is not quite right.
    # Indices: U is (n*m), V is (m*p), W is (p*n).
    # T[(i,j), (j',k), (i',k')] = 1 if i==i', j==j', k==k' else 0.
    # We want sum_c U[c, i*m+j] * V[c, j'*p+k] * W[c, k'*n+i'] == delta(i,i')delta(j,j')delta(k,k')
    
    # We will search for rank r
    # Lower bound is max(n*m, m*p, p*n) = max(12, 28, 21) = 28.
    # Let's try ranks from 30 to 80.
    
    def tensor_residual(U, V, W):
        # Calculate the residual tensor implicitly? 
        # It's easier to flatten the target tensor and the computed tensor.
        # Target tensor size: (n*m) x (m*p) x (p*n)
        # This is 12 * 28 * 21 = 7056 elements.
        # We can construct the target tensor T_target as a 3D array.
        pass
        
    # Precompute target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T_target[u_idx, v_idx, w_idx] = 1.0
                
    def compute_residual(U, V, W):
        # Residual = T_target - sum_c U[:,c] outer V[:,c] outer W[:,c]
        # We can compute the tensor product sum efficiently using einsum or loops.
        # Given small size, a loop over rank is fine.
        R = np.zeros_like(T_target)
        for c in range(U.shape[1]):
            # R += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
            # This is slow in python loops. Use broadcasting.
            # U[:,c] is (nm,), V[:,c] is (mp,), W[:,c] is (pn,)
            # Outer product:
            R += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.linalg.norm(T_target - R)

    def search_rank(r, num_restarts=10, max_iter=100):
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Normalize to avoid divergence
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            
            best_res = compute_residual(U, V, W)
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                    
                # Update U
                # Minimize || T - sum_c U_c V_c W_c ||
                # For a fixed c, the term is U_c (V_c outer W_c)
                # This is a least squares problem for each column of U?
                # Actually, we can update all U at once using a matrix equation.
                # Let A_c = V[:,c] outer W[:,c] (size nm x mp x pn? No, V is mp, W is pn)
                # The residual is T_target - sum_c U[:,c] * A_c
                # We want to find U to minimize || T_target - U * A ||_F
                # This is not a standard LS because A is a set of 3D tensors.
                # However, we can flatten T_target and A into a matrix.
                # T_flat (size nm*mp*pn)
                # A_flat (size nm*mp*pn, r) where A_flat[:,c] is flattened V_c outer W_c
                # Then minimize || T_flat - U_flat * A_flat ||_F
                # This is a standard LS problem: U_flat = (A_flat^T A_flat)^{-1} A_flat^T T_flat
                # But A_flat is huge (7056 x r). r is ~30-40.
                # 7056x30 matrix multiplication is fast.
                
                # Construct A_flat
                # A_flat[:, c] = flatten( V[:,c] outer W[:,c] )
                # V[:,c] is (mp,), W[:,c] is (pn,)
                # Outer is (mp, pn). Flatten is (mp*pn,)
                # But U is (nm,). The tensor is (nm, mp, pn).
                # Flatten order: i, j, k.
                # T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
                # Flatten T to (nm*mp*pn,).
                # Flatten A_c to (nm*mp*pn,).
                # A_c[i, j, k] = U[i, c] * V[j, c] * W[k, c]
                # This means A_flat[:, c] is not just V outer W.
                # The equation is: T = sum_c U_c \otimes V_c \otimes W_c
                # Flattened: T_flat = U_flat @ (V_flat \otimes W_flat)^T ? No.
                # T_flat[i*mp*pn + j*pn + k] = sum_c U[i,c] V[j,c] W[k,c]
                # This is a linear system for U if V,W are fixed.
                # Let M be a matrix of size (nm * (mp*pn), r).
                # M[row, c] = V[j,c] * W[k,c] where row corresponds to (i,j,k).
                # But i is the index of U.
                # So for each i, we have a system:
                # sum_c U[i,c] * (V[j,c] W[k,c]) = T[i,j,k]
                # Let B_i be a matrix of size (mp*pn, r) where B_i[j*pn+k, c] = V[j,c] W[k,c].
                # Then U[i, :] = argmin || T[i, :, :] - B_i U[i, :]^T ||
                # This is a LS problem: U[i, :]^T = (B_i^T B_i)^{-1} B_i^T T[i, :, :].flattened
                # We can solve this for each i.
                
                # Update U
                for i in range(n):
                    # Build B_i
                    # V is (mp, r), W is (pn, r)
                    # B_i shape: (mp*pn, r)
                    # B_i[j*pn+k, c] = V[j, c] * W[k, c]
                    # This is the element-wise product of V and W broadcast?
                    # V[:, c] is (mp,). W[:, c] is (pn,).
                    # Outer product of V[:,c] and W[:,c] is (mp, pn).
                    # Flatten to (mp*pn,).
                    # So B_i is the same for all i! It doesn't depend on i.
                    # B is (mp*pn, r).
                    # B[j*pn+k, c] = V[j, c] * W[k, c]
                    
                    # Precompute B once outside the i-loop?
                    # Yes, B depends only on V and W.
                    pass # Move B computation outside

            # Optimization: Precompute B for current V, W
            # B shape: (mp*pn, r)
            # B[j*pn+k, c] = V[j, c] * W[k, c]
            # This is equivalent to:
            # B = (V.T * W.T).T ? No.
            # V is (mp, r). W is (pn, r).
            # We want B such that B[j, k, c] = V[j, c] * W[k, c].
            # This is the tensor product of V and W along the feature axis.
            # B = np.einsum('jc,kc->jkc', V, W) -> shape (mp, pn, r)
            # Reshape to (mp*pn, r)
            
            # Let's implement the full iteration loop properly.
            
        return None

    # Corrected Search Logic
    def run_search(r):
        for restart in range(20):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Normalize
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            
            for it in range(50):
                if time.time() > deadline:
                    return None
                
                # Update U
                # B shape (mp*pn, r)
                B = np.einsum('jc,kc->jkc', V, W).reshape(m * p * p * n, r)
                # T_target shape (nm, mp, pn)
                # For each i, solve LS:
                # T_i = T_target[i].reshape(mp*pn)
                # U_i = (B^T B)^{-1} B^T T_i
                # Since B is same for all i, compute inverse once?
                # B is (mp*pn, r). r is small.
                # B^T B is (r, r).
                BtB = B.T @ B
                # Add regularization for stability
                BtB += 1e-6 * np.eye(r)
                try:
                    inv_BtB = np.linalg.inv(BtB)
                except:
                    continue
                
                for i in range(n):
                    T_i = T_target[i].reshape(m * p * p * n)
                    # U[i] = inv_BtB @ (B.T @ T_i)
                    U[i] = inv_BtB @ (B.T @ T_i)
                
                # Update V
                # Symmetric
                # A shape (nm*pn, r)
                # A[i*pn+k, c] = U[i, c] * W[k, c]
                A = np.einsum('ic,kc->ikc', U, W).reshape(n * m * p * n, r)
                AtA = A.T @ A
                AtA += 1e-6 * np.eye(r)
                try:
                    inv_AtA = np.linalg.inv(AtA)
                except:
                    continue
                
                for j in range(m):
                    # T_target[i, j, k]
                    # We need to gather T_target[:, j, :]
                    # Shape (nm, pn) -> flatten to (nm*pn)
                    T_j = T_target[:, j, :].reshape(n * m * p * n)
                    V[j] = inv_AtA @ (A.T @ T_j)
                    
                # Update W
                # C shape (nm*mp, r)
                # C[i*mp+j, c] = U[i, c] * V[j, c]
                C = np.einsum('ic,jc->ijc', U, V).reshape(n * m * m * p, r)
                CtC = C.T @ C
                CtC += 1e-6 * np.eye(r)
                try:
                    inv_CtC = np.linalg.inv(CtC)
                except:
                    continue
                
                for k in range(p):
                    # T_target[i, j, k]
                    # Gather T_target[:, :, k]
                    # Shape (nm, mp) -> flatten to (nm*mp)
                    T_k = T_target[:, :, k].reshape(n * m * m * p)
                    W[k] = inv_CtC @ (C.T @ T_k)
                    
                # Check residual
                # Compute residual norm
                # R = T_target - sum_c U_c V_c W_c
                # We can compute this efficiently
                # Sum = np.einsum('ic,jc,kc->ijk', U, V, W)
                Sum = np.einsum('ic,jc,kc->ijk', U, V, W)
                Res = T_target - Sum
                res_norm = np.linalg.norm(Res)
                
                if res_norm < 1e-5:
                    return U, V, W
                    
            # End iterations
        return None

    # Try ranks
    # Theoretical lower bound 28.
    # Let's try 30 to 60.
    for r in range(30, 61):
        if time.time() > deadline:
            break
        result = run_search(r)
        if result is not None:
            U, V, W = result
            # Verify
            Sum = np.einsum('ic,jc,kc->ijk', U, V, W)
            if np.allclose(T_target, Sum, atol=1e-5):
                # Convert to list of lists for output format
                # The task requires 'U', 'V', 'W' as factor matrices
                # Shapes: (n*m, rank), (m*p, rank), (p*n, rank)
                # Entries should be integer/half-integer? 
                # "integer/half-integer entries"
                # My search produces floats.
                # I need to round to nearest half-integer?
                # Or scale to integers?
                # If I scale U by 2, V by 1, W by 1, entries might be integers if they were half-integers.
                # Let's check if the solution is "nice".
                # Often these decompositions have integer entries.
                # Let's try to round to nearest integer.
                U_int = np.round(U).astype(int)
                V_int = np.round(V).astype(int)
                W_int = np.round(W).astype(int)
                
                # Verify integer solution
                Sum_int = np.einsum('ic,jc,kc->ijk', U_int.astype(float), V_int.astype(float), W_int.astype(float))
                if np.allclose(T_target, Sum_int, atol=1e-5):
                    best_rank = r
                    best_decomp = (U_int, V_int, W_int)
                    break # Found a valid one, stop searching higher ranks? 
                         # Actually, we want minimal rank. So if we find one for r, we can stop if we are iterating upwards.
                else:
                    # Try half-integers
                    U_half = np.round(U * 2) / 2
                    V_half = np.round(V * 2) / 2
                    W_half = np.round(W * 2) / 2
                    Sum_half = np.einsum('ic,jc,kc->ijk', U_half, V_half, W_half)
                    if np.allclose(T_target, Sum_half, atol=1e-5):
                        # Convert to integers by multiplying by 2? 
                        # The interface says "integer/half-integer entries".
                        # If I return half-integers, I must return them as floats?
                        # "integer/half-integer entries" usually implies the values are in {0, 0.5, 1, 1.5...}
                        # I will return the float representation.
                        best_rank = r
                        best_decomp = (U_half, V_half, W_half)
                        break

    # If no better found, return schoolbook
    if best_rank == rank_school:
        U, V, W = best_decomp
    else:
        U, V, W = best_decomp
        
    # Format output
    # The task says: {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # Entries integer/half-integer.
    
    # Convert numpy arrays to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

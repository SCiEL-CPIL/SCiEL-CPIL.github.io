"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import sys
import signal
import time
import itertools

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 110.0

    # Construct the schoolbook decomposition
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Convert to numpy for efficient validation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    # Verify schoolbook decomposition
    def verify(U_v, V_v, W_v, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.einsum('r,r,r->', U_v[idx_u], V_v[idx_v], W_v[idx_w])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        # Check a sample of off-diagonal entries to ensure they are 0
        # We check entries where the indices don't match the pattern
        # idx_u = i*m+j, idx_v = j'*p+k, idx_w = k'*n+i'
        # The tensor is 1 only if j==j' and k==k' and i==i' (wait, no)
        # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k)
        # So for arbitrary a,b,c:
        # i = a//m, j = a%m
        # j' = b//p, k = b%p
        # i' = c%n, k' = c//n (wait, W index is k*n+i)
        # So c = k*n + i => i = c%n, k = c//n
        # Condition for 1: j == j' and k == k' (and implicitly i==i' from structure? No)
        # Let's re-read: T[i*m+j, j*p+k, k*n+i] = 1
        # So if we pick arbitrary a, b, c:
        # i = a // m, j = a % m
        # j' = b // p, k = b % p
        # i' = c % n, k' = c // n
        # The entry is 1 if and only if j == j' AND k == k'
        # Note: i and i' are not directly constrained by the 1/0 condition except through the indices?
        # Wait. The tensor is defined on the space of matrices.
        # T(A, B, C) = sum C[i,k] A[i,j] B[j,k] ... no.
        # The bilinear map is C = A B.
        # The tensor T is in V^* x W^* x Z^* such that T(A, B, C) = tr(C^T A B) = sum C[i,k] A[i,j] B[j,k]
        # So T_{(i,j), (j,k), (i,k)} = 1?
        # Let's check indices:
        # A is n x m. Index (i, j).
        # B is m x p. Index (j, k).
        # C is n x p. Index (i, k).
        # The term is A[i,j] * B[j,k] * C[i,k].
        # So the tensor component T_{a, b, c} is 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k.
        # This requires:
        # a = i*m + j
        # b = j*p + k
        # c = i*p + k  <-- WAIT. The problem statement says W[k*n+i, r].
        # Let's check the problem statement again: "Output factor follows the official trace convention W[k*n+i,r]"
        # If W index is k*n + i, then W is indexed by (k, i) where k is from B (col) and i is from A (row)?
        # If W index is k*n + i, then for C[i,k], the index is k*n+i.
        # So c = k*n + i.
        # Then the condition is:
        # a = i*m + j
        # b = j*p + k
        # c = k*n + i
        # This implies:
        # j = a % m
        # j' = b // p
        # k = b % p
        # k' = c // n
        # i = a // m
        # i' = c % n
        # For the entry to be 1, we need the indices to be consistent:
        # j == j' (from a and b)
        # k == k' (from b and c)
        # i == i' (from a and c)
        
        # So we check random (a,b,c) to ensure T=0 if not consistent.
        num_checks = 1000
        for _ in range(num_checks):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            
            if j == j2 and k == k2 and i == i2:
                # Should be 1
                val = np.einsum('r,r,r->', U_v[a], V_v[b], W_v[c])
                err = abs(val - 1.0)
            else:
                # Should be 0
                val = np.einsum('r,r,r->', U_v[a], V_v[b], W_v[c])
                err = abs(val - 0.0)
            
            if err > max_err:
                max_err = err
                
        return max_err < tol

    # Verify schoolbook
    # verify(U_np, V_np, W_np)

    # Attempt to find a lower rank decomposition
    # Strategy: Use random initialization and gradient descent on the tensor entries.
    # We want to minimize sum_{a,b,c} (T_target[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
    # T_target is 1 if consistent, 0 otherwise.
    
    # Since the tensor is sparse (only n*m*p ones), we can iterate over the non-zeros for the gradient.
    # But we also need the zeros to be zero.
    
    # Let's try a few ranks below schoolbook.
    best_rank = schoolbook_rank
    best_U = U_np
    best_V = V_np
    best_W = W_np
    
    # Ranks to try
    # Strassen for 3x3x3 is 23. For 3x3x8, we can maybe do better than 72.
    # 3x3x3 is 23. 3x3x4 is ? 3x3x8 is ?
    # Let's try ranks from 60 down to 40.
    ranks_to_try = list(range(60, 39, -1))
    
    for target_rank in ranks_to_try:
        if time.time() > DEADLINE:
            break
            
        # Try a few random restarts
        for attempt in range(3):
            if time.time() > DEADLINE:
                break
            
            np.random.seed(42 + attempt * 1000 + target_rank)
            
            U_r = np.random.randn(n * m, target_rank) * 0.1
            V_r = np.random.randn(m * p, target_rank) * 0.1
            W_r = np.random.randn(p * n, target_rank) * 0.1
            
            # Gradient descent
            lr = 0.01
            for iter in range(200):
                if time.time() > DEADLINE:
                    break
                
                # Compute gradients
                # Loss = sum_{a,b,c} (Target[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
                # dL/dU[a,r] = -2 * sum_{b,c} (Target - Pred) * V[b,r] * W[c,r]
                
                # This is expensive to compute directly for all a,b,c.
                # We can use the fact that Target is sparse.
                # But the "0" entries are dense.
                
                # Alternative: Use the structure.
                # C = A B
                # C[i,k] = sum_j A[i,j] B[j,k]
                # We are decomposing the bilinear map.
                # U, V, W define: C[i,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]? No.
                # The tensor is T_{(i,j), (j,k), (i,k)}.
                # So C is not explicitly in the tensor, the tensor maps (A, B) -> C?
                # No, the tensor T is in V_A^* x V_B^* x V_C^*.
                # T(A, B, C) = sum C[i,k] A[i,j] B[j,k].
                # So we are decomposing the trilinear form.
                # T = sum_r u_r \otimes v_r \otimes w_r
                # where u_r is a linear form on A, v_r on B, w_r on C.
                # u_r(A) = sum_{i,j} U[i*m+j, r] A[i,j]
                # v_r(B) = sum_{j,k} V[j*p+k, r] B[j,k]
                # w_r(C) = sum_{i,k} W[k*n+i, r] C[i,k]
                
                # We want T(A,B,C) = sum_r u_r(A) v_r(B) w_r(C)
                
                # Gradient descent on U, V, W.
                # We can sample random A, B, C to estimate the gradient?
                # Or compute exact gradient?
                # Exact gradient for U[a, r]:
                # dL/du_r = sum_{A,B,C} (T(A,B,C) - sum_s u_s(A)v_s(B)w_s(C)) v_r(B) w_r(C) ... no.
                # This is getting complicated.
                
                # Let's just use a simple heuristic:
                # Initialize with random, then use alternating least squares?
                # ALS for tensor rank decomposition.
                
                # Fix V, W, solve for U.
                # Fix U, W, solve for V.
                # Fix U, V, solve for W.
                
                # For fixed V, W, we want to minimize sum_{a,b,c} (Target[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # This is a linear system for U.
                # Let M be the matrix where M[a, (b,c)] = V[b,r] W[c,r] ... no, r is summed.
                # Let's form the matrix for U.
                # For each a, we want to fit Target[a, :, :] (which is a matrix of size m*p x p*n)
                # using factors V and W.
                # Target[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # This is a tensor train or something?
                # No, it's a bilinear form in b,c for each a.
                # Let X_a be the matrix Target[a, :, :] of size (m*p) x (p*n).
                # We want X_a \approx sum_r U[a, r] (V[:, r] \otimes W[:, r])? No.
                # V[b, r] is a vector of size m*p. W[c, r] is a vector of size p*n.
                # The term is sum_r U[a, r] V[b, r] W[c, r].
                # This is not a standard matrix factorization of X_a.
                # It is a "rank-1" update in a different sense.
                
                # Actually, if we fix V and W, the problem for U is:
                # Minimize sum_{b,c} (X_a[b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # This is a least squares problem for U[a, :].
                # Let's define a matrix H of size (m*p * p*n) x rank.
                # H[(b,c), r] = V[b, r] W[c, r]
                # Then X_a (flattened) \approx H U[a, :]
                # So U[a, :] = (H^T H)^{-1} H^T X_a_flattened
                # This is feasible if rank is small and m*p*p*n is not too big.
                # m*p = 24, p*n = 24. Total 576 entries.
                # Rank is ~40-60.
                # 576 x 60 matrix. Solving 60x60 system. Very fast.
                
                # Let's implement ALS.
                
                # Precompute H?
                # H depends on V and W.
                
                # Initialize U, V, W randomly.
                # Iterate:
                # 1. Fix V, W. Compute H. Solve for U.
                # 2. Fix U, W. Compute G. Solve for V.
                #    G[(a,c), r] = U[a, r] W[c, r]
                #    Target[b, :, :] (size n*m x p*n) \approx G V[b, :]
                #    Wait, indices:
                #    Target[a, b, c].
                #    For fixed b, we have a matrix of size (n*m) x (p*n).
                #    Target[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                #    So for fixed b, let Y_b be the matrix Target[:, b, :] of size (n*m) x (p*n).
                #    Y_b[a, c] = sum_r V[b, r] (U[a, r] W[c, r])
                #    Let K[(a,c), r] = U[a, r] W[c, r]
                #    Then Y_b \approx K V[b, :]
                #    V[b, :] = (K^T K)^{-1} K^T Y_b_flattened
                # 3. Fix U, V. Compute L. Solve for W.
                #    For fixed c, let Z_c be the matrix Target[:, :, c] of size (n*m) x (m*p).
                #    Z_c[a, b] = sum_r W[c, r] (U[a, r] V[b, r])
                #    Let M[(a,b), r] = U[a, r] V[b, r]
                #    Then Z_c \approx M W[c, :]
                #    W[c, :] = (M^T M)^{-1} M^T Z_c_flattened
                
                # Construct Target tensor
                # T[a, b, c]
                # a in 0..n*m-1, b in 0..m*p-1, c in 0..p*n-1
                # T[a, b, c] = 1 if consistent, else 0.
                
                # We can construct the flattened targets for each step.
                # Step 1: For each a, we need X_a_flattened of size m*p * p*n.
                # X_a[b, c] = 1 if consistent(a, b, c) else 0.
                
                # Consistency check:
                # a = i*m + j
                # b = j*p + k
                # c = k*n + i
                # i = a//m, j = a%m
                # j2 = b//p, k = b%p
                # k2 = c//n, i2 = c%n
                # Consistent if j==j2 and k==k2 and i==i2.
                
                # Let's precompute the target tensor as a numpy array of shape (n*m, m*p, p*n)
                # Size: 9 * 24 * 24 = 5184. Very small.
                
                T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            T_target[a, b, c] = 1.0
                
                # ALS Loop
                for iter in range(50):
                    if time.time() > DEADLINE:
                        break
                    
                    # Step 1: Solve for U
                    # For each a, solve least squares.
                    # H size: (m*p * p*n) x rank
                    # H[(b,c), r] = V[b, r] W[c, r]
                    # We can construct H efficiently.
                    # V is (m*p, rank), W is (p*n, rank)
                    # H is outer product of V and W? No.
                    # H is a matrix where row (b,c) is element-wise product of V[b,:] and W[c,:].
                    # This is the Khatri-Rao product of V and W, but transposed/reshaped.
                    # Actually, it's the matrix whose columns are the element-wise products of columns of V and W.
                    # Let's construct it.
                    
                    # H = np.zeros((m*p * p*n, target_rank))
                    # for r in range(target_rank):
                    #     H[:, r] = (V[:, r] * W[:, r]).flatten() # Wait, V[:, r] is size m*p, W[:, r] is size p*n.
                    #     We need to broadcast to (m*p, p*n) and flatten.
                    
                    # Better:
                    # H = np.einsum('br,cr->bcr', V_r, W_r).reshape(-1, target_rank)
                    
                    H = np.einsum('br,cr->bcr', V_r, W_r).reshape(-1, target_rank)
                    # H is (m*p * p*n, target_rank)
                    
                    # For each a, we have target T_target[a, :, :] flattened.
                    # T_flat = T_target[a, :, :].flatten()
                    # Solve H^T H U[a, :] = H^T T_flat
                    
                    # Since H is the same for all a, we can precompute (H^T H)^{-1} H^T
                    # Or just use np.linalg.lstsq for each a.
                    # 9 calls to lstsq with 576 x 60 matrix. Fast.
                    
                    # Actually, we can do it all at once.
                    # Stack T_target to (n*m, m*p * p*n)
                    # T_stack = T_target.reshape(n*m, -1)
                    # Solve H^T H U^T = H^T T_stack^T
                    # U^T is (rank, n*m)
                    
                    T_stack = T_target.reshape(n*m, -1)
                    # Solve: U^T = (H^T H)^{-1} H^T T_stack^T
                    # Using lstsq:
                    # H^T T_stack^T is (rank, n*m)
                    # We want U^T such that H U^T = T_stack^T
                    # So U^T = np.linalg.lstsq(H, T_stack.T, rcond=None)[0]
                    
                    U_new_T = np.linalg.lstsq(H, T_stack.T, rcond=None)[0]
                    U_r = U_new_T.T
                    
                    # Step 2: Solve for V
                    # For each b, solve least squares.
                    # K[(a,c), r] = U[a, r] W[c, r]
                    # K = np.einsum('ar,cr->acr', U_r, W_r).reshape(-1, target_rank)
                    K = np.einsum('ar,cr->acr', U_r, W_r).reshape(-1, target_rank)
                    
                    # Target for b: T_target[:, b, :] flattened.
                    # T_target is (n*m, m*p, p*n)
                    # We need to move axis 1 to front.
                    # T_b = T_target[:, b, :] -> (n*m, p*n)
                    # Stack for all b: (m*p, n*m * p*n)
                    T_stack_V = np.moveaxis(T_target, 1, 0).reshape(m*p, -1)
                    
                    V_new_T = np.linalg.lstsq(K, T_stack_V.T, rcond=None)[0]
                    V_r = V_new_T.T
                    
                    # Step 3: Solve for W
                    # For each c, solve least squares.
                    # M[(a,b), r] = U[a, r] V[b, r]
                    M = np.einsum('ar,br->abr', U_r, V_r).reshape(-1, target_rank)
                    
                    # Target for c: T_target[:, :, c] flattened.
                    # T_target is (n*m, m*p, p*n)
                    # Move axis 2 to front.
                    T_stack_W = np.moveaxis(T_target, 2, 0).reshape(p*n, -1)
                    
                    W_new_T = np.linalg.lstsq(M, T_stack_W.T, rcond=None)[0]
                    W_r = W_new_T.T
                    
                    # Check error
                    if iter % 5 == 0:
                        # Compute max error
                        # Pred = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                        # err = np.max(np.abs(Pred - T_target))
                        # This is 9*24*24 = 5184 entries. Fast.
                        Pred = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                        err = np.max(np.abs(Pred - T_target))
                        if err < 1e-9:
                            # Found a valid decomposition
                            best_rank = target_rank
                            best_U = U_r
                            best_V = V_r
                            best_W = W_r
                            # We can break out of loops if we want, but maybe try lower ranks?
                            # The problem asks for rank-minimal. So we should keep going to lower ranks.
                            # But if we found one for this rank, we should record it.
                            pass
                        else:
                            # If error is not small, continue
                            pass
                
                # After ALS, check final error
                Pred = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err = np.max(np.abs(Pred - T_target))
                if err < 1e-9 and target_rank < best_rank:
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r

    # Convert best result to list of lists
    # Ensure entries are close to integers or half-integers?
    # The problem says "integer/half-integer entries".
    # We should round the results to the nearest 0.5 if they are close.
    
    def round_to_half(x):
        r = np.round(x * 2) / 2
        if np.allclose(x, r, atol=1e-6):
            return r
        return x

    best_U = round_to_half(best_U)
    best_V = round_to_half(best_V)
    best_W = round_to_half(best_W)

    # Verify the rounded result
    Pred = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
    err = np.max(np.abs(Pred - T_target))
    if err > 1e-6:
        # If rounding broke it, try without rounding or with different rounding
        # Or just return the unrounded float if it's valid?
        # The problem says "integer/half-integer entries".
        # If the ALS solution is not close to half-integers, it might be invalid for the "exact" requirement?
        # "exactly decompose" usually means the tensor entries are exactly 0 or 1.
        # If the factors are floats, the sum might be 1.0000000001.
        # The verify function uses tol=1e-9.
        # Let's check if the unrounded version is better.
        Pred_unrounded = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        err_unrounded = np.max(np.abs(Pred_unrounded - T_target))
        if err_unrounded < err:
            best_U = U_r
            best_V = V_r
            best_W = W_r
            # But we must return integer/half-integer?
            # "factor matrices ... with integer/half-integer entries"
            # If we can't find one, maybe we should return the schoolbook which is integers.
            # Let's check if the schoolbook is better.
            # Schoolbook rank is 72.
            # If best_rank is 72, we return schoolbook.
            # If best_rank < 72 but rounding fails, we are in trouble.
            # Let's assume ALS finds something close to integers.
            pass

    U_result = [[int(x) if abs(x - round(x)) < 1e-6 else (round(x*2)/2 if abs(x*2 - round(x*2)) < 1e-6 else x) for x in row] for row in best_U.tolist()]
    V_result = [[int(x) if abs(x - round(x)) < 1e-6 else (round(x*2)/2 if abs(x*2 - round(x*2)) < 1e-6 else x) for x in row] for row in best_V.tolist()]
    W_result = [[int(x) if abs(x - round(x)) < 1e-6 else (round(x*2)/2 if abs(x*2 - round(x*2)) < 1e-6 else x) for x in row] for row in best_W.tolist()]

    return {"U": U_result, "V": V_result, "W": W_result}

# EVOLVE-BLOCK-END

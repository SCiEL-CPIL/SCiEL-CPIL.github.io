import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    n, m, p = 2, 4, 7
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    def tensor_from_factors(U, V, W):
        return np.einsum('ad,bd,cd->abc', U, V, W)
        
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    def check_and_update(U, V, W, r):
        nonlocal best_U, best_V, best_W, best_rank
        # Check if current factors are close to exact
        T_rec = tensor_from_factors(U, V, W)
        if np.allclose(T_rec, T, atol=1e-9, rtol=1e-9):
            # Try to snap to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            T_rec_r = tensor_from_factors(U_r, V_r, W_r)
            if np.allclose(T_rec_r, T, atol=1e-9, rtol=1e-9):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r
                return True
        return False

    def try_rank(r, max_trials=10):
        nonlocal best_rank
        if r >= best_rank:
            return False
            
        for trial in range(max_trials):
            if time.time() > deadline:
                return False
            
            # Random sparse initialization
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # Initialize with random +/- 1 in sparse positions
            # Heuristic: each vector should have ~ sqrt(dim) non-zeros
            for d in range(r):
                # Randomly select indices
                u_idx = np.random.choice(nm, size=max(1, int(np.sqrt(nm))), replace=False)
                v_idx = np.random.choice(mp, size=max(1, int(np.sqrt(mp))), replace=False)
                w_idx = np.random.choice(pn, size=max(1, int(np.sqrt(pn))), replace=False)
                
                U[u_idx, d] = np.random.choice([-1, 1], size=len(u_idx))
                V[v_idx, d] = np.random.choice([-1, 1], size=len(v_idx))
                W[w_idx, d] = np.random.choice([-1, 1], size=len(w_idx))
                
                # Symmetry breaking: fix first element of first column if possible
                # This helps convergence
                if d == 0:
                    if U[0, 0] == 0: U[0, 0] = 1
                    if V[0, 0] == 0: V[0, 0] = 1
                    if W[0, 0] == 0: W[0, 0] = 1

            # Alternating minimization
            prev_err = float('inf')
            for it in range(200):
                if time.time() > deadline:
                    return False
                    
                # Update W
                # Minimize || T - U V W^T ||_F^2 wrt W
                # Gradient: 2 (U V^T)^T (T - U V W^T)
                # Hessian: 2 (U V^T)^T (U V^T)
                # Using normal equations: (U^T U * V^T V) W = U^T V^T T
                # Note: (U V^T) is (nm, pn) matrix? No.
                # T is (nm, mp, pn).
                # U is (nm, r), V is (mp, r), W is (pn, r).
                # T_rec = sum_d U[:,d] V[:,d] W[:,d]^T
                # This is a tensor.
                
                # Efficient update:
                # Fix U, V. Update W.
                # Residual R = T - sum_d U[:,d] V[:,d] W[:,d]^T
                # For each d, W[:,d] is updated.
                # Actually, we can treat W as a matrix (pn, r).
                # The term involving W_d is U[:,d] V[:,d] W[:,d]^T.
                # This is an outer product of (U[:,d] V[:,d]) and W[:,d].
                # Let M_d = U[:,d] V[:,d] (shape nm x mp? No, U[:,d] is nm, V[:,d] is mp. Outer product is nm x mp. But T is nm x mp x pn.
                # Wait, T_rec[i,j,k] = sum_d U[i,d] V[j,d] W[k,d].
                # So for fixed d, the contribution is U[:,d] (as col) * V[:,d] (as col) * W[:,d] (as col)?
                # No, it's a rank-1 tensor in the 3 modes.
                # T_rec = sum_d np.einsum('a,b,c->abc', U[:,d], V[:,d], W[:,d])
                
                # To update W, we fix U, V.
                # We want to minimize || T - sum_{d' != d} ... - U[:,d] V[:,d] W[:,d]^T ||^2
                # This is equivalent to a linear least squares problem for each column of W?
                # No, W[:,d] is a vector of length pn.
                # The term is sum_k W[k,d] * (U[:,d] V[:,d] at k?) No.
                # T_rec[i,j,k] = sum_d U[i,d] V[j,d] W[k,d].
                # For a fixed d, the term is W[k,d] * U[i,d] * V[j,d].
                # This is not separable in k for the update of W[:,d] alone if we consider all i,j.
                # Actually, we can rearrange.
                # Let's flatten T to a matrix of size (nm*mp, pn).
                # T_flat[i*mp+j, k] = T[i,j,k].
                # Then T_rec_flat[i*mp+j, k] = sum_d (U[i,d] V[j,d]) W[k,d].
                # Let A_d be a vector of size nm*mp, with A_d[i*mp+j] = U[i,d] V[j,d].
                # Then T_rec_flat = sum_d A_d W[:,d]^T.
                # This is a matrix factorization T_flat = A W^T where A is (nm*mp, r).
                # So we can update W by solving:
                # (A^T A) W = A^T T_flat
                # where A[:,d] = U[:,d] V[:,d] (flattened).
                
                # Construct A
                # A is (nm*mp, r)
                # A[i*mp+j, d] = U[i,d] * V[j,d]
                # This can be computed as:
                # A = np.einsum('id,jd->ijd', U, V).reshape(nm*mp, r)
                # This is expensive if done every iteration.
                # But nm*mp = 8*28 = 224. r <= 56.
                # 224*56 = 12544 operations. Very fast.
                
                A = np.einsum('id,jd->ijd', U, V).reshape(nm*mp, r)
                T_flat = T.reshape(nm*mp, pn)
                
                # Solve (A^T A + lambda I) W = A^T T_flat
                # Use ridge regression for stability
                lambda_reg = 1e-6
                try:
                    W_new = np.linalg.solve(A.T @ A + lambda_reg * np.eye(r), A.T @ T_flat)
                except np.linalg.LinAlgError:
                    break
                
                # Update U
                # Fix V, W.
                # T_rec[i,j,k] = sum_d U[i,d] (V[j,d] W[k,d]).
                # Flatten T to (nm, mp*pn).
                # T_flat2[i, j*pn+k] = T[i,j,k].
                # T_rec_flat2[i, j*pn+k] = sum_d U[i,d] (V[j,d] W[k,d]).
                # Let B_d be a vector of size mp*pn, with B_d[j*pn+k] = V[j,d] W[k,d].
                # Then T_rec_flat2 = sum_d U[:,d] B_d^T.
                # T_flat2 = U B^T.
                # Solve (B^T B) U = B^T T_flat2.
                
                B = np.einsum('jd,kd->jkd', V, W).reshape(mp, pn, r) # Wait, shape (mp, pn, r)
                # We need B to be (mp*pn, r)
                B_flat = np.einsum('jd,kd->jkd', V, W).reshape(mp*pn, r)
                T_flat2 = T.reshape(nm, mp*pn)
                
                try:
                    U_new = np.linalg.solve(B_flat.T @ B_flat + lambda_reg * np.eye(r), B_flat.T @ T_flat2)
                except np.linalg.LinAlgError:
                    break
                    
                # Update V
                # Fix U, W.
                # T_rec[i,j,k] = sum_d (U[i,d] W[k,d]) V[j,d].
                # Flatten T to (mp, nm*pn).
                # T_flat3[j, i*pn+k] = T[i,j,k].
                # T_rec_flat3[j, i*pn+k] = sum_d V[j,d] (U[i,d] W[k,d]).
                # Let C_d be a vector of size nm*pn, with C_d[i*pn+k] = U[i,d] W[k,d].
                # Then T_rec_flat3 = sum_d V[:,d] C_d^T.
                # T_flat3 = V C^T.
                # Solve (C^T C) V = C^T T_flat3.
                
                C_flat = np.einsum('id,kd->ikd', U_new, W).reshape(nm*pn, r)
                # Note: T is (nm, mp, pn). We want to flatten (nm, pn) for each j?
                # T[i,j,k]. For fixed j, we have T[:,j,:] which is (nm, pn).
                # So T_flat3 should be (mp, nm*pn).
                # T_flat3[j, i*pn+k] = T[i,j,k].
                # This is a transpose/reshape of T.
                T_flat3 = T.transpose(1, 0, 2).reshape(mp, nm*pn)
                
                try:
                    V_new = np.linalg.solve(C_flat.T @ C_flat + lambda_reg * np.eye(r), C_flat.T @ T_flat3)
                except np.linalg.LinAlgError:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence and exactness
                T_rec = tensor_from_factors(U, V, W)
                err = np.linalg.norm(T_rec - T)
                
                if check_and_update(U, V, W, r):
                    return True
                    
                if err < 1e-12:
                    break
                if abs(prev_err - err) < 1e-14 and it > 20:
                    break
                prev_err = err
                
            # Final check
            if check_and_update(U, V, W, r):
                return True
                
        return False

    # Try ranks from lower bound upward
    # Lower bound is max(nm, mp, pn) = 28.
    # Start from 30 to save time, but check 28-32 first.
    for r in range(28, best_rank + 1):
        if time.time() > deadline:
            break
        # Increase trials for lower ranks as they are harder
        max_trials = 20 if r < 40 else 10
        success = try_rank(r, max_trials)
        if success:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

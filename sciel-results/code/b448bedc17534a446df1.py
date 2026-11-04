import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # 5s margin before 150s kill

    # Build the target tensor T as a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Pre-compute reshaped tensors for closed-form ALS
    # T_U: shape (n*m, m*p * p*n), rows a, cols (b, c)
    T_U = T.reshape(n*m, m*p * p*n)
    # T_V: shape (m*p, n*m * p*n), rows b, cols (a, c)
    # Need to permute T to (b, a, c) then reshape
    T_V = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
    # T_W: shape (p*n, n*m * m*p), rows c, cols (a, b)
    T_W = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)

    # Fallback: Schoolbook rank 56
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Rank 50 Strassen-based decomposition
    def strassen_rank50():
        rank = 50
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        
        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
            U[i0*m + j0, col] += 1
            U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M2: (A21+A22)B11 -> C21 += M2, C22 -= M2
            U[(i0+1)*m + j0, col] += 1
            U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            
            # M3: A11(B12-B22) -> C12 += M3, C22 += M3
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M4: A22(B21-B11) -> C11 += M4, C21 += M4
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1
            W[k0*n + (i0+1), col] += 1
            col += 1
            
            # M5: (A11+A12)B22 -> C11 -= M5, C12 += M5
            U[i0*m + j0, col] += 1
            U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1
            W[(k0+1)*n + i0, col] += 1
            col += 1
            
            # M6: (A21-A11)(B11+B12) -> C22 += M6
            U[(i0+1)*m + j0, col] += 1
            U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M7: (A12-A22)(B21+B22) -> C11 += M7
            U[i0*m + (j0+1), col] += 1
            U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1

        # Part 1: 2x4x4 (k=0..3) - 4 Strassen blocks
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        
        # Part 2: 2x4x2 (k=4..5) - 2 Strassen blocks
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)
        
        # Part 3: 2x4x1 (k=6) - 8 rank-1 terms
        for i in range(2):
            for j in range(4):
                U[i*m + j, col] = 1
                V[j*p + 6, col] = 1
                W[6*n + i, col] = 1
                col += 1
                
        return U, V, W, rank

    # Verify a decomposition (vectorized)
    def verify(U, V, W):
        prod = U[:, np.newaxis, np.newaxis, :] * V[np.newaxis, :, np.newaxis, :] * W[np.newaxis, np.newaxis, :, :]
        result = np.sum(prod, axis=3)
        return np.allclose(result, T, atol=1e-8)

    # Closed-form ALS with Kronecker structure
    def closed_form_als(U_init, V_init, W_init, target_rank, max_iters=200, tol=1e-10):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: T_U @ Z_VW @ inv(Z_VW^T Z_VW)
            # Z_VW[:, r] = vec(V[:, r] @ W[:, r]) -> np.kron(V[:, r], W[:, r])
            Z_VW = np.zeros((m*p * p*n, target_rank))
            for r in range(target_rank):
                Z_VW[:, r] = np.kron(V[:, r], W[:, r])
            
            # Solve U Z_VW^T = T_U  =>  U = T_U Z_VW (Z_VW^T Z_VW)^-1
            # Using lstsq for stability
            try:
                U_new, _, _, _ = np.linalg.lstsq(Z_VW.T, T_U.T, rcond=None)
                U = U_new.T
            except np.linalg.LinAlgError:
                pass
            
            # Update V: T_V @ Z_UW @ inv(Z_UW^T Z_UW)
            Z_UW = np.zeros((n*m * p*n, target_rank))
            for r in range(target_rank):
                Z_UW[:, r] = np.kron(U[:, r], W[:, r])
            
            try:
                V_new, _, _, _ = np.linalg.lstsq(Z_UW.T, T_V.T, rcond=None)
                V = V_new.T
            except np.linalg.LinAlgError:
                pass
            
            # Update W: T_W @ Z_UV @ inv(Z_UV^T Z_UV)
            Z_UV = np.zeros((n*m * m*p, target_rank))
            for r in range(target_rank):
                Z_UV[:, r] = np.kron(U[:, r], V[:, r])
            
            try:
                W_new, _, _, _ = np.linalg.lstsq(Z_UV.T, T_W.T, rcond=None)
                W = W_new.T
            except np.linalg.LinAlgError:
                pass
            
            # Check convergence
            R = np.sum(U[:, np.newaxis, np.newaxis, :] * V[np.newaxis, :, np.newaxis, :] * W[np.newaxis, np.newaxis, :, :], axis=3)
            err = np.linalg.norm(T - R)
            if err < tol:
                break
        
        return U, V, W, R

    # Rounding and verification step
    def try_round_and_verify(U, V, W):
        # Try rounding to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        
        # Try rounding to nearest integer
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        
        if verify(U_i, V_i, W_i):
            return U_i, V_i, W_i
            
        return None

    # Start with rank 50 Strassen
    U_s, V_s, W_s, rank_s = strassen_rank50()
    if not verify(U_s, V_s, W_s):
        U_s, V_s, W_s, rank_s = schoolbook()
    
    best_U, best_V, best_W, best_rank = U_s, V_s, W_s, rank_s

    # Try to reduce rank
    for target_rank in range(best_rank - 1, max(best_rank - 20, 20), -1):
        if time.time() > deadline:
            break
        
        success = False
        # Try multiple initialization strategies
        for restart in range(50): # Increased restarts due to speed
            if time.time() > deadline:
                break
            
            if restart == 0:
                # Use top columns of previous best
                U_init = best_U[:, :target_rank].copy()
                V_init = best_V[:, :target_rank].copy()
                W_init = best_W[:, :target_rank].copy()
            elif restart == 1:
                # Random subset of previous best columns
                idx = np.random.choice(best_rank, target_rank, replace=False)
                U_init = best_U[:, idx].copy()
                V_init = best_V[:, idx].copy()
                W_init = best_W[:, idx].copy()
            else:
                # Random restarts with noise
                np.random.seed(1000 + restart * 7)
                U_init = np.random.randn(n*m, target_rank) * 0.1
                V_init = np.random.randn(m*p, target_rank) * 0.1
                W_init = np.random.randn(p*n, target_rank) * 0.1
                # Mix in some structure from the best solution
                if best_rank > target_rank:
                    idx = np.random.choice(best_rank, min(target_rank, best_rank), replace=False)
                    U_init[:, :min(target_rank, best_rank)] = best_U[:, idx]
                    V_init[:, :min(target_rank, best_rank)] = best_V[:, idx]
                    W_init[:, :min(target_rank, best_rank)] = best_W[:, idx]
            
            U_try, V_try, W_try, R_try = closed_form_als(U_init, V_init, W_try, target_rank, max_iters=100, tol=1e-10)
            
            # Check if the solution is close to exact
            if np.linalg.norm(T - R_try) < 1e-8:
                # Try to round to half-integers/integers
                rounded = try_round_and_verify(U_try, V_try, W_try)
                if rounded is not None:
                    best_U, best_V, best_W, best_rank = rounded
                    success = True
                    break
                else:
                    # If rounding fails, we cannot use this solution if exactness is required
                    # But we might use it as a starting point for next rank? No, rank is fixed.
                    # Just continue searching.
                    pass
        
        if not success:
            continue

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

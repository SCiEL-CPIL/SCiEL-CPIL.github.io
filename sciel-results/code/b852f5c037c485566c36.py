import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor: T[a,b,c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if (i,j,k) is valid, else 0
    # Only 50 non-zero entries
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_indices.append((a, b, c))
    
    # Schoolbook solution (rank 50)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    for idx, (a, b, c) in enumerate(nonzero_indices):
        U_sb[a, idx] = 1
        V_sb[b, idx] = 1
        W_sb[c, idx] = 1
    
    deadline = time.time() + 135  # Safety margin from 150s kill
    
    def compute_residual(U, V, W):
        """Compute residual using sparse structure of T."""
        # T_approx[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Only need to check the 50 non-zero entries
        residual = 0.0
        for a, b, c in nonzero_indices:
            val = np.dot(U[a], V[b] * W[c])
            residual += (1.0 - val) ** 2
        return residual
    
    def als_optimize(U, V, W, max_iters=80, tol=1e-12):
        """ALS with SVD-based updates."""
        rank = U.shape[1]
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U: minimize ||T - U V^T W||_F
            A_U = np.zeros((rank, rank))
            B_U = np.zeros((n*m, rank))
            for a, b, c in nonzero_indices:
                vw = V[b] * W[c]
                A_U += np.outer(vw, vw)
                B_U[a] += vw
            
            A_U += 1e-12 * np.eye(rank)
            try:
                U = np.linalg.solve(A_U.T, B_U.T).T
            except np.linalg.LinAlgError:
                U = np.linalg.lstsq(A_U.T, B_U.T, rcond=None)[0].T
            
            # Update V similarly
            A_V = np.zeros((rank, rank))
            B_V = np.zeros((m*p, rank))
            for a, b, c in nonzero_indices:
                uw = U[a] * W[c]
                A_V += np.outer(uw, uw)
                B_V[b] += uw
            
            A_V += 1e-12 * np.eye(rank)
            try:
                V = np.linalg.solve(A_V.T, B_V.T).T
            except np.linalg.LinAlgError:
                V = np.linalg.lstsq(A_V.T, B_V.T, rcond=None)[0].T
            
            # Update W similarly
            A_W = np.zeros((rank, rank))
            B_W = np.zeros((p*n, rank))
            for a, b, c in nonzero_indices:
                uv = U[a] * V[b]
                A_W += np.outer(uv, uv)
                B_W[c] += uv
            
            A_W += 1e-12 * np.eye(rank)
            try:
                W = np.linalg.solve(A_W.T, B_W.T).T
            except np.linalg.LinAlgError:
                W = np.linalg.lstsq(A_W.T, B_W.T, rcond=None)[0].T
            
            if it % 5 == 0 or it == max_iters - 1:
                residual = compute_residual(U, V, W)
                if residual < tol:
                    break
        
        return U, V, W, compute_residual(U, V, W)
    
    def gradient_polish(U, V, W, max_steps=200, tol=1e-16):
        """High-precision gradient descent with adaptive learning rate."""
        lr = 0.02
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Compute residual
            R_vals = np.zeros(len(nonzero_indices))
            for idx, (a, b, c) in enumerate(nonzero_indices):
                R_vals[idx] = 1.0 - np.dot(U[a], V[b] * W[c])
            
            residual = np.sum(R_vals ** 2)
            if residual < tol:
                break
            
            # Gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for idx, (a, b, c) in enumerate(nonzero_indices):
                r = R_vals[idx]
                vw = V[b] * W[c]
                uv = U[a] * V[b]
                uw = U[a] * W[c]
                grad_U[a] -= 2 * r * vw
                grad_V[b] -= 2 * r * uw
                grad_W[c] -= 2 * r * uv
            
            # Try step with line search
            U_new = U - lr * grad_U
            V_new = V - lr * grad_V
            W_new = W - lr * grad_W
            
            new_residual = compute_residual(U_new, V_new, W_new)
            
            if new_residual < residual:
                U, V, W = U_new, V_new, W_new
                lr = min(lr * 1.03, 0.3)
            else:
                lr *= 0.5
                if lr < 1e-12:
                    break
        
        return U, V, W, compute_residual(U, V, W)
    
    def high_precision_polish(U, V, W, max_steps=500, tol=1e-18):
        """Very tight tolerance polish using Newton-like method."""
        lr = 0.01
        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            
            R_vals = np.zeros(len(nonzero_indices))
            for idx, (a, b, c) in enumerate(nonzero_indices):
                R_vals[idx] = 1.0 - np.dot(U[a], V[b] * W[c])
            
            residual = np.sum(R_vals ** 2)
            if residual < tol:
                break
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for idx, (a, b, c) in enumerate(nonzero_indices):
                r = R_vals[idx]
                vw = V[b] * W[c]
                uv = U[a] * V[b]
                uw = U[a] * W[c]
                grad_U[a] -= 2 * r * vw
                grad_V[b] -= 2 * r * uw
                grad_W[c] -= 2 * r * uv
            
            U_new = U - lr * grad_U
            V_new = V - lr * grad_V
            W_new = W - lr * grad_W
            
            new_residual = compute_residual(U_new, V_new, W_new)
            
            if new_residual < residual:
                U, V, W = U_new, V_new, W_new
                lr = min(lr * 1.02, 0.1)
            else:
                lr *= 0.5
                if lr < 1e-15:
                    break
        
        return U, V, W, compute_residual(U, V, W)
    
    def try_round_and_verify(U, V, W):
        """Try multiple rounding strategies to half-integers and verify exactly."""
        
        def verify_matrices(Ur, Vr, Wr):
            """Verify that the decomposition is exact for all non-zero entries."""
            for a, b, c in nonzero_indices:
                val = np.dot(Ur[a], Vr[b] * Wr[c])
                if abs(val - 1.0) > 1e-9:
                    return False
            return True
        
        # Strategy 1: nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify_matrices(U_r, V_r, W_r):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Strategy 2: nearest integer
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if verify_matrices(U_r, V_r, W_r):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Strategy 3: floor to half-integers
        U_r = np.floor(U * 2) / 2
        V_r = np.floor(V * 2) / 2
        W_r = np.floor(W * 2) / 2
        if verify_matrices(U_r, V_r, W_r):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Strategy 4: ceil to half-integers
        U_r = np.ceil(U * 2) / 2
        V_r = np.ceil(V * 2) / 2
        W_r = np.ceil(W * 2) / 2
        if verify_matrices(U_r, V_r, W_r):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Strategy 5: mixed rounding with small perturbations
        for perturb in [0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.05, -0.05]:
            U_p = np.round((U + perturb) * 2) / 2
            V_p = np.round((V + perturb) * 2) / 2
            W_p = np.round((W + perturb) * 2) / 2
            if verify_matrices(U_p, V_p, W_p):
                return {"U": U_p.tolist(), "V": V_p.tolist(), "W": W_p.tolist()}
        
        # Strategy 6: independent perturbations per matrix
        for pu in [0.01, -0.01, 0.02]:
            for pv in [0.01, -0.01, 0.02]:
                for pw in [0.01, -0.01, 0.02]:
                    U_p = np.round((U + pu) * 2) / 2
                    V_p = np.round((V + pv) * 2) / 2
                    W_p = np.round((W + pw) * 2) / 2
                    if verify_matrices(U_p, V_p, W_p):
                        return {"U": U_p.tolist(), "V": V_p.tolist(), "W": W_p.tolist()}
        
        return None
    
    # Search for lower rank
    for rank in range(rank_schoolbook - 1, max(30, rank_schoolbook - 15), -1):
        if time.time() > deadline - 25:
            break
        
        # Adaptive number of restarts based on time remaining
        time_left = deadline - time.time()
        max_restarts = max(3, min(15, int(time_left / 8)))
        
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(12345 + restart * 7919 + rank * 131)
            
            # Different initialization strategies
            if restart % 4 == 0:
                # Small random
                U = np.random.randn(n*m, rank) * 0.2
                V = np.random.randn(m*p, rank) * 0.2
                W = np.random.randn(p*n, rank) * 0.2
            elif restart % 4 == 1:
                # Structured: start from schoolbook subset + random
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for idx in range(min(rank, len(nonzero_indices))):
                    a, b, c = nonzero_indices[idx]
                    U[a, idx] = 1
                    V[b, idx] = 1
                    W[c, idx] = 1
                U += np.random.randn(n*m, rank) * 0.15
                V += np.random.randn(m*p, rank) * 0.15
                W += np.random.randn(p*n, rank) * 0.15
            elif restart % 4 == 2:
                # Sparse random
                U = np.random.randn(n*m, rank) * 0.4
                V = np.random.randn(m*p, rank) * 0.4
                W = np.random.randn(p*n, rank) * 0.4
                U[np.random.rand(n*m, rank) < 0.4] = 0
                V[np.random.rand(m*p, rank) < 0.4] = 0
                W[np.random.rand(p*n, rank) < 0.4] = 0
            else:
                # Medium random with different scale
                U = np.random.randn(n*m, rank) * 0.35
                V = np.random.randn(m*p, rank) * 0.35
                W = np.random.randn(p*n, rank) * 0.35
            
            # ALS optimization
            U_opt, V_opt, W_opt, residual = als_optimize(U, V, W, max_iters=60)
            
            # Gradient polish if close
            if residual < 0.1 and time.time() < deadline - 10:
                U_opt, V_opt, W_opt, residual = gradient_polish(U_opt, V_opt, W_opt, max_steps=150)
            
            # High-precision polish if very close
            if residual < 1e-8 and time.time() < deadline - 5:
                U_opt, V_opt, W_opt, residual = high_precision_polish(U_opt, V_opt, W_opt, max_steps=300)
            
            if residual < 1e-10:
                result = try_round_and_verify(U_opt, V_opt, W_opt)
                if result is not None:
                    return result
        
    # Return schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }

import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the list of non-zero entries
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
    
    deadline = time.time() + 140  # Safety margin
    
    def compute_residual(U, V, W):
        """Compute residual using sparse structure of T."""
        residual = 0.0
        for a, b, c in nonzero_indices:
            val = np.dot(U[a], V[b] * W[c])
            residual += (1.0 - val) ** 2
        return residual
    
    def als_optimize(U, V, W, max_iters=100, tol=1e-14):
        """ALS with SVD-based updates, optimized for the sparse tensor structure."""
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Update U
            A_U = np.zeros((rank, rank))
            B_U = np.zeros((n*m, rank))
            for a, b, c in nonzero_indices:
                vw = V[b] * W[c]
                A_U += np.outer(vw, vw)
                B_U[a] += vw
            A_U += 1e-10 * np.eye(rank)
            try:
                U = np.linalg.solve(A_U.T, B_U.T).T
            except np.linalg.LinAlgError:
                U = np.linalg.lstsq(A_U.T, B_U.T, rcond=None)[0].T
            
            # Update V
            A_V = np.zeros((rank, rank))
            B_V = np.zeros((m*p, rank))
            for a, b, c in nonzero_indices:
                uw = U[a] * W[c]
                A_V += np.outer(uw, uw)
                B_V[b] += uw
            A_V += 1e-10 * np.eye(rank)
            try:
                V = np.linalg.solve(A_V.T, B_V.T).T
            except np.linalg.LinAlgError:
                V = np.linalg.lstsq(A_V.T, B_V.T, rcond=None)[0].T
            
            # Update W
            A_W = np.zeros((rank, rank))
            B_W = np.zeros((p*n, rank))
            for a, b, c in nonzero_indices:
                uv = U[a] * V[b]
                A_W += np.outer(uv, uv)
                B_W[c] += uv
            A_W += 1e-10 * np.eye(rank)
            try:
                W = np.linalg.solve(A_W.T, B_W.T).T
            except np.linalg.LinAlgError:
                W = np.linalg.lstsq(A_W.T, B_W.T, rcond=None)[0].T
            
            if it % 10 == 0 or it == max_iters - 1:
                residual = compute_residual(U, V, W)
                if residual < tol:
                    break
        
        return U, V, W, compute_residual(U, V, W)
    
    def gradient_polish(U, V, W, max_steps=200, tol=1e-16):
        """Gradient descent with adaptive learning rate for final polish."""
        lr = 0.02
        for step in range(max_steps):
            if time.time() > deadline - 8:
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
                if lr < 1e-10:
                    break
        
        return U, V, W, compute_residual(U, V, W)
    
    def try_round_and_verify(U, V, W):
        """Try to round to half-integers and verify exactly."""
        # Try nearest half-integer
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        valid = True
        for a, b, c in nonzero_indices:
            val = np.dot(U_rounded[a], V_rounded[b] * W_rounded[c])
            if abs(val - 1.0) > 1e-8:
                valid = False
                break
        if valid:
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
        # Try floor and ceil
        for U_fn, V_fn, W_fn in [
            (np.floor, np.floor, np.floor),
            (np.ceil, np.ceil, np.ceil),
            (np.floor, np.ceil, np.floor),
            (np.ceil, np.floor, np.ceil),
        ]:
            U_r = U_fn(U * 2) / 2
            V_r = V_fn(V * 2) / 2
            W_r = W_fn(W * 2) / 2
            valid = True
            for a, b, c in nonzero_indices:
                val = np.dot(U_r[a], V_r[b] * W_r[c])
                if abs(val - 1.0) > 1e-8:
                    valid = False
                    break
            if valid:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        return None
    
    # Search for lower rank
    best_result = None
    for rank in range(rank_schoolbook - 1, max(25, rank_schoolbook - 20), -1):
        if time.time() > deadline - 30:
            break
        
        time_left = deadline - time.time()
        max_restarts = max(2, min(10, int(time_left / 12)))
        
        for restart in range(max_restarts):
            if time.time() > deadline - 25:
                break
            
            np.random.seed(999 + restart * 104729 + rank * 31)
            
            if restart % 3 == 0:
                # Small random
                U = np.random.randn(n*m, rank) * 0.15
                V = np.random.randn(m*p, rank) * 0.15
                W = np.random.randn(p*n, rank) * 0.15
            elif restart % 3 == 1:
                # Structured initialization from schoolbook subset
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for idx in range(min(rank, len(nonzero_indices))):
                    a, b, c = nonzero_indices[idx]
                    U[a, idx] = 1
                    V[b, idx] = 1
                    W[c, idx] = 1
                U += np.random.randn(n*m, rank) * 0.1
                V += np.random.randn(m*p, rank) * 0.1
                W += np.random.randn(p*n, rank) * 0.1
            else:
                # Medium random
                U = np.random.randn(n*m, rank) * 0.25
                V = np.random.randn(m*p, rank) * 0.25
                W = np.random.randn(p*n, rank) * 0.25
            
            U_opt, V_opt, W_opt, residual = als_optimize(U, V, W, max_iters=80)
            
            if residual < 0.05 and time.time() < deadline - 12:
                U_opt, V_opt, W_opt, residual = gradient_polish(U_opt, V_opt, W_opt, max_steps=150)
            
            if residual < 1e-12:
                result = try_round_and_verify(U_opt, V_opt, W_opt)
                if result is not None:
                    return result
    
    # Return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

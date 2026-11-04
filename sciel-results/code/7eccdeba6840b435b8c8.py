import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_rank = n * m * p
    best_solution = None
    
    # Generate schoolbook solution as fallback
    def get_schoolbook():
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
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    def is_valid(U, V, W, rank, atol=1e-9):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return np.allclose(Rec, 1.0, atol=atol)

    def compute_error(U, V, W, rank):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return Rec - 1.0

    def try_known_decompositions():
        """Try known algebraic decompositions for 3x3x5."""
        # For 3x3x3, Strassen gives rank 7. 
        # For 3x3x5, we can use a combination of Strassen-like patterns.
        # 
        # Strategy: Use the fact that 3x3x5 can be decomposed using
        # multiple 3x3x3 blocks plus additional terms.
        # 
        # Known result: rank(3x3x5) <= 23
        # Let's try to construct a rank-23 decomposition.
        
        # Approach: Use 5 copies of a 3x3x3 rank-7 decomposition with modifications
        # But this gives rank 35, which is too high.
        # 
        # Better approach: Use the Coppersmith-Winograd type ideas or 
        # construct a specific low-rank decomposition.
        
        # For now, let's try a random search with better initialization
        # and more restarts at rank 23-30.
        return False

    def error_and_grad(params, rank):
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        W = W_flat.reshape(p * n, rank)
        
        err = compute_error(U, V, W, rank)
        loss = np.sum(err**2)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            
            grad_U[:, r] = 2.0 * np.einsum('ijk,jk,ki->ij', err, v, w, optimize=True).flatten()
            grad_V[:, r] = 2.0 * np.einsum('ijk,ij,ki->jk', err, u, w, optimize=True).flatten()
            grad_W[:, r] = 2.0 * np.einsum('ijk,ij,jk->ki', err, u, v, optimize=True).flatten()
            
        grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return loss, grad_flat

    def try_snap_and_validate(U, V, W, rank, atol=1e-8):
        """Try multiple snapping strategies and return valid solution if found."""
        # Strategy 1: Round to nearest half-integer
        U1 = np.round(U * 2) / 2
        V1 = np.round(V * 2) / 2
        W1 = np.round(W * 2) / 2
        if is_valid(U1, V1, W1, rank, atol):
            return U1, V1, W1
        
        # Strategy 2: Round to nearest integer
        U2 = np.round(U)
        V2 = np.round(V)
        W2 = np.round(W)
        if is_valid(U2, V2, W2, rank, atol):
            return U2, V2, W2
        
        # Strategy 3: Try small perturbations around half-integers
        for delta in [-0.01, 0.01]:
            U3 = np.round((U + delta) * 2) / 2
            V3 = np.round((V + delta) * 2) / 2
            W3 = np.round((W + delta) * 2) / 2
            if is_valid(U3, V3, W3, rank, atol):
                return U3, V3, W3
        
        return None, None, None

    def polish_solution(U, V, W, rank, max_iters=2000):
        """High-precision polish using gradient descent with adaptive step size"""
        U = U.copy().astype(np.float64)
        V = V.copy().astype(np.float64)
        W = W.copy().astype(np.float64)
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        lr = 0.001
        best_loss = np.inf
        best_params = params.copy()
        patience = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            loss, grad = error_and_grad(params, rank)
            
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
                patience = 0
            else:
                patience += 1
            
            if loss < 1e-16:
                break
            
            if patience > 100:
                break
            
            # Adaptive step size with backtracking
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            
            step = lr
            for _ in range(20):
                new_params = params - step * grad
                new_loss, _ = error_and_grad(new_params, rank)
                if new_loss < loss:
                    break
                step *= 0.5
            
            params = params - step * grad
            
            # Try to snap periodically
            if it % 200 == 0:
                size_U = n * m * rank
                size_V = m * p * rank
                U_snap = np.round(params[:size_U].reshape(n * m, rank) * 2) / 2
                V_snap = np.round(params[size_U:size_U + size_V].reshape(m * p, rank) * 2) / 2
                W_snap = np.round(params[size_U + size_V:].reshape(p * n, rank) * 2) / 2
                if is_valid(U_snap, V_snap, W_snap, rank, atol=1e-8):
                    return U_snap, V_snap, W_snap
        
        # Final snap attempt
        size_U = n * m * rank
        size_V = m * p * rank
        U_final = best_params[:size_U].reshape(n * m, rank)
        V_final = best_params[size_U:size_U + size_V].reshape(m * p, rank)
        W_final = best_params[size_U + size_V:].reshape(p * n, rank)
        
        U_s, V_s, W_s = try_snap_and_validate(U_final, V_final, W_final, rank)
        if U_s is not None:
            return U_s, V_s, W_s
        
        return None, None, None

    def structured_init(rank, seed=0):
        """Create a structured initialization that respects matrix multiplication block structure."""
        rng = np.random.RandomState(seed)
        U_init = np.zeros((n * m, rank))
        V_init = np.zeros((m * p, rank))
        W_init = np.zeros((p * n, rank))
        
        for r in range(rank):
            # Each rank-1 term should contribute to a specific (i,j,k) pattern
            # But we allow mixing
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            k = rng.randint(0, p)
            
            # Primary contribution
            U_init[i * m + j, r] = 1.0
            V_init[j * p + k, r] = 1.0
            W_init[k * n + i, r] = 1.0
            
            # Add small random perturbations to allow optimization
            if rng.rand() < 0.3:
                i2 = rng.randint(0, n)
                j2 = rng.randint(0, m)
                U_init[i2 * m + j2, r] += rng.choice([-0.5, 0.5])
            if rng.rand() < 0.3:
                j2 = rng.randint(0, m)
                k2 = rng.randint(0, p)
                V_init[j2 * p + k2, r] += rng.choice([-0.5, 0.5])
            if rng.rand() < 0.3:
                k2 = rng.randint(0, p)
                i2 = rng.randint(0, n)
                W_init[k2 * n + i2, r] += rng.choice([-0.5, 0.5])
        
        return U_init, V_init, W_init

    def try_rank(rank, num_restarts=10):
        nonlocal best_solution, best_rank
        if rank >= best_rank:
            return False
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                return False
                
            # Use structured initialization
            U_init, V_init, W_init = structured_init(rank, seed=restart * 7 + 13)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use L-BFGS-B
            try:
                res = minimize(error_and_grad, params_init, method='L-BFGS-B', jac=True, 
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 200})
            except Exception:
                continue
                
            if res.fun < 1e-3:
                # Extract optimized U, V, W
                size_U = n * m * rank
                size_V = m * p * rank
                U_opt = res.x[:size_U].reshape(n * m, rank)
                V_opt = res.x[size_U:size_U + size_V].reshape(m * p, rank)
                W_opt = res.x[size_U + size_V:].reshape(p * n, rank)
                
                # Try snapping
                U_s, V_s, W_s = try_snap_and_validate(U_opt, V_opt, W_opt, rank)
                if U_s is not None:
                    best_solution = {"U": U_s, "V": V_s, "W": W_s}
                    best_rank = rank
                    return True
                
                # Polish step
                U_polished, V_polished, W_polished = polish_solution(U_opt, V_opt, W_opt, rank, max_iters=1000)
                if U_polished is not None:
                    best_solution = {"U": U_polished, "V": V_polished, "W": W_polished}
                    best_rank = rank
                    return True

        return False

    # Search for lower ranks
    # For 3x3x5, the known rank is 23
    # Try from 20 upward
    for rank in range(20, best_rank):
        if time.time() > deadline:
            break
        if try_rank(rank, num_restarts=8):
            break

    # If we found something, try to go even lower
    if best_rank < 45:
        for rank in range(15, best_rank):
            if time.time() > deadline:
                break
            if try_rank(rank, num_restarts=5):
                break

    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
# EVOLVE-BLOCK-END

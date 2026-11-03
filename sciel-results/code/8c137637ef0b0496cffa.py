import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Target tensor T of shape (n, m, p) with T[i,j,k] = 1
    T = np.ones((n, m, p))
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook

    # Try to find lower ranks
    # Known lower bounds or guesses: 21, 22, 23
    # We'll try a range
    for rank in range(23, 15, -1):
        if time.time() > deadline - 10.0:
            break
            
        success = False
        # Multiple restarts for this rank
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
                
            # Initialize randomly
            np.random.seed(restart * 100 + rank)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Define objective and gradient
            def objective_and_grad(params):
                U_flat = params[:n * m * rank].reshape(n * m, rank)
                V_flat = params[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                W_flat = params[n * m * rank + m * p * rank :].reshape(p * n, rank)
                
                # Compute R_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
                # R shape (n, m, p)
                # Use einsum for speed
                # U: (n*m, rank) -> (n, m, rank)
                # V: (m*p, rank) -> (m, p, rank)
                # W: (p*n, rank) -> (p, n, rank)
                
                U_reshaped = U_flat.reshape(n, m, rank)
                V_reshaped = V_flat.reshape(m, p, rank)
                W_reshaped = W_flat.reshape(p, n, rank)
                
                # R_{ijk} = sum_c U_{ijc} V_{jkc} W_{kic}
                # Note W index is k*n + i, so W_reshaped[k, i, c]
                R = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
                
                error = R - T
                loss = np.sum(error ** 2)
                
                # Gradient
                # dL/dU_{ijc} = sum_k error_{ijk} V_{jkc} W_{kic}
                grad_U = np.einsum('ijk,jkc,kic->ijc', error, V_reshaped, W_reshaped)
                # dL/dV_{jkc} = sum_i error_{ijk} U_{ijc} W_{kic}
                grad_V = np.einsum('ijk,ijc,kic->jkc', error, U_reshaped, W_reshaped)
                # dL/dW_{kic} = sum_j error_{ijk} U_{ijc} V_{jkc}
                grad_W = np.einsum('ijk,ijc,jkc->kic', error, U_reshaped, V_reshaped)
                
                grad_flat = np.concatenate([
                    grad_U.flatten(),
                    grad_V.flatten(),
                    grad_W.flatten()
                ])
                
                return loss, grad_flat

            initial_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(
                    objective_and_grad,
                    initial_params,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < 1e-6:
                    # Extract solution
                    U_sol = result.x[:n * m * rank].reshape(n * m, rank)
                    V_sol = result.x[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                    W_sol = result.x[n * m * rank + m * p * rank :].reshape(p * n, rank)
                    
                    # Snap to nearest 0.5
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify exactness
                    U_reshaped = U_snap.reshape(n, m, rank)
                    V_reshaped = V_snap.reshape(m, p, rank)
                    W_reshaped = W_snap.reshape(p, n, rank)
                    R_check = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
                    
                    if np.allclose(R_check, T, atol=1e-9):
                        # Valid solution found
                        best_result = {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        best_rank = rank
                        success = True
                        break # Break restart loop, try next (lower) rank? No, we want min rank.
                        # Since we iterate rank downwards, if we find one, we should check if lower is possible.
                        # But finding lower is harder. Let's just return the first valid one found in descending order?
                        # No, the loop is `for rank in range(23, 15, -1)`. So it tries 23, then 22...
                        # If we find 23, we break the restart loop, but continue to 22?
                        # Yes, we want the MINIMAL rank. So we should continue searching for lower ranks.
                        # However, if 23 is found, 22 might not exist.
                        # To save time, if we find a rank, we might want to stop if it's very low.
                        # But let's try to go down to 20 if time permits.
                        
            except Exception:
                pass

        # If we found a solution for this rank, we can try to find a lower one.
        # If not, we continue.
        
    return best_result

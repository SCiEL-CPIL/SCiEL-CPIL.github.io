import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_solve_W(U, V, target_rank):
        """Given U and V, solve linear system for W."""
        num_eqs = n * m * p
        num_unk = p * n * target_rank
        
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(target_rank):
                        u_val = U[i*m+j, c]
                        v_val = V[j*p+k, c]
                        if u_val != 0 and v_val != 0:
                            w_row = k * n + i
                            w_col = c
                            unk_idx = w_row * target_rank + w_col
                            A[eq_idx, unk_idx] = u_val * v_val
                    eq_idx += 1
        
        try:
            W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        
        W_mat = W_vec.reshape(p*n, target_rank)
        return W_mat
    
    def snap_to_half_integers(mat, tol=1e-8):
        """Try to snap matrix entries to half-integers."""
        doubled = 2.0 * mat
        rounded = np.round(doubled)
        max_err = np.max(np.abs(doubled - rounded))
        if max_err < tol:
            return rounded / 2.0
        return None
    
    best_solution = None
    
    # Try rank 15 first (known optimal for 2x3x3)
    for target_rank in [15, 16]:
        if time.time() > deadline:
            break
        
        # Strategy 1: Random {-1,0,1} matrices for U and V, solve for W
        num_trials = 50000 if target_rank == 15 else 30000
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            np.random.seed(trial * 7919 + 13)
            
            # Use sparse random patterns
            density = 0.4 if target_rank == 15 else 0.5
            U = np.random.randint(-1, 2, size=(n*m, target_rank))
            V = np.random.randint(-1, 2, size=(m*p, target_rank))
            
            # Ensure some non-trivial structure
            if np.all(U == 0) or np.all(V == 0):
                continue
            
            W_mat = try_solve_W(U, V, target_rank)
            if W_mat is None:
                continue
            
            # Check if W is close to half-integers
            W_snapped = snap_to_half_integers(W_mat, tol=1e-6)
            if W_snapped is not None:
                # Verify the full decomposition
                if verify(U.tolist(), V.tolist(), W_snapped.tolist()):
                    best_solution = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_snapped.tolist()
                    }
                    break
        
        if best_solution:
            break
    
    # Strategy 2: Gradient descent with better initialization if search failed
    if best_solution is None and time.time() < deadline - 20:
        target_rank = 15
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 104729 + 42)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Gradient descent with adaptive learning rate
            lr = 0.05
            for step in range(500):
                if time.time() > deadline:
                    break
                
                T_current = np.zeros_like(T_target)
                for r in range(target_rank):
                    T_current += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
                error = T_current - T_target
                err_norm = np.linalg.norm(error)
                if err_norm < 1e-8:
                    break
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for r in range(target_rank):
                    grad_U[:, r] = np.einsum('ijk,j,k->i', error, V[:, r], W[:, r])
                    grad_V[:, r] = np.einsum('ijk,i,k->j', error, U[:, r], W[:, r])
                    grad_W[:, r] = np.einsum('ijk,i,j->k', error, U[:, r], V[:, r])
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive LR
                if step % 50 == 0:
                    lr *= 0.95
            
            # Final check and snap
            T_final = np.zeros_like(T_target)
            for r in range(target_rank):
                T_final += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            if np.linalg.norm(T_final - T_target) < 1e-6:
                U_snap = snap_to_half_integers(U, tol=1e-5)
                V_snap = snap_to_half_integers(V, tol=1e-5)
                W_snap = snap_to_half_integers(W, tol=1e-5)
                
                if U_snap is not None and V_snap is not None and W_snap is not None:
                    if verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                        best_solution = {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        break
    
    if best_solution:
        return best_solution
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END

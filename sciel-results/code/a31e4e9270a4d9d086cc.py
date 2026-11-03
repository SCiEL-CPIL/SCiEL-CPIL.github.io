"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    schoolbook_rank = n * m * p  # 36
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W, rank
    
    def try_rank_with_fixed_W(W_pattern, rank, max_iters=800):
        """
        Given a fixed W pattern (list of lists of shape (p*n, rank)),
        solve for U and V such that the decomposition holds using alternating least squares.
        """
        W = [row[:] for row in W_pattern]
        
        # Initialize U and V randomly
        np.random.seed(12345)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        
        # Precompute equation indices for efficiency
        # For V: equation (i,j,k) -> coefficient for V[j*p+k, c] is U[i*m+j, c] * W[k*n+i, c]
        # For U: equation (i,j,k) -> coefficient for U[i*m+j, c] is V[j*p+k, c] * W[k*n+i, c]
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Fix U, solve for V
            num_v_unknowns = m * p * rank
            A_v = np.zeros((n*m*p, num_v_unknowns))
            b_v = np.ones(n*m*p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            w_val = W[k*n+i][c]
                            if abs(w_val) > 1e-12:
                                v_idx = (j*p + k) * rank + c
                                A_v[eq_idx, v_idx] = U[i*m+j][c] * w_val
                        eq_idx += 1
            
            try:
                V_sol, residuals, rank_sv, sv = np.linalg.lstsq(A_v, b_v, rcond=None)
            except:
                return None
            V = V_sol.reshape(m*p, rank)
            
            # Fix V, solve for U
            num_u_unknowns = n * m * rank
            A_u = np.zeros((n*m*p, num_u_unknowns))
            b_u = np.ones(n*m*p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            w_val = W[k*n+i][c]
                            if abs(w_val) > 1e-12:
                                u_idx = (i*m + j) * rank + c
                                A_u[eq_idx, u_idx] = V[j*p+k][c] * w_val
                        eq_idx += 1
            
            try:
                U_sol, residuals, rank_sv, sv = np.linalg.lstsq(A_u, b_u, rcond=None)
            except:
                return None
            U = U_sol.reshape(n*m, rank)
            
            # Check convergence periodically
            if iteration % 20 == 0:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            max_err = max(max_err, abs(s - 1.0))
                if max_err < 1e-10:
                    # Snap to nearest half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    if verify_decomposition(U_snap.tolist(), V_snap.tolist(), W, rank):
                        return U_snap.tolist(), V_snap.tolist(), W
        
        # Final check
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    max_err = max(max_err, abs(s - 1.0))
        if max_err < 1e-10:
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            if verify_decomposition(U_snap.tolist(), V_snap.tolist(), W, rank):
                return U_snap.tolist(), V_snap.tolist(), W
        
        return None
    
    # Try different W patterns starting from low rank
    W_patterns = []
    
    # Try ranks from 18 up to 36
    for target_rank in range(18, 37):
        if time.time() > deadline - 15:
            break
        
        # Generate multiple random W patterns for this rank
        num_trials = max(1, min(5, int((deadline - time.time()) / 15)))
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(1000 + target_rank * 100 + trial)
            W_rand = np.random.choice([-1.0, 0.0, 1.0], size=(p*n, target_rank))
            # Ensure each (i,k) has at least one nonzero in W
            for i in range(n):
                for k in range(p):
                    if all(W_rand[k*n+i] == 0):
                        W_rand[k*n+i][trial % target_rank] = 1.0
            
            W_patterns.append((f'rank{target_rank}_{trial}', W_rand.tolist(), target_rank))
        
        # Also try a structured pattern: spread the 1s more evenly
        W_struct = np.zeros((p*n, target_rank))
        for i in range(n):
            for k in range(p):
                # Assign 2-3 nonzeros per (i,k) row
                num_nonzero = min(3, target_rank)
                cols = np.random.choice(target_rank, size=num_nonzero, replace=False)
                for col in cols:
                    W_struct[k*n+i, col] = 1.0
        W_patterns.append((f'struct_rank{target_rank}', W_struct.tolist(), target_rank))
    
    # Sort by rank (try lowest first)
    W_patterns.sort(key=lambda x: x[2])
    
    # Try each pattern
    for name, W_pat, rank in W_patterns:
        if time.time() > deadline - 5:
            break
        
        result = try_rank_with_fixed_W(W_pat, rank)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END

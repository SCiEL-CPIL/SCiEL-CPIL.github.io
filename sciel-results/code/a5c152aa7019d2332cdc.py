"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import random
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 145  # Safety margin before 150s kill

    # Schoolbook as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    def verify_half(U, V, W_half_int, rank):
        """Verify if W is half-integer such that sum U V W = 1.
        W_half_int is 2*W. So we check sum U V W_half_int = 2.
        """
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W_half_int, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 2)

    # We try to find a decomposition of rank R.
    # Lower bound for rank is max(nm, mp, pn) = 21.
    
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 5:
            break
            
        # Allocate more attempts to lower ranks
        if target_rank <= 23:
            max_attempts = 5000
        elif target_rank <= 26:
            max_attempts = 2000
        else:
            max_attempts = 500

        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Generate random U and V with small integers
            # Wider range might help find valid structures
            U = np.array([[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(nm)], dtype=np.float64)
            V = np.array([[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(mp)], dtype=np.float64)
            
            # Build the linear system for W
            # Unknowns: W has shape (pn, target_rank) -> pn * target_rank variables
            # Equations: n * m * p equations
            num_vars = pn * target_rank
            num_eqs = n * m * p
            
            M = np.zeros((num_eqs, num_vars), dtype=np.float64)
            b = np.ones(num_eqs, dtype=np.float64)
            
            # Fill M
            # Equation index e corresponds to (i, j, k)
            # Variable index v corresponds to (k, i, c) in W -> (k*n + i) * target_rank + c
            # M[e, v] = U[i*m + j, c] * V[j*p + k, c]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for c in range(target_rank):
                            w_row = k * n + i
                            var_idx = w_row * target_rank + c
                            u_val = U[i * m + j, c]
                            v_val = V[j * p + k, c]
                            M[eq_idx, var_idx] = u_val * v_val
            
            # Solve M W_flat = b
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            W_sol = W_flat.reshape(pn, target_rank)
            
            # Check if W_sol is close to integer
            W_rounded = np.round(W_sol)
            diff_int = np.abs(W_sol - W_rounded)
            if np.all(diff_int < 1e-6):
                W_int = W_rounded.astype(int).tolist()
                if verify_fast(U.astype(int).tolist(), V.astype(int).tolist(), W_int, target_rank):
                    return {"U": U.astype(int).tolist(), "V": V.astype(int).tolist(), "W": W_int}
            
            # Check if W_sol is close to half-integer
            # Multiply by 2
            W_check = W_sol * 2
            W_rounded_half = np.round(W_check)
            diff_half = np.abs(W_check - W_rounded_half)
            if np.all(diff_half < 1e-6):
                W_half_int = W_rounded_half.astype(int).tolist()
                if verify_half(U.astype(int).tolist(), V.astype(int).tolist(), W_half_int, target_rank):
                    # Return W = W_half_int / 2.0
                    W_final = [[val / 2.0 for val in row] for row in W_half_int]
                    return {"U": U.astype(int).tolist(), "V": V.astype(int).tolist(), "W": W_final}
                        
    return best
# EVOLVE-BLOCK-END

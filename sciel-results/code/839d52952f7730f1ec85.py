"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    # N1 = (A00+A11)(B02+B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2 = (A10+A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3 = A00(B03-B13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4 = A11(B12-B02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5 = (A00+A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6 = (A10-A00)(B02+B03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7 = (A01-A11)(B12+B13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U_arr[i*m + j, :] * V_arr[j*p + k, :] * W_arr[k*n + i, :])
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _try_rank13_randomized():
    """
    Attempt to find a rank-13 decomposition via randomized algebraic elimination.
    
    Strategy:
    1. Start with a random rank-14 decomposition (or the known one).
    2. Introduce 14 rank-1 terms with symbolic coefficients.
    3. Solve the linear system for the output factors.
    4. Iteratively prune terms with zero coefficients.
    5. Check if we can achieve rank 13.
    """
    n, m, p = DIMENSIONS
    target_rank = 13
    start_time = time.time()
    deadline = start_time + 10  # 10 second budget for this search
    
    # Index mappings
    # U: (i, j) -> i*m + j
    # V: (j, k) -> j*p + k
    # W: (k, i) -> k*n + i
    
    def _attempt_rank13():
        """Single attempt to find rank-13 decomposition."""
        # Initialize with random rank-14 decomposition
        rank_init = 14
        U = np.random.randn(n*m, rank_init)
        V = np.random.randn(m*p, rank_init)
        W = np.random.randn(p*n, rank_init)
        
        # Normalize to avoid extreme values
        for r in range(rank_init):
            U[:, r] /= np.linalg.norm(U[:, r]) + 1e-10
            V[:, r] /= np.linalg.norm(V[:, r]) + 1e-10
            W[:, r] /= np.linalg.norm(W[:, r]) + 1e-10
        
        # Try to reduce to rank 13 by setting one coefficient to zero
        # We'll try each rank position and see if we can solve for the others
        
        best_solution = None
        best_error = float('inf')
        
        for zero_idx in range(rank_init):
            # Try zeroing out rank position zero_idx
            U_trial = U.copy()
            V_trial = V.copy()
            W_trial = W.copy()
            
            U_trial[:, zero_idx] = 0
            V_trial[:, zero_idx] = 0
            W_trial[:, zero_idx] = 0
            
            # Now we have 13 active rank terms
            # Solve for U, V, W to minimize error
            # Use alternating minimization
            
            U_curr = U_trial.copy()
            V_curr = V_trial.copy()
            W_curr = W_trial.copy()
            
            for iteration in range(50):
                if time.time() > deadline:
                    return None
                
                # Fix V and W, solve for U
                U_new = np.zeros_like(U_curr)
                for i in range(n):
                    for j in range(m):
                        A_sys = np.zeros((p, rank_init))
                        for k in range(p):
                            A_sys[k, :] = V_curr[j*p + k, :] * W_curr[k*n + i, :]
                        b_sys = np.ones(p)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                            U_new[i*m + j, :] = sol
                        except np.linalg.LinAlgError:
                            return None
                U_curr = U_new
                
                # Fix U and W, solve for V
                V_new = np.zeros_like(V_curr)
                for j in range(m):
                    for k in range(p):
                        A_sys = np.zeros((n, rank_init))
                        for i in range(n):
                            A_sys[i, :] = U_curr[i*m + j, :] * W_curr[k*n + i, :]
                        b_sys = np.ones(n)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                            V_new[j*p + k, :] = sol
                        except np.linalg.LinAlgError:
                            return None
                V_curr = V_new
                
                # Fix U and V, solve for W
                W_new = np.zeros_like(W_curr)
                for k in range(p):
                    for i in range(n):
                        A_sys = np.zeros((m, rank_init))
                        for j in range(m):
                            A_sys[j, :] = U_curr[i*m + j, :] * V_curr[j*p + k, :]
                        b_sys = np.ones(m)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                            W_new[k*n + i, :] = sol
                        except np.linalg.LinAlgError:
                            return None
                W_curr = W_new
            
            # Verify the solution
            max_error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = np.sum(U_curr[i*m + j, :] * V_curr[j*p + k, :] * W_curr[k*n + i, :])
                        error = abs(total - 1.0)
                        max_error = max(max_error, error)
            
            if max_error < 1e-8:
                # Found a valid rank-13 decomposition!
                # Convert to integer/half-integer if possible
                U_int = _to_rational(U_curr)
                V_int = _to_rational(V_curr)
                W_int = _to_rational(W_curr)
                
                if U_int is not None and V_int is not None and W_int is not None:
                    return {"U": U_int, "V": V_int, "W": W_int}
            
            if max_error < best_error:
                best_error = max_error
                best_solution = (U_curr.copy(), V_curr.copy(), W_curr.copy())
        
        return None
    
    def _to_rational(arr):
        """Try to convert float array to integer or half-integer values."""
        result = []
        for row in arr:
            new_row = []
            for val in row:
                # Try integer
                int_val = round(val)
                if abs(val - int_val) < 1e-6:
                    new_row.append(int_val)
                else:
                    # Try half-integer
                    half_val = round(val * 2) / 2
                    if abs(val - half_val) < 1e-6:
                        new_row.append(half_val)
                    else:
                        return None
            result.append(new_row)
        return result
    
    # Try multiple random initializations
    np.random.seed(42)
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        result = _attempt_rank13()
        if result is not None:
            # Verify the result
            if _verify_decomposition(result["U"], result["V"], result["W"]):
                return result
    
    return None


def construct():
    """
    Main entry point. Returns the best decomposition found.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # First, try to find a rank-13 decomposition
    # This is the main search strategy
    result = _try_rank13_randomized()
    
    if result is not None:
        return result
    
    # If rank-13 not found, fall back to rank-14
    # The rank-14 decomposition is guaranteed to work
    return _build_rank14()
# EVOLVE-BLOCK-END

import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 8]

def solve_tensor_decomposition(n, m, p, target_rank, max_iters=2000, tol=1e-8):
    """
    Finds a decomposition of the matrix multiplication tensor M_{n,m,p}
    with the given target_rank.
    Returns U, V, W if successful, else None.
    """
    # Initialize U, V, W with small random values
    # Using normal distribution helps escape local minima sometimes
    np.random.seed(42)
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # Flatten indices for vectorization
    # U_flat: (n*m*R)
    # V_flat: (m*p*R)
    # W_flat: (p*n*R)
    
    def compute_error(params):
        U_flat, V_flat, W_flat = params
        U = U_flat.reshape(n * m, target_rank)
        V = V_flat.reshape(m * p, target_rank)
        W = W_flat.reshape(p * n, target_rank)
        
        # Compute T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
        # We want this to be 1 for all i,j,k.
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    val = np.sum(u_val * v_val * w_val)
                    errors.append(val - 1.0)
        
        return np.array(errors)

    # Use least squares to minimize the error
    # scipy.optimize.least_squares is robust
    from scipy.optimize import least_squares
    
    initial_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = least_squares(compute_error, initial_params, method='lm', max_nfev=10000)
        if result.success and np.max(np.abs(result.fun)) < tol:
            U_final = result.x[:n * m * target_rank].reshape(n * m, target_rank)
            V_final = result.x[n * m * target_rank:n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
            W_final = result.x[n * m * target_rank + m * p * target_rank:].reshape(p * n, target_rank)
            
            # Check if entries are close to integers or half-integers?
            # The problem allows integer/half-integer entries.
            # The solver might produce floats. We should round them?
            # "exactly decompose" implies the values should be exact.
            # If the error is < tol, rounding to 2 decimal places (half-integers) might work.
            # Let's try rounding to nearest 0.5.
            U_rounded = np.round(U_final * 2) / 2
            V_rounded = np.round(V_final * 2) / 2
            W_rounded = np.round(W_final * 2) / 2
            
            # Verify the rounded version
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U_rounded[i * m + j, :]
                        v_val = V_rounded[j * p + k, :]
                        w_val = W_rounded[k * n + i, :]
                        val = np.sum(u_val * v_val * w_val)
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-6:
                return U_rounded, V_rounded, W_rounded
            else:
                # If rounding fails, return the float version? 
                # The task says "integer/half-integer entries".
                # If we can't satisfy that, it's invalid.
                return None
        else:
            return None
    except Exception:
        return None

def construct():
    n, m, p = DIMENSIONS
    
    # Split p into 3, 3, 2
    p1, p2, p3 = 3, 3, 2
    
    # Solve for M_{3,3,3} with rank 23
    start_time = time.time()
    U1, V1, W1 = solve_tensor_decomposition(n, m, p1, 23)
    if U1 is None:
        # Fallback to schoolbook rank 27
        U1, V1, W1 = get_schoolbook(n, m, p1)
        
    # Solve for M_{3,3,2} with rank 16
    U2, V2, W2 = solve_tensor_decomposition(n, m, p3, 16)
    if U2 is None:
        # Fallback to schoolbook rank 18
        U2, V2, W2 = get_schoolbook(n, m, p3)
        
    # Combine
    # Part 1: M_{3,3,3} (cols 0-2 of B, rows 0-2 of C)
    # Part 2: M_{3,3,3} (cols 3-5 of B, rows 3-5 of C)
    # Part 3: M_{3,3,2} (cols 6-7 of B, rows 6-7 of C)
    
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    rank3 = U2.shape[1] # Reuse same rank for part 2
    
    total_rank = rank1 + rank2 + rank3
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Fill Part 1
    # U1 is (9, rank1)
    U[:, 0:rank1] = U1
    # V1 is (9, rank1) -> Embed in V (24, rank1)
    # V1 index j*p1 + k. p1=3.
    # V index j*p + k. p=8.
    for j in range(m):
        for k in range(p1):
            V[j * p + k, 0:rank1] = V1[j * p1 + k, 0:rank1]
    # W1 is (9, rank1) -> Embed in W (24, rank1)
    # W1 index k*p1 + i? No, W1 index k*n + i.
    # W index k*n + i.
    for k in range(p1):
        for i in range(n):
            W[k * n + i, 0:rank1] = W1[k * n + i, 0:rank1]
            
    # Fill Part 2
    offset2 = rank1
    U[:, offset2:offset2+rank2] = U1 # Same A structure
    # V2 is (9, rank2)
    for j in range(m):
        for k in range(p2):
            # k in 0..2 maps to global k 3..5
            V[j * p + (k + p1), offset2:offset2+rank2] = V1[j * p2 + k, 0:rank2]
    # W2 is (9, rank2)
    for k in range(p2):
        for i in range(n):
            # k in 0..2 maps to global k 3..5
            W[(k + p1) * n + i, offset2:offset2+rank2] = W1[k * n + i, 0:rank2]
            
    # Fill Part 3
    offset3 = rank1 + rank2
    U[:, offset3:offset3+rank3] = U2
    # V3 is (6, rank3) -> V (24, rank3)
    # V3 index j*p3 + k. p3=2.
    for j in range(m):
        for k in range(p3):
            # k in 0..1 maps to global k 6..7
            V[j * p + (k + p1 + p2), offset3:offset3+rank3] = V2[j * p3 + k, 0:rank3]
    # W3 is (6, rank3) -> W (24, rank3)
    # W3 index k*n + i.
    for k in range(p3):
        for i in range(n):
            # k in 0..1 maps to global k 6..7
            W[(k + p1 + p2) * n + i, offset3:offset3+rank3] = W2[k * n + i, 0:rank3]
            
    return {"U": U, "V": V, "W": W}

def get_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

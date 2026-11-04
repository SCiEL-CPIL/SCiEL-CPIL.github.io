import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For the tensor: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    where a = i*m+j, b = j*p+k, c = k*n+i
    T should be 1 when indices are consistent, 0 otherwise.
    """
    rank = U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor and check
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that all other entries are 0
    # We only need to check a subset for efficiency, but let's do full check
    # For efficiency, we can check that the tensor is correct by verifying
    # that for each (a,b), the vector over c is correct
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j_b = b // p
            k = b % p
            if j_b != j:
                # For this (a, b), all c should give 0
                for c in range(p * n):
                    val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(val) > tol:
                        return False
            else:
                # j_b == j, so only c = k*n+i should be 1
                c_expected = k * n + i
                for c in range(p * n):
                    val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if c == c_expected:
                        if abs(val - 1.0) > tol:
                            return False
                    else:
                        if abs(val) > tol:
                            return False
    return True

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Start with schoolbook as best
    best = _make_schoolbook(n, m, p)
    best_rank = rank_school
    
    # Precompute target tensor indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    
    # Strategy 1: Try random V, W and solve for U via least squares
    # For each rank R, try multiple random initializations
    for R in range(rank_school - 1, 24, -1):
        if not still_going():
            break
        
        max_trials = 5
        for trial in range(max_trials):
            if not still_going():
                break
            
            np.random.seed(trial * 10000 + R * 100 + 42)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Solve for U using least squares on the target constraints
            # We have n*m*p constraints: for each (a,b,c) in target_indices,
            # sum_r U[a,r] * V[b,r] * W[c,r] = 1
            # This is linear in U!
            
            num_constraints = n * m * p
            # Build the system: A * U_flat = b
            # where A is (num_constraints, n*m*R) and U_flat is U reshaped
            
            A = np.zeros((num_constraints, n * m * R))
            b_vec = np.ones(num_constraints)
            
            for idx, (a, b_idx, c) in enumerate(target_indices):
                # For this constraint: sum_r U[a,r] * V[b_idx,r] * W[c,r] = 1
                # The coefficients for U[a, r] are V[b_idx, r] * W[c, r]
                for r in range(R):
                    A[idx, a * R + r] = V[b_idx, r] * W[c, r]
            
            # Solve the linear system
            try:
                U_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                U = U_flat.reshape(n * m, R)
            except:
                continue
            
            # Check residual
            residual = 0.0
            for idx, (a, b_idx, c) in enumerate(target_indices):
                val = np.sum(U[a, :] * V[b_idx, :] * W[c, :])
                residual += (val - 1.0) ** 2
            
            if residual < 1e-10:
                # Verify full decomposition
                if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-8):
                    # Try to snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if _verify_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p, tol=1e-6):
                        best_rank = R
                        best = {
                            "U": [[round(val * 2) / 2 for val in row] for row in U_snap],
                            "V": [[round(val * 2) / 2 for val in row] for row in V_snap],
                            "W": [[round(val * 2) / 2 for val in row] for row in W_snap]
                        }
                    else:
                        # Use the floating point solution if it's exact enough
                        # But we need integer/half-integer entries
                        # Try rounding to nearest integer
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        
                        if _verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p, tol=1e-6):
                            best_rank = R
                            best = {
                                "U": [[int(val) for val in row] for row in U_int],
                                "V": [[int(val) for val in row] for row in V_int],
                                "W": [[int(val) for val in row] for row in W_int]
                            }
                    break  # Found for this rank, try lower
    
    # Strategy 2: Try structured sparse decompositions
    if still_going():
        import random
        random.seed(42)
        
        for R in range(best_rank - 1, 24, -1):
            if not still_going():
                break
            
            max_attempts = 10
            for attempt in range(max_attempts):
                if not still_going():
                    break
                
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                
                # Place random ±1 entries
                nnz_U = random.randint(R, R * 2)
                nnz_V = random.randint(R, R * 3)
                nnz_W = random.randint(R, R * 2)
                
                for _ in range(nnz_U):
                    r = random.randint(0, R - 1)
                    idx = random.randint(0, n * m - 1)
                    U[idx, r] = random.choice([-1, 1])
                
                for _ in range(nnz_V):
                    r = random.randint(0, R - 1)
                    idx = random.randint(0, m * p - 1)
                    V[idx, r] = random.choice([-1, 1])
                
                for _ in range(nnz_W):
                    r = random.randint(0, R - 1)
                    idx = random.randint(0, p * n - 1)
                    W[idx, r] = random.choice([-1, 1])
                
                if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-9):
                    best_rank = R
                    best = {
                        "U": [[int(val) for val in row] for row in U],
                        "V": [[int(val) for val in row] for row in V],
                        "W": [[int(val) for val in row] for row in W]
                    }
                    break
    
    return best

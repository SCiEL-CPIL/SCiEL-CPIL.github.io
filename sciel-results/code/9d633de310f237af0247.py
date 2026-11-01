import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all valid indices."""
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _check_decomposition(U, V, W, n, m, p, num_trials=20):
    """Check that the decomposition is exact for random inputs."""
    rank = len(U[0])
    for seed in range(num_trials):
        random.seed(seed)
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        AB = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Compute via decomposition
        C = [[0] * p for _ in range(n)]
        for r in range(rank):
            # X_r[i] = sum_j U[i*m+j][r] * A[i][j]
            X = [0] * n
            for i in range(n):
                s = 0
                for j in range(m):
                    s += U[i * m + j][r] * A[i][j]
                X[i] = s
            # Y_r[k] = sum_j V[j*p+k][r] * B[j][k]
            Y = [0] * p
            for k in range(p):
                s = 0
                for j in range(m):
                    s += V[j * p + k][r] * B[j][k]
                Y[k] = s
            # C[i][k] += W[k*n+i][r] * X[i] * Y[k]
            for i in range(n):
                for k in range(p):
                    C[i][k] += W[k * n + i][r] * X[i] * Y[k]
        
        if C != AB:
            return False
    return True

def _schoolbook(n, m, p):
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

def _strassen_2x2x6():
    """Three independent Strassen 2x2x2 blocks, rank 21."""
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Standard Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+f), M2 = (c+d)e, M3 = a(f-d), M4 = b(e-f)
    # M5 = (a+b)(d+f), M6 = (a-c)(e+d), M7 = (b-d)(c+e)
    U_s = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [1, 0, -1, 0],  # M6: a-c
        [0, 1, 0, -1],  # M7: b-d
    ]
    V_s = [
        [1, 1, 0, 0],   # M1: e+f
        [1, 0, 0, 0],   # M2: e
        [0, 0, 1, -1],  # M3: f-d
        [0, 0, 0, 1],   # M4: d  -- wait, this is wrong
    ]
    
    # Let me redo Strassen properly.
    # For 2x2x2: C = A*B
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    #
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)h
    # M3 = a(f-h)
    # M4 = b(e-f)
    # M5 = (a+b)(d+h)
    # M6 = (a-c)(e+d)
    # M7 = (b-d)(c+g)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 + M6
    
    # U rows indexed by (i,j) where A[i][j]: a=00, b=01, c=10, d=11
    # V rows indexed by (j,k) where B[j][k]: e=00, f=01, g=10, h=11
    # W rows indexed by (k,i) where C[i][k]: C00, C01, C10, C11
    
    U_s = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [1, 0, -1, 0],  # M6: a-c
        [0, 1, 0, -1],  # M7: b-d
    ]
    V_s = [
        [1, 1, 0, 0],   # M1: e+h  -> wait, B indices: e=00,f=01,g=10,h=11
        # M1 = (a+d)(e+h): V for M1 = [1,0,0,1] (e and h)
        [1, 0, 0, 1],   # M1: e+h
        [0, 0, 0, 1],   # M2: h
        [0, 1, 0, -1],  # M3: f-h
        [1, -1, 0, 0],  # M4: e-f
        [0, 0, 0, 1],   # M5: d+h -- no, M5 = (a+b)(d+h), V part is d+h
        [1, 1, 0, 0],   # M6: e+d -- no, M6 = (a-c)(e+d), V part is e+d
        [0, 0, 1, 1],   # M7: c+g -- no, M7 = (b-d)(c+g), V part is c+g
    ]
    
    # This is getting confusing. Let me use a known correct Strassen formulation.
    # 
    # Standard Strassen for 2x2:
    # M1 = (a + d)(e + h)
    # M2 = (c + d)h
    # M3 = a(f - h)
    # M4 = b(e - f)
    # M5 = (a + b)(d + h)
    # M6 = (a - c)(e + d)
    # M7 = (b - d)(c + g)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 + M6
    
    # U: factor for A (2x2), rows indexed by (i,j) in A
    # V: factor for B (2x2), rows indexed by (j,k) in B
    # W: factor for C (2x2), rows indexed by (k,i) in C (trace convention W[k*n+i])
    
    U_s = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [1, 0, -1, 0],  # M6: a-c
        [0, 1, 0, -1],  # M7: b-d
    ]
    V_s = [
        [1, 0, 0, 1],   # M1: e+h
        [0, 0, 0, 1],   # M2: h
        [0, 1, 0, -1],  # M3: f-h
        [1, -1, 0, 0],  # M4: e-f
        [0, 0, 0, 1],   # M5: d+h -- wait, d is in A, h is in B
        # M5 = (a+b)(d+h): the B part is h only? No, d is from A, h is from B.
        # So V for M5 should be [0,0,0,1] (just h)
        [1, 0, 0, 0],   # M6: e+d -- d is from A, e is from B. V part is e = [1,0,0,0]
        [0, 0, 1, 0],   # M7: c+g -- c is from A, g is from B. V part is g = [0,0,1,0]
    ]
    
    # Wait, I'm making errors. Let me be very careful.
    # A = [[a, b], [c, d]] with indices: A[0][0]=a, A[0][1]=b, A[1][0]=c, A[1][1]=d
    # B = [[e, f], [g, h]] with indices: B[0][0]=e, B[0][1]=f, B[1][0]=g, B[1][1]=h
    #
    # M1 = (a+d)(e+h): U part = [1,0,0,1] (a and d), V part = [1,0,0,1] (e and h)
    # M2 = (c+d)h: U part = [0,0,1,1] (c and d), V part = [0,0,0,1] (h)
    # M3 = a(f-h): U part = [1,0,0,0] (a), V part = [0,1,0,-1] (f-h)
    # M4 = b(e-f): U part = [0,1,0,0] (b), V part = [1,-1,0,0] (e-f)
    # M5 = (a+b)(d+h): U part = [1,1,0,0] (a+b), V part = [0,0,0,1] (h)
    # M6 = (a-c)(e+d): U part = [1,0,-1,0] (a-c), V part = [1,0,0,0] (e)
    # M7 = (b-d)(c+g): U part = [0,1,0,-1] (b-d), V part = [0,0,1,0] (g)
    
    U_s = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 1, 0, 0],   # M4
        [1, 1, 0, 0],   # M5
        [1, 0, -1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    V_s = [
        [1, 0, 0, 1],   # M1
        [0, 0, 0, 1],   # M2
        [0, 1, 0, -1],  # M3
        [1, -1, 0, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 0, 0, 0],   # M6
        [0, 0, 1, 0],   # M7
    ]
    W_s = [
        [1, 0, 0, 1],   # M1: C11 and C22
        [0, 0, 1, 0],   # M2: C21
        [0, 1, 0, -1],  # M3: C12 and C22
        [0, 1, 0, 0],   # M4: C11
        [-1, 1, 0, 0],  # M5: -C11 and C12
        [0, 0, 1, 1],   # M6: C21 and C22
        [1, 0, 0, 0],   # M7: C11
    ]
    
    # W is indexed by (k, i) where C[i][k], so W[k*n+i]
    # C[0][0] = M1 + M4 - M5 + M7 -> W[0*2+0] = [1, 0, 0, 0, -1, 0, 1]
    # C[0][1] = M3 + M5 -> W[1*2+0] = [0, 0, 1, 0, 1, 0, 0]
    # C[1][0] = M2 + M6 -> W[0*2+1] = [0, 1, 0, 0, 0, 1, 0]
    # C[1][1] = M1 - M2 + M3 + M6 -> W[1*2+1] = [1, -1, 1, 0, 0, 1, 0]
    
    W_s = [
        [1, 0, 0, 0, -1, 0, 1],   # C[0][0]
        [0, 0, 1, 0, 1, 0, 0],    # C[0][1]
        [0, 1, 0, 0, 0, 1, 0],    # C[1][0]
        [1, -1, 1, 0, 0, 1, 0],   # C[1][1]
    ]
    
    # Now build the 2x2x6 decomposition by placing three 2x2x2 blocks
    for block in range(3):
        for s in range(7):
            r = block * 7 + s
            # U: 4 entries (i,j) for 2x2
            for ij in range(4):
                U[ij][r] = U_s[s][ij]
            # V: 12 entries (j,k) for 2x6, block covers k=block*2 to block*2+1
            for j in range(2):
                for k in range(2):
                    v_idx = j * 6 + (block * 2 + k)
                    v_s_idx = j * 2 + k
                    V[v_idx][r] = V_s[s][v_s_idx]
            # W: 12 entries (k,i) for 6x2, block covers k=block*2 to block*2+1
            for k in range(2):
                for i in range(2):
                    w_idx = (block * 2 + k) * 2 + i
                    w_s_idx = k * 2 + i
                    W[w_idx][r] = W_s[w_s_idx][s]
                    
    return {"U": U, "V": V, "W": W}

def _try_optimize_rank(U, V, W, target_rank, max_iters=5000, timeout=None):
    """
    Try to reduce the rank by optimizing a lower-rank decomposition.
    We parameterize a rank-r decomposition and minimize the tensor reconstruction error.
    """
    n, m, p = 2, 2, 6
    T = _build_tensor(n, m, p)
    
    # Initialize with random rank-r factors
    np.random.seed(42)
    
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    U_r = np.random.randn(n * m, target_rank) * 0.1
    V_r = np.random.randn(m * p, target_rank) * 0.1
    W_r = np.random.randn(p * n, target_rank) * 0.1
    
    # Initialize by taking first target_rank columns of the 21-rank Strassen
    if len(U[0]) >= target_rank:
        for i in range(n * m):
            for r in range(target_rank):
                U_r[i, r] = U[i][r]
        for i in range(m * p):
            for r in range(target_rank):
                V_r[i, r] = V[i][r]
        for i in range(p * n):
            for r in range(target_rank):
                W_r[i, r] = W[i][r]
    
    def tensor_error(Uf, Vf, Wf):
        """Compute the tensor reconstruction error."""
        # Reconstruct tensor: T_recon[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(target_rank):
                        s += Uf[i * m + j, r] * Vf[j * p + k, r] * Wf[k * n + i, r]
                    err += (s - T[i, j, k]) ** 2
        return err
    
    # Simple gradient-free optimization using Nelder-Mead style or random perturbation
    # Given time constraints, use a simple iterative approach
    
    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
    best_err = tensor_error(best_U, best_V, best_W)
    
    for it in range(max_iters):
        if timeout is not None and time.time() > timeout:
            break
        
        # Random perturbation
        U_new = best_U + np.random.randn(n * m, target_rank) * 0.01
        V_new = best_V + np.random.randn(m * p, target_rank) * 0.01
        W_new = best_W + np.random.randn(p * n, target_rank) * 0.01
        
        err = tensor_error(U_new, V_new, W_new)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_new, V_new, W_new
    
    # Convert to list of lists with rounded values
    def to_lists(arr):
        return [[float(x) for x in row] for row in arr]
    
    result_U = to_lists(best_U)
    result_V = to_lists(best_V)
    result_W = to_lists(best_W)
    
    return result_U, result_V, result_W, best_err

def _try_rank_reduction_search(max_rank=20, timeout=None):
    """
    Try to find a lower-rank decomposition by searching over different structures.
    """
    n, m, p = 2, 2, 6
    T = _build_tensor(n, m, p)
    
    # First, get the 21-rank Strassen decomposition
    strassen = _strassen_2x2x6()
    U_s = strassen["U"]
    V_s = strassen["V"]
    W_s = strassen["W"]
    
    # Convert to numpy
    U_np = np.array(U_s, dtype=float)
    V_np = np.array(V_s, dtype=float)
    W_np = np.array(W_s, dtype=float)
    
    # Try to find a rank-20 decomposition by checking if any column can be eliminated
    # through linear combination
    
    # Alternative approach: use the fact that the tensor is a sum of 3 rank-7 tensors
    # and try to find a more efficient representation
    
    # For now, just return the Strassen decomposition
    return strassen

def construct():
    n, m, p = 2, 2, 6
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try Strassen-based decomposition first
    result = _strassen_2x2x6()
    if _check_decomposition(result["U"], result["V"], result["W"], n, m, p):
        rank = len(result["U"][0])
        
        # Try to reduce rank further
        for target_rank in range(rank - 1, 15, -1):
            if time.time() > deadline - 10:
                break
            try:
                U_r, V_r, W_r, err = _try_optimize_rank(
                    result["U"], result["V"], result["W"], 
                    target_rank, 
                    max_iters=2000,
                    timeout=deadline - 5
                )
                # Check if the optimized decomposition is exact
                if _check_decomposition(U_r, V_r, W_r, n, m, p):
                    return {"U": U_r, "V": V_r, "W": W_r}
            except Exception:
                continue
        
        return result
    
    # Fall back to schoolbook
    result = _schoolbook(n, m, p)
    return result

# EVOLVE-BLOCK-END

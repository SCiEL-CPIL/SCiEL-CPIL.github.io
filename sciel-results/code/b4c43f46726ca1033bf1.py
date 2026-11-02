import numpy as np
import time

DIMENSIONS = [2, 3, 8]
_DEADLINE = 145.0
_START_TIME = None

def _strassen_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U: a = i*2 + j
    V: b = j*2 + k
    W: d = k*2 + i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # m3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # m4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # m5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # m6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # m7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_rank44():
    """
    Construct Rank 44 decomposition.
    Splits 2x3 * 3x8 into 4 blocks of 2x3 * 3x2.
    Each 2x3 * 3x2 is split into 2x2*2x2 (Rank 7) + 2x1*1x2 (Rank 4).
    Total 11 per block. 4 * 11 = 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    S_U, S_V, S_W = _strassen_2x2()
    
    col = 0
    for b in range(4):
        k0 = 2*b
        k1 = 2*b + 1
        
        # Part 1: A[:, 0:2] * B[0:2, k0:k1+1]
        # A_L is 2x2, B_L is 2x2.
        # Strassen U, V, W are 4x7.
        # Map indices:
        # U_local[i, j] -> U_global[i*3 + j, col+r]
        # V_local[j, k] -> V_global[j*8 + (k0+k), col+r]  (Note: V index is j*p + k)
        # W_local[k, i] -> W_global[(k0+k)*2 + i, col+r] (Note: W index is k*n + i)
        
        for r in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    val = S_U[i*2+j, r]
                    if val != 0:
                        U[i*3 + j, col+r] += val
            
            # V
            for j in range(2):
                for k in range(2):
                    val = S_V[j*2+k, r]
                    if val != 0:
                        V[j*8 + (k0+k), col+r] += val
            
            # W
            for k in range(2):
                for i in range(2):
                    val = S_W[k*2+i, r]
                    if val != 0:
                        W[(k0+k)*2 + i, col+r] += val
        
        col += 7
        
        # Part 2: A[:, 2] * B[2, k0:k1+1]
        # A_R is 2x1 (indices i*3+2), B_R is 1x2 (indices 2*8+k0, 2*8+k1)
        # Outputs: C[i, k0], C[i, k1]
        
        # Term: A[0,2] * B[2,k0] -> C[0,k0]
        U[0*3+2, col] = 1.0
        V[2*8+k0, col] = 1.0
        W[k0*2+0, col] = 1.0
        col += 1
        
        # Term: A[0,2] * B[2,k1] -> C[0,k1]
        U[0*3+2, col] = 1.0
        V[2*8+k1, col] = 1.0
        W[k1*2+0, col] = 1.0
        col += 1
        
        # Term: A[1,2] * B[2,k0] -> C[1,k0]
        U[1*3+2, col] = 1.0
        V[2*8+k0, col] = 1.0
        W[k0*2+1, col] = 1.0
        col += 1
        
        # Term: A[1,2] * B[2,k1] -> C[1,k1]
        U[1*3+2, col] = 1.0
        V[2*8+k1, col] = 1.0
        W[k1*2+1, col] = 1.0
        col += 1

    return U, V, W

def _verify(U, V, W, n, m, p, rank, tol=1e-6):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_tensors(n, m, p):
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    T0 = np.zeros((dim_U, dim_V * dim_W))
    T1 = np.zeros((dim_V, dim_U * dim_W))
    T2 = np.zeros((dim_W, dim_U * dim_V))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T0[a, b * dim_W + c] = 1.0
                T1[b, a * dim_W + c] = 1.0
                T2[c, a * dim_V + b] = 1.0
    return T0, T1, T2

def _als_solve(T0, T1, T2, dim_U, dim_V, dim_W, rank, U0, V0, W0, max_iter=100, tol=1e-9, time_budget=None, start_time=None):
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    
    for iteration in range(max_iter):
        if time_budget is not None and time.time() - start_time > time_budget:
            break
        
        U_old = U.copy()
        
        VW = np.zeros((dim_V * dim_W, rank))
        for r in range(rank):
            VW[:, r] = np.kron(V[:, r], W[:, r])
        try:
            U = np.linalg.lstsq(VW.T, T0.T, rcond=None)[0].T
        except:
            break
        
        UW = np.zeros((dim_U * dim_W, rank))
        for r in range(rank):
            UW[:, r] = np.kron(U[:, r], W[:, r])
        try:
            V = np.linalg.lstsq(UW.T, T1.T, rcond=None)[0].T
        except:
            break
        
        UV = np.zeros((dim_U * dim_V, rank))
        for r in range(rank):
            UV[:, r] = np.kron(U[:, r], V[:, r])
        try:
            W = np.linalg.lstsq(UV.T, T2.T, rcond=None)[0].T
        except:
            break
        
        if np.linalg.norm(U - U_old) < tol:
            break
    
    return U, V, W

def _compute_error(U, V, W, n, m, p, rank):
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c, r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _to_list(mat):
    rows = []
    for row in mat:
        r_list = []
        for val in row:
            r2 = np.round(val * 2)
            if abs(val - r2 / 2.0) < 1e-4:
                if r2 % 2 == 0:
                    r_list.append(int(r2 // 2))
                else:
                    r_list.append(r2 / 2.0)
            else:
                r_list.append(float(val))
        rows.append(r_list)
    return rows

def _try_als_search(best_rank, time_budget=40):
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    T0, T1, T2 = _build_tensors(n, m, p)
    start_time = time.time()
    
    U44, V44, W44 = _build_rank44()
    
    for rank in range(max(30, best_rank - 10), best_rank):
        if time.time() - start_time > time_budget:
            break
        
        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() - start_time > time_budget:
                break
            
            np.random.seed(42 + restart * 131 + rank * 7)
            
            # Initialize by truncating the rank-44 solution
            U = np.zeros((dim_U, rank))
            V = np.zeros((dim_V, rank))
            W = np.zeros((dim_W, rank))
            
            # Copy first 'rank' columns of the 44 solution
            U[:, :rank] = U44[:, :rank]
            V[:, :rank] = V44[:, :rank]
            W[:, :rank] = W44[:, :rank]
            
            # Add small noise to help escape local minima
            U += np.random.randn(dim_U, rank) * 0.01
            V += np.random.randn(dim_V, rank) * 0.01
            W += np.random.randn(dim_W, rank) * 0.01
            
            U, V, W = _als_solve(T0, T1, T2, dim_U, dim_V, dim_W, rank,
                                  U, V, W, max_iter=150, tol=1e-10,
                                  time_budget=time_budget, start_time=start_time)
            
            max_err = _compute_error(U, V, W, n, m, p, rank)
            
            if max_err < 1e-6:
                return rank, _to_list(U), _to_list(V), _to_list(W)
    
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    
    # Build Rank 44
    U44, V44, W44 = _build_rank44()
    
    if _verify(U44, V44, W44, n, m, p, 44):
        best_result = {"U": _to_list(U44), "V": _to_list(V44), "W": _to_list(W44)}
        best_rank = 44
    else:
        # Fallback to schoolbook
        rank_school = 48
        U_school = np.zeros((n*m, rank_school))
        V_school = np.zeros((m*p, rank_school))
        W_school = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col] = 1
                    V_school[j*p+k, col] = 1
                    W_school[k*n+i, col] = 1
                    col += 1
        best_result = {"U": _to_list(U_school), "V": _to_list(V_school), "W": _to_list(W_school)}
        best_rank = 48
    
    # Try to find better rank
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - 30 and best_rank > 30:
        search_result = _try_als_search(best_rank, time_budget=min(30, _DEADLINE - elapsed - 10))
        if search_result is not None:
            rank, U, V, W = search_result
            if _verify(np.array(U), np.array(V), np.array(W), n, m, p, rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
    
    return best_result

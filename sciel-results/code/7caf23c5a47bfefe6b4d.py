import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0]) if U else 0
    if rank == 0:
        return True
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _als_decompose(T, n, m, p, rank, deadline, max_sweeps=200, seed=0):
    rng = np.random.RandomState(seed)
    dim1, dim2, dim3 = T.shape
    
    # Initialize with small random values
    U = rng.randn(dim1, rank) * 0.1
    V = rng.randn(dim2, rank) * 0.1
    W = rng.randn(dim3, rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for sweep in range(max_sweeps):
        if time.time() > deadline - 5.0:
            break
        
        # Update U
        VW = np.einsum('jc,kc->jkc', V, W).reshape(-1, rank)
        T1 = T.reshape(dim1, -1)
        try:
            U_new, _, _, _ = np.linalg.lstsq(VW, T1.T, rcond=None)
            U = U_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Update V
        UW = np.einsum('ic,kc->ikc', U, W).reshape(-1, rank)
        T2 = np.transpose(T, (1, 0, 2)).reshape(dim2, -1)
        try:
            V_new, _, _, _ = np.linalg.lstsq(UW, T2.T, rcond=None)
            V = V_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Update W
        UV = np.einsum('ic,jc->ijc', U, V).reshape(-1, rank)
        T3 = np.transpose(T, (2, 0, 1)).reshape(dim3, -1)
        try:
            W_new, _, _, _ = np.linalg.lstsq(UV, T3.T, rcond=None)
            W = W_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Compute loss
        reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
        loss = np.sum((reconstructed - T) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            break
    
    if best_loss < 1e-8:
        U_list = [[round(float(x), 8) for x in row] for row in best_U]
        V_list = [[round(float(x), 8) for x in row] for row in best_V]
        W_list = [[round(float(x), 8) for x in row] for row in best_W]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank(T, n, m, p, rank, deadline, max_trials=50):
    for seed in range(max_trials):
        if time.time() > deadline - 10.0:
            break
        result = _als_decompose(T, n, m, p, rank, deadline, max_sweeps=100, seed=seed)
        if result is not None:
            return result
    return None

def _construct_2x3x6_from_2x3x3(res_2x3x3):
    """
    Construct 2x3x6 decomposition from 2x3x3 decomposition.
    res_2x3x3: dict with U (6xR), V (9xR), W (6xR) for 2x3x3.
    We need U (6x2R), V (18x2R), W (12x2R) for 2x3x6.
    
    The 2x3x6 tensor T_236 can be viewed as two 2x3x3 tensors T_A and T_B
    corresponding to the first and second 3 columns of the B matrix (and first/second 
    3 rows of C).
    
    T_236[i, j, k] = 1 if indices match.
    For k in 0..2, this is T_A. For k in 3..5, this is T_B.
    
    The indices for V are (j*p + k) for p=6.
    The indices for W are (k*n + i) for n=2.
    
    For the first block (k=0,1,2):
    V indices are j*6 + k (0..8)
    W indices are k*2 + i (0..5)
    
    For the second block (k=3,4,5):
    V indices are j*6 + k (9..17)
    W indices are k*2 + i (6..11)
    
    If we have a rank R decomposition for 2x3x3:
    U (6xR), V_33 (9xR), W_33 (6xR)
    
    We can construct the 2x3x6 decomposition with rank 2R:
    U_new = [U, U] (6x2R)
    V_new = 18x2R
    V_new[0:9, 0:R] = V_33
    V_new[9:18, R:2R] = V_33
    
    W_new = 12x2R
    W_new[0:6, 0:R] = W_33
    W_new[6:12, R:2R] = W_33
    
    Let's verify:
    Sum_c U_new[i,j,c] V_new[j,k,c] W_new[k,i,c]
    
    For c in 0..R-1:
    U[i,j,c] V_33[j, k%3, c] W_33[k%3, i, c] (Wait, V index mapping)
    
    Let's check V indices carefully.
    In 2x3x3, V is (m*p)xR = 9xR. Index is j*3 + k.
    In 2x3x6, V is (m*p)x2R = 18x2R. Index is j*6 + k.
    
    For the first block (k=0,1,2):
    V_new[j*6 + k, c] = V_33[j*3 + k, c]
    
    For the second block (k=3,4,5):
    V_new[j*6 + k, c+R] = V_33[j*3 + (k-3), c]
    
    Similarly for W.
    In 2x3x3, W is (p*n)xR = 6xR. Index is k*2 + i.
    In 2x3x6, W is (p*n)x2R = 12x2R. Index is k*2 + i.
    
    For the first block (k=0,1,2):
    W_new[k*2 + i, c] = W_33[k*2 + i, c]
    
    For the second block (k=3,4,5):
    W_new[k*2 + i, c+R] = W_33[(k-3)*2 + i, c]
    
    U is (n*m)xR = 6xR. Index is i*3 + j.
    In 2x3x6, U is (n*m)x2R = 6x2R. Index is i*3 + j.
    U_new[i*3 + j, c] = U[i*3 + j, c]
    U_new[i*3 + j, c+R] = U[i*3 + j, c]
    
    So:
    U_new = np.hstack([U, U])
    V_new = np.zeros((18, 2*R))
    for j in range(3):
        for k in range(3):
            V_new[j*6 + k, 0:R] = V_33[j*3 + k, 0:R]
            V_new[j*6 + (k+3), R:2*R] = V_33[j*3 + k, 0:R]
            
    W_new = np.zeros((12, 2*R))
    for k in range(3):
        for i in range(2):
            W_new[k*2 + i, 0:R] = W_33[k*2 + i, 0:R]
            W_new[(k+3)*2 + i, R:2*R] = W_33[k*2 + i, 0:R]
            
    This construction is correct.
    """
    U_233 = np.array(res_2x3x3['U'])
    V_233 = np.array(res_2x3x3['V'])
    W_233 = np.array(res_2x3x3['W'])
    
    R = U_233.shape[1]
    
    U_new = np.hstack([U_233, U_233])
    
    V_new = np.zeros((18, 2*R))
    for j in range(3):
        for k in range(3):
            V_new[j*6 + k, 0:R] = V_233[j*3 + k, 0:R]
            V_new[j*6 + (k+3), R:2*R] = V_233[j*3 + k, 0:R]
            
    W_new = np.zeros((12, 2*R))
    for k in range(3):
        for i in range(2):
            W_new[k*2 + i, 0:R] = W_233[k*2 + i, 0:R]
            W_new[(k+3)*2 + i, R:2*R] = W_233[k*2 + i, 0:R]
    
    U_list = [[round(float(x), 8) for x in row] for row in U_new]
    V_list = [[round(float(x), 8) for x in row] for row in V_new]
    W_list = [[round(float(x), 8) for x in row] for row in W_new]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Strategy: Find rank 11 for 2x3x3, then lift to 2x3x6 with rank 22.
    T_233 = _build_tensor(2, 3, 3)
    
    # Try ranks 11, 12, 13 for 2x3x3
    res_233 = None
    for rank in [11, 12, 13]:
        if time.time() > deadline - 60.0:
            break
        res_233 = _try_rank(T_233, 2, 3, 3, rank, deadline, max_trials=20)
        if res_233 is not None:
            break
            
    if res_233 is not None:
        if _verify_decomposition(res_233['U'], res_233['V'], res_233['W'], 2, 3, 3, tol=1e-6):
            res_236 = _construct_2x3x6_from_2x3x3(res_233)
            if _verify_decomposition(res_236['U'], res_236['V'], res_236['W'], 2, 3, 6, tol=1e-6):
                return res_236
                
    # Fallback: Direct search for 2x3x6 with ranks 22-36
    T_236 = _build_tensor(2, 3, 6)
    for rank in range(22, 37):
        if time.time() > deadline - 20.0:
            break
        res = _try_rank(T_236, 2, 3, 6, rank, deadline, max_trials=10)
        if res is not None:
            return res
            
    # Fallback: Schoolbook
    return _schoolbook_decomposition(2, 3, 6)

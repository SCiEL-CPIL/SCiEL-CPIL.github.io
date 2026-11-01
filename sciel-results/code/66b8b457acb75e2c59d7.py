"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Returns U, V, W for schoolbook decomposition (rank 52)."""
    n, m, p = 2, 2, 13
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, col] = 1.0
                V[k * p + j, col] = 1.0
                W[j * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_target_tensor():
    """Builds the full target tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 13
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i * m + k, k * p + j, j * n + i] = 1.0
    return T

def _find_strassen_rank7():
    """
    Find rank-7 decomposition of 2x2x2 matmul tensor via numerical optimization.
    
    Local indices:
      U: 0=(i=0,k=0), 1=(i=0,k=1), 2=(i=1,k=0), 3=(i=1,k=1)
      V: 0=(k=0,j=0), 1=(k=0,j=1), 2=(k=1,j=0), 3=(k=1,j=1)
      W: 0=(j=0,i=0), 1=(j=0,i=1), 2=(j=1,i=0), 3=(j=1,i=1)
    
    Non-zero entries of T_loc:
      (0,0,0), (0,1,2), (1,2,0), (1,3,2), (2,0,1), (2,1,3), (3,2,1), (3,3,3)
    """
    T_loc = np.zeros((4, 4, 4))
    nonzeros = [(0,0,0), (0,1,2), (1,2,0), (1,3,2), (2,0,1), (2,1,3), (3,2,1), (3,3,3)]
    for a, b, c in nonzeros:
        T_loc[a, b, c] = 1.0
    
    return _solve_rank7_numerically(T_loc)

def _solve_rank7_numerically(T_loc):
    """Numerically solve for rank-7 decomposition of 2x2x2 matmul tensor."""
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for seed in range(30):
        np.random.seed(seed)
        U_t = np.random.randn(4, 7) * 0.5
        V_t = np.random.randn(4, 7) * 0.5
        W_t = np.random.randn(4, 7) * 0.5
        
        for iteration in range(300):
            T_approx = np.einsum('at,bt,ct->abc', U_t, V_t, W_t)
            R = T_loc - T_approx
            
            grad_U = np.einsum('abc,bt,ct->at', R, V_t, W_t)
            grad_V = np.einsum('abc,at,ct->bt', R, U_t, W_t)
            grad_W = np.einsum('abc,at,bt->ct', R, U_t, V_t)
            
            lr = 0.02 / (1 + iteration * 0.003)
            
            U_t -= lr * grad_U
            V_t -= lr * grad_V
            W_t -= lr * grad_W
        
        err = np.max(np.abs(np.einsum('at,bt,ct->abc', U_t, V_t, W_t) - T_loc))
        if err < best_err:
            best_err = err
            best_U = U_t.copy()
            best_V = V_t.copy()
            best_W = W_t.copy()
        
        if best_err < 1e-10:
            break
    
    return best_U, best_V, best_W

def _strassen_block_decomposition(k0, k1):
    """
    Returns 7 rank-1 terms (u, v, w) for a 2x2x2 sub-block covering
    B columns k0, k1.
    """
    n, m, p = 2, 2, 13
    v_map = [k0, k1, m + k0, m + k1]
    w_map = [k0 * n + 0, k0 * n + 1, k1 * n + 0, k1 * n + 1]
    
    U_t, V_t, W_t = _find_strassen_rank7()
    
    terms = []
    for t in range(7):
        u_global = np.zeros(4)
        v_global = np.zeros(m * p)
        w_global = np.zeros(p * n)
        
        u_global[:] = U_t[:, t]
        
        for local_idx, global_idx in enumerate(v_map):
            v_global[global_idx] = V_t[local_idx, t]
        
        for local_idx, global_idx in enumerate(w_map):
            w_global[global_idx] = W_t[local_idx, t]
        
        terms.append((u_global, v_global, w_global))
    
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 2, 13
    
    U_school, V_school, W_school = _build_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    try:
        U_strassen = np.zeros((n * m, 46))
        V_strassen = np.zeros((m * p, 46))
        W_strassen = np.zeros((p * n, 46))
        
        col = 0
        for block in range(6):
            k0 = block * 2
            k1 = block * 2 + 1
            
            terms = _strassen_block_decomposition(k0, k1)
            for u, v, w in terms:
                U_strassen[:, col] = u
                V_strassen[:, col] = v
                W_strassen[:, col] = w
                col += 1
        
        k = 12
        for i in range(n):
            for kk in range(m):
                U_strassen[i * m + kk, col] = 1.0
                V_strassen[kk * p + k, col] = 1.0
                W_strassen[k * n + i, col] = 1.0
                col += 1
        
        T_target = _build_target_tensor()
        T_approx = np.einsum('it,jt,kt->ijk', U_strassen, V_strassen, W_strassen)
        
        error = np.max(np.abs(T_approx - T_target))
        
        if error < 1e-8:
            best_rank = 46
            best_U, best_V, best_W = U_strassen, V_strassen, W_strassen
    
    except Exception:
        pass
    
    if time.time() > deadline:
        pass
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

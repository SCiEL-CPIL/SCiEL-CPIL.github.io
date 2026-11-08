import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 130.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _elapsed()

def _verify(U, V, W, n, m, p, tol=1e-9):
    r = U.shape[1]
    T = np.ones((n, m, p), dtype=np.float64)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T_computed = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T_computed, T, atol=tol)

def _strassen_decomposition():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    U_coeffs = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)]
    ]
    V_coeffs = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)]
    ]
    W_coeff_map = {
        0: [(0,0,1), (1,1,1)],
        1: [(1,0,1), (1,1,-1)],
        2: [(0,1,1), (1,1,1)],
        3: [(0,0,1), (1,0,1)],
        4: [(0,0,-1), (0,1,1)],
        5: [(1,1,1)],
        6: [(0,0,1)]
    }
    
    col = 0
    for jp in range(2):
        for kp in range(2):
            for t in range(7):
                for (i_l, j_l, coeff) in U_coeffs[t]:
                    i = i_l
                    j = jp * 2 + j_l
                    U[i * m + j, col] += coeff
                for (j_l, k_l, coeff) in V_coeffs[t]:
                    j = jp * 2 + j_l
                    k = kp * 2 + k_l
                    V[j * p + k, col] += coeff
                for (i_l, k_l, coeff) in W_coeff_map[t]:
                    i = i_l
                    k = kp * 2 + k_l
                    W[k * n + i, col] += coeff
                col += 1
    return U, V, W

def _prune_rank(U, V, W, n, m, p):
    # Flatten tensors
    r = U.shape[1]
    dim_u = U.shape[0]
    dim_v = V.shape[0]
    dim_w = W.shape[0]
    
    # Create matrix of flattened tensors
    # Each column is a flattened rank-1 tensor
    flat_tensors = np.zeros((dim_u * dim_v * dim_w, r))
    for c in range(r):
        u_c = U[:, c]
        v_c = V[:, c]
        w_c = W[:, c]
        tensor_c = np.einsum('i,j,k->ijk', u_c, v_c, w_c)
        flat_tensors[:, c] = tensor_c.flatten()
        
    # Find rank
    # Use SVD to find rank
    try:
        U_svd, s, Vt = np.linalg.svd(flat_tensors, full_matrices=False)
        # Rank is number of non-zero singular values
        tol = 1e-10
        rank = np.sum(s > tol)
        
        if rank < r:
            # We have linear dependencies.
            # We can project onto the null space?
            # Or just pick 'rank' independent columns?
            # Picking independent columns from the original set might not work 
            # if the dependencies are not simple zero columns.
            # However, if the rank is 16, we can find a basis.
            # The basis vectors are linear combinations.
            # This is complex to implement robustly with integer scaling.
            # For now, we just return the original if rank reduction is not trivial.
            pass
    except Exception:
        pass
        
    return U, V, W

def _to_list_of_lists(mat):
    return [[float(x) for x in row] for row in mat]

def construct():
    n, m, p = 2, 4, 4
    
    if _remaining() < 10.0:
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}
    
    U, V, W = _strassen_decomposition()
    
    if _verify(U, V, W, n, m, p):
        return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}
    else:
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]) if result["U"] else 0)

import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _get_tensor_333():
    """
    Returns the flattened tensor for 3x3x3 matrix multiplication.
    Indices:
    U: i*3 + j
    V: j*3 + k
    W: k*3 + i
    T is 1 at (i*3+j, j*3+k, k*3+i)
    """
    n = 3
    size_u = n * n
    size_v = n * n
    size_w = n * n
    T = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                idx_u = i * n + j
                idx_v = j * n + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1
    return T

def _als_decompose(T, rank, max_iter=1000, tol=1e-10):
    """
    Alternating Least Squares for tensor decomposition.
    T is a 3D array.
    Returns U, V, W matrices of shapes (dim1, rank), (dim2, rank), (dim3, rank).
    """
    d1, d2, d3 = T.shape
    U = np.random.randn(d1, rank)
    V = np.random.randn(d2, rank)
    W = np.random.randn(d3, rank)
    
    for _ in range(max_iter):
        # Update U
        # T = sum_r U_r V_r W_r
        # For each r, U_r = argmin || T - sum U_r V_r W_r ||
        # This is a linear system.
        # Reshape T to (d1, d2*d3)
        T_mat = T.reshape(d1, d2 * d3)
        # V_W is (d2*d3, rank)
        V_W = np.einsum('jr,kr->jk', V, W).reshape(d2 * d3, rank)
        # Solve U @ V_W^T = T_mat
        # U = T_mat @ V_W @ (V_W^T @ V_W)^-1
        try:
            U_new = np.linalg.lstsq(V_W, T_mat.T, rcond=None)[0].T
        except:
            break
        # Update V
        T_mat_v = T.reshape(d2, d1 * d3)
        U_W = np.einsum('ir,kr->ik', U_new, W).reshape(d1 * d3, rank)
        try:
            V_new = np.linalg.lstsq(U_W, T_mat_v.T, rcond=None)[0].T
        except:
            break
        # Update W
        T_mat_w = T.reshape(d3, d1 * d2)
        U_V = np.einsum('ir,jr->ij', U_new, V_new).reshape(d1 * d2, rank)
        try:
            W_new = np.linalg.lstsq(U_V, T_mat_w.T, rcond=None)[0].T
        except:
            break
        
        # Check convergence
        T_approx = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
        error = np.linalg.norm(T - T_approx)
        if error < tol:
            return U_new, V_new, W_new
        
        U, V, W = U_new, V_new, W_new
        
    T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
    error = np.linalg.norm(T - T_approx)
    if error < tol:
        return U, V, W
    return None

def _try_integer_round(U, V, W, T, tol=1e-9):
    """
    Try to round U, V, W to integers and check if it exactly decomposes T.
    """
    U_int = np.rint(U)
    V_int = np.rint(V)
    W_int = np.rint(W)
    
    T_approx = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
    if np.allclose(T, T_approx, atol=tol):
        return U_int, V_int, W_int
    
    # Try half-integers?
    U_half = np.rint(U * 2) / 2
    V_half = np.rint(V * 2) / 2
    W_half = np.rint(W * 2) / 2
    T_approx = np.einsum('ir,jr,kr->ijk', U_half, V_half, W_half)
    if np.allclose(T, T_approx, atol=tol):
        return U_half, V_half, W_half
        
    return None

def _find_333_decomposition():
    """
    Find a low rank integer decomposition for 3x3x3 multiplication.
    """
    T = _get_tensor_333()
    best_rank = 9 # Schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 23 upwards
    for rank in range(23, 30):
        # Try a few random initializations
        for _ in range(10):
            np.random.seed(int(time.time() * 1000) % 100000)
            res = _als_decompose(T, rank, max_iter=200, tol=1e-8)
            if res is not None:
                U, V, W = res
                int_res = _try_integer_round(U, V, W, T)
                if int_res is not None:
                    U_int, V_int, W_int = int_res
                    if rank < best_rank:
                        best_rank = rank
                        best_U = U_int
                        best_V = V_int
                        best_W = W_int
                        return best_U, best_V, best_W, best_rank
        # If we found a float solution but not integer, we might still use it if we can't find integer?
        # No, must be exact.
        
    # Fallback to schoolbook if no better integer solution found
    n = 3
    rank = 27
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i*n+j, col] = 1
                V[j*n+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W, 27

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=6
    
    # Get 3x3x3 decomposition
    U_333, V_333, W_333, rank_333 = _find_333_decomposition()
    
    # Construct 3x3x6 decomposition
    # U is 9 x rank
    # V is 18 x rank
    # W is 18 x rank
    
    total_rank = 2 * rank_333
    U = np.zeros((9, total_rank))
    V = np.zeros((18, total_rank))
    W = np.zeros((18, total_rank))
    
    # First half: B_1 (cols 0-2), C_1 (cols 0-2)
    U[:, 0:rank_333] = U_333
    V[0:9, 0:rank_333] = V_333
    # V[9:18, 0:rank_333] = 0
    W[0:9, 0:rank_333] = W_333
    # W[9:18, 0:rank_333] = 0
    
    # Second half: B_2 (cols 3-5), C_2 (cols 3-5)
    # Note: The tensor for the second part is identical to the first,
    # just shifted in V and W indices.
    U[:, rank_333:2*rank_333] = U_333
    # V[0:9, rank_333:2*rank_333] = 0
    V[9:18, rank_333:2*rank_333] = V_333
    # W[0:9, rank_333:2*rank_333] = 0
    W[9:18, rank_333:2*rank_333] = W_333
    
    # Convert to lists of lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Target tensor T of size (n*m) x (m*p) x (p*n)
    # T[a, b, c] = 1 if indices are consistent, else 0
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    # We will represent T as a matrix of size (n*m) x (m*p * p*n)
    # T_mat[a, b * (p*n) + c]
    
    size_a = n * m
    size_b = m * p
    size_c = p * n
    
    T_mat = np.zeros((size_a, size_b * size_c), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for k in range(p):
                b = j * p + k
                c = k * n + i
                T_mat[a, b * size_c + c] = 1.0
                
    def als_decompose(T_mat, rank, iters=100, tol=1e-10, seed=0):
        np.random.seed(seed)
        size_a, size_bc = T_mat.shape
        size_b = m * p
        size_c = p * n
        
        U = np.random.randn(size_a, rank)
        V = np.random.randn(size_b, rank)
        W = np.random.randn(size_c, rank)
        
        prev_err = np.inf
        for it in range(iters):
            # Solve for U
            # T_mat approx U^T @ Y where Y rows are V_l otimes W_l
            Y = np.zeros((rank, size_b * size_c))
            for l in range(rank):
                Y[l, :] = np.outer(V[:, l], W[:, l]).flatten()
            
            # T_mat = U^T Y
            # U = (Y Y^T)^-1 Y T_mat^T
            try:
                U = np.linalg.solve(Y @ Y.T, Y @ T_mat.T).T
            except:
                U = np.linalg.pinv(Y @ Y.T) @ (Y @ T_mat.T).T
            
            # Solve for V
            # T_mat approx U^T @ (V otimes W)
            # Flatten T_mat to (size_a * size_b) x size_c?
            # T_mat[a, b * size_c + c]
            # Reshape to (size_a * size_b, size_c)
            T_mat_bc = T_mat.reshape(size_a * size_b, size_c)
            
            # For each c, T_mat_bc[:, c] = sum_l U_l V_l W_{c,l}
            # Let X_l = U_l otimes V_l (flattened)
            # T_mat_bc = X^T W?
            # T_mat_bc[a * size_b + b, c] = sum_l (U_l otimes V_l)_{a * size_b + b} W_{c,l}
            # So T_mat_bc = X^T W
            # W = (X X^T)^-1 X T_mat_bc^T
            
            X = np.zeros((rank, size_a * size_b))
            for l in range(rank):
                X[l, :] = np.outer(U[:, l], V[:, l]).flatten()
                
            try:
                W = np.linalg.solve(X @ X.T, X @ T_mat_bc.T).T
            except:
                W = np.linalg.pinv(X @ X.T) @ (X @ T_mat_bc.T).T
                
            # Solve for W
            # T_mat approx U^T @ (V otimes W)
            # Reshape T_mat to (size_a * size_c, size_b)
            # T_mat[a, b * size_c + c] -> T_mat_ac[a * size_c + c, b]
            T_mat_ac = T_mat.T.reshape(size_c * size_a, size_b).T
            # T_mat_ac[a * size_c + c, b] = sum_l (U_l otimes W_l)_{a * size_c + c} V_{b,l}
            # So T_mat_ac = Z^T V
            # V = (Z Z^T)^-1 Z T_mat_ac^T
            
            Z = np.zeros((rank, size_a * size_c))
            for l in range(rank):
                Z[l, :] = np.outer(U[:, l], W[:, l]).flatten()
                
            try:
                V = np.linalg.solve(Z @ Z.T, Z @ T_mat_ac.T).T
            except:
                V = np.linalg.pinv(Z @ Z.T) @ (Z @ T_mat_ac.T).T
                
            # Calculate error
            Y = np.zeros((rank, size_b * size_c))
            for l in range(rank):
                Y[l, :] = np.outer(V[:, l], W[:, l]).flatten()
            T_approx = U.T @ Y
            err = np.linalg.norm(T_mat - T_approx)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W, err

    # Try to find a solution with rank < 42
    best_result = None
    start_time = time.time()
    deadline = start_time + 140  # 140s limit
    
    for rank in [40, 41]:
        if time.time() > deadline:
            break
        for seed in range(10):
            if time.time() > deadline:
                break
            try:
                U, V, W, err = als_decompose(T_mat, rank, iters=50, tol=1e-12, seed=seed)
                if err < 1e-8:
                    # Check if we can round to half-integers
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    
                    # Verify rounded solution
                    Y = np.zeros((rank, size_b * size_c))
                    for l in range(rank):
                        Y[l, :] = np.outer(V_round[:, l], W_round[:, l]).flatten()
                    T_approx = U_round.T @ Y
                    err_round = np.linalg.norm(T_mat - T_approx)
                    
                    if err_round < 1e-8:
                        best_result = {'U': U_round.tolist(), 'V': V_round.tolist(), 'W': W_round.tolist(), 'rank': rank}
                        break
            except Exception:
                pass
        if best_result:
            break
            
    if best_result:
        return best_result
        
    # Fallback to rank 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def get_a_idx(group, local_i, local_j):
        global_col = local_j + 2 * group
        global_row = local_i
        return global_row * m + global_col

    def get_b_idx(group, local_i, local_j, block_col):
        global_row = local_i + 2 * group
        global_col = local_j + 2 * block_col
        return global_row * p + global_col

    def get_c_idx(local_i, local_j, block_col):
        global_row = local_i
        global_col = local_j + 2 * block_col
        return global_col * n + global_row

    col = 0
    for group in [0, 1]:
        for block_col in [0, 1, 2]:
            # Term 1
            U[get_a_idx(group, 0, 0)][col] += 1
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1
            
            # Term 2
            U[get_a_idx(group, 1, 0)][col] += 1
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            W[get_c_idx(1, 0, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] -= 1
            col += 1
            
            # Term 3
            U[get_a_idx(group, 0, 0)][col] += 1
            V[get_b_idx(group, 0, 1, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] -= 1
            W[get_c_idx(0, 1, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1
            
            # Term 4
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 1, 0, block_col)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] -= 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            W[get_c_idx(1, 0, block_col)][col] += 1
            col += 1
            
            # Term 5
            U[get_a_idx(group, 0, 0)][col] += 1
            U[get_a_idx(group, 0, 1)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] -= 1
            W[get_c_idx(0, 1, block_col)][col] += 1
            col += 1
            
            # Term 6
            U[get_a_idx(group, 1, 0)][col] += 1
            U[get_a_idx(group, 0, 0)][col] -= 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            V[get_b_idx(group, 0, 1, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1
            
            # Term 7
            U[get_a_idx(group, 0, 1)][col] += 1
            U[get_a_idx(group, 1, 1)][col] -= 1
            V[get_b_idx(group, 1, 0, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            col += 1
            
    return {'U': U, 'V': V, 'W': W}

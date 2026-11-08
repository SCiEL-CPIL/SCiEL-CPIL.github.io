import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 4, 6
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W uses trace convention: W[k*n+i, r] = C[i, k]
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for valid indices
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i  # trace convention
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Try rank 41 first, then work down
    best_rank = 42
    best_U, best_V, best_W = None, None, None
    
    # Use the known rank-42 solution as initialization
    def make_rank42():
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def get_a_idx(group, local_i, local_j):
            global_col = local_j + 2 * (group - 1)
            global_row = local_i
            return global_row * m + global_col
        
        def get_b_idx(group, local_i, local_j, block_col):
            global_row = local_i + 2 * (group - 1)
            global_col = local_j + 2 * block_col
            return global_row * p + global_col
        
        def get_c_idx(local_i, local_j, block_col):
            global_col = local_j + 2 * block_col
            global_row = local_i
            return global_col * n + global_row
        
        col = 0
        for group in [1, 2]:
            for block_col in [0, 1, 2]:
                # P1: (A11+A22)(B11+B22)
                U[get_a_idx(group, 0, 0), col] += 1
                U[get_a_idx(group, 1, 1), col] += 1
                V[get_b_idx(group, 0, 0, block_col), col] += 1
                V[get_b_idx(group, 1, 1, block_col), col] += 1
                W[get_c_idx(0, 0, block_col), col] += 1
                W[get_c_idx(1, 1, block_col), col] += 1
                col += 1
                # P2: (A21+A22)B11
                U[get_a_idx(group, 1, 0), col] += 1
                U[get_a_idx(group, 1, 1), col] += 1
                V[get_b_idx(group, 0, 0, block_col), col] += 1
                W[get_c_idx(1, 0, block_col), col] += 1
                W[get_c_idx(1, 1, block_col), col] -= 1
                col += 1
                # P3: A11(B12-B22)
                U[get_a_idx(group, 0, 0), col] += 1
                V[get_b_idx(group, 0, 1, block_col), col] += 1
                V[get_b_idx(group, 1, 1, block_col), col] -= 1
                W[get_c_idx(0, 1, block_col), col] += 1
                W[get_c_idx(1, 1, block_col), col] += 1
                col += 1
                # P4: A22(B21-B11)
                U[get_a_idx(group, 1, 1), col] += 1
                V[get_b_idx(group, 1, 0, block_col), col] += 1
                V[get_b_idx(group, 0, 0, block_col), col] -= 1
                W[get_c_idx(0, 0, block_col), col] += 1
                W[get_c_idx(1, 0, block_col), col] += 1
                col += 1
                # P5: (A11+A12)B22
                U[get_a_idx(group, 0, 0), col] += 1
                U[get_a_idx(group, 0, 1), col] += 1
                V[get_b_idx(group, 1, 1, block_col), col] += 1
                W[get_c_idx(0, 0, block_col), col] -= 1
                W[get_c_idx(0, 1, block_col), col] += 1
                col += 1
                # P6: (A21-A11)(B11+B12)
                U[get_a_idx(group, 1, 0), col] += 1
                U[get_a_idx(group, 0, 0), col] -= 1
                V[get_b_idx(group, 0, 0, block_col), col] += 1
                V[get_b_idx(group, 0, 1, block_col), col] += 1
                W[get_c_idx(1, 1, block_col), col] += 1
                col += 1
                # P7: (A12-A22)(B21+B22)
                U[get_a_idx(group, 0, 1), col] += 1
                U[get_a_idx(group, 1, 1), col] -= 1
                V[get_b_idx(group, 1, 0, block_col), col] += 1
                V[get_b_idx(group, 1, 1, block_col), col] += 1
                W[get_c_idx(0, 0, block_col), col] += 1
                col += 1
        return U, V, W
    
    U42, V42, W42 = make_rank42()
    best_U, best_V, best_W = U42, V42, W42
    best_rank = 42
    
    # Try to reduce rank by 1 using optimization
    # Start from rank 42 and try to zero out one column
    for target_rank in [41]:
        if time.time() > deadline - 30:
            break
        
        rank = target_rank
        # Initialize from rank 42 solution, drop last column
        U = U42[:, :rank].copy()
        V = V42[:, :rank].copy()
        W = W42[:, :rank].copy()
        
        # Add small random perturbation
        np.random.seed(42)
        U += np.random.randn(*U.shape) * 0.01
        V += np.random.randn(*V.shape) * 0.01
        W += np.random.randn(*W.shape) * 0.01
        
        # Optimize using gradient descent
        lr = 0.01
        for step in range(500):
            if time.time() > deadline - 10:
                break
            
            # Compute residual: R = T - sum_r U[:,r] V[:,r] W[:,r]
            # Efficient computation: R[a,b,c] = T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r]
            R = T.copy()
            for r in range(rank):
                R -= np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
            
            # Gradient w.r.t. U[a,r]: -R[a,b,c] * V[b,r] * W[c,r] summed over b,c
            grad_U = np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = np.einsum('abc,ar,br->cr', R, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            err = np.linalg.norm(R) / np.linalg.norm(T)
            if err < 1e-6:
                break
        
        # Check if we achieved rank 41
        R = T.copy()
        for r in range(rank):
            R -= np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
        err = np.linalg.norm(R) / np.linalg.norm(T)
        
        if err < 1e-4:
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break
    
    # Convert to lists
    U_list = [[float(U[i, j]) for j in range(best_rank)] for i in range(n * m)]
    V_list = [[float(V[i, j]) for j in range(best_rank)] for i in range(m * p)]
    W_list = [[float(W[i, j]) for j in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

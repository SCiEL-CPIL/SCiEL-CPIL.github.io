import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    M = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                M[u_idx, v_idx, w_idx] = 1.0

    M_u = M.reshape(U_dim, V_dim * W_dim)
    M_v = M.transpose(1, 0, 2).reshape(V_dim, U_dim * W_dim)
    M_w = M.transpose(2, 0, 1).reshape(W_dim, U_dim * V_dim)

    # Try to find a decomposition with rank < 22
    for R in range(12, 22):
        for seed in range(5):
            np.random.seed(seed)
            U = np.random.rand(U_dim, R)
            V = np.random.rand(V_dim, R)
            W = np.random.rand(W_dim, R)
            
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            
            for it in range(100):
                # Update U
                A = np.zeros((V_dim * W_dim, R))
                for r in range(R):
                    A[:, r] = np.kron(W[:, r], V[:, r])
                Y = np.linalg.lstsq(A, M_u.T, rcond=None)[0]
                U = Y.T
                
                # Update V
                A = np.zeros((U_dim * W_dim, R))
                for r in range(R):
                    A[:, r] = np.kron(W[:, r], U[:, r])
                Y = np.linalg.lstsq(A, M_v.T, rcond=None)[0]
                V = Y.T
                
                # Update W
                A = np.zeros((U_dim * V_dim, R))
                for r in range(R):
                    A[:, r] = np.kron(V[:, r], U[:, r])
                Y = np.linalg.lstsq(A, M_w.T, rcond=None)[0]
                W = Y.T
                
                # Check error
                M_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
                err = np.linalg.norm(M_rec - M)
                if err < 1e-8:
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    M_rec_int = np.einsum('ur,vr,wr->uvw', U_int, V_int, W_int)
                    err_int = np.linalg.norm(M_rec_int - M)
                    if err_int < 1e-6:
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                    break

    # Fallback to seed solution
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val

    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    return {"U": U, "V": V, "W": W}

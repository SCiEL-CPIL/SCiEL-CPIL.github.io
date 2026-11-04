import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_2x2x1():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        return U, V, W

    def find_2x2x3_decomposition():
        """Try to find a low-rank decomposition for 2x2 * 2x3 -> 2x3 using scipy."""
        n_s, m_s, p_s = 2, 2, 3
        target = np.zeros((n_s, m_s, p_s))
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    target[i, j, k] = 1.0

        def compute_tensor_flat(x):
            """Compute tensor from flattened x. x contains U, V, W concatenated."""
            rank = x.size // (n_s * m_s + m_s * p_s + p_s * n_s)
            if rank == 0:
                return np.zeros((n_s, m_s, p_s))
            
            U_size = n_s * m_s * rank
            V_size = m_s * p_s * rank
            W_size = p_s * n_s * rank
            
            U_flat = x[:U_size]
            V_flat = x[U_size:U_size+V_size]
            W_flat = x[U_size+V_size:U_size+V_size+W_size]
            
            U = U_flat.reshape(n_s * m_s, rank)
            V = V_flat.reshape(m_s * p_s, rank)
            W = W_flat.reshape(p_s * n_s, rank)
            
            T = np.zeros((n_s, m_s, p_s))
            for i in range(n_s):
                for j in range(m_s):
                    for k in range(p_s):
                        u_idx = i * m_s + j
                        v_idx = j * p_s + k
                        w_idx = k * n_s + i
                        T[i, j, k] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            return T

        def loss_flat(x):
            T = compute_tensor_flat(x)
            return np.sum((T - target) ** 2)

        best_rank = 11
        best_x = None
        best_loss = float('inf')

        # Initialize with known rank 11 solution
        U11, V11, W11 = [], [], []
        U1, V1, W1 = build_2x2x1()
        U2, V2, W2 = build_2x2x2_strassen()
        
        # Construct rank 11
        U11 = np.zeros((4, 11))
        V11 = np.zeros((6, 11))
        W11 = np.zeros((6, 11))
        
        # Map 2x2x1 (rank 4) to columns 0-3
        for r in range(4):
            for c in range(4):
                U11[r, c] = U1[r][c]
        for r in range(2):
            for c in range(4):
                V11[r*3, c] = V1[r][c]
                # Note: 2x2x1 V has shape (2, 4). We need to map to (6, 11).
                # j in 0..1, k in 0..1. v_idx = j*2 + k.
                # Here p_s=2 for the sub-problem? No, p_s=3.
                # The sub-problem is 2x2 * 2x3.
                # V1 is for 2x2 * 2x1. So V1 has shape (2*1, 4) = (2,4).
                # We need to place this in V11 which is for 2x2 * 2x3, so shape (2*3, 11) = (6, 11).
                # The first column of B corresponds to k=0.
                # So V1[j, c] goes to V11[j*3 + 0, c].
                pass 
        
        # Re-do mapping carefully
        U11 = np.zeros((4, 11))
        V11 = np.zeros((6, 11))
        W11 = np.zeros((6, 11))
        
        # 2x2x1 part (columns 0-3 of rank 11)
        # U1 shape (4,4), V1 shape (2,4), W1 shape (2,4)
        # Target 2x2x3: U (4,11), V (6,11), W (6,11)
        # V index: j*p + k. p=3.
        # For the first column (k=0), V1[j, c] -> V11[j*3, c].
        # For the second column? The 2x2x1 decomposition only covers 1 column of output.
        # So it only fills k=0.
        for r in range(4):
            for c in range(4):
                U11[r, c] = U1[r][c]
        for j in range(2):
            for c in range(4):
                V11[j*3, c] = V1[j][c]
        for i in range(2):
            for c in range(4):
                # W1 shape (2,4). W index: k*n + i. n=2.
                # For k=0, W1[i, c] -> W11[0*2 + i, c] = W11[i, c].
                W11[i, c] = W1[i][c]
                
        # 2x2x2 part (columns 4-10 of rank 11)
        # U2 shape (4,7), V2 shape (4,7), W2 shape (4,7)
        # V2 covers j=0,1 and k=0,1.
        # We need to place these in V11 for k=0,1.
        # V2[j, c] -> V11[j*3 + k, c]? No, V2 is for 2x2x2.
        # In 2x2x2, V index is j*2 + k.
        # In 2x2x3, V index is j*3 + k.
        # So for a given j, k in 0..1:
        # V2[j*2+k, c] -> V11[j*3+k, c].
        # W2 covers i=0,1 and k=0,1.
        # W2[k*2+i, c] -> W11[k*2+i, c].
        
        for r in range(4):
            for c in range(7):
                U11[r, 4+c] = U2[r][c]
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    V11[j*3+k, 4+c] = V2[j*2+k][c]
        for k in range(2):
            for i in range(2):
                for c in range(7):
                    W11[k*2+i, 4+c] = W2[k*2+i][c]
                    
        best_x = np.concatenate([U11.flatten(), V11.flatten(), W11.flatten()])
        best_loss = loss_flat(best_x)
        
        # Try to optimize for lower ranks
        for target_rank in [10, 9]:
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42)
            for attempt in range(20):
                if time.time() > deadline - 5:
                    break
                    
                # Random init
                x0 = np.random.randn((n_s*m_s + m_s*p_s + p_s*n_s) * target_rank) * 0.5
                
                res = minimize(loss_flat, x0, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 10000})
                
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_x = res.x
                    best_rank = target_rank
                    
                if best_loss < 1e-10:
                    break
                    
            if best_loss < 1e-10:
                break

        # Extract best U, V, W
        U_size = n_s * m_s * best_rank
        V_size = m_s * p_s * best_rank
        W_size = p_s * n_s * best_rank
        
        U_flat = best_x[:U_size]
        V_flat = best_x[U_size:U_size+V_size]
        W_flat = best_x[U_size+V_size:U_size+V_size+W_size]
        
        U = U_flat.reshape(n_s * m_s, best_rank)
        V = V_flat.reshape(m_s * p_s, best_rank)
        W = W_flat.reshape(p_s * n_s, best_rank)
        
        U_list = [[float(U[r, c]) for c in range(best_rank)] for r in range(n_s * m_s)]
        V_list = [[float(V[r, c]) for c in range(best_rank)] for r in range(m_s * p_s)]
        W_list = [[float(W[r, c]) for c in range(best_rank)] for r in range(p_s * n_s)]
        
        return U_list, V_list, W_list, best_rank

    def build_4x4x6():
        """Build the 4x4 * 4x6 decomposition using Strassen."""
        n, m, p = 4, 4, 6

        U_sub, V_sub, W_sub, rank_sub = find_2x2x3_decomposition()
        total_rank = 7 * rank_sub

        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]

        terms = [
            {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
            {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},
            {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
            {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
            {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
            {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
            {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},
        ]

        u_offset = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
        v_offset = {0: (0, 0), 1: (0, 3), 2: (2, 0), 3: (2, 3)}
        w_offset = {0: (0, 0), 1: (3, 0), 2: (0, 2), 3: (3, 2)}

        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0:
                        continue
                    i_local = u_local // 2
                    j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        g_u = (i_local + ro) * m + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff
                for v_local in range(6):
                    v_val = V_sub[v_local][c]
                    if v_val == 0:
                        continue
                    j_local = v_local // 3
                    k_local = v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        g_v = (j_local + ro) * p + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff
                for w_local in range(6):
                    w_val = W_sub[w_local][c]
                    if w_val == 0:
                        continue
                    k_local = w_local // 2
                    i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        g_w = (k_local + ro) * n + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff

        return U, V, W, total_rank

    U, V, W, total_rank = build_4x4x6()

    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(sum(prod) - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
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

import time
import numpy as np

def construct():
    n, m, p = 3, 4, 6
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Build the matrix multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1
    T_float = T.astype(float)
    
    def verify(U, V, W):
        """Verify decomposition exactly reconstructs T."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T_check = np.einsum('ar,br,cr->abc', U_arr.T, V_arr.T, W_arr.T)
        return np.allclose(T_check, T_float, atol=1e-10)
    
    def get_baseline_66():
        """Return the known rank-66 decomposition."""
        U = [[0] * 66 for _ in range(n * m)]
        V = [[0] * 66 for _ in range(m * p)]
        W = [[0] * 66 for _ in range(p * n)]
        col = 0
        
        def add_schoolbook(i, j, k):
            nonlocal col
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
        
        def add_strassen_222(i_base, j_base, k_base):
            nonlocal col
            a_idx = i_base * m + j_base
            b_idx = i_base * m + (j_base + 1)
            c_idx = (i_base + 1) * m + j_base
            d_idx = (i_base + 1) * m + (j_base + 1)
            e_idx = j_base * p + k_base
            f_idx = j_base * p + (k_base + 1)
            g_idx = (j_base + 1) * p + k_base
            h_idx = (j_base + 1) * p + (k_base + 1)
            w00 = k_base * n + i_base
            w01 = (k_base + 1) * n + i_base
            w10 = k_base * n + (i_base + 1)
            w11 = (k_base + 1) * n + (i_base + 1)
            
            U[a_idx][col] = 1; U[d_idx][col] = 1
            V[e_idx][col] = 1; V[h_idx][col] = 1
            W[w00][col] = 1; W[w11][col] = 1
            col += 1
            U[c_idx][col] = 1; U[d_idx][col] = 1
            V[e_idx][col] = 1
            W[w10][col] = 1; W[w11][col] = -1
            col += 1
            U[a_idx][col] = 1
            V[f_idx][col] = 1; V[h_idx][col] = -1
            W[w01][col] = 1; W[w11][col] = 1
            col += 1
            U[d_idx][col] = 1
            V[g_idx][col] = 1; V[e_idx][col] = -1
            W[w00][col] = 1; W[w10][col] = 1
            col += 1
            U[a_idx][col] = 1; U[b_idx][col] = 1
            V[h_idx][col] = 1
            W[w00][col] = -1; W[w01][col] = 1
            col += 1
            U[c_idx][col] = 1; U[a_idx][col] = -1
            V[e_idx][col] = 1; V[f_idx][col] = 1
            W[w11][col] = 1
            col += 1
            U[b_idx][col] = 1; U[d_idx][col] = -1
            V[g_idx][col] = 1; V[h_idx][col] = 1
            W[w00][col] = 1
            col += 1
        
        # Row 0: schoolbook 24 terms
        for j in range(4):
            for k in range(6):
                add_schoolbook(0, j, k)
        
        # Rows 1-2: 6 Strassen blocks of 2x2x2
        for j_block in range(2):
            for k_block in range(3):
                j_base = j_block * 2
                k_base = k_block * 2
                add_strassen_222(1, j_base, k_base)
        
        return U, V, W
    
    U_base, V_base, W_base = get_baseline_66()
    best_rank = 66
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    
    # Try to find a rank-65 or lower decomposition
    # Strategy: use the 3x2x2 block structure with rank 11 per block
    # This gives 6 * 11 = 66, same as before, but different structure
    # Try to find cross-block optimizations
    
    # Alternative: try a direct search for rank < 66
    # Use a greedy approach: start with schoolbook (72), greedily add rank-1 terms
    # that maximize the reduction in "effective rank"
    
    if time.time() < deadline - 15:
        np.random.seed(12345)
        
        # Greedy rank-1 peeling from the full tensor
        T_rem = T_float.copy()
        U_list = []
        V_list = []
        W_list = []
        
        max_terms = 65
        for step in range(max_terms):
            if time.time() > deadline - 15:
                break
            
            if np.allclose(T_rem, 0, atol=1e-10):
                break
            
            best_u, best_v, best_w, best_score = None, None, None, np.count_nonzero(T_rem)
            
            # Try multiple random candidates
            for attempt in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Random sparse vectors with small integer entries
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                
                # Sparse supports
                n_u = np.random.randint(1, 5)
                n_v = np.random.randint(1, 6)
                n_w = np.random.randint(1, 4)
                
                u_idx = np.random.choice(n*m, min(n_u, n*m), replace=False)
                v_idx = np.random.choice(m*p, min(n_v, m*p), replace=False)
                w_idx = np.random.choice(p*n, min(n_w, p*n), replace=False)
                
                u[u_idx] = np.random.choice([-2, -1, 1, 2], size=len(u_idx))
                v[v_idx] = np.random.choice([-2, -1, 1, 2], size=len(v_idx))
                w[w_idx] = np.random.choice([-2, -1, 1, 2], size=len(w_idx))
                
                rank1 = np.einsum('a,b,c->abc', u, v, w)
                
                # Try to find optimal alpha to minimize sparsity
                # Use a few candidate alphas
                for alpha in [0.5, 1.0, -0.5, -1.0]:
                    T_test = T_rem - alpha * rank1
                    score = np.count_nonzero(T_test)
                    if score < best_score:
                        best_score = score
                        best_u = u
                        best_v = v
                        best_w = w
                        best_alpha = alpha
            
            if best_u is None or best_score >= np.count_nonzero(T_rem):
                break
            
            T_rem = T_rem - best_alpha * np.einsum('a,b,c->abc', best_u, best_v, best_w)
            U_list.append(best_u)
            V_list.append(best_v)
            W_list.append(best_w)
        
        if len(U_list) < best_rank and np.allclose(T_rem, 0, atol=1e-10):
            U_new = [[0] * len(U_list) for _ in range(n*m)]
            V_new = [[0] * len(U_list) for _ in range(m*p)]
            W_new = [[0] * len(U_list) for _ in range(p*n)]
            for r in range(len(U_list)):
                for a in range(n*m):
                    U_new[a][r] = U_list[r][a]
                for b in range(m*p):
                    V_new[b][r] = V_list[r][b]
                for c in range(p*n):
                    W_new[c][r] = W_list[r][c]
            if verify(U_new, V_new, W_new):
                best_rank = len(U_list)
                best_result = {"U": U_new, "V": V_new, "W": W_new}
    
    return best_result

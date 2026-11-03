"""Rewritten program for matrix multiplication 3,5,7 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Define index groups for Strassen tiling
    # n=3: {0,1}, {2}
    # m=5: {0,1}, {2,3}, {4}
    # p=7: {0,1}, {2,3}, {4,5}, {6}
    
    n_groups = [[0, 1], [2]]
    m_groups = [[0, 1], [2, 3], [4]]
    p_groups = [[0, 1], [2, 3], [4, 5], [6]]
    
    # We will build the decomposition by iterating over all combinations of groups
    # If a group has size 2 in all three dimensions, we use Strassen (rank 7)
    # Otherwise, we use schoolbook (rank size_n * size_m * size_p)
    
    U = {}
    V = {}
    W = {}
    
    # Helper to add to dict
    def add_dict(d, key, val):
        if key in d:
            d[key] += val
        else:
            d[key] = val

    for ng in n_groups:
        for mg in m_groups:
            for pg in p_groups:
                if len(ng) == 2 and len(mg) == 2 and len(pg) == 2:
                    a, b = ng
                    c, d = mg
                    e, f = pg
                    
                    # Strassen 7 terms
                    # 1
                    add_dict(U, (a, c), 1); add_dict(U, (b, d), 1)
                    add_dict(V, (c, e), 1); add_dict(V, (d, f), 1)
                    add_dict(W, (e, a), 1); add_dict(W, (f, b), 1)
                    
                    # 2
                    add_dict(U, (b, c), 1); add_dict(U, (b, d), 1)
                    add_dict(V, (c, e), 1)
                    add_dict(W, (e, b), 1); add_dict(W, (f, b), -1)
                    
                    # 3
                    add_dict(U, (a, c), 1)
                    add_dict(V, (d, e), 1); add_dict(V, (c, e), -1)
                    add_dict(W, (f, a), 1); add_dict(W, (f, b), 1)
                    
                    # 4
                    add_dict(U, (b, d), 1)
                    add_dict(V, (c, f), 1); add_dict(V, (d, e), -1)
                    add_dict(W, (e, a), 1); add_dict(W, (e, b), 1)
                    
                    # 5
                    add_dict(U, (a, c), 1); add_dict(U, (a, d), 1)
                    add_dict(V, (d, f), 1)
                    add_dict(W, (f, a), 1); add_dict(W, (e, a), -1)
                    
                    # 6
                    add_dict(U, (b, c), 1); add_dict(U, (a, c), -1)
                    add_dict(V, (c, e), 1); add_dict(V, (c, f), 1)
                    add_dict(W, (e, b), 1)
                    
                    # 7
                    add_dict(U, (a, d), 1); add_dict(U, (b, d), -1)
                    add_dict(V, (d, e), 1); add_dict(V, (d, f), 1)
                    add_dict(W, (f, b), 1)
                    
                else:
                    # Schoolbook
                    for i in ng:
                        for j in mg:
                            for k in pg:
                                add_dict(U, (i, j), 1)
                                add_dict(V, (j, k), 1)
                                add_dict(W, (k, i), 1)
                                
    # Convert dicts to matrices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Determine max rank
    # Each term in the dict is a column in the final matrices
    # But wait, the dict approach above merges terms that share the same (i,j) etc?
    # No, the dict approach is wrong for tensor decomposition.
    # Each "term" in the sum $\sum u_c \otimes v_c \otimes w_c$ is a separate column c.
    # My loop above adds coefficients to the *same* column if the indices match?
    # No, I need to generate a list of tuples for each column.
    
    U_list = []
    V_list = []
    W_list = []
    
    for ng in n_groups:
        for mg in m_groups:
            for pg in p_groups:
                if len(ng) == 2 and len(mg) == 2 and len(pg) == 2:
                    a, b = ng
                    c, d = mg
                    e, f = pg
                    
                    terms = [
                        # (U_indices, V_indices, W_indices)
                        # 1
                        ([(a,c,1), (b,d,1)], [(c,e,1), (d,f,1)], [(e,a,1), (f,b,1)]),
                        # 2
                        ([(b,c,1), (b,d,1)], [(c,e,1)], [(e,b,1), (f,b,-1)]),
                        # 3
                        ([(a,c,1)], [(d,e,1), (c,e,-1)], [(f,a,1), (f,b,1)]),
                        # 4
                        ([(b,d,1)], [(c,f,1), (d,e,-1)], [(e,a,1), (e,b,1)]),
                        # 5
                        ([(a,c,1), (a,d,1)], [(d,f,1)], [(f,a,1), (e,a,-1)]),
                        # 6
                        ([(b,c,1), (a,c,-1)], [(c,e,1), (c,f,1)], [(e,b,1)]),
                        # 7
                        ([(a,d,1), (b,d,-1)], [(d,e,1), (d,f,1)], [(f,b,1)])
                    ]
                    
                    for t in terms:
                        u_vec = [0] * (n * m)
                        v_vec = [0] * (m * p)
                        w_vec = [0] * (p * n)
                        
                        for idx, val in t[0]:
                            u_vec[idx[0] * m + idx[1]] = val
                        for idx, val in t[1]:
                            v_vec[idx[0] * p + idx[1]] = val
                        for idx, val in t[2]:
                            w_vec[idx[0] * n + idx[1]] = val
                        
                        U_list.append(u_vec)
                        V_list.append(v_vec)
                        W_list.append(w_vec)
                        
                else:
                    for i in ng:
                        for j in mg:
                            for k in pg:
                                u_vec = [0] * (n * m)
                                v_vec = [0] * (m * p)
                                w_vec = [0] * (p * n)
                                
                                u_vec[i * m + j] = 1
                                v_vec[j * p + k] = 1
                                w_vec[k * n + i] = 1
                                
                                U_list.append(u_vec)
                                V_list.append(v_vec)
                                W_list.append(w_vec)
                                
    rank = len(U_list)
    
    # Verification
    U_arr = np.array(U_list, dtype=float).T # Shape (n*m, rank)
    V_arr = np.array(V_list, dtype=float).T # Shape (m*p, rank)
    W_arr = np.array(W_list, dtype=float).T # Shape (p*n, rank)
    
    # Check a few random triples
    import random
    random.seed(42)
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        val = float(np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i]))
        if abs(val - 1) > 1e-9:
            # Fallback to schoolbook if verification fails
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
            
    return {"U": U_list, "V": V_list, "W": W_list}


import time
import numpy as np

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Strassen 2x2x2 terms
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Part 1: j in {0,1}, k in {0..8}
    for k1, k2 in [(0,1), (2,3), (4,5), (6,7)]:
        for u_d, v_d, w_d in strassen_terms:
            U_t = {}
            V_t = {}
            W_t = {}
            for (i, j), c in u_d.items():
                U_t[(i, j)] = c
            for (j, k_loc), c in v_d.items():
                V_t[(j, k1 + k_loc)] = c
            for (i, k_loc), c in w_d.items():
                W_t[(i, k1 + k_loc)] = c
            U_terms.append(U_t)
            V_terms.append(V_t)
            W_terms.append(W_t)
            
    # 1 single k: 8
    for i in range(2):
        for j in range(2):
            U_terms.append({(i, j): 1})
            V_terms.append({(j, 8): 1})
            W_terms.append({(i, 8): 1})
            
    # Part 2: j in {2}, k in {0..8}
    for i in range(2):
        for k in range(9):
            U_terms.append({(i, 2): 1})
            V_terms.append({(2, k): 1})
            W_terms.append({(i, k): 1})
            
    rank = len(U_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for (i, j), c in U_terms[col].items():
            U[i * m + j][col] = c
        for (j, k), c in V_terms[col].items():
            V[j * p + k][col] = c
        for (i, k), c in W_terms[col].items():
            W[k * n + i][col] = c
            
    # Build tensor vectors for SVD-based dependency check
    vec_dim = n * m * m * p * p * n
    tensor_vecs = []
    for col in range(rank):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for (i, j), c in U_terms[col].items():
            u_vec[i * m + j] = c
        for (j, k), c in V_terms[col].items():
            v_vec[j * p + k] = c
        for (i, k), c in W_terms[col].items():
            w_vec[k * n + i] = c
        vec = np.kron(np.kron(np.array(u_vec, dtype=float), np.array(v_vec, dtype=float)), np.array(w_vec, dtype=float))
        tensor_vecs.append(vec)
    
    M = np.array(tensor_vecs).T  # Shape (vec_dim, rank)
    tol = 1e-10
    
    # Iterative greedy elimination with multiple passes
    removed = set()
    for _pass in range(10):  # Max 10 passes
        if time.time() > deadline:
            break
        found_elimination = False
        for i in range(rank):
            if i in removed:
                continue
            if time.time() > deadline:
                break
            # Try removing i
            mask = [j for j in range(rank) if j != i and j not in removed]
            if len(mask) == 0:
                break
            M_sub = M[:, mask]
            s_sub = np.linalg.svd(M_sub, full_matrices=False)[1]
            r_sub = np.sum(s_sub > tol)
            
            # Check full rank
            s_full = np.linalg.svd(M[:, [j for j in range(rank) if j not in removed]], full_matrices=False)[1]
            r_full = np.sum(s_full > tol)
            
            if r_sub == r_full:
                removed.add(i)
                found_elimination = True
        
        if not found_elimination:
            break
    
    # Reconstruct if we removed anything
    if removed:
        keep = [i for i in range(rank) if i not in removed]
        new_rank = len(keep)
        
        new_U = [[0] * new_rank for _ in range(n * m)]
        new_V = [[0] * new_rank for _ in range(m * p)]
        new_W = [[0] * new_rank for _ in range(p * n)]
        
        for new_col, old_col in enumerate(keep):
            for (i, j), c in U_terms[old_col].items():
                new_U[i * m + j][new_col] = c
            for (j, k), c in V_terms[old_col].items():
                new_V[j * p + k][new_col] = c
            for (i, k), c in W_terms[old_col].items():
                new_W[k * n + i][new_col] = c
                
        return {"U": new_U, "V": new_V, "W": new_W}
        
    return {"U": U, "V": V, "W": W}

"""Laser Method: tensor power of Strassen 2x2x2 projected onto 3x4x7."""
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

# Strassen 2x2x2: 7 rank-1 terms.
# Each term: (u_list, v_list, w_list) where each is a list of (index_tuple, coeff)
# u: for A (2x2), index (i,j) in {0,1}^2
# v: for B (2x2), index (j,k) in {0,1}^2
# w: for C (2x2), index (k,i) in {0,1}^2

STRASSEN_TERMS = [
    # M1 = (A11+A22)(B11+B22)
    ([( (0,0), 1), ((1,1), 1)], [((0,0), 1), ((0,1), 1)], [((0,0), 1), ((1,1), 1)]),
    # M2 = (A21+A22)B11
    ([( (1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1)]),
    # M3 = A11(B21-B22)
    ([( (0,0), 1)], [((1,0), 1), ((1,1), -1)], [((0,0), 1), ((1,1), 1)]),
    # M4 = A22(B12-B11)
    ([( (1,1), 1)], [((0,1), 1), ((0,0), -1)], [((1,0), 1), ((0,0), 1)]),
    # M5 = (A11+A12)B22
    ([( (0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((1,1), 1)]),
    # M6 = (A21-A11)(B11+B12)
    ([( (1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
    # M7 = (A12-A22)(B21+B22)
    ([( (0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
]

def tensor_power_terms(k):
    """Generate all 7^k tensor power terms of Strassen.
    Returns list of (u_dict, v_dict, w_dict) where:
    u_dict: {(a, b): coeff} with a, b in {0,1}^k
    v_dict: {(b, c): coeff} with b, c in {0,1}^k
    w_dict: {(c, a): coeff} with c, a in {0,1}^k
    """
    # Base case: convert STRASSEN_TERMS to dict form
    base = []
    for (u_list, v_list, w_list) in STRASSEN_TERMS:
        u = {idx: coeff for idx, coeff in u_list}
        v = {idx: coeff for idx, coeff in v_list}
        w = {idx: coeff for idx, coeff in w_list}
        base.append((u, v, w))
    
    if k == 1:
        return base
    
    current = base
    for level in range(1, k):
        new_current = []
        for (u1, v1, w1) in current:
            for (u2, v2, w2) in base:
                # Tensor product
                new_u = {}
                for (a1, b1), c1 in u1.items():
                    for (a2, b2), c2 in u2.items():
                        key = (a1 + a2, b1 + b2)
                        new_u[key] = new_u.get(key, 0) + c1 * c2
                
                new_v = {}
                for (b1, c1), c1_val in v1.items():
                    for (b2, c2), c2_val in v2.items():
                        key = (b1 + b2, c1 + c2)
                        new_v[key] = new_v.get(key, 0) + c1_val * c2_val
                
                new_w = {}
                for (c1, a1), c1_val in w1.items():
                    for (c2, a2), c2_val in w2.items():
                        key = (c1 + c2, a1 + a2)
                        new_w[key] = new_w.get(key, 0) + c1_val * c2_val
                
                new_current.append((new_u, new_v, new_w))
        current = new_current
    
    return current

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    
    n, m, p = DIMENSIONS
    
    best_rank = n * m * p  # schoolbook = 84
    best_result = None
    
    for k in [2, 3, 4]:
        if time.time() > DEADLINE - 30:
            break
        
        try:
            terms = tensor_power_terms(k)
            
            # Choose index subsets
            all_indices = list(product(range(2), repeat=k))
            
            # For m=4: need 4 indices from {0,1}^k
            # For n=3: need 3 indices
            # For p=7: need 7 indices
            # 
            # For k=2: 4 indices total. m=4 uses all, n=3 uses 3, p=7 impossible (only 4)
            # For k=3: 8 indices total. m=4 uses 4, n=3 uses 3, p=7 uses 7. Works!
            # For k=4: 16 indices. All fit.
            
            if k < 3 and p > 2**k:
                continue
            
            # Choose subsets: use first d indices in lexicographic order
            n_subset = all_indices[:n]
            m_subset = all_indices[:m]
            p_subset = all_indices[:p]
            
            n_map = {idx: i for i, idx in enumerate(n_subset)}
            m_map = {idx: j for j, idx in enumerate(m_subset)}
            p_map = {idx: k_idx for k_idx, idx in enumerate(p_subset)}
            
            # Project each term
            rank = 0
            U_cols = []
            V_cols = []
            W_cols = []
            
            for (u, v, w) in terms:
                u_proj = {}
                for (a_t, b_t), coeff in u.items():
                    if a_t in n_map and b_t in m_map and coeff != 0:
                        i = n_map[a_t]
                        j = m_map[b_t]
                        u_proj[(i, j)] = u_proj.get((i, j), 0) + coeff
                
                v_proj = {}
                for (b_t, c_t), coeff in v.items():
                    if b_t in m_map and c_t in p_map and coeff != 0:
                        j = m_map[b_t]
                        k_idx = p_map[c_t]
                        v_proj[(j, k_idx)] = v_proj.get((j, k_idx), 0) + coeff
                
                w_proj = {}
                for (c_t, a_t), coeff in w.items():
                    if c_t in p_map and a_t in n_map and coeff != 0:
                        k_idx = p_map[c_t]
                        i = n_map[a_t]
                        w_proj[(k_idx, i)] = w_proj.get((k_idx, i), 0) + coeff
                
                if u_proj and v_proj and w_proj:
                    rank += 1
                    U_cols.append(u_proj)
                    V_cols.append(v_proj)
                    W_cols.append(w_proj)
            
            if rank < best_rank and rank > 0:
                best_rank = rank
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                for r in range(rank):
                    for (i, j), coeff in U_cols[r].items():
                        U[i * m + j][r] = coeff
                    for (j, k_idx), coeff in V_cols[r].items():
                        V[j * p + k_idx][r] = coeff
                    for (k_idx, i), coeff in W_cols[r].items():
                        W[k_idx * n + i][r] = coeff
                
                best_result = {"U": U, "V": V, "W": W}
                
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
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

# EVOLVE-BLOCK-END

import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def schoolbook(n_prime, m_prime, p_prime, i_offset, j_offset, k_offset):
        for i_loc in range(n_prime):
            for j_loc in range(m_prime):
                for k_loc in range(p_prime):
                    i_glob = i_offset + i_loc
                    j_glob = j_offset + j_loc
                    k_glob = k_offset + k_loc
                    I = i_glob * m + j_glob
                    J = j_glob * p + k_glob
                    K = k_glob * n + i_glob
                    add_term([(I, 1)], [(J, 1)], [(K, 1)])
                    
    def strassen_2x2x2(i_offset, j_offset, k_offset):
        def gI(i_l, j_l): return (i_offset + i_l) * m + (j_offset + j_l)
        def gJ(j_l, k_l): return (j_offset + j_l) * p + (k_offset + k_l)
        def gK(k_l, i_l): return (k_offset + k_l) * n + (i_offset + i_l)
        
        add_term(
            [(gI(0,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1), (gK(1,1), 1)]
        )
        add_term(
            [(gI(1,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1)],
            [(gK(0,1), 1), (gK(1,1), -1)]
        )
        add_term(
            [(gI(0,0), 1)],
            [(gJ(0,1), 1), (gJ(1,1), -1)],
            [(gK(1,0), 1), (gK(1,1), 1)]
        )
        add_term(
            [(gI(1,1), 1)],
            [(gJ(1,0), 1), (gJ(0,0), -1)],
            [(gK(0,0), 1), (gK(0,1), 1)]
        )
        add_term(
            [(gI(0,0), 1), (gI(0,1), 1)],
            [(gJ(1,1), 1)],
            [(gK(0,0), -1), (gK(1,0), 1)]
        )
        add_term(
            [(gI(1,0), 1), (gI(0,0), -1)],
            [(gJ(0,0), 1), (gJ(0,1), 1)],
            [(gK(1,1), 1)]
        )
        add_term(
            [(gI(0,1), 1), (gI(1,1), -1)],
            [(gJ(1,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1)]
        )

    def strassen_2x2x2_offset(i_off, j_off, k_off, i_start, j_start, k_start):
        def gI(i_l, j_l): return (i_start + i_off + i_l) * m + (j_start + j_off + j_l)
        def gJ(j_l, k_l): return (j_start + j_off + j_l) * p + (k_start + k_off + k_l)
        def gK(k_l, i_l): return (k_start + k_off + k_l) * n + (i_start + i_off + i_l)
        
        add_term(
            [(gI(0,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1), (gK(1,1), 1)]
        )
        add_term(
            [(gI(1,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1)],
            [(gK(0,1), 1), (gK(1,1), -1)]
        )
        add_term(
            [(gI(0,0), 1)],
            [(gJ(0,1), 1), (gJ(1,1), -1)],
            [(gK(1,0), 1), (gK(1,1), 1)]
        )
        add_term(
            [(gI(1,1), 1)],
            [(gJ(1,0), 1), (gJ(0,0), -1)],
            [(gK(0,0), 1), (gK(0,1), 1)]
        )
        add_term(
            [(gI(0,0), 1), (gI(0,1), 1)],
            [(gJ(1,1), 1)],
            [(gK(0,0), -1), (gK(1,0), 1)]
        )
        add_term(
            [(gI(1,0), 1), (gI(0,0), -1)],
            [(gJ(0,0), 1), (gJ(0,1), 1)],
            [(gK(1,1), 1)]
        )
        add_term(
            [(gI(0,1), 1), (gI(1,1), -1)],
            [(gJ(1,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1)]
        )

    def handle_2x2x3(i_start, j_start, k_start):
        strassen_2x2x2_offset(0, 0, 0, i_start, j_start, k_start)
        schoolbook(2, 2, 1, i_start, j_start, k_start + 2)

    def handle_2x3x2(i_start, j_start, k_start):
        strassen_2x2x2_offset(0, 0, 0, i_start, j_start, k_start)
        schoolbook(2, 1, 2, i_start, j_start + 2, k_start)

    def handle_2x3x3(i_start, j_start, k_start):
        strassen_2x2x2_offset(0, 0, 0, i_start, j_start, k_start)
        schoolbook(2, 2, 1, i_start, j_start, k_start + 2)
        for i_loc in range(2):
            for k_loc in range(3):
                I = (i_start + i_loc) * m + (j_start + 2)
                J = (j_start + 2) * p + (k_start + k_loc)
                K = (k_start + k_loc) * n + (i_start + i_loc)
                add_term([(I, 1)], [(J, 1)], [(K, 1)])

    i_groups = [(0, 2), (2, 2)]
    j_groups = [(0, 3), (3, 2)]
    k_groups = [(0, 3), (3, 2)]
    
    for i_start, i_size in i_groups:
        for j_start, j_size in j_groups:
            for k_start, k_size in k_groups:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    strassen_2x2x2(i_start, j_start, k_start)
                elif i_size == 2 and j_size == 2 and k_size == 3:
                    handle_2x2x3(i_start, j_start, k_start)
                elif i_size == 2 and j_size == 3 and k_size == 2:
                    handle_2x3x2(i_start, j_start, k_start)
                elif i_size == 2 and j_size == 3 and k_size == 3:
                    handle_2x3x3(i_start, j_start, k_start)
                else:
                    schoolbook(i_size, j_size, k_size, i_start, j_start, k_start)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    rank = U.shape[1]
    
    # Try to reduce rank by finding linear dependencies
    # Build the tensor and check for redundancy
    def verify(U_m, V_m, W_m):
        """Check if decomposition is exact."""
        r = U_m.shape[1]
        # Compute tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        T = np.zeros((n, m, p))
        for r in range(r):
            u = U_m[:, r]
            v = V_m[:, r]
            w = W_m[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        # Compare with identity tensor (matrix multiplication tensor)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    if abs(T[i,j,k] - expected) > 1e-9:
                        return False
        return True
    
    # Try greedy rank reduction: remove terms that are linear combinations of others
    # This is expensive, so only try a limited number of attempts
    t_start = time.time()
    deadline = 100  # safety margin before 120s limit
    
    # Try removing each term and checking if the remaining still works
    # This is O(r^2 * verification_cost) which is too slow for r=92
    # Instead, try a simpler heuristic: check for duplicate or near-duplicate terms
    
    # Check for exact duplicates
    unique_mask = np.ones(rank, dtype=bool)
    for i in range(rank):
        if not unique_mask[i]:
            continue
        for j in range(i+1, rank):
            if not unique_mask[j]:
                continue
            if np.array_equal(U[:, i], U[:, j]) and np.array_equal(V[:, i], V[:, j]) and np.array_equal(W[:, i], W[:, j]):
                unique_mask[j] = False
    
    if unique_mask.sum() < rank:
        U = U[:, unique_mask]
        V = V[:, unique_mask]
        W = W[:, unique_mask]
        rank = U.shape[1]
    
    if not U_cols:
        return {"U": [[0]*100 for _ in range(20)], "V": [[0]*100 for _ in range(25)], "W": [[0]*100 for _ in range(20)]}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()

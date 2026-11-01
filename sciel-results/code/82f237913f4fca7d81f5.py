"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS

    def verify_decomposition(U, V, W, rank):
        """Verify that the tensor decomposition exactly represents matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build the tensor T[i,j,k] = delta (1 if i,j,k form a valid multiplication term)
        # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # Actually, the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
        # The decomposition must satisfy: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Compute the decomposed tensor
        # T_decomposed[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For efficiency, compute in chunks
        T_decomposed = np.zeros((n, m, p))
        
        for r in range(rank):
            u_vec = U_arr[:, r]
            v_vec = V_arr[:, r]
            w_vec = W_arr[:, r]
            
            # u_vec has shape (n*m,): index i*m+j
            # v_vec has shape (m*p,): index j*p+k
            # w_vec has shape (p*n,): index k*n+i
            
            # For each (i,j,k): contribution = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
            # This is an outer-product-like operation
            
            # Reshape u_vec to (n, m)
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            
            # T_decomposed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # This is a 3-way contraction
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_decomposed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        # The target tensor is all ones (since A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1)
        T_target = np.ones((n, m, p))
        
        return np.allclose(T_decomposed, T_target, atol=1e-10)

    def make_strassen_block(j1, j2, k1, k2, U, V, W, col):
        """Add Strassen rank-7 terms, return new col."""
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        steps = [
            # m1 = (a00+a11)(b00+b11)
            ([(u_idx(0, j1), 1), (u_idx(1, j2), 1)],
             [(v_idx(j1, k1), 1), (v_idx(j2, k2), 1)],
             [(w_idx(k1, 0), 1), (w_idx(k2, 1), 1)]),
            # m2 = (a10+a11)*b00
            ([(u_idx(1, j1), 1), (u_idx(1, j2), 1)],
             [(v_idx(j1, k1), 1)],
             [(w_idx(k1, 1), 1), (w_idx(k2, 1), -1)]),
            # m3 = a00*(b01-b11)
            ([(u_idx(0, j1), 1)],
             [(v_idx(j1, k2), 1), (v_idx(j2, k2), -1)],
             [(w_idx(k2, 0), 1), (w_idx(k2, 1), 1)]),
            # m4 = a11*(b10-b00)
            ([(u_idx(1, j2), 1)],
             [(v_idx(j2, k1), 1), (v_idx(j1, k1), -1)],
             [(w_idx(k1, 0), 1), (w_idx(k1, 1), 1)]),
            # m5 = (a00+a01)*b11
            ([(u_idx(0, j1), 1), (u_idx(0, j2), 1)],
             [(v_idx(j2, k2), 1)],
             [(w_idx(k1, 0), -1), (w_idx(k2, 0), 1)]),
            # m6 = (a10-a00)*(b00+b01)
            ([(u_idx(1, j1), 1), (u_idx(0, j1), -1)],
             [(v_idx(j1, k1), 1), (v_idx(j1, k2), 1)],
             [(w_idx(k2, 1), 1)]),
            # m7 = (a01-a11)*(b10+b11)
            ([(u_idx(0, j2), 1), (u_idx(1, j2), -1)],
             [(v_idx(j2, k1), 1), (v_idx(j2, k2), 1)],
             [(w_idx(k1, 0), 1)]),
        ]
        
        for u_pairs, v_pairs, w_pairs in steps:
            for idx, val in u_pairs:
                U[idx][col] = val
            for idx, val in v_pairs:
                V[idx][col] = val
            for idx, val in w_pairs:
                W[idx][col] = val
            col += 1
        return col

    def make_schoolbook_2x1x2(j, k1, k2, U, V, W, col):
        """Add schoolbook rank-4 terms for A[:, j] * B[j, k1:k2+1]."""
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        for i in range(n):
            for k in [k1, k2]:
                U[u_idx(i, j)][col] = 1
                V[v_idx(j, k)][col] = 1
                W[w_idx(k, i)][col] = 1
                col += 1
        return col

    def construct_decomposition(j_groups, k_groups):
        """
        j_groups: list of tuples, e.g., [(0,1), (2,3), (4,)]
        k_groups: list of tuples, e.g., [(0,1), (2,3), (4,5)]
        Returns (U, V, W, rank) or None if invalid.
        """
        # Calculate rank
        rank = 0
        for jg in j_groups:
            for kg in k_groups:
                if len(jg) == 2 and len(kg) == 2:
                    rank += 7
                elif len(jg) == 1 and len(kg) == 2:
                    rank += 4
                elif len(jg) == 2 and len(kg) == 1:
                    rank += 4
                elif len(jg) == 1 and len(kg) == 1:
                    rank += 1
                else:
                    return None
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for jg in j_groups:
            for kg in k_groups:
                if len(jg) == 2 and len(kg) == 2:
                    col = make_strassen_block(jg[0], jg[1], kg[0], kg[1], U, V, W, col)
                elif len(jg) == 1 and len(kg) == 2:
                    col = make_schoolbook_2x1x2(jg[0], kg[0], kg[1], U, V, W, col)
                elif len(jg) == 2 and len(kg) == 1:
                    # Schoolbook 2x2x1: rank 4
                    j1, j2 = jg
                    k = kg[0]
                    for i in range(n):
                        for j in [j1, j2]:
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
                            col += 1
                elif len(jg) == 1 and len(kg) == 1:
                    j, k = jg[0], kg[0]
                    for i in range(n):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
        
        if col != rank:
            return None
        
        return U, V, W, rank

    # Current best: rank 54 with j_groups=[(0,1),(2,3),(4,)], k_groups=[(0,1),(2,3),(4,5)]
    best_result = None
    best_rank = float('inf')
    
    # Try the known good decomposition first
    j_groups = [(0, 1), (2, 3), (4,)]
    k_groups = [(0, 1), (2, 3), (4, 5)]
    result = construct_decomposition(j_groups, k_groups)
    if result:
        U, V, W, rank = result
        if verify_decomposition(U, V, W, rank):
            best_result = (U, V, W, rank)
            best_rank = rank
    
    # Try alternative groupings
    # All possible ways to partition {0,1,2,3,4} into groups of size 1 or 2
    def get_partitions(elements):
        """Generate all partitions of elements into groups of size 1 or 2."""
        if not elements:
            yield []
            return
        first = elements[0]
        rest = elements[1:]
        
        # Put first alone
        for partition in get_partitions(rest):
            yield [(first,)] + partition
        
        # Put first with another element
        for i in range(len(rest)):
            pair = (first, rest[i])
            remaining = rest[:i] + rest[i+1:]
            for partition in get_partitions(remaining):
                yield [pair] + partition
    
    j_elements = [0, 1, 2, 3, 4]
    k_elements = [0, 1, 2, 3, 4, 5]
    
    j_partitions = list(get_partitions(j_elements))
    k_partitions = list(get_partitions(k_elements))
    
    # Limit search to avoid timeout
    max_attempts = 200
    attempts = 0
    
    for jp in j_partitions:
        for kp in k_partitions:
            if attempts >= max_attempts:
                break
            if time.time() - start_time > deadline - 5:
                break
            
            # Sort groups for consistency
            jp_sorted = tuple(sorted(tuple(sorted(g)) for g in jp))
            kp_sorted = tuple(sorted(tuple(sorted(g)) for g in kp))
            
            # Skip if same as already tried
            result = construct_decomposition(list(jp), list(kp))
            attempts += 1
            
            if result:
                U, V, W, rank = result
                if rank < best_rank:
                    if verify_decomposition(U, V, W, rank):
                        best_result = (U, V, W, rank)
                        best_rank = rank
                        if best_rank <= 50:
                            break
        
        if time.time() - start_time > deadline - 5:
            break
    
    if best_result is None:
        # Fallback to schoolbook
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
        best_result = (U, V, W, rank)
    
    U, V, W, rank = best_result
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

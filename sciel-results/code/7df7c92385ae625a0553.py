import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 4, 4, 7
    
    # The matrix multiplication tensor T has entries T[i, j, k, l] = 1 if i==l and j==k, else 0
    # We want to decompose: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # In tensor form: T_{i,j,k} = delta_{i,k} delta_{j,l}... wait, let me be more careful.
    # 
    # The multiplication tensor is a 3-tensor T in U ⊗ V ⊗ W where:
    # U = R^{n*m} (A flattened), V = R^{m*p} (B flattened), W = R^{n*p} (C flattened)
    # T[a, b, c] = 1 if the a-th element of A, b-th element of B, and c-th element of C
    # are related by the multiplication formula, 0 otherwise.
    #
    # Specifically, if A is flattened as A[i*m + j], B as B[j*p + k], C as C[i*p + k]
    # then C[i*p + k] = sum_j A[i*m + j] * B[j*p + k]
    # So T[i*m+j, j*p+k, i*p+k] = 1 for all valid i, j, k
    #
    # The task says W follows trace convention W[k*n + i, r]. So W is indexed by (k, i) -> k*n + i
    # where k is the column of C (0..p-1) and i is the row of C (0..n-1).
    # So C_{i,k} corresponds to W index k*n + i.
    
    # Let's build the tensor T as a 3D array of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if there exist i, j, k such that:
    #   a = i*m + j
    #   b = j*p + k
    #   c = k*n + i
    # and the product A[i,j]*B[j,k] contributes to C[i,k]
    
    T = np.zeros((n * m, m * p, p * n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Now we need to find a rank decomposition:
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    #
    # This is the tensor rank problem. We can approach it by treating it as:
    # For each pair (a, b), T[a, b, :] is a vector in R^{p*n}.
    # We want to express T as a sum of rank-1 tensors.
    #
    # One approach: flatten T into a matrix. For example, fix the first index a and
    # reshape T[a, :, :] into a (m*p) x (p*n) matrix. But that gives the rank for fixed a.
    #
    # Better: use the "matrix flattening" duality. The tensor rank is at least the rank
    # of any matrix flattening. We can try to find a decomposition by:
    # 1. Reshape T into a matrix M of size (n*m) x (m*p * p*n)
    # 2. Find the column space basis
    # 3. Express each column as a sum of basis vectors
    # 4. Factor each basis vector as an outer product
    #
    # Actually, a cleaner approach: 
    # Reshape T into a matrix of size (n*m) x (m*p * p*n). Each row corresponds to an 'a' index.
    # The rank of this matrix is an upper bound on the tensor rank.
    # 
    # But actually, the standard approach for finding tensor decompositions is:
    # - The tensor can be seen as a linear map from U to V ⊗ W (or any other flattening)
    # - The rank of this linear map gives an upper bound
    # - We can extract a basis from the image and use that to build the decomposition
    #
    # Let me think about this differently. The tensor T defines a bilinear map:
    # (A, B) -> C where C_{i,k} = sum_j A_{i,j} B_{j,k}
    #
    # We want to write this as sum_r (u_r^T A) (v_r^T B) w_r
    # where u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    #
    # This is equivalent to finding a low-rank decomposition of the tensor.
    #
    # One concrete method: 
    # 1. Consider the tensor as a matrix M of size (n*m) x (m*p * p*n)
    # 2. Find a basis for the column space of M
    # 3. For each basis vector, factor it as u ⊗ v ⊗ w if possible
    #
    # Actually, let me use a more direct approach. I'll use the fact that the multiplication
    # tensor has a known structure and try to find a minimal decomposition using linear algebra.
    #
    # Here's a concrete plan:
    # 1. Build the tensor T as a 3D array
    # 2. Reshape it into a 2D matrix by flattening the last two indices: M of size (n*m) x (m*p*p*n)
    # 3. Compute the rank of M using exact arithmetic (Bareiss algorithm)
    # 4. Find a basis for the row space
    # 5. Each row of T can be expressed as a linear combination of basis rows
    # 6. Factor each basis row into a product of vectors
    #
    # Wait, that's not quite right either. Let me think more carefully.
    #
    # The tensor T in U ⊗ V ⊗ W can be flattened in three ways:
    # - T_U: (n*m) x (m*p * p*n) - flatten V and W
    # - T_V: (m*p) x (n*m * p*n) - flatten U and W  
    # - T_W: (p*n) x (n*m * m*p) - flatten U and V
    #
    # The tensor rank is at least max(rank(T_U), rank(T_V), rank(T_W))
    # and at most min(rank(T_U), rank(T_V), rank(T_W)) * something
    #
    # Actually, the tensor rank is exactly the minimum number of rank-1 tensors needed.
    # A rank-1 tensor is u ⊗ v ⊗ w.
    #
    # Let me try a different approach: use the known Strassen decomposition but optimize it.
    # The current program uses 49 + 28 + 16 = 93 terms. The score is 1.2043 which means
    # the seed had rank ~ 77 (since score = seed/value, 1.2043 = seed/rank, so seed ≈ 93 * 1.2043 ≈ 112? No...)
    # Wait, score = seed/value where value is the rank. If score is 1.2043 and the program has 93 terms,
    # then seed = 93 * 1.2043 ≈ 112. That doesn't make sense.
    #
    # Actually, re-reading: "score is the rank" and "search score is seed/value, so 1.0 equals the seed program".
    # So if the seed program has rank R_seed, and this program has rank R, then score = R_seed / R.
    # Score 1.2043 means R_seed / R = 1.2043, so R = R_seed / 1.2043.
    # If R = 93, then R_seed = 93 * 1.2043 ≈ 112. That seems high for a seed.
    #
    # Hmm, maybe the seed is a simple schoolbook decomposition with 4*4*7 = 112 terms.
    # 112 / 93 = 1.2043... Yes! That matches.
    #
    # So the seed is schoolbook with 112 terms, and we have 93 terms.
    # We need to do better than 93.
    #
    # Let me try to find a better decomposition. The theoretical lower bound for 4x4x7 multiplication
    # is not well-known, but we can try to reduce the number of terms.
    #
    # One idea: use a more efficient decomposition for the 4x4x4 part.
    # The current approach uses 49 terms for 4x4x4 (recursive Strassen).
    # The best known for 4x4x4 is 49 (Strassen), but there might be better.
    # Actually, the minimum rank for 4x4x4 is known to be 49 (Strassen is optimal for 4x4x4? No, 
    # I think the minimum is actually 49 for 4x4x4, but let me check).
    #
    # Actually, I recall that the minimum rank for multiplying two 4x4 matrices is 49, 
    # achieved by Strassen. So that part is optimal.
    #
    # For the 4x4x2 part (multiplying 4x4 by 4x2), the current approach uses 28 terms
    # (4 blocks of 2x2x2 with 7 terms each). Can we do better?
    #
    # For 4x4x2, we need to compute C_{i,k} = sum_j A_{i,j} B_{j,k} for k=4,5.
    # This is equivalent to multiplying a 4x4 matrix by a 4x2 matrix.
    # The tensor rank for this is at least... let's think.
    #
    # Actually, let me try a completely different approach. Instead of using block Strassen,
    # let me try to find a minimal decomposition using linear algebra directly.
    #
    # Here's my plan:
    # 1. Build the full tensor T of shape (16, 28, 28)
    # 2. Use the "matrix flattening" approach: flatten T into a matrix of size 16 x (28*28)
    # 3. Find the rank of this matrix
    # 4. If the rank is small, we can use it to build a decomposition
    #
    # But actually, the tensor rank is not simply the matrix rank. Let me use a more systematic approach.
    #
    # I'll try to find a decomposition by solving a system of linear equations.
    # Assume the rank is r. We want to find U (16 x r), V (28 x r), W (28 x r) such that:
    # T[a, b, c] = sum_{r=0}^{r-1} U[a, r] * V[b, r] * W[c, r]
    #
    # This is a nonlinear system. But we can linearize it by fixing one of the matrices.
    #
    # Alternative: use the fact that the tensor is defined by a bilinear map.
    # The bilinear map is: (A, B) -> C where C = A @ B
    # We want to write this as a sum of r rank-1 bilinear maps.
    #
    # Let me try a greedy approach:
    # 1. Start with the full tensor T
    # 2. Find the best rank-1 approximation
    # 3. Subtract it from T
    # 4. Repeat until T is zero
    #
    # But finding the best rank-1 approximation is NP-hard in general.
    #
    # Let me try a simpler approach: use the known structure of the multiplication tensor.
    # The tensor T has the property that T[a, b, c] is nonzero only when the indices are consistent.
    # Specifically, if a = i*m + j, b = j'*p + k, c = k'*n + i', then T[a, b, c] = 1 if i=i', j=j', k=k'.
    #
    # So T is a "permutation-like" tensor with 4*4*7 = 112 nonzero entries.
    #
    # The minimum rank of such a tensor is at least the maximum of the ranks of its matrix flattening.
    #
    # Let me compute the rank of the matrix flattening T_U (16 x 784):
    # Each row corresponds to an (i, j) pair. The row has 1s at positions corresponding to
    # (j, k) for all k, and (k, i) for all k.
    #
    # Actually, let me just compute it numerically and see what rank we get.
    
    # Build T as a numpy array
    T_arr = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_arr[a, b, c] = 1
    
    # Flatten to matrix: (n*m) x (m*p * p*n)
    M = T_arr.reshape(n * m, -1)
    
    # Compute rank using SVD (numerical)
    try:
        U_svd, S_svd, Vt_svd = np.linalg.svd(M, full_matrices=False)
        rank_num = np.sum(S_svd > 1e-10)
    except:
        rank_num = M.shape[1]
    
    # The tensor rank is at least rank_num
    # Let's see what rank_num is
    
    # For now, let's use the Strassen-based decomposition but try to optimize it.
    # Current: 49 (4x4x4) + 28 (4x4x2) + 16 (4x4x1) = 93
    
    # Let me try a different decomposition for the tail parts.
    # For 4x4x2: instead of 4 blocks of 2x2x2 (28 terms), can we do better?
    # The tensor for 4x4x2 has 4*4*2 = 32 nonzero entries.
    # The matrix flattening rank is at most 16 (since there are 16 rows).
    # Actually, let me compute it.
    
    T_tail = np.zeros((n * m, m * 2, 2 * n))
    for i in range(n):
        for j in range(m):
            for k in range(2):
                a = i * m + j
                b = j * 2 + k
                c = k * n + i
                T_tail[a, b, c] = 1
    
    M_tail = T_tail.reshape(n * m, -1)
    try:
        U_svd_t, S_svd_t, Vt_svd_t = np.linalg.svd(M_tail, full_matrices=False)
        rank_tail = np.sum(S_svd_t > 1e-10)
    except:
        rank_tail = M_tail.shape[1]
    
    # For 4x4x1:
    T_last = np.zeros((n * m, m * 1, 1 * n))
    for i in range(n):
        for j in range(m):
            a = i * m + j
            b = j * 1 + 0
            c = 0 * n + i
            T_last[a, b, c] = 1
    
    M_last = T_last.reshape(n * m, -1)
    try:
        U_svd_l, S_svd_l, Vt_svd_l = np.linalg.svd(M_last, full_matrices=False)
        rank_last = np.sum(S_svd_l > 1e-10)
    except:
        rank_last = M_last.shape[1]
    
    # Let me try to find a better decomposition using a direct linear algebra approach.
    # I'll try to find a decomposition of the full tensor with rank as low as possible.
    #
    # Approach: use the "rank-1 extraction" method.
    # 1. Start with T
    # 2. Find a rank-1 tensor u ⊗ v ⊗ w that best approximates T
    # 3. Subtract it
    # 4. Repeat
    #
    # For step 2, I can use the following trick:
    # Fix u and v, then w is determined by w[c] = T[a, b, c] / (u[a] * v[b]) for nonzero entries.
    # But this is overconstrained.
    #
    # Alternative: use the fact that the tensor is sparse and has a specific structure.
    # Let me try to find a decomposition by solving a linear system.
    #
    # Assume rank r. We want U (16 x r), V (28 x r), W (28 x r).
    # The constraint is: for all (a, b, c), T[a, b, c] = sum_{r} U[a, r] V[b, r] W[c, r]
    #
    # This is equivalent to: for each r, the tensor U[:, r] ⊗ V[:, r] ⊗ W[:, r]
    # and their sum equals T.
    #
    # Let me try a different approach: use the known Strassen decomposition for the 4x4x4 part
    # and try to optimize the tail parts.
    
    # Actually, let me just try to reduce the number of terms by combining some of them.
    # The current decomposition has 93 terms. Let me see if I can do better.
    
    # Let me try a decomposition where I treat the whole 4x4x7 as a single tensor
    # and use a greedy rank-1 extraction.
    
    T_work = T_arr.copy()
    terms = []
    
    def find_rank1(T_in):
        """Find a rank-1 approximation of T_in."""
        # Flatten to matrix (n*m) x (m*p * p*n)
        M_in = T_in.reshape(n * m, -1)
        # Get the leading singular vector
        try:
            U_s, S_s, Vt_s = np.linalg.svd(M_in, full_matrices=False)
            if len(S_s) == 0 or S_s[0] < 1e-10:
                return None
            # The leading singular vector gives a matrix approximation
            # But we need a tensor approximation
            # For now, let's just return the leading singular vectors
            u_vec = U_s[:, 0]
            v_w_mat = (S_s[0] * Vt_s[0]).reshape(m * p, p * n)
            # Now factor v_w_mat as v ⊗ w
            U_vw, S_vw, Vt_vw = np.linalg.svd(v_w_mat, full_matrices=False)
            if len(S_vw) == 0 or S_vw[0] < 1e-10:
                return None
            v_vec = U_vw[:, 0] * np.sqrt(S_vw[0])
            w_vec = Vt_vw[0] * np.sqrt(S_vw[0])
            return u_vec, v_vec, w_vec
        except:
            return None
    
    # Try greedy extraction
    max_terms = 100
    for _ in range(max_terms):
        if time.time() > deadline - 10:
            break
        # Check if T_work is close to zero
        if np.max(np.abs(T_work)) < 1e-10:
            break
        r1 = find_rank1(T_work)
        if r1 is None:
            break
        u, v, w = r1
        # Round to nearest integers/half-integers
        u_r = np.round(u * 2) / 2
        v_r = np.round(v * 2) / 2
        w_r = np.round(w * 2) / 2
        # Subtract
        T_work -= np.einsum('a,b,c->abc', u_r, v_r, w_r)
        terms.append((u_r, v_r, w_r))
        # Check if we're making progress
        if len(terms) > 93:
            break
    
    # If the greedy approach didn't work well, fall back to Strassen
    if len(terms) > 93 or np.max(np.abs(T_work)) > 1e-8:
        terms = []
        
        def add_term(u, v, w):
            terms.append((u, v, w))
        
        def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
            u = [0] * (n * m)
            u[i_off * m + j_off] = a_scale
            u[(i_off+1) * m + (j_off+1)] = a_scale
            v = [0] * (m * p)
            v[j_off * p + k_off] = b_scale
            v[(j_off+1) * p + (k_off+1)] = b_scale
            w = [0] * (p * n)
            w[k_off * n + i_off] = c_scale
            w[(k_off+1) * n + (i_off+1)] = c_scale
            add_term(u, v, w)
            
            u = [0] * (n * m)
            u[(i_off+1) * m + j_off] = a_scale
            u[(i_off+1) * m + (j_off+1)] = a_scale
            v = [0] * (m * p)
            v[j_off * p + k_off] = b_scale
            w = [0] * (p * n)
            w[k_off * n + (i_off+1)] = c_scale
            w[(k_off+1) * n + (i_off+1)] = -c_scale
            add_term(u, v, w)
            
            u = [0] * (n * m)
            u[i_off * m + j_off] = a_scale
            v = [0] * (m * p)
            v[j_off * p + (k_off+1)] = b_scale
            v[(j_off+1) * p + (k_off+1)] = -b_scale
            w = [0] * (p * n)
            w[(k_off+1) * n + i_off] = c_scale
            w[(k_off+1) * n + (i_off+1)] = c_scale
            add_term(u, v, w)
            
            u = [0] * (n * m)
            u[(i_off+1) * m + (j_off+1)] = a_scale
            v = [0] * (m * p)
            v[(j_off+1) * p + k_off] = b_scale
            v[j_off * p + k_off] = -b_scale
            w = [0] * (p * n)
            w[k_off * n + i_off] = c_scale
            w[k_off * n + (i_off+1)] = c_scale
            add_term(u, v, w)
            
            u = [0] * (n * m)
            u[i_off * m + j_off] = a_scale
            u[i_off * m + (j_off+1)] = a_scale
            v = [0] * (m * p)
            v[(j_off+1) * p + (k_off+1)] = b_scale
            w = [0] * (p * n)
            w[(k_off+1) * n + i_off] = c_scale
            w[k_off * n + i_off] = -c_scale
            add_term(u, v, w)
            
            u = [0] * (n * m)
            u[(i_off+1) * m + j_off] = a_scale
            u[i_off * m + j_off] = -a_scale
            v = [0] * (m * p)
            v[j_off * p + k_off] = b_scale
            v[j_off * p + (k_off+1)] = b_scale
            w = [0] * (p * n)
            w[(k_off+1) * n + (i_off+1)] = c_scale
            add_term(u, v, w)
            
            u = [0] * (n * m)
            u[i_off * m + (j_off+1)] = a_scale
            u[(i_off+1) * m + (j_off+1)] = -a_scale
            v = [0] * (m * p)
            v[(j_off+1) * p + k_off] = b_scale
            v[(j_off+1) * p + (k_off+1)] = b_scale
            w = [0] * (p * n)
            w[k_off * n + i_off] = c_scale
            add_term(u, v, w)
        
        def strassen_4x4_head():
            outer_terms = [
                ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
                ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
                ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
                ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
                ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
                ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
                ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
            ]
            inner_terms = [
                ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
                ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
                ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
                ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
                ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
                ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
                ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
            ]
            
            def get_block_indices(block_idx, local_idx, dim_a, dim_b, offset_k=0):
                br = block_idx // 2
                bc = block_idx % 2
                lr = local_idx // 2
                lc = local_idx % 2
                row = br * 2 + lr
                col = bc * 2 + lc + offset_k
                if dim_a == 'A':
                    return row * m + col
                elif dim_a == 'B':
                    return row * p + col
                elif dim_a == 'C':
                    return col * n + row
            
            for outer_a, outer_b, outer_c in outer_terms:
                for inner_x, inner_y, inner_z in inner_terms:
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    for x_local, x_coeff in inner_x:
                        for a_block, a_coeff in outer_a:
                            idx = get_block_indices(a_block, x_local, 'A', None)
                            u[idx] += a_coeff * x_coeff
                    for y_local, y_coeff in inner_y:
                        for b_block, b_coeff in outer_b:
                            idx = get_block_indices(b_block, y_local, 'B', None, offset_k=0)
                            v[idx] += b_coeff * y_coeff
                    for z_local, z_coeff in inner_z:
                        for c_block, c_coeff in outer_c:
                            idx = get_block_indices(c_block, z_local, 'C', None, offset_k=0)
                            w[idx] += c_coeff * z_coeff
                    add_term(u, v, w)
        
        def strassen_4x4x2_tail():
            for i_b in range(2):
                for j_b in range(2):
                    i_off = i_b * 2
                    j_off = j_b * 2
                    k_off = 4
                    strassen_2x2_block(i_off, j_off, k_off)
        
        def schoolbook_4x4x1_tail():
            for i in range(n):
                for j in range(m):
                    u = [0] * (n * m)
                    u[i * m + j] = 1
                    v = [0] * (m * p)
                    v[j * p + 6] = 1
                    w = [0] * (p * n)
                    w[6 * n + i] = 1
                    add_term(u, v, w)
        
        strassen_4x4_head()
        strassen_4x4x2_tail()
        schoolbook_4x4x1_tail()
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}

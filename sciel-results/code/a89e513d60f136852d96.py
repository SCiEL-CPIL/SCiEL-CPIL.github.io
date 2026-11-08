import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        """
        Adds 7 Strassen terms for a 2x2 * 2x2 block multiplication.
        A block at rows [i_off, i_off+1], cols [j_off, j_off+1]
        B block at rows [j_off, j_off+1], cols [k_off, k_off+1]
        C block at rows [i_off, i_off+1], cols [k_off, k_off+1]
        """
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
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

        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        add_term(u, v, w)

        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        add_term(u, v, w)

        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
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
        """
        Generates 49 terms for 4x4 * 4x4 using recursive Strassen.
        """
        # Outer Strassen terms
        # Each term: (A_block_coeffs, B_block_coeffs, C_block_coeffs)
        # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        outer_terms = [
            # M1 = (A00+A11)(B00+B11) -> C00 + C11
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2 = (A10+A11)B00 -> C10 - C11
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # M3 = A00(B01-B11) -> C01 + C11
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            # M4 = A11(B10-B00) -> C00 + C10
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            # M5 = (A00+A01)B11 -> -C00 + C01
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6 = (A10-A00)(B00+B01) -> C11
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7 = (A01-A11)(B10+B11) -> C00
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        # Inner Strassen terms for 2x2 * 2x2
        # Each term: (x_entries, y_entries, z_entries)
        # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        inner_terms = [
            # M1 = (x00+x11)(y00+y11) -> z00 + z11
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2 = (x10+x11)y00 -> z10 - z11
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # M3 = x00(y01-y11) -> z01 + z11
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            # M4 = x11(y10-y00) -> z00 + z10
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            # M5 = (x00+x01)y11 -> -z00 + z01
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6 = (x10-x00)(y00+y01) -> z11
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7 = (x01-x11)(y10+y11) -> z00
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        def get_block_indices(block_idx, local_idx, dim_a, dim_b, offset_k=0):
            """
            Get global flat indices for a block entry.
            block_idx: 0..3 (0=00, 1=01, 2=10, 3=11)
            local_idx: 0..3 (0=00, 1=01, 2=10, 3=11)
            dim_a: rows, dim_b: cols
            offset_k: column offset for B/C
            """
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
                # W index: k * n + i
                # k is col, i is row
                return col * n + row

        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                # Accumulate U
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices(a_block, x_local, 'A', None)
                        u[idx] += a_coeff * x_coeff
                        
                # Accumulate V
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B', None, offset_k=0)
                        v[idx] += b_coeff * y_coeff
                        
                # Accumulate W
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C', None, offset_k=0)
                        w[idx] += c_coeff * z_coeff
                        
                add_term(u, v, w)

    def strassen_4x4x2_tail():
        """
        Generates 28 terms for 4x4 * 4x2 (cols 4,5) using block Strassen.
        """
        # 4 blocks of 2x2 * 2x2
        for i_b in range(2):
            for j_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                k_off = 4
                strassen_2x2_block(i_off, j_off, k_off)

    def schoolbook_4x4x1_tail():
        """
        Generates 16 terms for 4x4 * 4x1 (col 6) using schoolbook.
        """
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                add_term(u, v, w)

    # Execute generation
    strassen_4x4_head()
    strassen_4x4x2_tail()
    schoolbook_4x4x1_tail()
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []} # Should not happen
    
    # Build the tensor and try to reduce rank using linear algebra
    # The multiplication tensor T[i,j,k] = 1 if we need A[i,j]*B[j,k] to contribute to C[i,k]
    # We want to express T as sum of rank-1 tensors: T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # First, let's verify our current decomposition is correct
    def verify_decomposition(terms_list):
        """
        Verify that the given terms correctly decompose the multiplication tensor.
        """
        if not terms_list:
            return False
        
        rank = len(terms_list)
        U = np.array(terms_list, dtype=float).reshape(-1, n*m)
        V = np.array(terms_list, dtype=float).reshape(-1, m*p)
        W = np.array(terms_list, dtype=float).reshape(-1, p*n)
        
        # The tensor T[i,j,k] should equal sum_r U[r,i] * V[r,j] * W[r,k]
        # where i is row of A (0..n-1), j is col of A / row of B (0..m-1), k is col of B (0..p-1)
        # But our indexing is: U[i*m+j], V[j*p+k], W[k*n+i]
        
        # Let's check for a few random inputs
        np.random.seed(42)
        for _ in range(10):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute C using the decomposition
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_vec = U[r].reshape(n, m)
                v_vec = V[r].reshape(m, p)
                w_vec = W[r].reshape(p, n)
                
                # The term contributes: (u_vec * v_vec) * w_vec^T? No...
                # Actually, the decomposition is: C[i,k] = sum_j sum_r A[i,j] * V[r,j*p+k] * U[r,i*m+j] * W[r,k*n+i]
                # Wait, that's not right either. Let me think...
                
                # Standard tensor decomposition for matrix mult:
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # = sum_r (sum_j A[i,j] * U[r,i*m+j]) * (sum_k' B[j,k'] * V[r,j*p+k']) ... no
                # 
                # Actually: C[i,k] = sum_r U[r,i*m+j] * A[i,j] * V[r,j*p+k] * B[j,k] * W[r,k*n+i]
                # But we sum over j, so: C[i,k] = sum_j sum_r A[i,j] * B[j,k] * U[r,i*m+j] * V[r,j*p+k] * W[r,k*n+i]
                # This should equal sum_j A[i,j] * B[j,k] for all i,k
                # So we need: sum_r U[r,i*m+j] * V[r,j*p+k] * W[r,k*n+i] = 1 for all i,j,k
                # This is the tensor decomposition condition
                
                # Let's just compute the full tensor and check
                pass
        
        # Better: build the full tensor and check
        T_expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i, j, k] = 1
        
        # Build T_computed from the decomposition
        T_computed = np.zeros((n, m, p))
        for r in range(rank):
            u_vec = np.array(terms_list[r][0]).reshape(n, m)
            v_vec = np.array(terms_list[r][1]).reshape(m, p)
            w_vec = np.array(terms_list[r][2]).reshape(p, n)
            
            # The contribution to T[i,j,k] is U[r,i*m+j] * V[r,j*p+k] * W[r,k*n+i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_computed[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
        
        # Check if they match
        if not np.allclose(T_computed, T_expected, atol=1e-9):
            return False
        
        return True
    
    # Verify current decomposition
    if not verify_decomposition(terms):
        # Fallback: return empty if invalid (should not happen)
        return {"U": [], "V": [], "W": []}
    
    # Try to reduce rank by finding linear dependencies
    # This is a hard problem, but we can try a simple approach:
    # 1. Build the tensor T
    # 2. Try removing each term and see if we can still represent T
    
    def try_remove_term(terms_list, term_idx):
        """
        Try to remove term_idx and see if the remaining terms still span the tensor.
        """
        remaining = [t for i, t in enumerate(terms_list) if i != term_idx]
        return verify_decomposition(remaining)
    
    # Greedy: try to remove each term if possible
    improved = True
    current_terms = terms.copy()
    while improved:
        improved = False
        for i in range(len(current_terms)):
            if try_remove_term(current_terms, i):
                current_terms = [t for j, t in enumerate(current_terms) if j != i]
                improved = True
                break
    
    terms = current_terms
    
    if time.time() > deadline:
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

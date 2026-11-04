import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    Nm = n * m  # 20
    Mp = m * p  # 25
    Pn = p * n  # 20
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Target indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    target_set = set(target_indices)

    # Schoolbook decomposition as fallback (Rank 100)
    rank_sb = n * m * p
    U_sb = np.zeros((Nm, rank_sb))
    V_sb = np.zeros((Mp, rank_sb))
    W_sb = np.zeros((Pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1.0
                V_sb[b, col] = 1.0
                W_sb[c, col] = 1.0
                col += 1

    def verify(U_mat, V_mat, W_mat, tol=1e-6, full_check=False):
        R = U_mat.shape[1]
        max_err = 0.0
        for (a, b, c) in target_indices:
            val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
            if max_err > tol:
                return False, max_err
        if full_check:
            for a in range(Nm):
                for b in range(Mp):
                    for c in range(Pn):
                        if (a, b, c) in target_set:
                            continue
                        val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
                        err = abs(val)
                        if err > max_err:
                            max_err = err
                        if max_err > tol:
                            return False, max_err
        else:
            np.random.seed(12345)
            for _ in range(500):
                a = np.random.randint(Nm)
                b = np.random.randint(Mp)
                c = np.random.randint(Pn)
                if (a, b, c) in target_set:
                    continue
                val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
                err = abs(val)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False, max_err
        return max_err < tol, max_err

    # Block Decomposition Strategy
    # Split n=4 into 2+2, m=5 into 2+3, p=5 into 2+3
    # A is 4x5, B is 5x5, C is 4x5
    # A = [[A11(2x2), A12(2x3)], [A21(2x2), A22(2x3)]]
    # B = [[B11(2x2), B12(2x3)], [B21(3x2), B22(3x3)]]
    # C = [[C11(2x2), C12(2x3)], [C21(2x2), C22(2x3)]]
    
    # We will accumulate the decomposition terms.
    # Each term is a set of (a_idx, b_idx, c_idx, coeff)
    # But it's easier to just build U, V, W directly.
    
    # Max possible rank estimate: 
    # 2x2x2 (Strassen): 7
    # 2x3x2 (Schoolbook): 12
    # 2x2x3 (Schoolbook): 12
    # 2x3x3 (Schoolbook): 18
    # Total: 7+12+12+18+7+12+12+18 = 98
    
    max_rank = 100
    U = np.zeros((Nm, max_rank))
    V = np.zeros((Mp, max_rank))
    W = np.zeros((Pn, max_rank))
    
    current_rank = 0
    
    def add_term(u_indices, v_indices, w_indices, coeff):
        """
        Add a rank-1 term to the decomposition.
        u_indices: list of (idx, coeff) for U
        v_indices: list of (idx, coeff) for V
        w_indices: list of (idx, coeff) for W
        """
        nonlocal current_rank
        if current_rank >= max_rank:
            return
        for idx, c_u in u_indices:
            U[idx, current_rank] += c_u
        for idx, c_v in v_indices:
            V[idx, current_rank] += c_v
        for idx, c_w in w_indices:
            W[idx, current_rank] += c_w
        current_rank += 1

    # Helper to get global indices
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(k, i): return k * n + i

    # Define Strassen decomposition for 2x2x2 block
    # Local indices 0,1 for rows/cols
    # A11 is at global A(0,0), A(0,1), A(1,0), A(1,1)
    # B11 is at global B(0,0), B(0,1), B(1,0), B(1,1)
    # C11 is at global C(0,0), C(0,1), C(1,0), C(1,1)
    
    def strassen_2x2x2(offset_i, offset_j, offset_k):
        """
        Adds Strassen decomposition for block at A[offset_i:offset_i+2, offset_j:offset_j+2]
        B[offset_j:offset_j+2, offset_k:offset_k+2]
        C[offset_k:offset_k+2, offset_i:offset_i+2]
        """
        # M1 = (A11+A22)(B11+B22)
        # A11: (offset_i, offset_j), A22: (offset_i+1, offset_j+1)
        # B11: (offset_j, offset_k), B22: (offset_j+1, offset_k+1)
        # C11: (offset_k, offset_i), C22: (offset_k+1, offset_i+1)
        
        # Term 1: M1
        u = [(a_idx(offset_i, offset_j), 1.0), (a_idx(offset_i+1, offset_j+1), 1.0)]
        v = [(b_idx(offset_j, offset_k), 1.0), (b_idx(offset_j+1, offset_k+1), 1.0)]
        w = [(c_idx(offset_k, offset_i), 1.0), (c_idx(offset_k+1, offset_i+1), 1.0)]
        add_term(u, v, w, 1.0)
        
        # Term 2: M2 = (A21+A22)B11
        # A21: (offset_i+1, offset_j), A22: (offset_i+1, offset_j+1)
        # B11: (offset_j, offset_k)
        # C21: (offset_k+1, offset_i), C22: (offset_k+1, offset_i+1)
        u = [(a_idx(offset_i+1, offset_j), 1.0), (a_idx(offset_i+1, offset_j+1), 1.0)]
        v = [(b_idx(offset_j, offset_k), 1.0)]
        w = [(c_idx(offset_k+1, offset_i), 1.0), (c_idx(offset_k+1, offset_i+1), 1.0)]
        add_term(u, v, w, 1.0)
        
        # Term 3: M3 = A11(B12-B22)
        # A11: (offset_i, offset_j)
        # B12: (offset_j, offset_k+1), B22: (offset_j+1, offset_k+1)
        # C12: (offset_k, offset_i+1), C22: (offset_k+1, offset_i+1)
        u = [(a_idx(offset_i, offset_j), 1.0)]
        v = [(b_idx(offset_j, offset_k+1), 1.0), (b_idx(offset_j+1, offset_k+1), -1.0)]
        w = [(c_idx(offset_k, offset_i+1), 1.0), (c_idx(offset_k+1, offset_i+1), 1.0)]
        add_term(u, v, w, 1.0)
        
        # Term 4: M4 = A22(B21-B11)
        # A22: (offset_i+1, offset_j+1)
        # B21: (offset_j+1, offset_k), B11: (offset_j, offset_k)
        # C21: (offset_k+1, offset_i), C11: (offset_k, offset_i)
        u = [(a_idx(offset_i+1, offset_j+1), 1.0)]
        v = [(b_idx(offset_j+1, offset_k), 1.0), (b_idx(offset_j, offset_k), -1.0)]
        w = [(c_idx(offset_k+1, offset_i), 1.0), (c_idx(offset_k, offset_i), 1.0)]
        add_term(u, v, w, 1.0)
        
        # Term 5: M5 = (A11+A12)B22
        # A11: (offset_i, offset_j), A12: (offset_i, offset_j+1)
        # B22: (offset_j+1, offset_k+1)
        # C11: (offset_k, offset_i), C12: (offset_k, offset_i+1)
        u = [(a_idx(offset_i, offset_j), 1.0), (a_idx(offset_i, offset_j+1), 1.0)]
        v = [(b_idx(offset_j+1, offset_k+1), 1.0)]
        w = [(c_idx(offset_k, offset_i), -1.0), (c_idx(offset_k, offset_i+1), 1.0)] # Note: C11 gets -M5, C12 gets +M5
        add_term(u, v, w, 1.0)
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        # A21: (offset_i+1, offset_j), A11: (offset_i, offset_j)
        # B11: (offset_j, offset_k), B12: (offset_j, offset_k+1)
        # C22: (offset_k+1, offset_i+1)
        u = [(a_idx(offset_i+1, offset_j), 1.0), (a_idx(offset_i, offset_j), -1.0)]
        v = [(b_idx(offset_j, offset_k), 1.0), (b_idx(offset_j, offset_k+1), 1.0)]
        w = [(c_idx(offset_k+1, offset_i+1), 1.0)]
        add_term(u, v, w, 1.0)
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        # A12: (offset_i, offset_j+1), A22: (offset_i+1, offset_j+1)
        # B21: (offset_j+1, offset_k), B22: (offset_j+1, offset_k+1)
        # C11: (offset_k, offset_i), C21: (offset_k+1, offset_i)
        u = [(a_idx(offset_i, offset_j+1), 1.0), (a_idx(offset_i+1, offset_j+1), -1.0)]
        v = [(b_idx(offset_j+1, offset_k), 1.0), (b_idx(offset_j+1, offset_k+1), 1.0)]
        w = [(c_idx(offset_k, offset_i), 1.0), (c_idx(offset_k+1, offset_i), 1.0)]
        add_term(u, v, w, 1.0)

    # Helper for Schoolbook decomposition of rxk x kxc -> rxc
    def schoolbook_block(r, k, c, offset_i, offset_j, offset_k):
        """
        A block: rows offset_i..offset_i+r-1, cols offset_j..offset_j+k-1
        B block: rows offset_j..offset_j+k-1, cols offset_k..offset_k+c-1
        C block: rows offset_k..offset_k+c-1, cols offset_i..offset_i+r-1
        """
        for ii in range(r):
            for jj in range(k):
                for kk in range(c):
                    a_i = offset_i + ii
                    a_j = offset_j + jj
                    b_j = offset_j + jj
                    b_k = offset_k + kk
                    c_k = offset_k + kk
                    c_i = offset_i + ii
                    
                    u = [(a_idx(a_i, a_j), 1.0)]
                    v = [(b_idx(b_j, b_k), 1.0)]
                    w = [(c_idx(c_k, c_i), 1.0)]
                    add_term(u, v, w, 1.0)

    # 1. C11 = A11 B11 (2x2x2) -> Strassen
    strassen_2x2x2(0, 0, 0)
    
    # 2. C11 += A12 B21 (2x3x2) -> Schoolbook
    # A12: 2x3 (rows 0-1, cols 2-4)
    # B21: 3x2 (rows 2-4, cols 0-1)
    # C11: 2x2 (rows 0-1, cols 0-1)
    schoolbook_block(2, 3, 2, 0, 2, 0)
    
    # 3. C12 = A11 B12 (2x2x3) -> Schoolbook
    # A11: 2x2 (rows 0-1, cols 0-1)
    # B12: 2x3 (rows 0-1, cols 2-4)
    # C12: 2x3 (rows 2-4, cols 0-1)
    schoolbook_block(2, 2, 3, 0, 0, 2)
    
    # 4. C12 += A12 B22 (2x3x3) -> Schoolbook
    # A12: 2x3 (rows 0-1, cols 2-4)
    # B22: 3x3 (rows 2-4, cols 2-4)
    # C12: 2x3 (rows 2-4, cols 0-1)
    schoolbook_block(2, 3, 3, 0, 2, 2)
    
    # 5. C21 = A21 B11 (2x2x2) -> Strassen
    # A21: 2x2 (rows 2-3, cols 0-1)
    # B11: 2x2 (rows 0-1, cols 0-1)
    # C21: 2x2 (rows 0-1, cols 2-3)
    strassen_2x2x2(2, 0, 0)
    
    # 6. C21 += A22 B21 (2x3x2) -> Schoolbook
    # A22: 2x3 (rows 2-3, cols 2-4)
    # B21: 3x2 (rows 2-4, cols 0-1)
    # C21: 2x2 (rows 0-1, cols 2-3)
    schoolbook_block(2, 3, 2, 2, 2, 0)
    
    # 7. C22 = A21 B12 (2x2x3) -> Schoolbook
    # A21: 2x2 (rows 2-3, cols 0-1)
    # B12: 2x3 (rows 0-1, cols 2-4)
    # C22: 2x3 (rows 2-4, cols 2-3)
    schoolbook_block(2, 2, 3, 2, 0, 2)
    
    # 8. C22 += A22 B22 (2x3x3) -> Schoolbook
    # A22: 2x3 (rows 2-3, cols 2-4)
    # B22: 3x3 (rows 2-4, cols 2-4)
    # C22: 2x3 (rows 2-4, cols 2-3)
    schoolbook_block(2, 3, 3, 2, 2, 2)

    # Trim to actual rank used
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]
    
    is_valid, err = verify(U, V, W, full_check=True)
    if is_valid:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
# EVOLVE-BLOCK-END

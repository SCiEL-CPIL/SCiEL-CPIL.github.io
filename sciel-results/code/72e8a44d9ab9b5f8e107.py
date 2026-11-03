"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    Each factor is (u_vec, v_vec, w_vec) where vectors are in row-major order.
    """
    factors = []
    
    # p1 = (A00+A11)(B00+B11)
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # p2 = (A10+A11)*B00
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # p3 = A00*(B01-B11)
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # p4 = A11*(B10-B00)
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    # p5 = (A00+A01)*B11
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    # p6 = (A10-A00)*(B00+B01)
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # p7 = (A01-A11)*(B10+B11)
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    
    return factors

def _strassen_4x4_factors():
    """
    Return the 49 rank-1 factors for 4x4*4x4 matrix multiplication using Strassen.
    Each factor is (u_vec, v_vec, w_vec) where vectors are 16-element (row-major).
    """
    s2 = _strassen_2x2_factors()
    factors = []
    
    for s in range(7):
        for r in range(7):
            u_block = s2[s][0]  # 4-element: coefficients for A blocks [00,01,10,11]
            v_block = s2[s][1]  # 4-element: coefficients for B blocks [00,01,10,11]
            w_block = s2[s][2]  # 4-element: coefficients for C blocks [00,01,10,11]
            
            u_scalar = s2[r][0]  # 4-element: coefficients within each 2x2 block
            v_scalar = s2[r][1]
            w_scalar = s2[r][2]
            
            u_vec = [0] * 16
            v_vec = [0] * 16
            w_vec = [0] * 16
            
            # U: A is 4x4, indexed by (i,j) where i=0..3, j=0..3
            # Block (bi,bj) of A is at rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    c_block = u_block[block_idx]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            scalar_idx = ei * 2 + ej
                            c_scalar = u_scalar[scalar_idx]
                            if c_scalar == 0:
                                continue
                            i = 2 * bi + ei
                            j = 2 * bj + ej
                            u_vec[i * 4 + j] += c_block * c_scalar
            
            # V: B is 4x4, indexed by (j,k) where j=0..3, k=0..3
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    c_block = v_block[block_idx]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            scalar_idx = ei * 2 + ej
                            c_scalar = v_scalar[scalar_idx]
                            if c_scalar == 0:
                                continue
                            j = 2 * bi + ei
                            k = 2 * bj + ej
                            v_vec[j * 4 + k] += c_block * c_scalar
            
            # W: C is 4x4, indexed by (i,k) where i=0..3, k=0..3
            for ci in range(2):
                for cj in range(2):
                    block_idx = ci * 2 + cj
                    c_block = w_block[block_idx]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            scalar_idx = ei * 2 + ej
                            c_scalar = w_scalar[scalar_idx]
                            if c_scalar == 0:
                                continue
                            i = 2 * ci + ei
                            k = 2 * cj + ej
                            w_vec[i * 4 + k] += c_block * c_scalar
            
            factors.append((u_vec, v_vec, w_vec))
    
    return factors

def _strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using Strassen.
    Split B into two 4x4 blocks, use Strassen (rank 49) for each.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    s4_factors = _strassen_4x4_factors()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]   # (16, 98)
    V = [[0] * rank for _ in range(m * p)]   # (32, 98)
    W = [[0] * rank for _ in range(p * n)]   # (32, 98)
    
    for t in range(2):  # two 4x4 blocks of B
        for idx in range(49):
            col = t * 49 + idx
            u_vec, v_vec, w_vec = s4_factors[idx]
            
            # U: A is 4x4, index i*m+j = i*4+j
            for i in range(4):
                for j in range(4):
                    U[i * 4 + j][col] = u_vec[i * 4 + j]
            
            # V: B is 4x8, but for block t, columns are 4*t..4*t+3
            # B index is j*p+k = j*8+k, where j=0..3, k=4*t..4*t+3
            for j in range(4):
                for k in range(4):
                    V[j * 8 + (4 * t + k)][col] = v_vec[j * 4 + k]
            
            # W: C is 4x8, index k*n+i = k*4+i, where k=4*t..4*t+3, i=0..3
            for i in range(4):
                for k in range(4):
                    W[(4 * t + k) * 4 + i][col] = w_vec[i * 4 + k]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
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

def _try_rank_reduction(base, max_removals=10, time_limit=100):
    """
    Try to reduce the rank of the base decomposition by removing linearly dependent columns.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U_base = np.array(base["U"], dtype=float)
    V_base = np.array(base["V"], dtype=float)
    W_base = np.array(base["W"], dtype=float)
    R = len(U_base[0])
    
    def compute_tensor(U, V, W):
        rank = len(U[0])
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        return np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    def can_remove_column(col_idx, U, V, W, tol=1e-10):
        """
        Check if column col_idx can be removed while maintaining exactness.
        """
        Uc = U[:, col_idx]
        Vc = V[:, col_idx]
        Wc = W[:, col_idx]
        
        C = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C[i,j,k] = Uc[i*m+j] * Vc[j*p+k] * Wc[k*n+i]
        
        other_cols = [c for c in range(len(U[0])) if c != col_idx]
        
        num_unknowns = p * n * len(other_cols)
        num_eqs = n * m * p
        
        if num_unknowns < num_eqs:
            return False, None
        
        A = np.zeros((num_eqs, num_unknowns))
        b = np.zeros(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    b[eq_idx] = C[i,j,k]
                    
                    for r_pos, r in enumerate(other_cols):
                        unk_idx = (k * n + i) * len(other_cols) + r_pos
                        A[eq_idx, unk_idx] = U[i*m+j, r] * V[j*p+k, r]
                    
                    eq_idx += 1
        
        try:
            x, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=1e-12)
            
            if len(residuals) > 0:
                if residuals[0] > tol:
                    return False, None
            else:
                residual = np.linalg.norm(A @ x - b)
                if residual > tol:
                    return False, None
            
            W_new = W.copy()
            for r_pos, r in enumerate(other_cols):
                for k in range(p):
                    for i in range(n):
                        W_new[k*n+i, r] = x[(k*n+i)*len(other_cols) + r_pos]
            
            T_new = compute_tensor(U, V, W_new)
            if np.allclose(T_new, 1.0, atol=1e-10):
                return True, W_new
            else:
                return False, None
                
        except Exception:
            return False, None
    
    current_U = U_base.copy()
    current_V = V_base.copy()
    current_W = W_base.copy()
    current_rank = R
    
    removed_cols = []
    
    for attempt in range(max_removals):
        if time.time() - start_time > time_limit:
            break
        
        found_removal = False
        for col in range(current_rank):
            if time.time() - start_time > time_limit:
                break
            
            can_rm, W_new = can_remove_column(col, current_U, current_V, current_W)
            if can_rm:
                mask = np.ones(current_rank, dtype=bool)
                mask[col] = False
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = W_new[:, mask]
                removed_cols.append(col)
                current_rank -= 1
                found_removal = True
                break
        
        if not found_removal:
            break
    
    if current_rank < R:
        U_list = current_U.tolist()
        V_list = current_V.tolist()
        W_list = current_W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # First, build the base Strassen decomposition
    base = _strassen_4x4_4x8()
    
    # Try to reduce the rank
    try:
        reduced = _try_rank_reduction(base, max_removals=10, time_limit=110)
        if reduced is not None and time.time() < deadline:
            if _verify(reduced["U"], reduced["V"], reduced["W"]):
                return reduced
    except Exception:
        pass
    
    # Fall back to Strassen
    try:
        if _verify(base["U"], base["V"], base["W"]):
            return base
    except Exception:
        pass
    
    # Final fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END

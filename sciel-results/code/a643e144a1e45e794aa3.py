import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """
    Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is a tuple (u, v, w) of length 4 vectors (flattened 2x2 matrices).
    Indexing: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    """
    terms = []
    # M1 = (a00+a11)(b00+b11)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (a10+a11)b00
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # M3 = a00(b01-b11)
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # M4 = a11(b10-b00)
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]))
    # M5 = (a00+a01)b11
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    # M6 = (a10-a00)(b00+b01)
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # M7 = (a01-a11)(b10+b11)
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return terms

def _build_49_rank_4x4x4():
    """
    Builds a rank 49 decomposition for 4x4x4 matrix multiplication
    using the Kronecker product of two Strassen 2x2x2 decompositions.
    
    The 4x4 matrices are viewed as 2x2 blocks of 2x2 submatrices.
    A = [[A00, A01], [A10, A11]]
    B = [[B00, B01], [B10, B11]]
    C = A B
    
    The tensor T_{4,4,4} = T_{2,2,2} \otimes T_{2,2,2}.
    Rank is 7 * 7 = 49.
    
    Returns U, V, W of shape (16, 49).
    """
    terms_2x2 = _strassen_2x2_terms()
    rank = 49
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((16, rank), dtype=np.float64)
    W = np.zeros((16, rank), dtype=np.float64)
    
    # Map local 2x2 index (0..3) to global 4x4 index (0..15)
    # Global index g = (block_row * 2 + local_row) * 4 + (block_col * 2 + local_col)
    # Local index l = local_row * 2 + local_col
    # Block index b = block_row * 2 + block_col
    
    def expand_vector(vec_2x2, block_indices):
        """
        Expand a 2x2 vector (length 4) to a 4x4 vector (length 16)
        based on which 2x2 blocks it applies to.
        block_indices: list of (block_idx, coeff)
        """
        out = np.zeros(16, dtype=np.float64)
        for b_idx, coeff in block_indices:
            if coeff == 0:
                continue
            br = b_idx // 2
            bc = b_idx % 2
            for l in range(4):
                val = vec_2x2[l]
                if val == 0:
                    continue
                lr = l // 2
                lc = l % 2
                gr = br * 2 + lr
                gc = bc * 2 + lc
                g_idx = gr * 4 + gc
                out[g_idx] += coeff * val
        return out

    col = 0
    for k1 in range(7):
        u1, v1, w1 = terms_2x2[k1]
        for k2 in range(7):
            u2, v2, w2 = terms_2x2[k2]
            
            # The Kronecker product term is (u1 \otimes u2, v1 \otimes v2, w1 \otimes w2)
            # This corresponds to the outer product of the block-level coefficients.
            # u_full = u1 \otimes u2 ? No.
            # In the Kronecker product of tensors T1 \otimes T2:
            # T[a1, a2, b1, b2, c1, c2] = T1[a1, b1, c1] * T2[a2, b2, c2]
            # The rank-1 term is (u1 \otimes u2) (v1 \otimes v2) (w1 \otimes w2)
            # Here u1 is a vector in R^4 (flattened 2x2 A-blocks).
            # u2 is a vector in R^4 (flattened 2x2 A-sub-blocks).
            # The combined u_full is in R^16.
            
            # We can construct u_full by iterating over the non-zero entries of u1 and u2.
            # u_full[g_idx] = sum_{b1, l1, b2, l2} u1[b1] * u2[l2] ... 
            # Actually, it's simpler:
            # u_full is the Kronecker product of u1 and u2, reshaped appropriately.
            # u1 is 4x1, u2 is 4x1. u1 \otimes u2 is 16x1.
            # Does the index mapping match?
            # u1 index b (0..3) corresponds to block (br, bc).
            # u2 index l (0..3) corresponds to local (lr, lc).
            # Global index (br*2+lr, bc*2+lc).
            # Kronecker product index: b * 4 + l.
            # b = br*2 + bc. l = lr*2 + lc.
            # Index = (br*2+bc)*4 + lr*2+lc = br*8 + bc*4 + lr*2 + lc.
            # Global index = (br*2+lr)*4 + bc*2+lc = br*8 + lr*4 + bc*2 + lc.
            # These are NOT the same order.
            # Kronecker: br, bc, lr, lc.
            # Global: br, lr, bc, lc.
            # So we need to permute the Kronecker product.
            
            u_kron = np.kron(u1, u2)
            v_kron = np.kron(v1, v2)
            w_kron = np.kron(w1, w2)
            
            # Permute indices from (br, bc, lr, lc) to (br, lr, bc, lc)
            # Reshape to (2, 2, 2, 2)
            u_reshaped = u_kron.reshape(2, 2, 2, 2)
            # Axes: 0=br, 1=bc, 2=lr, 3=lc
            # Target axes: 0=br, 1=lr, 2=bc, 3=lc
            u_permuted = np.transpose(u_reshaped, (0, 2, 1, 3))
            U[:, col] = u_permuted.flatten()
            
            v_reshaped = v_kron.reshape(2, 2, 2, 2)
            v_permuted = np.transpose(v_reshaped, (0, 2, 1, 3))
            V[:, col] = v_permuted.flatten()
            
            w_reshaped = w_kron.reshape(2, 2, 2, 2)
            w_permuted = np.transpose(w_reshaped, (0, 2, 1, 3))
            W[:, col] = w_permuted.flatten()
            
            col += 1
            
    return U, V, W

def _verify(U, V, W, T, tol=1e-9):
    r = U.shape[1]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 4, 4, 8

    # Build the full 4x4x8 multiplication tensor
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Schoolbook decomposition (rank 128)
    rank_schoolbook = n * m * p
    U = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook

    # Try 49-rank 4x4x4 block, doubled to 98 for 4x4x8
    if time.time() < deadline - 5:
        try:
            U4, V4, W4 = _build_49_rank_4x4x4()
            rank4 = U4.shape[1]

            if rank4 == 49:
                # Construct 4x4x8 decomposition
                # We split the 4x8 B matrix into two 4x4 blocks: B_left (cols 0-3) and B_right (cols 4-7)
                # Similarly C is split into C_left (cols 0-3) and C_right (cols 4-7)
                # Note: C index is k*n + i.
                # For k=0..3, c = 0..11 (wait, n=4, so c = 0..15 for k=0..3? No, c = k*4 + i. i is 0..3.
                # So for k=0, c=0..3. For k=1, c=4..7. Etc.
                # The output C is 8x4.
                # The tensor T is 16 x 32 x 32.
                # A is 16x1. B is 32x1. C is 32x1.
                
                U_full = np.zeros((n*m, 2*rank4), dtype=np.float64)
                V_full = np.zeros((m*p, 2*rank4), dtype=np.float64)
                W_full = np.zeros((p*n, 2*rank4), dtype=np.float64)

                # Block 1: Left half of B (j=0..3, k=0..3) and Left half of C (k=0..3)
                # U is the same for both blocks (A is shared)
                U_full[:, 0:rank4] = U4
                
                # V4 is 16x49, indices j*4+k_local (0..15)
                # V_full is 32x98, indices j*8+k_global (0..31)
                # For left block, k_global = k_local (0..3)
                for r in range(rank4):
                    for local_idx in range(16):
                        val = V4[local_idx, r]
                        if val != 0:
                            j = local_idx // 4
                            k_local = local_idx % 4
                            global_idx = j * 8 + k_local
                            V_full[global_idx, r] += val

                # W4 is 16x49, indices k_local*4+i (0..15)
                # W_full is 32x98, indices k_global*4+i (0..31)
                # For left block, k_global = k_local (0..3)
                for r in range(rank4):
                    for local_idx in range(16):
                        val = W4[local_idx, r]
                        if val != 0:
                            k_local = local_idx // 4
                            i = local_idx % 4
                            global_idx = k_local * 4 + i
                            W_full[global_idx, r] += val

                # Block 2: Right half of B (j=0..3, k=4..7) and Right half of C (k=4..7)
                U_full[:, rank4:2*rank4] = U4

                for r in range(rank4):
                    for local_idx in range(16):
                        val = V4[local_idx, r]
                        if val != 0:
                            j = local_idx // 4
                            k_local = local_idx % 4
                            global_idx = j * 8 + (k_local + 4)
                            V_full[global_idx, r + rank4] += val

                for r in range(rank4):
                    for local_idx in range(16):
                        val = W4[local_idx, r]
                        if val != 0:
                            k_local = local_idx // 4
                            i = local_idx % 4
                            global_idx = (k_local + 4) * 4 + i
                            W_full[global_idx, r + rank4] += val

                if _verify(U_full, V_full, W_full, T):
                    best_U, best_V, best_W = U_full, V_full, W_full
                    best_rank = 2*rank4
        except Exception as e:
            pass # Fallback to schoolbook

    # Convert to half-integer lists
    def to_half_integer(mat):
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-6):
            return (rounded / 2).tolist()
        rounded_int = np.round(mat)
        if np.allclose(mat, rounded_int, atol=1e-6):
            return rounded_int.astype(int).tolist()
        return mat.tolist()

    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END

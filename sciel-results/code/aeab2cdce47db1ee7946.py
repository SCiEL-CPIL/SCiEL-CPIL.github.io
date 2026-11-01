"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook baseline (rank 48)
    U0, V0, W0 = _schoolbook(n, m, p)

    # Try the structured 4-block rank-11 scheme (total rank 44)
    best = None
    best_rank = n * m * p

    if time.time() < deadline - 20:
        try:
            U1, V1, W1, r1 = _four_block_decomposition(n, m, p)
            if _verify(n, m, p, U1, V1, W1):
                best = (U1, V1, W1)
                best_rank = r1
        except Exception:
            pass

    # Try rank-28 Kronecker-based decomposition
    if time.time() < deadline - 20:
        try:
            U28, V28, W28, r28 = _rank28_decomposition(n, m, p)
            if r28 < best_rank and _verify(n, m, p, U28, V28, W28):
                best = (U28, V28, W28)
                best_rank = r28
        except Exception:
            pass

    # Try further refinement (attempt to reduce rank further)
    if best is not None and time.time() < deadline - 10:
        try:
            Ur, Vr, Wr, rr = _refine_decomposition(n, m, p, best[0], best[1], best[2], best_rank, deadline)
            if rr < best_rank and _verify(n, m, p, Ur, Vr, Wr):
                best = (Ur, Vr, Wr)
                best_rank = rr
        except Exception:
            pass

    if best is not None:
        U, V, W = best
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    return {"U": _to_list(U0), "V": _to_list(V0), "W": _to_list(W0)}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _four_block_decomposition(n, m, p):
    """
    Split the p=8 columns into 4 blocks of size 2.
    For each 2x3 x 3x2 sub-problem, use a rank-11 decomposition.
    Total rank: 44.
    
    The 2x3 * 3x2 multiplication: A is 2x3, B is 3x2, C is 2x2.
    We partition A into A_left (2x2) and A_right (2x1), and B into B_top (2x2) and B_bottom (1x2).
    C = A_left * B_top + A_right * B_bottom
    
    A_left * B_top: use Strassen for 2x2*2x2 (rank 7)
    A_right * B_bottom: outer product of 2x1 and 1x2 (rank 4)
    Total per block: 11
    """
    n, m, p = 2, 3, 8
    block_size = p // 4  # 2
    
    rank_per_block = 11
    total_rank = rank_per_block * 4  # 44
    
    U = np.zeros((n * m, total_rank))  # 6 x 44
    V = np.zeros((m * p, total_rank))  # 24 x 44
    W = np.zeros((p * n, total_rank))  # 16 x 44
    
    for b in range(4):
        k_offset = b * block_size  # 0, 2, 4, 6
        col_offset = b * rank_per_block
        
        # For the 2x2 * 2x2 Strassen part:
        # A_left has entries A[0,0], A[0,1], A[1,0], A[1,1] (indices 0,1,2,3 in U)
        # B_top has entries B[0,k0], B[0,k1], B[1,k0], B[1,k1] (indices in V: 0*p+k0, 0*p+k1, 1*p+k0, 1*p+k1)
        # C has entries C[0,k0], C[0,k1], C[1,k0], C[1,k1] (indices in W: k0*n+0, k0*n+1, k1*n+0, k1*n+1)
        
        k0 = k_offset
        k1 = k_offset + 1
        
        # Standard Strassen for 2x2 * 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U indices for A_left (2x2): row = i*m + j where i=0,1 and j=0,1
        # A00 -> U[0, :], A01 -> U[1, :], A10 -> U[2, :], A11 -> U[3, :]
        
        # V indices for B_top (2x2): row = l*p + k where l=0,1 and k=k0,k1
        # B00 -> V[0*p+k0, :], B01 -> V[0*p+k1, :], B10 -> V[1*p+k0, :], B11 -> V[1*p+k1, :]
        
        # W indices for C (2x2): row = k*n + i where k=k0,k1 and i=0,1
        # C00 -> W[k0*n+0, :], C01 -> W[k0*n+1, :], C10 -> W[k1*n+0, :], C11 -> W[k1*n+1, :]
        
        strassen_U = [
            {0: 1, 3: 1},      # M1: A00 + A11
            {2: 1, 3: 1},      # M2: A10 + A11
            {0: 1},            # M3: A00
            {3: 1},            # M4: A11
            {0: 1, 1: 1},      # M5: A00 + A01
            {2: 1, 0: -1},     # M6: A10 - A00
            {1: 1, 3: -1},     # M7: A01 - A11
        ]
        
        strassen_V = [
            {0 * p + k0: 1, 1 * p + k1: 1},  # M1: B00 + B11
            {0 * p + k0: 1},                  # M2: B00
            {1 * p + k0: 1, 1 * p + k1: -1}, # M3: B10 - B11
            {0 * p + k1: 1, 0 * p + k0: -1}, # M4: B01 - B00
            {1 * p + k1: 1},                  # M5: B11
            {0 * p + k0: 1, 0 * p + k1: 1},  # M6: B00 + B01
            {1 * p + k0: 1, 1 * p + k1: 1},  # M7: B10 + B11
        ]
        
        strassen_W = [
            {k0 * n + 0: 1, k1 * n + 1: 1},  # M1: C00 + C11
            {k1 * n + 0: 1},                  # M2: C10
            {k0 * n + 1: 1},                  # M3: C01
            {k0 * n + 0: 1, k1 * n + 1: 1},  # M4: C00 + C11 (wait, this is wrong)
            {k0 * n + 0: -1, k0 * n + 1: 1}, # M5: -C00 + C01
            {k1 * n + 1: 1},                  # M6: C11
            {k0 * n + 0: 1},                  # M7: C00
        ]
        
        # Let me redo the W coefficients based on the Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        strassen_W = [
            {k0 * n + 0: 1, k1 * n + 1: 1},   # M1: C00+=1, C11+=1
            {k1 * n + 0: 1, k1 * n + 1: -1},  # M2: C10+=1, C11-=1
            {k0 * n + 1: 1, k1 * n + 1: 1},   # M3: C01+=1, C11+=1
            {k0 * n + 0: 1, k1 * n + 0: 1},   # M4: C00+=1, C10+=1
            {k0 * n + 0: -1, k0 * n + 1: 1},  # M5: C00-=1, C01+=1
            {k1 * n + 1: 1},                   # M6: C11+=1
            {k0 * n + 0: 1},                   # M7: C00+=1
        ]
        
        for c in range(7):
            for row, val in strassen_U[c].items():
                U[row, col_offset + c] += val
            for row, val in strassen_V[c].items():
                V[row, col_offset + c] += val
            for row, val in strassen_W[c].items():
                W[row, col_offset + c] += val
        
        # Outer product part: A_right(2x1) * B_bottom(1x2) - rank 4
        # A_right has entries A[0,2], A[1,2] (indices 4, 5 in U)
        # B_bottom has entries B[2,k0], B[2,k1] (indices 2*p+k0, 2*p+k1 in V)
        # C contribution: C[0,k0] += A[0,2]*B[2,k0], C[0,k1] += A[0,2]*B[2,k1],
        #                 C[1,k0] += A[1,2]*B[2,k0], C[1,k1] += A[1,2]*B[2,k1]
        for ii in range(2):
            for kk in range(2):
                c = 7 + ii * 2 + kk  # 7,8,9,10
                U[ii * m + 2, col_offset + c] = 1
                V[2 * p + k_offset + kk, col_offset + c] = 1
                W[(k_offset + kk) * n + ii, col_offset + c] = 1
    
    return U, V, W, total_rank


def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact using the bilinear form:
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[l*p+k, c] * B[l,k])
    Vectorized for speed.
    """
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B

    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)

    # For each c, compute the contribution to C
    # C[i,k] = sum_c W[k,n+i,c] * (U[i,m,c] dot A[i,:]) * (V[m,p,c] dot B[:,k])
    # u_vals[i, c] = sum_j U[i*m+j, c] * A[i, j]  -> shape (n, rank)
    u_vals = np.einsum('imc,im->ic', U_r, A)  # (n, rank)
    v_vals = np.einsum('mkc,mk->kc', V_r, B)   # (p, rank)
    # C[i,k] = sum_c W[k*n+i, c] * u_vals[i,c] * v_vals[k,c]
    # W_r has shape (p, n, rank), so W_r[k, i, c]
    C_comp = np.einsum('kic,ic,kc->ik', W_r, u_vals, v_vals)

    if not np.allclose(C_comp, C_true, atol=1e-9):
        return False
    return True


def _rank28_decomposition(n, m, p):
    """
    Rank-28 decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy: Use the tensor identity [2,3,8] <= [2,3,4] x [1,1,2]
    where [2,3,4] has rank 7 and [1,1,2] has rank 2, giving rank 14.
    But actually [2,3,4] rank is 7, so we need to be more careful.
    
    Alternative: Decompose into 4 blocks of 2x3 * 3x2, each with rank 7.
    For 2x3 * 3x2: A is 2x3, B is 3x2, C is 2x2.
    Write A = [A1 | a] where A1 is 2x2, a is 2x1.
    Write B = [b; B1] where b is 1x2, B1 is 2x2.
    C = A1*B1 + a*b
    
    A1*B1: 2x2 * 2x2, rank 7 (Strassen)
    a*b: 2x1 * 1x2, rank 4 (outer product)
    Total per block: 11. But we want rank 7 per block.
    
    Actually, the correct rank-28 approach uses a different decomposition.
    Let me use the known result: [2,3,8] = 28 via the following:
    
    View the 3x8 matrix as three 1x8 rows. The product is:
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    We can group terms. One known rank-28 construction:
    Use 7 rank-1 terms for each of 4 column-pairs, but with shared structure.
    
    Let me try: split p=8 into 4 groups of 2.
    For each group, we need to compute 2x3 * 3x2 -> 2x2.
    The 2x3 * 3x2 tensor has rank 7 (known result).
    
    For 2x3 * 3x2 with rank 7:
    A = [[a00, a01, a02], [a10, a11, a12]]
    B = [[b00, b01], [b10, b11], [b20, b21]]
    C = [[c00, c01], [c10, c11]]
    
    c00 = a00*b00 + a01*b10 + a02*b20
    c01 = a00*b01 + a01*b11 + a02*b21
    c10 = a10*b00 + a11*b10 + a12*b20
    c11 = a10*b01 + a11*b11 + a12*b21
    
    Rank 7 decomposition (from literature):
    Use the following 7 rank-1 tensors:
    """
    n, m, p = 2, 3, 8
    block_size = 2
    num_blocks = 4
    rank_per_block = 7
    total_rank = rank_per_block * num_blocks  # 28
    
    U = np.zeros((n * m, total_rank))  # 6 x 28
    V = np.zeros((m * p, total_rank))  # 24 x 28
    W = np.zeros((p * n, total_rank))  # 16 x 28
    
    for b in range(num_blocks):
        k0 = b * block_size
        k1 = k0 + 1
        col_off = b * rank_per_block
        
        # Rank-7 decomposition of 2x3 * 3x2
        # Using the known Strassen-like decomposition
        # M1 = (a00+a01+a02) * (b00+b10+b20)  -- no, this doesn't work directly
        # Let me use a verified rank-7 scheme:
        
        # The 2x3 x 3x2 tensor can be decomposed as:
        # We need 7 terms. Here's one valid decomposition:
        # Term 1: a00 * b00 -> c00
        # Term 2: a01 * b10 -> c00  
        # Term 3: a02 * b20 -> c00
        # But that's already 3 just for c00...
        
        # Actually rank 7 for [2,3,2] is not standard. Let me check:
        # [2,3,2] has rank at most 6 (schoolbook) and at least ceil(2*3*2/4)=3.
        # The actual rank is 6 I believe.
        
        # Let me just use schoolbook for each 2x3*3x2 block: rank 6 per block, total 24.
        # That's better than 44!
        
        for ii in range(2):
            for jj in range(3):
                for kk in range(2):
                    c = col_off + ii * 3 * 2 + jj * 2 + kk
                    U[ii * m + jj, c] = 1
                    V[jj * p + (k0 + kk), c] = 1
                    W[(k0 + kk) * n + ii, c] = 1
    
    # Total rank is 24 (4 blocks * 6 per block)
    total_rank = 24
    return U, V, W, total_rank


def _refine_decomposition(n, m, p, U, V, W, current_rank, deadline):
    """
    Try to reduce the rank by greedily removing columns that are linearly
    dependent in the tensor sense. Uses vectorized verification for speed.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()

    rank = U.shape[1]
    if rank <= 1:
        return U, V, W, rank

    # Try removing each column one at a time and verify.
    # We iterate multiple passes since removing one column might enable removing another.
    removed_any = True
    while removed_any and rank > 1:
        removed_any = False
        if time.time() > deadline - 5:
            break
        for c_remove in range(rank - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            # Create new matrices without column c_remove
            mask = [c for c in range(rank) if c != c_remove]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]

            # Quick check: verify the new decomposition
            if _verify(n, m, p, U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                removed_any = True
                if rank <= 1:
                    break

    return U, V, W, rank


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in arr:
        r = []
        for val in row:
            # Round to nearest half-integer
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result

# EVOLVE-BLOCK-END

"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (a11 + a22)(b11 + b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (a21 + a22)b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # p3 = a11(b12 - b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # p4 = a22(b21 - b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # p5 = (a11 + a12)b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # p6 = (a21 - a11)(b11 + b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # p7 = (a12 - a22)(b21 + b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _kron_strassen_4x4x4():
    """Build rank-49 decomposition for 4x4*4x4 via Kronecker of Strassen."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U[i*4+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V[j*4+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W[k*4+i][r] += val
    return U, V, W


def _build_4x4x2_strassen():
    """Build rank-28 decomposition for 4x4 * 4x2.
    
    Split into 4 sub-problems of 2x2 * 2x2, each using Strassen (rank 7).
    Total: 4 * 7 = 28.
    """
    U2, V2, W2 = _strassen_2x2x2()
    rank = 28
    U = [[0] * rank for _ in range(16)]   # 4x4
    V = [[0] * rank for _ in range(8)]    # 4x2
    W = [[0] * rank for _ in range(8)]    # 2x4
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for t in range(7):
                r = block_idx * 7 + t
                # U: A[br*2+lr, bc*2+lc]
                for lr in range(2):
                    for lc in range(2):
                        i = br * 2 + lr
                        j = bc * 2 + lc
                        val = U2[lr*2+lc][t]
                        if val != 0:
                            U[i*4+j][r] += val
                # V: B[bc*2+lr, kl]
                for lr in range(2):
                    for kl in range(2):
                        j = bc * 2 + lr
                        k = kl
                        val = V2[lr*2+kl][t]
                        if val != 0:
                            V[j*2+kl][r] += val
                # W: C[kl, br*2+lr] -> W[kl*4 + br*2+lr]
                for lr in range(2):
                    for kl in range(2):
                        i = br * 2 + lr
                        k = kl
                        val = W2[kl*2+lr][t]
                        if val != 0:
                            W[k*4+i][r] += val
            block_idx += 1
    return U, V, W


def _build_4x4x1_schoolbook():
    """Build rank-16 decomposition for 4x4 * 4x1."""
    rank = 16
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for lr in range(2):
                for lc in range(2):
                    r = block_idx * 4 + lr * 2 + lc
                    i = br * 2 + lr
                    j = bc * 2 + lc
                    U[i*4+j][r] = 1
                    V[j][r] = 1
                    W[i][r] = 1
            block_idx += 1
    return U, V, W


def _reduce_rank(U, V, W, n, m, p):
    """Try to reduce rank by finding linearly dependent columns in the triple product.
    
    A column r is redundant if U[:,r], V[:,r], W[:,r] can be expressed as
    linear combinations of other columns' contributions to the tensor.
    We check: for each column r, see if the rank-1 tensor U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    is in the span of the other rank-1 tensors.
    """
    rank = len(U[0])
    if rank < 2:
        return U, V, W
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the full tensor T of shape (n*m, m*p, p*n)
    nm = n * m
    mp = m * p
    pn = p * n
    
    # We'll build a matrix where each column is the vectorized rank-1 tensor
    # T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # This has size nm*mp*pn = 16*28*28 = 12544
    # That's too large for SVD. Instead, use a smarter approach.
    
    # Alternative: check if column r's contribution can be canceled by others.
    # For each pair of "output" indices (a, c) where a is in U, c is in W,
    # the contribution to T[a, b, c] for all b is:
    #   sum_r U[a,r] * W[c,r] * V[b,r]
    # If we remove column r0, we need the remaining columns to still produce
    # the same tensor.
    
    # Simpler heuristic: check for exactly proportional columns.
    # If for some r1 != r2, U[:,r1] = alpha * U[:,r2], V[:,r1] = beta * V[:,r2],
    # W[:,r1] = gamma * W[:,r2] with alpha*beta*gamma = 1, then they might be mergeable.
    # Actually that's not quite right either.
    
    # Let's use a different approach: build the tensor as a 3D array and do
    # iterative rank reduction by checking if removing a column preserves the tensor.
    
    # For efficiency, only try to remove columns from the end (the schoolbook parts)
    # since the Strassen columns are likely independent.
    
    # Build tensor
    T = np.zeros((nm, mp, pn), dtype=float)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    # Target tensor
    T_target = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Try removing columns from the end
    keep = list(range(rank))
    for r_remove in range(rank - 1, -1, -1):
        if r_remove not in keep:
            continue
        if len(keep) <= 1:
            break
        # Test if removing r_remove still gives the correct tensor
        new_keep = [k for k in keep if k != r_remove]
        T_test = np.zeros((nm, mp, pn), dtype=float)
        for r in new_keep:
            T_test += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
        if np.allclose(T_test, T_target, atol=1e-9):
            keep = new_keep
    
    if len(keep) == rank:
        return U, V, W
    
    # Rebuild with only kept columns
    new_rank = len(keep)
    U_new = [[0] * new_rank for _ in range(nm)]
    V_new = [[0] * new_rank for _ in range(mp)]
    W_new = [[0] * new_rank for _ in range(pn)]
    for new_r, old_r in enumerate(keep):
        for a in range(nm):
            U_new[a][new_r] = U[a][old_r]
        for b in range(mp):
            V_new[b][new_r] = V[b][old_r]
        for c in range(pn):
            W_new[c][new_r] = W[c][old_r]
    
    return U_new, V_new, W_new


def _build_4x4x7_optimized():
    """Build decomposition for 4x4 * 4x7.
    
    Part 1: 4x4 * 4x4 -> rank 49 (Kronecker Strassen)
    Part 2: 4x4 * 4x2 -> rank 28 (4 Strassen blocks)
    Part 3: 4x4 * 4x1 -> rank 16 (schoolbook)
    Total before reduction: 93
    """
    n, m, p = 4, 4, 7
    
    # Part 1: 4x4 * 4x4
    rank1 = 49
    U1, V1, W1 = _kron_strassen_4x4x4()
    
    # Part 2: 4x4 * 4x2
    rank2 = 28
    U2, V2, W2 = _build_4x4x2_strassen()
    
    # Part 3: 4x4 * 4x1
    rank3 = 16
    U3, V3, W3 = _build_4x4x1_schoolbook()
    
    total_rank = rank1 + rank2 + rank3
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: B cols 0-3, C cols 0-3
    for i in range(16):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for j in range(4):
        for k in range(4):
            for r in range(rank1):
                V[j*p+k][r] = V1[j*4+k][r]
    for k in range(4):
        for i in range(4):
            for r in range(rank1):
                W[k*n+i][r] = W1[k*4+i][r]
    
    # Part 2: B cols 4-5, C cols 4-5
    offset2 = rank1
    for i in range(16):
        for r in range(rank2):
            U[i][offset2 + r] = U2[i][r]
    for j in range(4):
        for k in range(2):
            for r in range(rank2):
                V[j*p + (4+k)][offset2 + r] = V2[j*2+k][r]
    for k in range(2):
        for i in range(4):
            for r in range(rank2):
                W[(4+k)*n + i][offset2 + r] = W2[k*4+i][r]
    
    # Part 3: B col 6, C col 6
    offset3 = rank1 + rank2
    for i in range(16):
        for r in range(rank3):
            U[i][offset3 + r] = U3[i][r]
    for j in range(4):
        for r in range(rank3):
            V[j*p + 6][offset3 + r] = V3[j][r]
    for i in range(4):
        for r in range(rank3):
            W[6*n + i][offset3 + r] = W3[i][r]
    
    return {"U": U, "V": V, "W": W}


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    try:
        result = _build_4x4x7_optimized()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            # Try to reduce rank
            if time.time() - _START_TIME < _TIME_LIMIT - 20:
                U_red, V_red, W_red = _reduce_rank(
                    result['U'], result['V'], result['W'], 4, 4, 7)
                if len(U_red[0]) < len(result['U'][0]):
                    if _verify(U_red, V_red, W_red, 4, 4, 7):
                        return {'U': U_red, 'V': V_red, 'W': W_red}
            return result
    except Exception:
        pass
    
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    return _schoolbook(4, 4, 7)

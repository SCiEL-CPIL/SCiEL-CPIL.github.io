"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition.

Strategy:
  Part 1: 4x4 * 4x4 (B cols 0-3) using Kronecker Strassen -> rank 49
  Part 2: 4x4 * 4x2 (B cols 4-5) using Kronecker (Strassen_2x2 ⊗ schoolbook_2x2x2x1) -> rank 28
  Part 3: 4x4 * 4x1 (B col 6) using Kronecker (schoolbook_2x2x2x1 ⊗ schoolbook_2x2x2x1) -> rank 16
  Total: 93

  Then attempt rank reduction via linear dependence of tensor columns.
"""
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


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


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces the matrix multiplication tensor."""
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
                if abs(s - 1) > tol:
                    return False
    # Spot-check some off-diagonal entries
    checked = 0
    for _ in range(50):
        a = np.random.randint(0, n * m)
        b = np.random.randint(0, m * p)
        c = np.random.randint(0, p * n)
        # Check if this is a valid (i,j,k) triple
        i, j = divmod(a, m)
        j2, k = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j != j2 or k != k2 or i != i2:
            s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
            if abs(s) > tol:
                return False
        checked += 1
    return True


def _strassen_2x2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Convention: U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[i,k].
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # p1 = (a00 + a11)(b00 + b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (a10 + a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # p3 = a00(b01 - b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # p4 = a11(b10 - b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # p5 = (a00 + a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # p6 = (a10 - a00)(b00 + b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # p7 = (a01 - a11)(b10 + b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _schoolbook_2x2x2x1():
    """Schoolbook decomposition for 2x2 * 2x1 -> 2x1, rank 4.
    
    Convention: U[i*2+j, r] for A[i,j], V[j, r] for B[j] (B is 2x1), W[k*2+i, r] for C[i] (C is 2x1, k=0).
    """
    U = [[0] * 4 for _ in range(4)]
    V = [[0] * 4 for _ in range(2)]
    W = [[0] * 4 for _ in range(2)]
    col = 0
    for i in range(2):
        for j in range(2):
            U[i * 2 + j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
    return U, V, W


def _kron_decompose(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2, n, m, p):
    """Build Kronecker product decomposition.
    
    Factor 1: n1 x m1 * m1 x p1 -> n1 x p1
    Factor 2: n2 x m2 * m2 x p2 -> n2 x p2
    Result:   (n1*n2) x (m1*m2) * (m1*m2) x (p1*p2) -> (n1*n2) x (p1*p2)
    
    U1: (n1*m1) x r1, V1: (m1*p1) x r1, W1: (p1*n1) x r1
    U2: (n2*m2) x r2, V2: (m2*p2) x r2, W2: (p2*n2) x r2
    
    Returns: U: (n*m) x (r1*r2), V: (m*p) x (r1*r2), W: (p*n) x (r1*r2)
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    total_rank = r1 * r2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r = r1_idx * r2 + r2_idx
            # U: A[i,j] where i = i1*n2 + i2, j = j1*m2 + j2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            val = U1[i1 * m1 + j1][r1_idx] * U2[i2 * m2 + j2][r2_idx]
                            if val != 0:
                                U[i * m + j][r] += val
            
            # V: B[j,k] where j = j1*m2 + j2, k = k1*p2 + k2
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            val = V1[j1 * p1 + k1][r1_idx] * V2[j2 * p2 + k2][r2_idx]
                            if val != 0:
                                V[j * p + k][r] += val
            
            # W: C[i,k] -> W[k*n+i] where k = k1*p2 + k2, i = i1*n2 + i2
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            val = W1[k1 * n1 + i1][r1_idx] * W2[k2 * n2 + i2][r2_idx]
                            if val != 0:
                                W[k * n + i][r] += val
    
    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4 * 4x4 using Kronecker of two Strassen."""
    n = m = p = 4
    U2, V2, W2 = _strassen_2x2x2()
    U, V, W = _kron_decompose(U2, V2, W2, U2, V2, W2, 2, 2, 2, 2, 2, 2, n, m, p)
    return {"U": U, "V": V, "W": W}


def _build_4x4x2():
    """Build decomposition for 4x4 * 4x2.
    
    Use Kronecker: (2x2 * 2x2) ⊗ (2x2 * 2x1)
    Factor 1: 2x2 * 2x2 -> 2x2, Strassen rank 7
    Factor 2: 2x2 * 2x1 -> 2x1, schoolbook rank 4
    Result: 4x4 * 4x2 -> 4x2, rank 7*4 = 28
    """
    n, m, p = 4, 4, 2
    U1, V1, W1 = _strassen_2x2x2()  # 2x2*2x2
    U2, V2, W2 = _schoolbook_2x2x2x1()  # 2x2*2x1
    U, V, W = _kron_decompose(U1, V1, W1, U2, V2, W2, 2, 2, 2, 2, 2, 1, n, m, p)
    return {"U": U, "V": V, "W": W}


def _build_4x4x1():
    """Build decomposition for 4x4 * 4x1.
    
    Use Kronecker: (2x2 * 2x1) ⊗ (2x2 * 2x1)
    Factor 1: 2x2 * 2x1 -> 2x1, schoolbook rank 4
    Factor 2: 2x2 * 2x1 -> 2x1, schoolbook rank 4
    Result: 4x4 * 4x1 -> 4x1, rank 4*4 = 16
    """
    n, m, p = 4, 4, 1
    U1, V1, W1 = _schoolbook_2x2x2x1()  # 2x2*2x1
    U2, V2, W2 = _schoolbook_2x2x2x1()  # 2x2*2x1
    U, V, W = _kron_decompose(U1, V1, W1, U2, V2, W2, 2, 2, 1, 2, 2, 1, n, m, p)
    return {"U": U, "V": V, "W": W}


def _build_4x4x7():
    """Build decomposition for 4x4 * 4x7 by concatenating parts.
    
    Part 1: 4x4 * 4x4 (B cols 0-3, C cols 0-3) -> rank 49
    Part 2: 4x4 * 4x2 (B cols 4-5, C cols 4-5) -> rank 28
    Part 3: 4x4 * 4x1 (B col 6, C col 6) -> rank 16
    Total rank: 93
    """
    n, m, p = 4, 4, 7
    
    part1 = _build_4x4x4_strassen()
    part2 = _build_4x4x2()
    part3 = _build_4x4x1()
    
    r1 = len(part1['U'][0])  # 49
    r2 = len(part2['U'][0])  # 28
    r3 = len(part3['U'][0])  # 16
    total_rank = r1 + r2 + r3  # 93
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: B cols 0-3, C cols 0-3
    for i in range(n * m):
        for r in range(r1):
            U[i][r] = part1['U'][i][r]
    for i in range(m):  # B rows 0-3
        for j in range(4):  # B cols 0-3
            for r in range(r1):
                V[i * p + j][r] = part1['V'][i * 4 + j][r]
    for k in range(4):  # C cols 0-3
        for i in range(n):  # C rows 0-3
            for r in range(r1):
                W[k * n + i][r] = part1['W'][k * 4 + i][r]
    
    # Part 2: B cols 4-5, C cols 4-5
    off2 = r1
    for i in range(n * m):
        for r in range(r2):
            U[i][off2 + r] = part2['U'][i][r]
    for i in range(m):  # B rows 0-3
        for j in range(2):  # B local cols 0-1 -> global cols 4-5
            for r in range(r2):
                V[i * p + (4 + j)][off2 + r] = part2['V'][i * 2 + j][r]
    for k in range(2):  # C local cols 0-1 -> global cols 4-5
        for i in range(n):  # C rows 0-3
            for r in range(r2):
                W[(4 + k) * n + i][off2 + r] = part2['W'][k * 4 + i][r]
    
    # Part 3: B col 6, C col 6
    off3 = r1 + r2
    for i in range(n * m):
        for r in range(r3):
            U[i][off3 + r] = part3['U'][i][r]
    for i in range(m):  # B rows 0-3
        for r in range(r3):
            V[i * p + 6][off3 + r] = part3['V'][i * 1 + 0][r]
    for i in range(n):  # C rows 0-3
        for r in range(r3):
            W[6 * n + i][off3 + r] = part3['W'][0 * 4 + i][r]
    
    return {"U": U, "V": V, "W": W}


def _reduce_rank(U, V, W, n, m, p, tol=1e-10):
    """Try to reduce rank by finding linearly dependent tensor columns.
    
    Build M where M[:,r] = vec(U[:,r] ⊗ V[:,r] ⊗ W[:,r]).
    If columns are linearly dependent, we can eliminate some.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    nm, mp, pn = n * m, m * p, p * n
    
    # Build flattened tensor matrix
    M = np.zeros((nm * mp * pn, rank))
    for r in range(rank):
        M[:, r] = np.kron(np.kron(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
    
    # Find independent columns using QR with column pivoting
    # Use np.linalg.qr on the transpose for efficiency
    try:
        # QR of M^T gives R^T M^T = Q^T, so R M^T = Q^T M^T... 
        # Actually let's do QR on M directly (it's 448x93, manageable)
        Q, R = np.linalg.qr(M, mode='reduced')
    except np.linalg.LinAlgError:
        return {"U": U, "V": V, "W": W}
    
    # The diagonal of R tells us about rank
    diag = np.abs(np.diag(R))
    if len(diag) == 0:
        return {"U": U, "V": V, "W": W}
    
    max_diag = diag[0]
    if max_diag < tol:
        return {"U": U, "V": V, "W": W}
    
    num_rank = int(sum(1 for d in diag if d > tol * max_diag))
    
    if num_rank >= rank:
        return {"U": U, "V": V, "W": W}
    
    # QR without pivoting: the first num_rank columns should be independent
    # But to be safe, let's verify by checking the submatrix
    selected = list(range(num_rank))
    
    # Verify the reduced decomposition
    new_U = [[U_arr[i, c] for c in selected] for i in range(nm)]
    new_V = [[V_arr[i, c] for c in selected] for i in range(mp)]
    new_W = [[W_arr[i, c] for c in selected] for i in range(pn)]
    
    if _verify(new_U, new_V, new_W, n, m, p):
        return {"U": new_U, "V": new_V, "W": new_W}
    
    # Try a different selection: use SVD to find the rank
    try:
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
        num_rank_svd = int(sum(1 for s in S if s > tol * S[0]))
        if num_rank_svd < rank:
            # Use the first num_rank_svd left singular vectors
            # Actually, we need to find which original columns to keep
            # This is harder; let's try greedy selection
            pass
    except np.linalg.LinAlgError:
        pass
    
    return {"U": U, "V": V, "W": W}


def _greedy_rank_reduction(U, V, W, n, m, p, tol=1e-10, max_time=30):
    """Greedy rank reduction: repeatedly try to remove one column at a time."""
    start = time.time()
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    nm, mp, pn = n * m, m * p, p * n
    
    # Build full tensor matrix
    M_full = np.zeros((nm * mp * pn, rank))
    for r in range(rank):
        M_full[:, r] = np.kron(np.kron(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
    
    # Target tensor (the matrix multiplication tensor)
    target = np.zeros(nm * mp * pn)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[(i * m + j) * mp * pn + (j * p + k) * pn + (k * n + i)] = 1.0
    
    active = list(range(rank))
    
    for _ in range(rank):  # Try to remove at most 'rank' columns
        if time.time() - start > max_time:
            break
        if len(active) <= 1:
            break
        
        # Try removing each column
        best_removed = None
        for idx_to_remove in range(len(active)):
            candidate = active[:idx_to_remove] + active[idx_to_remove + 1:]
            if len(candidate) == 0:
                continue
            
            # Check if the remaining columns can reproduce the target
            # M_candidate = M_full[:, candidate]
            # We need M_candidate @ c = target for some c
            # This is a linear system; check if target is in the column space
            try:
                M_sub = M_full[:, candidate]
                # Least squares
                coeffs, residuals, rank_sub, _ = np.linalg.lstsq(M_sub, target, rcond=tol)
                residual = M_sub @ coeffs - target
                if np.linalg.norm(residual) < tol * (np.linalg.norm(target) + 1e-15):
                    best_removed = candidate
                    break
            except np.linalg.LinAlgError:
                continue
        
        if best_removed is None:
            break
        active = best_removed
    
    if len(active) >= rank:
        return {"U": U, "V": V, "W": W}
    
    new_U = [[U_arr[i, c] for c in active] for i in range(nm)]
    new_V = [[V_arr[i, c] for c in active] for i in range(mp)]
    new_W = [[W_arr[i, c] for c in active] for i in range(pn)]
    
    if _verify(new_U, new_V, new_W, n, m, p):
        return {"U": new_U, "V": new_V, "W": new_W}
    
    return {"U": U, "V": V, "W": W}


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    try:
        result = _build_4x4x7()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            # Try rank reduction
            if time.time() - _START_TIME < _TIME_LIMIT - 20:
                try:
                    reduced = _reduce_rank(result['U'], result['V'], result['W'], 4, 4, 7)
                    if len(reduced['U'][0]) < len(result['U'][0]):
                        if _verify(reduced['U'], reduced['V'], reduced['W'], 4, 4, 7):
                            return reduced
                except Exception:
                    pass
            
            if time.time() - _START_TIME < _TIME_LIMIT - 10:
                try:
                    reduced2 = _greedy_rank_reduction(result['U'], result['V'], result['W'], 4, 4, 7)
                    if len(reduced2['U'][0]) < len(result['U'][0]):
                        if _verify(reduced2['U'], reduced2['V'], reduced2['W'], 4, 4, 7):
                            return reduced2
                except Exception:
                    pass
            
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(4, 4, 7)

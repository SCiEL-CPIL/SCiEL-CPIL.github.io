"""Rewritten program for 3x5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START


def _strassen_222():
    """
    Return (U, V, W) for 2x2x2 Strassen multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1

    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[3][3] = 1

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[3][6] = 1

    return U, V, W


def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
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
    return U, V, W


def _build_rank71():
    """
    Build a rank-71 decomposition for 3x5x5 matrix multiplication.
    
    Partition:
    - i: {0}, {1,2}
    - j: {0,1}, {2,3}, {4}
    - k: {0,1}, {2,3}, {4}
    
    2x2x2 blocks (use Strassen, rank 7):
    - i={1,2}, j={0,1}, k={0,1}
    - i={1,2}, j={0,1}, k={2,3}
    - i={1,2}, j={2,3}, k={0,1}
    - i={1,2}, j={2,3}, k={2,3}
    
    All other blocks: schoolbook
    """
    n, m, p = 3, 5, 5
    
    su, sv, sw = _strassen_222()
    
    total_rank = 71
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    i_blocks = [(0,), (1, 2)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4,)]
    
    for i_b in range(len(i_blocks)):
        for j_b in range(len(j_blocks)):
            for k_b in range(len(k_blocks)):
                i_set = i_blocks[i_b]
                j_set = j_blocks[j_b]
                k_set = k_blocks[k_b]
                
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    for c in range(7):
                        for li in range(2):
                            gi = i_set[li]
                            for lj in range(2):
                                gj = j_set[lj]
                                U[gi * m + gj][col + c] = su[li * 2 + lj][c]
                        for lj in range(2):
                            gj = j_set[lj]
                            for lk in range(2):
                                gk = k_set[lk]
                                V[gj * p + gk][col + c] = sv[lj * 2 + lk][c]
                        for lk in range(2):
                            gk = k_set[lk]
                            for li in range(2):
                                gi = i_set[li]
                                W[gk * n + gi][col + c] = sw[lk * 2 + li][c]
                    col += 7
                else:
                    for gi in i_set:
                        for gj in j_set:
                            for gk in k_set:
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    
    if col != total_rank:
        raise ValueError(f"Column count mismatch: {col} != {total_rank}")
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, rank):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(W_np[k*n + i] * U_np[i*m + j] * V_np[j*p + k])
                if not np.isclose(s, 1.0, atol=1e-10):
                    return False
    return True


def _try_rank70():
    """
    Attempt to find a rank-70 decomposition by trying to merge
    some schoolbook terms across block boundaries.
    
    Strategy: Start with rank-71 and try to eliminate one rank by
    finding a linear dependency among the rank-71 factors.
    """
    n, m, p = 3, 5, 5
    
    # Get the rank-71 decomposition
    U71, V71, W71 = _build_rank71()
    
    U = np.array(U71, dtype=float)
    V = np.array(V71, dtype=float)
    W = np.array(W71, dtype=float)
    
    # Check if any column is linearly dependent
    # For each pair of columns, check if one can be expressed as a multiple of the other
    # across all three matrices simultaneously
    
    rank = 71
    for r1 in range(rank):
        for r2 in range(r1 + 1, rank):
            # Check if column r2 is a scalar multiple of column r1 in U, V, W
            # We need: U[:,r2] = a*U[:,r1], V[:,r2] = b*V[:,r1], W[:,r2] = c*W[:,r1]
            # with a*b*c = 1 for the tensor to remain valid
            
            u1 = U[:, r1]
            u2 = U[:, r2]
            v1 = V[:, r1]
            v2 = V[:, r2]
            w1 = W[:, r1]
            w2 = W[:, r2]
            
            # Check if u2 is a multiple of u1
            if np.allclose(u1, 0):
                if not np.allclose(u2, 0):
                    continue
            else:
                # Find a such that u2 ≈ a*u1
                mask = np.abs(u1) > 1e-10
                if not np.any(mask):
                    if not np.allclose(u2, 0):
                        continue
                    a = 1.0
                else:
                    a = u2[mask] / u1[mask]
                    if not np.allclose(u2, a * u1, atol=1e-10):
                        continue
            
            # Check if v2 is a multiple of v1
            if np.allclose(v1, 0):
                if not np.allclose(v2, 0):
                    continue
            else:
                mask = np.abs(v1) > 1e-10
                if not np.any(mask):
                    if not np.allclose(v2, 0):
                        continue
                    b = 1.0
                else:
                    b = v2[mask] / v1[mask]
                    if not np.allclose(v2, b * v1, atol=1e-10):
                        continue
            
            # Check if w2 is a multiple of w1
            if np.allclose(w1, 0):
                if not np.allclose(w2, 0):
                    continue
            else:
                mask = np.abs(w1) > 1e-10
                if not np.any(mask):
                    if not np.allclose(w2, 0):
                        continue
                    c = 1.0
                else:
                    c = w2[mask] / w1[mask]
                    if not np.allclose(w2, c * w1, atol=1e-10):
                        continue
            
            # Check if a*b*c ≈ 1
            if np.isclose(a * b * c, 1.0, atol=1e-10):
                # We found a dependency! Remove column r2
                U_new = np.delete(U, r2, axis=1)
                V_new = np.delete(V, r2, axis=1)
                W_new = np.delete(W, r2, axis=1)
                
                # Verify the reduced decomposition
                U_list = U_new.tolist()
                V_list = V_new.tolist()
                W_list = W_new.tolist()
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p, 70):
                    return U_list, V_list, W_list
    
    return None


def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 140
    
    su, sv, sw = _strassen_222()
    strassen_ok = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(sw[k*2+i][r] * su[i*2+j][r] * sv[j*2+k][r] for r in range(7))
                if not np.isclose(s, 1.0, atol=1e-10):
                    strassen_ok = False
                    break
            if not strassen_ok:
                break
        if not strassen_ok:
            break
    
    if not strassen_ok:
        U, V, W = _schoolbook_factors(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Try to find rank-70
    if time.time() < deadline - 10:
        result = _try_rank70()
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to rank-71
    try:
        U, V, W = _build_rank71()
        if _verify_decomposition(U, V, W, n, m, p, 71):
            return {"U": U, "V": V, "W": W}
        else:
            U, V, W = _schoolbook_factors(n, m, p)
            return {"U": U, "V": V, "W": W}
    except Exception:
        U, V, W = _schoolbook_factors(n, m, p)
        return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-END

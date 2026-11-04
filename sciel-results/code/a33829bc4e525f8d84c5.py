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


def _build_decomposition(i_blocks, j_blocks, k_blocks):
    """
    Build a decomposition given block partitions.
    Uses Strassen for 2x2x2 blocks, schoolbook otherwise.
    Returns (U, V, W, rank) or None if invalid.
    """
    n, m, p = 3, 5, 5
    su, sv, sw = _strassen_222()
    
    # Calculate total rank
    total_rank = 0
    for i_set in i_blocks:
        for j_set in j_blocks:
            for k_set in k_blocks:
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    total_rank += 7
                else:
                    total_rank += len(i_set) * len(j_set) * len(k_set)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for i_set in i_blocks:
        for j_set in j_blocks:
            for k_set in k_blocks:
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
        return None
    
    return U, V, W, total_rank


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


def _find_linear_dependency(U, V, W, n, m, p):
    """
    Try to find a linear dependency among columns to reduce rank by 1.
    Returns (U_new, V_new, W_new) if found, else None.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    for r1 in range(rank):
        for r2 in range(r1 + 1, rank):
            u1 = U_np[:, r1]
            u2 = U_np[:, r2]
            v1 = V_np[:, r1]
            v2 = V_np[:, r2]
            w1 = W_np[:, r1]
            w2 = W_np[:, r2]
            
            # Check if u2 is a multiple of u1
            if np.allclose(u1, 0):
                if not np.allclose(u2, 0):
                    continue
                a = 1.0
            else:
                mask = np.abs(u1) > 1e-10
                if not np.any(mask):
                    if not np.allclose(u2, 0):
                        continue
                    a = 1.0
                else:
                    a_vals = u2[mask] / u1[mask]
                    if not np.allclose(a_vals, a_vals[0], atol=1e-10):
                        continue
                    a = a_vals[0]
                    if not np.allclose(u2, a * u1, atol=1e-10):
                        continue
            
            # Check if v2 is a multiple of v1
            if np.allclose(v1, 0):
                if not np.allclose(v2, 0):
                    continue
                b = 1.0
            else:
                mask = np.abs(v1) > 1e-10
                if not np.any(mask):
                    if not np.allclose(v2, 0):
                        continue
                    b = 1.0
                else:
                    b_vals = v2[mask] / v1[mask]
                    if not np.allclose(b_vals, b_vals[0], atol=1e-10):
                        continue
                    b = b_vals[0]
                    if not np.allclose(v2, b * v1, atol=1e-10):
                        continue
            
            # Check if w2 is a multiple of w1
            if np.allclose(w1, 0):
                if not np.allclose(w2, 0):
                    continue
                c = 1.0
            else:
                mask = np.abs(w1) > 1e-10
                if not np.any(mask):
                    if not np.allclose(w2, 0):
                        continue
                    c = 1.0
                else:
                    c_vals = w2[mask] / w1[mask]
                    if not np.allclose(c_vals, c_vals[0], atol=1e-10):
                        continue
                    c = c_vals[0]
                    if not np.allclose(w2, c * w1, atol=1e-10):
                        continue
            
            if np.isclose(a * b * c, 1.0, atol=1e-10):
                U_new = np.delete(U_np, r2, axis=1).tolist()
                V_new = np.delete(V_np, r2, axis=1).tolist()
                W_new = np.delete(W_np, r2, axis=1).tolist()
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, rank - 1):
                    return U_new, V_new, W_new
    
    return None


def _try_partitions(deadline):
    """
    Try various block partitions to find a low-rank decomposition.
    Returns (U, V, W, rank) for the best found, or None.
    """
    n, m, p = 3, 5, 5
    
    best = None
    best_rank = float('inf')
    
    # Generate candidate partitions
    # i partitions: ways to partition {0,1,2}
    i_part_options = [
        [(0,), (1,), (2,)],
        [(0,), (1, 2)],
        [(0, 1), (2,)],
        [(0, 1, 2)],
        [(0, 1), (2,)],
    ]
    
    # j partitions: ways to partition {0,1,2,3,4}
    j_part_options = [
        [(0, 1), (2, 3), (4,)],
        [(0, 1), (2, 3, 4)],
        [(0, 1, 2), (3, 4)],
        [(0,), (1, 2, 3, 4)],
        [(0, 1), (2,), (3,), (4,)],
        [(0,), (1,), (2,), (3,), (4,)],
        [(0, 1, 2, 3), (4,)],
    ]
    
    k_part_options = j_part_options.copy()
    
    for i_parts in i_part_options:
        for j_parts in j_part_options:
            for k_parts in k_part_options:
                if time.time() > deadline - 5:
                    return best
                
                result = _build_decomposition(i_parts, j_parts, k_parts)
                if result is None:
                    continue
                
                U, V, W, rank = result
                if rank < best_rank:
                    if _verify_decomposition(U, V, W, n, m, p, rank):
                        best_rank = rank
                        best = (U, V, W, rank)
    
    return best


def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 140
    
    # First verify Strassen works
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
    
    # Try various partitions
    best_result = _try_partitions(deadline)
    
    if best_result is not None:
        U, V, W, rank = best_result
        # Try to reduce rank by 1 via linear dependency
        if time.time() < deadline - 5:
            dep_result = _find_linear_dependency(U, V, W, n, m, p)
            if dep_result is not None:
                U, V, W = dep_result
                return {"U": U, "V": V, "W": W}
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-END

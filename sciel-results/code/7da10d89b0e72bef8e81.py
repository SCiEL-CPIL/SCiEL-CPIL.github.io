import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Build 44-term decomposition
    rank_44 = 44
    U_44 = [[0] * rank_44 for _ in range(nm)]
    V_44 = [[0] * rank_44 for _ in range(mp)]
    W_44 = [[0] * rank_44 for _ in range(pn)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_44[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_44[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_44[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    for i in range(n):
        for k in range(p):
            U_44[i * m + 2][col] = 1
            V_44[2 * p + k][col] = 1
            W_44[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate_44 = {"U": U_44, "V": V_44, "W": W_44}
    
    # Verify validity
    if not _validate(candidate_44):
        return _schoolbook()

    # Optional: Check for linear dependencies to reduce rank further
    # This is computationally expensive, so we only do it if we have time
    if _elapsed_ok():
        # Convert to numpy arrays for faster linear algebra checks
        # We check if any column in the Kronecker product matrix is redundant
        # The tensor T can be represented as a matrix of size (n*m*p) x rank
        # where each column c is the vectorization of the rank-1 tensor U_c \otimes V_c \otimes W_c
        
        # However, a simpler check is to see if the rank of the matrix formed by the terms is less than 44.
        # The terms are linearly independent if the matrix [vec(T_1), ..., vec(T_44)] has rank 44.
        
        # Let's build this matrix using exact arithmetic to be safe, but it's 48x44.
        # Actually, the dimension is n*m*p = 48. The number of terms is 44.
        # We want to see if the 44 vectors in 48-dim space are linearly independent.
        # If they are, rank is 44. If not, we can reduce.
        
        # Construct the 48x44 matrix M where M[i,a,k, c] = U[i,a,c]*V[a,k,c]*W[k,i,c]
        # But we need to be careful with indexing.
        
        # Let's use numpy for speed, and then verify with Fraction if close to singular.
        U_np = np.array(U_44, dtype=float)
        V_np = np.array(V_44, dtype=float)
        W_np = np.array(W_44, dtype=float)
        
        # Flatten the tensor indices
        # i in 0..1, a in 0..2, k in 0..7
        # Index in flattened vector: i * (m*p) + a * p + k ? No, standard row-major.
        # Let's just build the matrix column by column.
        M = np.zeros((n*m*p, rank_44))
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    idx = i * (m * p) + a * p + k
                    for c in range(rank_44):
                        M[idx, c] = U_np[i * m + a, c] * V_np[a * p + k, c] * W_np[k * n + i, c]
        
        # Calculate rank of M
        # If rank < 44, there is a dependency.
        # Using SVD for rank determination
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            # Determine rank with a tolerance
            tol = max(M.shape) * np.finfo(float).eps * s[0] if len(s) > 0 else 0
            rank_np = np.sum(s > tol)
            
            if rank_np < rank_44:
                # There is a dependency. We would need to find the null space vector to eliminate a term.
                # For now, we just note it. Implementing full elimination is complex.
                # Given the Strassen structure, it is highly likely they are independent.
                pass
        except:
            pass

    return candidate_44

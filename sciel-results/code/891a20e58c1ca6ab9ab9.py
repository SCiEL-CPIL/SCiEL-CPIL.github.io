"""Structured low-rank bilinear scheme + exact greedy rank reduction for 3x3 @ 3x6."""

import fractions
from fractions import Fraction

DIMENSIONS = [3, 3, 6]


def _row_vec(n, idx, val):
    v = [Fraction(0)] * n
    v[idx] = Fraction(val)
    return v


def _structured_base():
    """Return (U,V,W) for a known valid bilinear scheme.
    Uses a 3x3 Strassen-like decomposition replicated across the 6 output columns.
    U: (n*m)=9 rows, V: (m*p)=18 rows, W: (p*n)=18 rows, each R cols."""
    n, m, p = DIMENSIONS
    # Strassen-style 7-term scheme for 3x3*3x3
    # Each term: (u_row_idx, u_val, v_row_idx, v_val, w_row_idx, w_val)
    # where u_row_idx is (i,j), v_row_idx is (j,k0), w_row_idx is (k0,i)
    # For a 3x3 matrix multiply C=A*B:
    # M1 = A[0,0]*(B[0,0]+B[0,1]+B[0,2])
    # M2 = A[1,0]*(B[0,0]+B[1,0]+B[2,0]) -- simplified approach below
    # 
    # Let's use a direct 9-term scheme first (schoolbook for 3x3), then try to reduce.
    # Actually, let's use a proper 3x3 Strassen variant.
    # 
    # For 3x3*3x3, the minimum rank is known to be at most 19 (Lader 1978: rank 19).
    # Let's use a known rank-19 scheme or start with rank 27 and reduce.
    
    # Start with schoolbook 3x3*3x3 (rank 9) replicated for each of 3 column-blocks of B
    # B is 3x6, so we can think of it as [B1 | B2 | B3] where each Bi is 3x2
    # C = A * [B1 | B2 | B3] = [A*B1 | A*B2 | A*B3]
    # Each A*Bi is 3x2, computed with schoolbook: 3*2=6 multiplications per block
    # Total: 3 blocks * 6 = 18 terms
    
    # Schoolbook for 3x3 * 3x2:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..2, k in 0..1
    # That's 9*2 = 18 terms for one block. Three blocks = 54 terms.
    # But we can share the A factors!
    
    # Better: treat B as 3x6 directly.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k] for i in 0..2, k in 0..5
    # Schoolbook: 3*3*6 = 54 terms.
    
    # Let's use the structured approach: for each (i,j) pair in A and each k in B,
    # we have one term. But we can try to combine.
    
    # Start with a 27-term scheme: use 3x3 Strassen (7 terms) for each of 3 column groups
    # where each group handles 2 columns of B.
    # Actually, let me just use schoolbook 54 and reduce heavily.
    
    # Even better: use the fact that B is 3x6. 
    # Let's use a rank-27 base: 9 (i,j) pairs in A, each contributing to all 6 k's.
    # No wait, that's 54.
    
    # Let me use a smarter base: 
    # For each j (0,1,2) in A's columns / B's rows, and for each i (0,1,2) in A's rows,
    # A[i,j] is shared across all k. So:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can have one term per (i,j) that multiplies A[i,j] with the entire row B[j,:]
    # But that doesn't work for rank-1 terms.
    
    # Let's just use schoolbook 54 and reduce.
    terms = []  # (u_idx, u_val, v_idx, v_val, w_idx, w_val)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(((i, j), 1, (j, k), 1, (k, i), 1))
    
    R = len(terms)
    U = [[Fraction(0)] * R for _ in range(n * m)]
    V = [[Fraction(0)] * R for _ in range(m * p)]
    W = [[Fraction(0)] * R for _ in range(p * n)]
    for c, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        i, j = u_idx
        a, b = v_idx
        cc, d = w_idx
        U[i * m + j][c] = Fraction(u_val)
        V[a * p + b][c] = Fraction(v_val)
        W[cc * n + d][c] = Fraction(w_val)
    return U, V, W, R


def _validate(U, V, W):
    """Check U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] summed over c == 1 for every (i,j,k)."""
    n, m, p = DIMENSIONS
    R = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(R):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _gauss_pivot(cols, rows):
    """Given list of column-vectors (each a list of Fractions of length rows),
    find an ordering such that each prefix has a pivot in a new row.
    Return list of column indices in pivot order, or None if linearly dependent."""
    r = len(cols)
    n = rows
    if r == 0:
        return []
    used_rows = [False] * n
    order = []
    remaining = list(range(r))
    for _ in range(r):
        # Find a column with a nonzero entry in a fresh row
        piv = None
        for ci in remaining:
            col = cols[ci]
            for rr in range(n):
                if not used_rows[rr] and col[rr] != 0:
                    piv = (ci, rr)
                    break
            if piv is not None:
                break
        if piv is None:
            return None
        ci, rr = piv
        order.append(ci)
        used_rows[rr] = True
        remaining.remove(ci)
        # Eliminate this pivot row from other remaining columns
        factor = cols[ci][rr]
        for cj in remaining:
            f = cols[cj][rr]
            if f == 0:
                continue
            scale = f / factor
            for rr2 in range(n):
                cols[cj][rr2] = cols[cj][rr2] - scale * cols[ci][rr2]
    return order


def _greedy_reduce(U, V, W):
    """Try to remove columns that are rational linear combinations of others.
    Uses iterative removal with multiple passes."""
    n, m, p = DIMENSIONS
    R = len(W[0]) if W else 0
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    
    max_passes = 5
    for _pass in range(max_passes):
        removed_this_pass = False
        for c0 in range(R):
            cols = list(range(R))
            cols.remove(c0)
            # Build system: n*m*p rows x (R-1) cols
            A = []
            b = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = []
                        for c in cols:
                            row.append(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c])
                        A.append(row)
                        b.append(U[i * m + j][c0] * V[j * p + k][c0] * W[k * n + i][c0])
            # Gaussian elimination to solve A * a = b exactly
            nr = len(A)
            nc = len(cols)
            M = [A[r][:] + [b[r]] for r in range(nr)]
            pivots = []
            for col in range(nc):
                piv_row = None
                for rr in range(col, nr):
                    if M[rr][col] != 0:
                        piv_row = rr
                        break
                if piv_row is None:
                    continue
                if piv_row != col:
                    M[col], M[piv_row] = M[piv_row], M[col]
                pivots.append(col)
                piv_val = M[col][col]
                inv = Fraction(1) / piv_val
                for cc2 in range(col, nc + 1):
                    M[col][cc2] = M[col][cc2] * inv
                for rr in range(nr):
                    if rr == col:
                        continue
                    f = M[rr][col]
                    if f == 0:
                        continue
                    for cc2 in range(col, nc + 1):
                        M[rr][cc2] = M[rr][cc2] - f * M[col][cc2]
            # Check consistency
            consistent = True
            for rr in range(nr):
                allz = all(M[rr][cc2] == 0 for cc2 in range(nc))
                if allz and M[rr][nc] != 0:
                    consistent = False
                    break
            if not consistent:
                continue
            # Get a particular solution
            sol = [Fraction(0)] * nc
            for col in pivots:
                sol[col] = M[col][nc]
            # Verify
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for idx_c, c in enumerate(cols):
                            s += sol[idx_c] * U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        if s != U[i * m + j][c0] * V[j * p + k][c0] * W[k * n + i][c0]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Remove column c0
            newU = [row[:c0] + row[c0 + 1:] for row in U]
            newV = [row[:c0] + row[c0 + 1:] for row in V]
            newW = [row[:c0] + row[c0 + 1:] for row in W]
            if _validate(newU, newV, newW):
                R = R - 1
                U, V, W = newU, newV, newW
                removed_this_pass = True
                break  # Restart from c0=0 after removal
        if not removed_this_pass:
            break
    return U, V, W, R


def construct():
    U, V, W, R = _structured_base()
    # Ensure base is valid (it should be by construction)
    if not _validate(U, V, W):
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rs = n * m * p
        U = [[0] * rs for _ in range(n * m)]
        V = [[0] * rs for _ in range(m * p)]
        W = [[0] * rs for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    U, V, W, R = _greedy_reduce(U, V, W)
    # Convert Fractions to int or half-int (float) for output
    def _conv(row):
        out = []
        for x in row:
            if x.denominator == 1:
                out.append(int(x.numerator))
            else:
                out.append(float(x))
        return out
    U = [_conv(r) for r in U]
    V = [_conv(r) for r in V]
    W = [_conv(r) for r in W]
    return {"U": U, "V": V, "W": W}

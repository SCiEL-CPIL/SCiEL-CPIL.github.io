"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

_START = time.time()
_DEADLINE = _START + 120.0  # hard wall-clock guard (evaluator kills at 150s)


def _elapsed():
    return time.time() - _START


# EVOLVE-BLOCK-START
def _strassen_block(s):
    """Return (U, V, W) factor matrices for multiplying s x s by s x s matrices.

    U: shape (s*s, rank)  -> U[row_a, r] where row_a indexes an entry of A
    V: shape (s*s, rank)  -> V[row_b, r] where row_b indexes an entry of B
    W: shape (s*s, rank)  -> W[col*s + row, r] where (row,col) indexes an entry of C
    Convention: C[row, col] = sum_r U[row_a, r] * V[row_b, r] * W[col*s + row, r]
    where row_a = i*s + j (A[i][j]) and row_b = j*s + k (B[j][k]).
    """
    if s == 1:
        U = np.zeros((1, 1))
        V = np.zeros((1, 1))
        W = np.zeros((1, 1))
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        return U, V, W

    h = s // 2
    # Recursive Strassen: 7 products of size h
    sub_U, sub_V, sub_W = _strassen_block(h)
    # sub_U shape (h*h, rank_sub), each product r uses sub_U[i,h+j], sub_V[j,h+k],
    # contributing to C block (bi, bj) at local (i, k) with coefficient sub_W[k*h + i, r].
    rank_sub = sub_U.shape[1]
    rank = 7 * rank_sub

    U = np.zeros((s * s, rank))
    V = np.zeros((s * s, rank))
    W = np.zeros((s * s, rank))

    # Strassen 7 products (A, B are h x h blocks; C block (bi,bj) local (i,k)):
    # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
    # P2 = (A10+A11)B00       -> C10 += P2
    # P3 = A00(B01-B11)       -> C01 += P3
    # P4 = A11(B10-B00)       -> C11 += P4
    # P5 = (A00+A01)B11       -> C00 += P5, C01 -= P5
    # P6 = (A10-A00)(B00+B01) -> C10 += P6
    # P7 = (A01-A11)(B10+B11) -> C11 -= P7
    #
    # Each product Pq = sum over local (i,j,k): A[bi][i][j] * B[bj'][j][k] * (coeff for C[bi][bj][i][k])
    # We express each product as a linear combination of the sub-decomposition terms.

    def block_row_a(bi, i, j):
        return (bi * h + i) * s + (bi * h + j)

    def block_row_b(bj, j, k):
        return (bj * h + j) * s + (bj * h + k)

    def block_row_c(bi, bj, i, k):
        # C row index = i*s + k, but we need W indexed by col*s + row = (bi*h+k)*s + (bi*h+i)
        row = bi * h + i
        col = bj * h + k
        return col * s + row

    # For each product q, the contributing A-blocks and B-blocks and C-blocks with signs.
    # product q: list of (A_block_i, A_block_j, B_block_j, B_block_k, sign, C_block_i, C_block_j)
    products = [
        # P1
        [(0, 0, 0, 0, +1, 0, 0), (0, 0, 1, 1, +1, 0, 0), (1, 1, 0, 0, +1, 1, 1), (1, 1, 1, 1, +1, 1, 1)],
        # P2
        [(1, 0, 0, 0, +1, 1, 0), (1, 1, 0, 0, +1, 1, 0)],
        # P3
        [(0, 0, 0, 1, +1, 0, 1), (0, 0, 1, 1, -1, 0, 1)],
        # P4
        [(1, 1, 1, 0, +1, 1, 1), (1, 1, 0, 0, -1, 1, 1)],
        # P5
        [(0, 0, 1, 1, +1, 0, 0), (0, 1, 1, 1, +1, 0, 0), (0, 0, 1, 1, -1, 0, 1), (0, 1, 1, 1, -1, 0, 1)],
        # P6
        [(1, 0, 0, 0, +1, 1, 0), (1, 0, 0, 1, +1, 1, 0), (0, 0, 0, 0, -1, 1, 0), (0, 0, 0, 1, -1, 1, 0)],
        # P7
        [(0, 1, 1, 0, +1, 1, 1), (0, 1, 1, 1, +1, 1, 1), (1, 1, 1, 0, -1, 1, 1), (1, 1, 1, 1, -1, 1, 1)],
    ]

    for q, terms in enumerate(products):
        base = q * rank_sub
        for (ai, aj, bj, bk, sign, ci, cj) in terms:
            ra = block_row_a(ai, aj, aj)  # A[ai][?][?] -> row index in A: (ai*h + ?) * s + (ai*h + ?)
            # Actually A block (ai, aj): A[ai*h + r][ai*h + c] for r,c in [0,h)
            # We need to add: U[A_row, base+r_sub] += sign * sub_U[local_r, r_sub]
            # where local_r = r*h + c for the sub-decomposition of the product matrix
            # Product matrix M_a = A_block(ai, aj) has entries M_a[r][c] = A[ai*h+r][ai*h+c]
            # sub_U for this product: sub_U[r*h + c, r_sub]
            for r in range(h):
                for c in range(h):
                    local_idx = r * h + c
                    A_row = (ai * h + r) * s + (ai * h + c)
                    B_row = (bj * h + c) * s + (bj * h + 0)  # placeholder, fixed below
            # The above inner loop is messy; rewrite cleanly below.

    # --- Clean reconstruction (overwrite the messy loop) ---
    U = np.zeros((s * s, rank))
    V = np.zeros((s * s, rank))
    W = np.zeros((s * s, rank))

    for q, terms in enumerate(products):
        base = q * rank_sub
        for (ai, aj, bj, bk, sign, ci, cj) in terms:
            # Product matrix M_a[r][c] = A[ai*h+r][ai*h+c]
            # Product matrix M_b[c][k] = B[bj*h+c][bj*h+k]
            # Contribution to C[ci*h+r][cj*h+k] += sign * M_a[r][c] * M_b[c][k]
            # In sub-decomposition of product (size h): sub_U[r*h+c, rsub], sub_V[c*h+k, rsub], sub_W[k*h+r, rsub]
            for r in range(h):
                for c in range(h):
                    for k in range(h):
                        rsub_idx = r * h + c
                        vsub_idx = c * h + k
                        wsub_val = sub_W[k * h + r]  # vector over rsub
                        if not np.any(wsub_val):
                            continue
                        A_row = (ai * h + r) * s + (ai * h + c)
                        B_row = (bj * h + c) * s + (bj * h + k)
                        C_row = (cj * h + k) * s + (ci * h + r)  # col*s + row
                        # For each sub rank index:
                        for rsub in range(rank_sub):
                            wv = wsub_val[rsub]
                            if wv == 0:
                                continue
                            U[A_row, base + rsub] += sign * sub_U[rsub_idx, rsub] * 1
                            V[B_row, base + rsub] += sub_V[vsub_idx, rsub] * 1
                            W[C_row, base + rsub] += wv
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check exact decomposition: C[i][k] = sum_j A[i][j]B[j][k] for symbolic basis.
    Uses the identity: the tensor T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    must equal 1 iff (i,j,k) is a valid triple (always valid here) -- actually T should equal the
    multiplication tensor: T[i,j,k] = 1 for all i,j,k (since C[i][k]=sum_j A[i][j]B[j][k]).
    """
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # Use einsum for speed.
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Reshape U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # Expected: all ones
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)


def construct():
    n, m, p = DIMENSIONS
    # Primary: recursive Strassen (rank 49 for 4x4)
    try:
        U, V, W = _strassen_block(n)
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook rank n*m*p (guaranteed correct)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

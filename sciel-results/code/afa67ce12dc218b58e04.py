"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]
import time

# EVOLVE-BLOCK-START
def _add_term(U, V, W, i, j, k, col):
    U[i * 4 + j][col] = 1
    V[j * 6 + k][col] = 1
    W[k * 3 + i][col] = 1

def _strassen_block(U, V, W, i0, i1, j0, j1, k0, k1, col):
    """Add Strassen terms for 2x2x2 block with indices [i0,i1]x[j0,j1]x[k0,k1].
    U rows indexed by i*m+j, V rows by j*p+k, W rows by k*n+i.
    n=3, m=4, p=6.
    """
    n, m, p = 3, 4, 6
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(([(i0,j0),(i1,j1)], [(j0,k0),(j1,k1)], [(k0,i0),(k1,i1)], 1))
    # M2 = (A10+A11)B00
    terms.append(([(i1,j0),(i1,j1)], [(j0,k0)], [(k0,i0),(k0,i1)], 1))
    # M3 = A00(B10-B11)
    terms.append(((i0,j0), [(j1,k0),(j1,k1)], [(k1,i0),(k1,i1)], 1))
    # M4 = A11(B01-B10)
    terms.append(((i1,j1), [(j0,k1),(j1,k0)], [(k0,i1),(k1,i0)], 1))
    # M5 = (A00+A01)B11
    terms.append(([(i0,j0),(i0,j1)], [(j1,k1)], [(k1,i0),(k1,i1)], 1))
    # M6 = (A10-A00)(B00+B01)
    terms.append(([(i1,j0),(i0,j0)], [(j0,k0),(j0,k1)], [(k0,i0),(k1,i0)], 1))
    # M7 = (A01-A11)(B10+B11)
    terms.append(([(i0,j1),(i1,j1)], [(j1,k0),(j1,k1)], [(k0,i1),(k1,i1)], 1))

    for u_list, v_list, w_list, sign in terms:
        if isinstance(u_list, tuple):
            u_list = [u_list]
        for (u_i, u_j) in u_list:
            for (v_j, v_k) in v_list:
                for (w_k, w_i) in w_list:
                    # u_i should be i, u_j should be j
                    # v_j should be j, v_k should be k
                    # w_k should be k, w_i should be i
                    if u_i != w_i or u_j != v_j:
                        continue  # shouldn't happen
                    _add_term(U, V, W, w_i, u_j, w_k, col)
                    col += 1
    return col

def construct():
    n, m, p = 3, 4, 6
    # Strassen decomposition:
    # Split into 2x2 blocks in i and j, with k split into 3 pairs
    # i: {0,1},{2}  j: {0,1},{2,3}  k: {0,1},{2,3},{4,5}
    # For 2x2x2 blocks: rank 7
    # For 1x2x2 blocks: rank 4 (just do 1*2*2 schoolbook)
    # Actually let's use Strassen where both i and j have size 2
    # Block (i={0,1}, j={0,1}, k={0,1}): Strassen, rank 7
    # Block (i={0,1}, j={0,1}, k={2,3}): Strassen, rank 7
    # Block (i={0,1}, j={0,1}, k={4,5}): Strassen, rank 7
    # Block (i={0,1}, j={2,3}, k={0,1}): Strassen, rank 7
    # Block (i={0,1}, j={2,3}, k={2,3}): Strassen, rank 7
    # Block (i={0,1}, j={2,3}, k={4,5}): Strassen, rank 7
    # Block (i={2}, j={0,1}, k={0,1}): 1x2x2 schoolbook, rank 4
    # Block (i={2}, j={0,1}, k={2,3}): 1x2x2 schoolbook, rank 4
    # Block (i={2}, j={0,1}, k={4,5}): 1x2x2 schoolbook, rank 4
    # Block (i={2}, j={2,3}, k={0,1}): 1x2x2 schoolbook, rank 4
    # Block (i={2}, j={2,3}, k={2,3}): 1x2x2 schoolbook, rank 4
    # Block (i={2}, j={2,3}, k={4,5}): 1x2x2 schoolbook, rank 4
    # Total: 6*7 + 6*4 = 42 + 24 = 66

    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Strassen blocks for i in {0,1}, j in {0,1} or {2,3}, k in {0,1},{2,3},{4,5}
    for i0, i1 in [(0, 1)]:
        for j0, j1 in [(0, 1), (2, 3)]:
            for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
                col = _strassen_block(U, V, W, i0, i1, j0, j1, k0, k1, col)

    # Schoolbook blocks for i={2}
    for j0, j1 in [(0, 1), (2, 3)]:
        for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
            for i in [2]:
                for j in range(j0, j1 + 1):
                    for k in range(k0, k1 + 1):
                        _add_term(U, V, W, i, j, k, col)
                        col += 1

    # col should be 66
    assert col == rank, f"Expected {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

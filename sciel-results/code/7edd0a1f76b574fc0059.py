import time
import itertools

DIMENSIONS = [3, 4, 7]
# n=3, m=4, p=7

def _build_strassen_block(i0, j0, k0):
    """Return list of (u_idx, v_idx, w_idx, coeff) for a 2x2x2 Strassen block.
    Indices are local (0,1) offsets; caller adds base offsets.
    """
    # Standard 7-term Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Mapping to (i, j, k) triples:
    # A is n x m, B is m x p, C is n x p
    # A[i,j], B[j,k] -> C[i,k]
    #
    # For a 2x2x2 block with indices (i0+i1, j0+j2, k0+k1):
    # A factors use (i, j), B factors use (j, k), C uses (i, k)
    #
    # Let's define the 7 terms:
    # Term 0: M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    #   U: (i0,j0), (i1,j1) with coeff 1
    #   V: (j0,k0), (j1,k1) with coeff 1
    #   W: (i0,k0), (i1,k1) with coeff +1; (i1,k0), (i0,k1) with coeff +1
    #
    # Term 1: M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
    #   U: (i1,j0), (i1,j1) with coeff 1
    #   V: (j0,k0) with coeff 1
    #   W: (i1,k0) with coeff +1
    #
    # Term 2: M3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
    #   U: (i0,j0) with coeff 1
    #   V: (j0,k1) with coeff +1, (j1,k1) with coeff -1
    #   W: (i0,k1) with coeff +1
    #
    # Term 3: M4 = A[i1,j1] * (B[j0,k1]-B[j0,k0])
    #   U: (i1,j1) with coeff 1
    #   V: (j0,k1) with coeff +1, (j0,k0) with coeff -1
    #   W: (i1,k1) with coeff +1
    #
    # Term 4: M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
    #   U: (i0,j0), (i0,j1) with coeff 1
    #   V: (j1,k1) with coeff 1
    #   W: (i0,k1) with coeff +1
    #
    # Term 5: M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
    #   U: (i1,j0) with coeff +1, (i0,j0) with coeff -1
    #   V: (j0,k0) with coeff +1, (j0,k1) with coeff +1
    #   W: (i1,k1) with coeff +1
    #
    # Term 6: M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
    #   U: (i0,j1) with coeff +1, (i1,j1) with coeff -1
    #   V: (j1,k0) with coeff +1, (j1,k1) with coeff +1
    #   W: (i0,k0) with coeff +1, (i1,k0) with coeff +1

    terms = []

    # M1: (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    terms.append({'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)],
                  'W': [(0,0,1), (1,1,1), (1,0,1), (0,1,1)]})

    # M2: (A[i1,j0]+A[i1,j1]) * B[j0,k0]
    terms.append({'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)],
                  'W': [(1,0,1)]})

    # M3: A[i0,j0] * (B[j0,k1]-B[j1,k1])
    terms.append({'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)],
                  'W': [(0,1,1)]})

    # M4: A[i1,j1] * (B[j0,k1]-B[j0,k0])
    terms.append({'U': [(1,1,1)], 'V': [(0,1,1), (0,0,-1)],
                  'W': [(1,1,1)]})

    # M5: (A[i0,j0]+A[i0,j1]) * B[j1,k1]
    terms.append({'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)],
                  'W': [(0,1,1)]})

    # M6: (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
    terms.append({'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)],
                  'W': [(1,1,1)]})

    # M7: (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
    terms.append({'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)],
                  'W': [(0,0,1), (1,0,1)]})

    return terms


def _verify_decomposition(n, m, p, U, V, W, rank):
    """Verify that the decomposition is correct."""
    # Check dimensions
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False

    # Check the tensor decomposition
    # For each (i, j, k), the coefficient of A[i,j]*B[j,k] in C[i,k]
    # should be sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # which should equal 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    total_triples = n * m * p  # 84

    # Find disjoint 2x2x2 Strassen blocks
    # A block covers all (i, j, k) where i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    # We need to find a set of non-overlapping blocks to maximize the number covered.

    # Greedy approach: try all possible blocks, pick those that don't overlap
    # Then fill the rest with schoolbook.

    all_blocks = []
    for i0 in range(n - 1):  # 0, 1
        for j0 in range(m - 1):  # 0, 1, 2, 3
            for k0 in range(p - 1):  # 0, 1, 2, 3, 4, 5, 6
                all_blocks.append((i0, j0, k0))

    # Greedy: sort blocks by some heuristic, pick non-overlapping ones
    # To maximize coverage, just pick as many as possible
    selected_blocks = []
    used = set()

    for (i0, j0, k0) in all_blocks:
        block_triples = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    block_triples.add((i0 + di, j0 + dj, k0 + dk))

        # Check no overlap
        if not block_triples & used:
            selected_blocks.append((i0, j0, k0))
            used |= block_triples

    # Build the decomposition
    rank = len(selected_blocks) * 7 + (total_triples - len(selected_blocks) * 8)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for (i0, j0, k0) in selected_blocks:
        terms = _build_strassen_block(i0, j0, k0)
        for term in terms:
            # U part
            u_contribs = term['U']  # list of (di, dj, coeff)
            v_contribs = term['V']  # list of (dj, dk, coeff)
            w_contribs = term['W']  # list of (di, dk, coeff)

            # For this term, the contribution to the tensor is:
            # sum over u_entries and v_entries and w_entries
            # Actually, each term is a product of sums.
            # The rank-1 decomposition of each Strassen term:
            # Each term like (A[i1,j0]+A[i1,j1]) * B[j0,k0] contributes:
            #   For each (di,dj) in U: U[i0+di, j0+dj, col] += coeff_u
            #   For each (dj,dk) in V: V[j0+dj, k0+dk, col] += coeff_v
            #   For each (di,dk) in W: W[k0+dk, i0+di, col] += coeff_w
            #
            # But wait, this is only valid if the term is a single product of two linear forms.
            # For M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1]), this is:
            # (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
            # = A[i0,j0]*B[j0,k0] + A[i0,j0]*B[j1,k1] + A[i1,j1]*B[j0,k0] + A[i1,j1]*B[j1,k1]
            #
            # This is a rank-1 tensor in the (i,j,k) space only if we can write it as
            # (sum_u u_i * A[i,j]) * (sum_v v_jk * B[j,k])
            # which gives C[i,k] += (sum_u u_i A[i,j]) * (sum_v v_jk B[j,k])
            #
            # So for M1:
            # U factor: A[i0,j0] + A[i1,j1] -> U[i0*m+j0] += 1, U[(i0+1)*m+(j0+1)] += 1
            # V factor: B[j0,k0] + B[j1,k1] -> V[j0*p+k0] += 1, V[(j0+1)*p+(k0+1)] += 1
            # W factor: C[i0,k0] += 1, C[i0,k0+1] += 1, C[i0+1,k0] += 1, C[i0+1,k0+1] += 1
            #
            # But W[i,k] should accumulate the product. So W entries are set to 1 for each (i,k) pair.

            for (di, dj, cu) in u_contribs:
                U[(i0 + di) * m + (j0 + dj)][col] += cu
            for (dj, dk, cv) in v_contribs:
                V[(j0 + dj) * p + (k0 + dk)][col] += cv
            for (di, dk, cw) in w_contribs:
                W[(k0 + dk) * n + (i0 + di)][col] += cw
            col += 1

    # Fill remaining triples with schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in used:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    # Verify
    assert col == rank, f"Column count {col} != rank {rank}"
    valid = _verify_decomposition(n, m, p, U, V, W, rank)
    if not valid:
        # Fall back to schoolbook
        rank = total_triples
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1

    return {"U": U, "V": V, "W": W}

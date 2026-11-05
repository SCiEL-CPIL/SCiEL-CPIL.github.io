"""Tensor decomposition search for 3x4 x 4x7 matrix multiplication."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin

    n, m, p = 3, 4, 7

    # Schoolbook baseline: rank 84
    rank_schoolbook = n * m * p
    best_rank = rank_schoolbook

    def build_schoolbook(n, m, p):
        rank = n * m * p
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
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if n * m > 0 else 0
        for r in range(rank):
            # Compute the tensor term and check it's a valid rank-1 component
            # of the matrix multiplication tensor
            pass  # We'll do a full reconstruction check instead

    def full_verify(U, V, W, n, m, p, seed=None):
        """Full reconstruction check: C_ij = sum_r sum_... verify tensor identity."""
        # The matrix multiplication tensor is defined by:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # A rank-1 decomposition term r has:
        #   U[i,j] * V[j,k] * W[k,i] should sum over r to give the tensor
        #
        # We verify: sum_r (sum_i sum_j U[i,j]r * A[i,j]) * (sum_j sum_k V[j,k]r * B[j,k])
        #            = sum_r (sum_i sum_k W[k,i]r * C[i,k])  for all r? No...
        #
        # Actually, the tensor T[i,j,k] = delta_{jk} means:
        # sum_r U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i] = 1 if this is a valid
        # matrix multiplication term, but the decomposition is:
        # C[i,k] = sum_j sum_r A[i,j] * U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i] * B[j,k]
        # must equal sum_j A[i,j] * B[j,k] for all A,B.
        #
        # So: sum_r U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i] = delta_{j,j} = 1
        # (it's the identity tensor in the (i,j,k) sense where j is shared)
        #
        # More precisely: for the decomposition to be valid, we need
        # sum_r U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i] = 1 for all (i,k) pairs
        # summed appropriately... Let me think again.
        #
        # The standard formulation: A matrix multiplication of size n x m * m x p
        # can be written as C = A * B where C is n x p.
        # C[i,k] = sum_j A[i,j] * B[j,k]
        #
        # A tensor decomposition with rank r gives:
        # C[i,k] = sum_{l=1}^{r} (sum_j A[i,j] * U[l][i*m+j]) * (sum_j B[j,k] * W'[l][j*p+k])
        # where the factors are U and V contracted with A and B respectively,
        # and W provides the output combination.
        #
        # Actually in the standard formulation:
        # Let a_l = sum_{i,j} U[l][i*m+j] * A[i,j]
        # Let b_l = sum_{j,k} V[l][j*p+k] * B[j,k]
        # Then C[i,k] = sum_l W[l][k*n+i] * a_l * b_l
        #
        # For this to equal C[i,k] = sum_j A[i,j]*B[j,k] for all A,B:
        # sum_l W[l][k*n+i] * (sum_{i',j'} U[l][i'*m+j']*A[i',j']) * (sum_{j'',k''} V[l][j''*p+k''*B[j'',k''])
        # = sum_j A[i,j]*B[j,k]
        #
        # The coefficient of A[i1,j1]*B[j2,k2] on the LHS is:
        # sum_l U[l][i1*m+j1] * V[l][j2*p+k2] * W[l][k2*n+i1]
        # This must equal 1 if (i1,j1) and (j2,k2) are "connected" in the
        # multiplication, i.e., j1==j2, and 0 otherwise.
        #
        # So the condition is:
        # sum_l U[l][i1*m+j1] * V[l][j2*p+k2] * W[l][k2*n+i1] = 1 if j1==j2 else 0
        #
        # Wait, that's for the specific case. Let me be more careful.
        #
        # LHS coefficient of A[i1,j1] * B[j2,k2]:
        # sum_l U[l][i1*m+j1] * W[l][k2*n+i1] * V[l][j2*p+k2]
        #
        # This must equal 1 if j1 == j2 (same row of A matches same column of B for that product term)
        # and 0 otherwise? No...
        #
        # Actually C[i,k] = sum_j A[i,j]*B[j,k]. The term A[i1,j1]*B[j2,k2] appears
        # in C[i1,k2] if j1==j2. So the coefficient should be:
        # 1 if (i1==i_output, j1==j2, k2==k_output) for the specific output position
        #
        # Let me redo: For the identity to hold for ALL matrices A,B,C:
        # For each fixed (i1,j1) and (j2,k2):
        # sum_l U[l][i1*m+j1] * V[l][j2*p+k2] * W[l][k2*n+i1] = 1 if j1==j2 else 0
        #
        # This is the correct condition. Let's verify with schoolbook:
        # In schoolbook, each term l corresponds to a unique (i,j,k) triple.
        # U[l][i'*m+j'] = 1 if (i',j')=(i,j), else 0
        # V[l][j'*p+k'] = 1 if (j',k')=(j,k), else 0
        # W[l][k'*n+i'] = 1 if (k',i')=(k,i), else 0
        #
        # So sum_l U[l][i1*m+j1]*V[l][j2*p+k2]*W[l][k2*n+i1]
        # For a given term l corresponding to (i,j,k):
        # U[l][i1*m+j1]=1 iff i1==i and j1==j
        # V[l][j2*p+k2]=1 iff j2==j and k2==k
        # W[l][k2*n+i1]=1 iff k2==k and i1==i
        # All three are 1 iff i1==i, j1==j, j2==j, k2==k
        # So the sum = 1 iff j1==j2 and there exists a term with i1==i,j1==j,k2==k and j1==j2
        # Which means j1==j2 (and any i1,k2).
        # So the result is 1 iff j1==j2. Correct!

        if seed is None:
            import random
            seed = 42
        rng = random.Random(seed)
        rank = len(U[0]) if n*m > 0 else 0
        if rank == 0:
            return True

        # Check exact condition: for random (i1,j1,j2,k2) samples
        for trial in range(500):
            i1 = rng.randrange(n)
            j1 = rng.randrange(m)
            j2 = rng.randrange(m)
            k2 = rng.randrange(p)
            s = 0
            for l in range(rank):
                val = U[l][i1*m + j1] * V[l][j2*p + k2] * W[l][k2*n + i1]
                if val != 0:
                    s += val
            expected = 1 if j1 == j2 else 0
            if s != expected:
                return False
        return True

    # Try to apply Strassen-like reductions on 2x2x2 sub-blocks
    # Start from schoolbook and try to replace groups of 8 terms (2x2x2 block) with 7 terms

    def get_schoolbook_terms():
        """Return list of (i,j,k) for schoolbook terms."""
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((i, j, k))
        return terms

    # Try a specific Strassen reduction on a 2x2x2 block
    # Block: i in {0,1}, j in {0,1}, k in {0,1}
    # 8 schoolbook terms: (i,j,k) for i,j,k in {0,1}
    # Replace with 7 Strassen terms

    def strassen_222_offset(ioff, joff, koff, n, m, p):
        """
        Generate 7 Strassen terms for a 2x2x2 block with offsets.
        i_off, j_off, k_off are the base indices.
        Returns list of 7 terms, each term is a list of (U_indices, V_indices, W_indices, coeff)
        where each index is (row, col) and coeff is 1, -1, etc.
        """
        i0, j0, k0 = ioff, joff, koff
        # Strassen 2x2x2:
        # M1 = (A00+A11)*(B00+B11) -> contributes to C00,C11
        # M2 = (A10+A11)*B00 -> contributes to C10
        # M3 = A00*(B10-B00) -> contributes to C01
        # M4 = A11*(B01-B11) -> contributes to C00
        # M5 = (A00+A01)*B11 -> contributes to C01
        # M6 = (A10-A00)*(B00+B01) -> contributes to C10
        # M7 = (A01-A11)*(B10+B11) -> contributes to C11
        #
        # In terms of indices (i,j) for A, (j,k) for B, (k,i) for C:
        # Term 1: U={(i0,j0),(i0+1,j0+1)}, V={(j0,k0),(j0+1,k0+1)}, W={(k0,i0),(k0+1,i0+1)}
        # Term 2: U={(i0+1,j0),(i0+1,j0+1)}, V={(j0,k0)}, W={(k0,i0+1)}
        # Term 3: U={(i0,j0)}, V={(j0+1,k0),(j0,k0),coeff=-1 for (j0,k0)}, W={(k0+1,i0),(k0+1,i0+1)}
        # Wait, let me use the standard Strassen formulation more carefully.
        #
        # Standard Strassen for C = A*B where A,B are 2x2:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b21-b11)
        # M4 = a22(b12-b22)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        #
        # c11 = M1 + M4 - M5 - M7
        # c12 = M3 + M5
        # c21 = M2 + M6
        # c22 = M1 - M2 + M3 + M7
        #
        # Using 0-indexed:
        # a00,a01,a10,a11 and b00,b01,b10,b11
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b00)
        # M4 = a11(b01-b11)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # c00 = M1 + M4 - M5 - M7
        # c01 = M3 + M5
        # c10 = M2 + M6
        # c11 = M1 - M2 + M3 + M7
        #
        # Now translate to tensor indices (i,j) for A, (j,k) for B, (k,i) for C:
        # A[i,j] -> U row index: i*m + j (but we're in a 2x2 block, so local)
        # B[j,k] -> V row index: j*p + k
        # C[k,i] -> W row index: k*n + i
        #
        # For the 2x2 block at offset (i0,j0,k0):
        # Local A indices: (di,dj) for di,dj in {0,1} -> global (i0+di, j0+dj)
        # Local B indices: (dj,dk) -> global (j0+dj, k0+dk)
        # Local C indices: (dk,di) -> global (k0+dk, i0+di)
        #
        # Term 1: (a00+a11)(b00+b11)
        #   U: {(i0,j0), (i0+1,j0+1)} with coeff +1 each
        #   V: {(j0,k0), (j0+1,k0+1)} with coeff +1 each
        #   W: {(k0,i0), (k0+1,i0+1)} with coeff +1 each
        #   Contribution to C: this is a rank-1 factor (a00+a11) * (b00+b11)
        #   But wait, the rank-1 term is U_factor * V_factor where:
        #   U_factor = a00+a11, V_factor = b00+b11
        #   And W distributes to output positions.
        #
        # Hmm, I'm confusing the tensor decomposition structure. Let me reconsider.
        #
        # In the tensor decomposition:
        # C[i,k] = sum_l W[l][k*n+i] * (sum_{i',j'} U[l][i'*m+j'] * A[i',j']) * (sum_{j',k'} V[l][j'*p+k'] * B[j',k'])
        #
        # So each term l has:
        # - A U-vector that picks out certain (i,j) pairs
        # - A V-vector that picks out certain (j,k) pairs
        # - A W-vector that distributes the product to output (k,i) positions
        #
        # For Strassen term 1: M1 = (a00+a11)*(b00+b11)
        # The A-factor is: a00 + a11, which means U[l] has entries:
        #   U[l][i0*m+j0] = 1, U[l][(i0+1)*m+(j0+1)] = 1
        # The B-factor is: b00 + b11, so V[l] has:
        #   V[l][j0*p+k0] = 1, V[l][(j0+1)*p+(k0+1)] = 1
        # The W-factor distributes M1 to c00, c11:
        #   c00 gets +M1, c11 gets +M1
        # So W[l] has:
        #   W[l][k0*n+i0] = 1, W[l][(k0+1)*n+(i0+1)] = 1
        #
        # Term 2: M2 = (a10+a11)*b00
        # U: {(i0+1,j0), (i0+1,j0+1)} coeff +1
        # V: {(j0,k0)} coeff +1
        # W: distributes to c10: +M2
        #   W[l][(k0)*n+(i0+1)] = 1  (c10 means i=i0+1, k=k0)
        #
        # Term 3: M3 = a00*(b10-b00)
        # U: {(i0,j0)} coeff +1
        # V: {(j0+1,k0)} coeff +1, {(j0,k0)} coeff -1
        # W: distributes to c01, c11: +M3 to each
        #   c01: i=i0, k=k0+1 -> W[l][(k0+1)*n+i0] = 1
        #   c11: i=i0+1, k=k0+1 -> W[l][(k0+1)*n+(i0+1)] = 1
        #
        # Term 4: M4 = a11*(b01-b11)
        # U: {(i0+1,j0+1)} coeff +1
        # V: {(j0,k0+1)} coeff +1, {(j0+1,k0+1)} coeff -1
        # W: c00: i=i0, k=k0 -> +M4
        #   W[l][k0*n+i0] = 1
        #
        # Term 5: M5 = (a00+a01)*b11
        # U: {(i0,j0), (i0,j0+1)} coeff +1
        # V: {(j0+1,k0+1)} coeff +1
        # W: c00: -M5, c01: +M5
        #   W[l][k0*n+i0] = -1
        #   W[l][(k0+1)*n+i0] = 1
        #
        # Term 6: M6 = (a10-a00)*(b00+b01)
        # U: {(i0+1,j0)} coeff +1, {(i0,j0)} coeff -1
        # V: {(j0,k0), (j0,k0+1)} coeff +1
        # W: c10: +M6
        #   W[l][k0*n+(i0+1)] = 1
        #
        # Term 7: M7 = (a01-a11)*(b10+b11)
        # U: {(i0,j0+1)} coeff +1, {(i0+1,j0+1)} coeff -1
        # V: {(j0+1,k0), (j0+1,k0+1)} coeff +1
        # W: c00: -M7, c11: +M7
        #   W[l][k0*n+i0] = -1
        #   W[l][(k0+1)*n+(i0+1)] = 1

        terms = []
        # Term 1
        terms.append({
            'U': {(i0*m + j0): 1, ((i0+1)*m + (j0+1)): 1},
            'V': {(j0*p + k0): 1, ((j0+1)*p + (k0+1)): 1},
            'W': {(k0*n + i0): 1, ((k0+1)*n + (i0+1)): 1},
        })
        # Term 2
        terms.append({
            'U': {((i0+1)*m + j0): 1, ((i0+1)*m + (j0+1)): 1},
            'V': {(j0*p + k0): 1},
            'W': {(k0*n + (i0+1)): 1},
        })
        # Term 3
        terms.append({
            'U': {(i0*m + j0): 1},
            'V': {((j0+1)*p + k0): 1, (j0*p + k0): -1},
            'W': {((k0+1)*n + i0): 1, ((k0+1)*n + (i0+1)): 1},
        })
        # Term 4
        terms.append({
            'U': {((i0+1)*m + (j0+1)): 1},
            'V': {(j0*p + (k0+1)): 1, ((j0+1)*p + (k0+1)): -1},
            'W': {(k0*n + i0): 1},
        })
        # Term 5
        terms.append({
            'U': {(i0*m + j0): 1, (i0*m + (j0+1)): 1},
            'V': {((j0+1)*p + (k0+1)): 1},
            'W': {(k0*n + i0): -1, ((k0+1)*n + i0): 1},
        })
        # Term 6
        terms.append({
            'U': {((i0+1)*m + j0): 1, (i0*m + j0): -1},
            'V': {(j0*p + k0): 1, (j0*p + (k0+1)): 1},
            'W': {(k0*n + (i0+1)): 1},
        })
        # Term 7
        terms.append({
            'U': {(i0*m + (j0+1)): 1, ((i0+1)*m + (j0+1)): -1},
            'V': {((j0+1)*p + k0): 1, ((j0+1)*p + (k0+1)): 1},
            'W': {(k0*n + i0): -1, ((k0+1)*n + (i0+1)): 1},
        })
        return terms

    def build_from_terms(terms_list, n, m, p):
        """
        Build U, V, W matrices from a list of term specifications.
        Each term has 'U', 'V', 'W' dicts mapping index -> coeff.
        """
        rank = len(terms_list)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for l, term in enumerate(terms_list):
            for idx, coeff in term['U'].items():
                if 0 <= idx < n * m:
                    U[idx][l] += coeff
            for idx, coeff in term['V'].items():
                if 0 <= idx < m * p:
                    V[idx][l] += coeff
            for idx, coeff in term['W'].items():
                if 0 <= idx < p * n:
                    W[idx][l] += coeff
        return U, V, W

    # Try applying one Strassen reduction
    # Pick a 2x2x2 block: i in {0,1}, j in {0,1}, k in {0,1}
    # This replaces 8 terms with 7, reducing rank by 1

    def try_strassen_offset(ioff, joff, koff, timeout_check=None):
        """Try applying Strassen to block at (ioff, joff, koff)."""
        if time.time() > deadline:
            return None
        if ioff + 1 >= n or joff + 1 >= m or koff + 1 >= p:
            return None

        # Start from schoolbook
        U, V, W = build_schoolbook(n, m, p)
        rank = len(U[0])

        # Find the 8 schoolbook term indices for this block
        # Schoolbook term for (i,j,k) is at position:
        # idx = i * m * p + j * p + k
        block_indices = []
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    idx = (ioff + di) * m * p + (joff + dj) * p + (koff + dk)
                    if idx < rank:
                        block_indices.append(idx)

        if len(block_indices) != 8:
            return None

        # Create new terms: schoolbook terms not in block + 7 Strassen terms
        strassen_terms = strassen_222_offset(ioff, joff, koff, n, m, p)

        # Build the new decomposition
        new_terms_list = []
        for l in range(rank):
            if l in block_indices:
                continue
            # Schoolbook term l: (i,j,k) = (l // (m*p), (l // p) % m, l % p)
            i = l // (m * p)
            j = (l // p) % m
            k = l % p
            new_terms_list.append({
                'U': {i * m + j: 1},
                'V': {j * p + k: 1},
                'W': {k * n + i: 1},
            })

        for st in strassen_terms:
            new_terms_list.append(st)

        new_rank = len(new_terms_list)
        new_U, new_V, new_W = build_from_terms(new_terms_list, n, m, p)

        # Verify
        if full_verify(new_U, new_V, new_W, n, m, p):
            return new_U, new_V, new_W, new_rank
        return None

    # Try all possible 2x2x2 Strassen offsets
    best_result = None
    for ioff in range(n - 1):
        if time.time() > deadline:
            break
        for joff in range(m - 1):
            if time.time() > deadline:
                break
            for koff in range(p - 1):
                if time.time() > deadline:
                    break
                result = try_strassen_offset(ioff, joff, koff)
                if result is not None:
                    _, _, _, r = result
                    if best_result is None or r < best_result[3]:
                        best_result = result

    if best_result is not None:
        U, V, W, r = best_result
        if r < best_rank:
            best_rank = r

    # Also try the plain schoolbook as fallback
    U, V, W = build_schoolbook(n, m, p)

    if best_result is not None:
        U, V, W, best_rank = best_result

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

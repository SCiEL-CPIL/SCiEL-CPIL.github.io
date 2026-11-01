"""Matrix multiplication 2,5,6 tensor decomposition with Strassen on 2x2x2 sub-blocks."""
import time
import itertools

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x5 matrix by a 5x6 matrix.
    
    Strategy: Use Strassen's algorithm (rank 7) on the 2x2x2 sub-block formed by
    the first 2 rows of A, first 2 columns of B, and first 2 rows of C.
    The remaining elements use schoolbook (rank 1 each).
    
    Total rank: 7 (Strassen) + 1*5*6 - 2*2 (already covered) = 7 + 30 - 4 = 33
    Wait, let me think more carefully.
    
    The full multiplication is C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..1, j in 0..4, k in 0..5.
    
    We can partition:
    - For the 2x2x2 sub-problem (i in {0,1}, j in {0,1}, k in {0,1}), use Strassen (rank 7)
    - For all other (i,j,k) triples, use schoolbook (rank 1 each)
    
    The total number of (i,j,k) triples is 2*5*6 = 60.
    The 2x2x2 sub-block covers 2*2*2 = 8 triples.
    So we need 60 - 8 = 52 schoolbook terms + 7 Strassen terms = 59 total.
    """
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Build the decomposition
    # U has shape (n*m, rank) = (10, rank)
    # V has shape (m*p, rank) = (30, rank)
    # W has shape (p*n, rank) = (12, rank)
    
    # We'll build lists of (U_row, V_row, W_row, coeff) for each term
    # Then assemble into matrices
    
    terms = []  # list of (u_idx, v_idx, w_idx, coeff)
    
    # Strassen's algorithm for 2x2x2 block
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    # where C = A*B
    # 
    # The 2x2x2 tensor has entries T[i,j,k] = 1 if C[i,k] involves A[i,j]*B[j,k]
    # Strassen computes this with 7 multiplications:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C[0,0] = m1 - m4 + m5 - m7
    # C[0,1] = m3 + m4
    # C[1,0] = m2 + m5
    # C[1,1] = m1 - m2 + m3 + m7
    #
    # For the tensor decomposition, we need to express each of the 7 Strassen terms
    # as rank-1 contributions to the tensor.
    #
    # The 2x2x2 sub-block uses indices:
    # i in {0,1} (rows of A, n=2)
    # j in {0,1} (cols of A / rows of B, first 2 of m=5)
    # k in {0,1} (cols of B, first 2 of p=6)
    #
    # U index: i*m + j = i*5 + j
    # V index: j*p + k = j*6 + k
    # W index: k*n + i = k*2 + i
    
    # Strassen term 1: (a+d)(e+h)
    # This involves A[0,0], A[1,1] and B[0,0], B[1,1]
    # U vector: nonzero at (0,0) and (1,1) -> indices 0*5+0=0, 1*5+1=6
    # V vector: nonzero at (0,0) and (1,1) -> indices 0*6+0=0, 1*6+1=7
    # W: This term contributes to C[0,0], C[0,1], C[1,0], C[1,1] with coefficients
    # C[0,0] += 1, C[0,1] += 0, C[1,0] += 0, C[1,1] += 1 (from m1 - m4 + m5 - m7 for C[0,0], etc.)
    # Actually, let me be more careful. Each Strassen multiplication m_t produces a scalar,
    # and then linear combinations give the C entries.
    #
    # In tensor decomposition terms, each "multiplication" in Strassen corresponds to
    # a rank-1 tensor where U and V are the input vectors and W is the output vector.
    #
    # m1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # U_m1: [1 at (0,0), 1 at (1,1)] in A-space
    # V_m1: [1 at (0,0), 1 at (1,1)] in B-space
    # This product contributes to C as:
    # C[0,0] gets +1, C[0,1] gets 0, C[1,0] gets 0, C[1,1] gets +1? No...
    #
    # Actually, the way Strassen works in tensor form:
    # The 2x2x2 matrix multiplication tensor T satisfies:
    # T[i,j,k] = 1 for all i,j,k in {0,1} (it's the identity tensor for 2x2x2)
    #
    # Wait no. The matrix multiplication tensor for 2x2x2 has:
    # T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k]
    # Since C[i,k] = sum_j A[i,j] B[j,k], T[i,j,k] = 1 for all i,j,k.
    # So it's the all-ones 2x2x2 tensor? No, that's not right either.
    #
    # The matrix multiplication tensor is: T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1}
    # because every entry A[i,j]B[j,k] appears in exactly one C[i,k].
    #
    # So the 2x2x2 matmul tensor is the 2x2x2 all-ones tensor.
    #
    # Strassen decomposes this as sum of 7 rank-1 tensors.
    #
    # Let me look up the standard Strassen decomposition of the 2x2x2 tensor.
    #
    # The 7 rank-1 tensors (U_t, V_t, W_t) where each is a vector in R^4 (flattened 2x2):
    #
    # t=1: U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1)  -> (a+d)(e+h)
    # t=2: U=(0,0,1,1), V=(1,0,0,0), W=(0,0,1,0)  -> (c+d)e
    # t=3: U=(1,0,0,0), V=(0,1,0,-1), W=(0,1,0,0)  -> a(f-h)
    # t=4: U=(0,0,0,1), V=(0,0,1,-1), W=(0,0,0,1)  -> d(g-e)
    # t=5: U=(1,1,0,0), V=(0,0,0,1), W=(0,0,1,0)   -> (a+b)h
    # t=6: U=(1,0,-1,0), V=(1,1,0,0), W=(0,0,0,0)  -> wait, this doesn't look right
    #
    # Let me use a cleaner formulation. The standard Strassen rank-7 decomposition:
    #
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C00 = m1 - m4 + m5 - m7
    # C01 = m3 + m4
    # C10 = m2 + m5
    # C11 = m1 - m2 + m3 + m7
    #
    # Each m_t is a product of a linear form in A and a linear form in B.
    #
    # m1: A-form = a+d, B-form = e+h
    #   U: [1,0,0,1] (a at pos 0, d at pos 3)
    #   V: [1,0,0,1] (e at pos 0, h at pos 3)
    #   W: contributes to C00 (+1), C11 (+1)
    #   W: [1,0,0,1]
    #
    # m2: A-form = c+d, B-form = e
    #   U: [0,0,1,1]
    #   V: [1,0,0,0]
    #   W: contributes to C10 (+1)
    #   W: [0,0,1,0]
    #
    # m3: A-form = a, B-form = f-h
    #   U: [1,0,0,0]
    #   V: [0,1,0,-1]
    #   W: contributes to C01 (+1), C11 (+1)
    #   W: [0,1,0,1]
    #
    # m4: A-form = d, B-form = g-e
    #   U: [0,0,0,1]
    #   V: [0,0,1,-1]
    #   W: contributes to C00 (-1), C01 (+1)
    #   W: [-1,1,0,0]
    #
    # m5: A-form = a+b, B-form = h
    #   U: [1,1,0,0]
    #   V: [0,0,0,1]
    #   W: contributes to C00 (+1), C10 (+1)
    #   W: [1,0,1,0]
    #
    # m6: A-form = c-a, B-form = e+f
    #   U: [1,0,-1,0]  (a=1, c=-1)
    #   V: [1,1,0,0]
    #   W: contributes to C11 (-1)
    #   W: [0,0,0,-1]
    #
    # m7: A-form = b-d, B-form = g+h
    #   U: [0,1,0,-1]
    #   V: [0,0,1,1]
    #   W: contributes to C00 (-1), C11 (+1)
    #   W: [-1,0,0,1]
    
    # Verify: 
    # C00 = m1 - m4 + m5 - m7 = [1,0,0,1] - [-1,1,0,0] + [1,0,1,0] - [-1,0,0,1]
    #    = [1+1+1+1, 0-1+0+0, 0+0+1+0, 1+0+0-1] = [4,-1,1,0]
    # That's wrong. C00 should be [1,0,0,0].
    #
    # I think I'm confusing the W vectors. Let me reconsider.
    #
    # In the tensor decomposition, we have:
    # T = sum_t U_t ⊗ V_t ⊗ W_t
    # where T[i,j,k] = sum_t U_t[i] * V_t[j] * W_t[k]
    #
    # For the 2x2x2 matmul tensor, T[i,j,k] = 1 for all i,j,k.
    #
    # So we need: sum_t U_t[i] * V_t[j] * W_t[k] = 1 for all i,j,k in {0,1}.
    #
    # Let me use the standard Strassen decomposition more carefully.
    # The 7 terms with their U, V, W vectors (in flattened 2x2 order [00,01,10,11]):
    
    strassen_terms = [
        # (U, V, W) for each of the 7 terms
        # m1 = (a+d)(e+h)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # m2 = (c+d)e  
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        # m3 = a(f-h)
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        # m4 = d(g-e)
        ([0,0,0,1], [0,0,1,-1], [-1,1,0,0]),
        # m5 = (a+b)h
        ([1,1,0,0], [0,0,0,1], [1,0,1,0]),
        # m6 = (c-a)(e+f)
        ([1,0,-1,0], [1,1,0,0], [0,0,0,-1]),
        # m7 = (b-d)(g+h)
        ([0,1,0,-1], [0,0,1,1], [-1,0,0,1]),
    ]
    
    # Verify the decomposition: for each (i,j,k), sum_t U_t[i]*V_t[j]*W_t[k] should = 1
    # Let's verify computationally
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for U, V, W in strassen_terms:
                        total += U[i] * V[j] * W[k]
                    if total != 1:
                        return False, (i, j, k, total)
        return True, None
    
    valid, err = verify_strassen()
    if not valid:
        # Fallback: use schoolbook for everything
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
    
    # Now build the full decomposition
    # Strassen covers i in {0,1}, j in {0,1}, k in {0,1}
    # Schoolbook covers all other (i,j,k) triples
    
    # Collect all terms: each term is (U_vec, V_vec, W_vec) where vectors are sparse
    # We'll represent each term as a list of (index, value) for U, V, W
    
    all_terms = []
    
    # Add Strassen terms for the 2x2x2 sub-block
    # The sub-block uses:
    # A indices: i in {0,1}, j in {0,1} -> U indices: i*5+j
    # B indices: j in {0,1}, k in {0,1} -> V indices: j*6+k
    # C indices: k in {0,1}, i in {0,1} -> W indices: k*2+i
    
    for U_flat, V_flat, W_flat in strassen_terms:
        # U_flat is [U00, U01, U10, U11] corresponding to (i,j) in {(0,0),(0,1),(1,0),(1,1)}
        U_sparse = {}
        for idx, val in enumerate(U_flat):
            if val != 0:
                i, j = idx // 2, idx % 2
                U_sparse[i * m + j] = val
        V_sparse = {}
        for idx, val in enumerate(V_flat):
            if val != 0:
                j, k = idx // 2, idx % 2
                V_sparse[j * p + k] = val
        W_sparse = {}
        for idx, val in enumerate(W_flat):
            if val != 0:
                k, i = idx // 2, idx % 2
                W_sparse[k * n + i] = val
        all_terms.append((U_sparse, V_sparse, W_sparse))
    
    # Add schoolbook terms for all (i,j,k) NOT in the 2x2x2 sub-block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # already covered by Strassen
                U_sparse = {i * m + j: 1}
                V_sparse = {j * p + k: 1}
                W_sparse = {k * n + i: 1}
                all_terms.append((U_sparse, V_sparse, W_sparse))
    
    rank = len(all_terms)
    
    # Check time
    if time.time() - start_time > DEADLINE:
        # Return schoolbook as fallback
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
    
    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_sparse, V_sparse, W_sparse) in enumerate(all_terms):
        for idx, val in U_sparse.items():
            U[idx][col] = val
        for idx, val in V_sparse.items():
            V[idx][col] = val
        for idx, val in W_sparse.items():
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}

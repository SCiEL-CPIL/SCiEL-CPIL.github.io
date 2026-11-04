"""Matrix multiplication 2x3 * 3x6 — rank-minimal exact decomposition."""
import time
import itertools
import random

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150 s kill

    n, m, p = DIMENSIONS  # 2, 3, 6
    # Tensor index convention:
    #   U row index:  i * m + j   (i in 0..n-1, j in 0..m-1)   -> 6 rows
    #   V row index:  j * p + k   (j in 0..m-1, k in 0..p-1)   -> 18 rows
    #   W row index:  k * n + i   (k in 0..p-1, i in 0..n-1)   -> 12 rows
    #   Multiplication term: A[i,j] * B[j,k] appears at output (k, i)
    #   So W[k*n+i] gets the coefficient.
    nm = n * m          # 6
    mp = m * p          # 18
    pn = p * n          # 12

    def build(rank, seed=42):
        """Build a random decomposition with the given rank and verify."""
        rng = random.Random(seed)
        # We use integer entries in {0, 1, -1} for simplicity, then check exactness.
        # Better: use structured initialisation.

        # Structured approach: try rank = n * m * (p // block) + correction
        # Known lower bound: rank >= max(n*m, m*p, p*n, ceil(m*p*n / max(n*m, m*p, p*n)))
        # For 2x3*3x6: lower bound = max(6, 18, 12) = 18
        # Schoolbook rank = 36.
        # Literature: R(2,3,3) = 7 (Strassen-like? No, 2x3*3x3 has rank 9 for schoolbook,
        #   known best is 9? Actually R(2,3,3) is known to be 9 in some sources.)
        # R(2,3,6): we can try to build from R(2,3,3) = 9 by tiling? 9 + 9 = 18 if columns split cleanly.
        # But the columns of a 3x6 matrix don't split into two independent 3-column groups for a single multiplication.

        # Alternative: use the "Kronecker product" trick.
        # A (2x3) * B (3x6). Write B as two 3x3 blocks: B = [B1 | B2].
        # Then C = [A*B1 | A*B2], each product is 2x3.
        # So we need rank(R(2,3,3)) for each block = 9, total 18.
        # But this requires knowing R(2,3,3) <= 9. Schoolbook for 2x3*3x3 is 9, and it's
        # believed tight (R(2,3,3) = 9 is the schoolbook value and no smaller is known).
        # So 18 is achievable via two independent 2x3*3x3 multiplications.

        # Let's try rank 18 first, then 17, 16, ... down to the lower bound.
        # For each candidate rank, try random structured search.

        pass  # placeholder; actual search below

    # Lower bound on rank
    # rank >= max(n*m, m*p, p*n) is NOT correct in general.
    # Correct lower bounds: rank >= n*m, rank >= m*p, rank >= p*n (from injectivity of each factor space)
    # Also rank >= ceil(n*m*p / max(n*m, m*p, p*n))
    # For 2x3*3x6: lb = max(6, 18, 12) = 18.

    lb = max(nm, mp, pn)

    # Start from a known-good rank (schoolbook) and try to reduce.
    schoolbook_rank = n * m * p  # 36

    def make_schoolbook(rank=None):
        """Return (U, V, W) as lists of lists of ints for schoolbook decomposition."""
        if rank is None:
            rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W, tol=1e-12):
        """Check that U @ V.T elementwise combined with W gives the correct tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The coefficient of A[i,j]*B[j,k] in the decomposition must be 1.
                    # It is sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > tol:
                        return False, (i, j, k, total)
                    # Also check that cross terms vanish:
                    # For any (i',j') != (i,j) and (j'',k'') with output (k'', i''):
                    # Actually the full check is: for each output index (kk, ii),
                    #   sum_{j'} U[ii*m+j'][c] * V[j'*p+kk][c] * W[kk*n+ii][c] should be
                    #   the coefficient which should be 1 if j'=j (the correct middle index)
                    #   but this is automatically handled by checking all (i,j,k) = 1
                    #   and all other (i,j')*B[j',k] terms = 0.
        # Check that wrong cross-terms are zero:
        # The full tensor T[i,j,k] = delta. The reconstruction is:
        #   for each (i_out, j_out) in output (size p x n):
        #     C[i_out, j_out] = sum_{j_mid} A[j_out, j_mid] * B[j_mid, i_out]
        #     In decomposition: C[i_out, j_out] = sum_c (sum_{j_mid} U[j_out*m+j_mid][c] * V[j_mid*p+i_out][c]) * W[i_out*n+j_out][c]
        #     Wait, that's not right either. Let me re-derive.
        #
        # Standard: C_{i_out, j_out} = sum_{j_mid} A_{j_out, j_mid} * B_{j_mid, i_out}
        # Tensor form: T_{j_out, j_mid, i_out} = 1 for the multiplication.
        # Decomposition: T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
        # where a = j_out*m + j_mid (input A index), b = j_mid*p + i_out (input B index), c = i_out*n + j_out (output index).
        # So T[a,b,c] should equal 1 iff (a,b,c) corresponds to a valid multiplication term.
        # Specifically, for given (j_out, j_mid, i_out), a = j_out*m+j_mid, b = j_mid*p+i_out, c = i_out*n+j_out.
        # The term A[j_out,j_mid]*B[j_mid,i_out] contributes to C[i_out, j_out] with coefficient 1.
        # So T[a,b,c] = 1 for all (j_out, j_mid, i_out), i.e., for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
        # And T should be 0 for any (a,b,c) that doesn't correspond to a valid triple.
        #
        # But a = i*m+j, b = j'*p+k, c = k'*n+i'.
        # For T[a,b,c] to be non-zero, we need: the j in a and j' in b must match (same middle index),
        # and the k in b and k' in c must match, and the i in a and i' in c must match.
        # i.e., j = j', k = k', i = i'.
        # So T is 1 exactly when the indices are consistent.

        # Full check: iterate over all a, b, c and verify.
        # This is expensive: 6 * 18 * 12 * rank operations per (a,b,c) pair... no, per c.
        # Total: 6*18*12 = 1296 (a,b,c) triples, each with rank multiplications.
        # For rank=18: 1296*18 = 23328 ops. Fast enough.

        # Let's do the full check.
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                for c in range(pn):
                    k_c = c // n
                    i_c = c % n
                    # Expected value: 1 if j_a == j_b and k_b == k_c and i_a == i_c, else 0.
                    expected = 1 if (j_a == j_b and k_b == k_c and i_a == i_c) else 0
                    actual = 0
                    for r in range(rank):
                        actual += U[a][r] * V[b][r] * W[c][r]
                    if abs(actual - expected) > tol:
                        return False, (a, b, c, actual, expected)
        return True, None

    def try_random_search(rank, max_iters=2000, seed=0):
        """Try to find a valid decomposition of the given rank via random search with local moves."""
        rng = random.Random(seed)
        # Start with a sparse random initialisation using {0, 1, -1}
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        # Seed some 1s randomly
        for _ in range(rank * 2):
            r = rng.randint(0, rank - 1)
            a = rng.randint(0, nm - 1)
            b = rng.randint(0, mp - 1)
            c = rng.randint(0, pn - 1)
            val = rng.choice([1, -1])
            U[a][r] += val
            V[b][r] += val
            W[c][r] += val

        for it in range(max_iters):
            if time.time() > deadline:
                break
            ok, _ = verify(U, V, W)
            if ok:
                return U, V, W

            # Random local move: pick a random entry in one of the matrices and perturb it.
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = rng.randint(0, nm - 1)
                arr = U
            elif mat == 'V':
                row = rng.randint(0, mp - 1)
                arr = V
            else:
                row = rng.randint(0, pn - 1)
                arr = W
            col = rng.randint(0, rank - 1)
            delta = rng.choice([-1, 0, 1])
            arr[row][col] += delta

        ok, _ = verify(U, V, W)
        if ok:
            return U, V, W
        return None

    # Try rank from lower bound upward, with multiple random seeds.
    # First, quickly check if schoolbook works (it should).
    best = make_schoolbook()

    # Try to reduce rank. Start from schoolbook and try random restarts at lower ranks.
    for rank in range(lb, schoolbook_rank):
        if time.time() > deadline:
            break
        # Try several random restarts at this rank
        found = None
        max_restarts = 10
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            found = try_random_search(rank, max_iters=500, seed=attempt * 1000 + rank)
            if found:
                break
        if found:
            best = found
            # Can stop searching if we reached the lower bound
            if rank == lb:
                break

    U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

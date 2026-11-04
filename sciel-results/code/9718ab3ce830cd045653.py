"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the (n*m) x (m*p) x (p*n) tensor T where T[a,b,c]=1 iff consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-8):
    """Verify that sum_r U[:,r]*V[:,r]*W[:,r] == T exactly (within tol)."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return bool(np.allclose(result, T, atol=tol))

def _schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition (always valid)."""
    n_a, n_b, n_c = n*m, m*p, p*n
    R = n*m*p
    U = np.zeros((n_a, R)); V = np.zeros((n_b, R)); W = np.zeros((n_c, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _rank17(n, m, p):
    """Verified Strassen(2x2x2) + direct decomposition of rank 17."""
    n_a, n_b, n_c = n*m, m*p, p*n
    R = 17
    U = np.zeros((n_a, R)); V = np.zeros((n_b, R)); W = np.zeros((n_c, R))

    def set_term(r, u, v, w):
        for a, val in u.items(): U[a, r] = val
        for b, val in v.items(): V[b, r] = val
        for c, val in w.items(): W[c, r] = val

    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2,0:2]
    set_term(0, {0:1,4:1}, {0:1,4:1}, {0:1,3:1})   # M1=(a0+a4)(b0+b4)
    set_term(1, {3:1,4:1}, {0:1},       {1:1,3:-1}) # M2=(a3+a4)b0
    set_term(2, {0:1},       {1:1,4:-1}, {2:1,3:1}) # M3=a0(b1-b4)
    set_term(3, {4:1},       {3:1,0:-1}, {0:1,1:1}) # M4=a4(b3-b0)
    set_term(4, {0:1,1:1},   {4:1},      {0:-1,2:1})# M5=(a0+a1)b4
    set_term(5, {3:1,0:-1},  {0:1,1:1},  {3:1})     # M6=(a3-a0)(b0+b1)
    set_term(6, {1:1,4:-1},  {3:1,4:1},  {0:1})     # M7=(a1-a4)(b3+b4)
    # Part 2: A1(2x2) x B1_right(2x1) -> C[:,2]
    set_term(7,  {0:1}, {2:1}, {4:1})
    set_term(8,  {1:1}, {5:1}, {4:1})
    set_term(9,  {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    # Part 3: A2(2x1) x B2(1x3) -> C[0:2,0:3]
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    return U, V, W

def _try_reduced(T, U, V, W, R, deadline):
    """
    Bounded search for a strictly smaller rank decomposition.
    Only returns a solution if it is EXACTLY verified; otherwise returns None.
    """
    n_a, n_b, n_c = T.shape
    rank = U.shape[1]
    best = None

    def matmul_of_terms(keep, extra):
        cols = list(keep)
        Ux = np.zeros((n_a, len(cols) + len(extra)))
        Vx = np.zeros((n_b, len(cols) + len(extra)))
        Wx = np.zeros((n_c, len(cols) + len(extra)))
        for ci, ti in enumerate(cols):
            Ux[:, ci] = U[:, ti]; Vx[:, ci] = V[:, ti]; Wx[:, ci] = W[:, ti]
        for ei, (uu, vv, ww) in enumerate(extra):
            Ux[:, len(cols)+ei] = uu; Vx[:, len(cols)+ei] = vv; Wx[:, len(cols)+ei] = ww
        return Ux, Vx, Wx

    grid = [-2, -1, 0, 1, 2]

    # (a) single-term drop: try removing each term
    for drop in range(rank):
        if time.time() > deadline: break
        keep = [t for t in range(rank) if t != drop]
        Ux, Vx, Wx = matmul_of_terms(keep, [])
        if Ux.shape[1] < rank and _verify_decomposition(Ux, Vx, Wx, T):
            best = (Ux, Vx, Wx)
            break

    if best is not None:
        return best

    # (b) drop two terms, add one new rank-1 term from a small structured grid.
    for d1 in range(rank):
        if time.time() > deadline: break
        for d2 in range(d1+1, rank):
            if time.time() > deadline: break
            keep = [t for t in range(rank) if t not in (d1, d2)]
            for ca in grid:
                for cb in grid:
                    if ca == 0 and cb == 0:
                        continue
                    uu = ca*U[:, d1] + cb*U[:, d2]
                    vv = ca*V[:, d1] + cb*V[:, d2]
                    ww = ca*W[:, d1] + cb*W[:, d2]
                    Ux, Vx, Wx = matmul_of_terms(keep, [(uu, vv, ww)])
                    if Ux.shape[1] < rank and _verify_decomposition(Ux, Vx, Wx, T):
                        return (Ux, Vx, Wx)
    
    # (c) Probabilistic algebraic approach: random support patterns + linear solve
    #     We look for rank-16 by randomly sampling which 16 of the 17 terms to keep
    #     and then checking if we can adjust coefficients (or find a new term) to complete the tensor.
    #     Here we specifically try: Keep 15 original terms, add 1 new term whose support is a 
    #     random subset of the union of supports of the 2 dropped terms. Then solve for coeffs.
    #     However, solving for exact integer/rational coeffs is hard. 
    #     Instead, we use the fact that if rank is 16, there exists a linear dependency among the 17 
    #     rank-1 tensors. We check if any subset of 16 can span the 17th.
    
    # A more direct approach: Try to find if the tensor lies in the span of 16 random rank-1 tensors
    # chosen from a large pool of candidate rank-1 tensors derived from the structure.
    
    # Let's try a targeted "swap" search:
    # For each pair of terms to drop (d1, d2), we tried linear combos.
    # Now, let's try dropping 1 term and seeing if the remaining 16 can be "re-tuned" slightly?
    # No, re-tuning changes the support structure usually.
    
    # Let's try a different structural reduction:
    # The 2x3*3x3 matmul tensor has a known border rank.
    # Is rank 16 possible?
    # Lower bounds for 2x3*3x3 are not as tight as 2x2*2x2.
    # But let's try a random search for a rank-16 decomposition with sparse supports.
    
    # We will generate random sparse vectors for U, V, W with entries in {-1, 0, 1}
    # and check if the sum of outer products equals T.
    # This is a SAT-like problem but we do it randomly.
    
    # To stay within time, we do a limited number of random restarts.
    # We fix the supports of U, V, W to be consistent with a 2x3*3x3 structure.
    # A rank-1 term (u, v, w) must satisfy:
    # u has non-zeros only at indices i*m+j
    # v has non-zeros only at indices j*p+k
    # w has non-zeros only at indices k*n+i
    # And for the product to contribute to T, the indices must align.
    
    # Actually, any rank-1 term (u, v, w) contributes to T[a,b,c] only if
    # u[a] != 0, v[b] != 0, w[c] != 0.
    # For the sum to equal T, the non-zero entries of T must be covered.
    
    # We will randomly generate 16 rank-1 terms with small integer coefficients
    # and check if they sum to T.
    
    np.random.seed(42)
    
    # Define the set of valid (a,b,c) indices for T
    # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    # i in [0,1], j in [0,2], k in [0,2]
    # a in [0,5], b in [0,8], c in [0,5]
    
    # We will try to find 16 terms.
    # Each term is defined by (u, v, w).
    # To ensure we can potentially form T, we should bias the random generation
    # towards terms that have non-zeros in positions that align with T's non-zeros.
    
    # A term (u, v, w) "matches" a component (i,j,k) if:
    # u[i*m+j] != 0, v[j*p+k] != 0, w[k*n+i] != 0.
    # If we pick u, v, w to have non-zeros at these specific indices, the term contributes
    # to the component (i,j,k).
    
    # Let's try to randomly construct 16 terms that cover all 12 components (i,j,k)
    # with coefficients that sum to 1 for each component, and 0 for others.
    
    # This is equivalent to finding a 16x12 matrix M such that M @ ones = ones,
    # and the rank-1 tensors corresponding to the rows of M (interpreted appropriately)
    # sum to T.
    
    # Actually, it's simpler to just randomly pick 16 terms from a large candidate pool
    # and check.
    
    # Candidate pool: All rank-1 terms that have non-zeros in a "compatible" set of indices.
    # For a term to be useful, it should have non-zeros in indices that are "aligned".
    # E.g., if u has non-zero at i*m+j, v at j*p+k, w at k*n+i, it contributes to (i,j,k).
    # If u has non-zero at i*m+j and v at j'*p+k' with j' != j, it contributes to 0 (since T is 0 there? No, T is 0 there but the product is non-zero, so it would create an error unless cancelled).
    # So, to keep it simple, we only generate terms that are "monomial" in the sense that
    # their non-zero indices all correspond to the same (i,j,k) or a small set of them.
    
    # Let's generate terms that are "pure" for a single (i,j,k) or a pair.
    # Pure term for (i,j,k): u[i*m+j]=1, v[j*p+k]=1, w[k*n+i]=1.
    # This gives 12 such terms.
    # We need 16 terms. So we need 4 more terms that are combinations.
    
    # Let's try to find 16 terms from the set of all "aligned" terms.
    # An aligned term has non-zeros only at indices (a,b,c) such that there exists (i,j,k)
    # with a=i*m+j, b=j*p+k, c=k*n+i.
    # This is a strong constraint.
    
    # Let's just try the random search with the constraint that each term is "aligned".
    
    # Generate 100 random aligned terms.
    # Check if any subset of 16 sums to T.
    # This is still hard.
    
    # Let's try a different tactic:
    # We know rank 17 works. Can we do 16?
    # The lower bound for 2x3*3x3 is not 17. It might be 16 or lower.
    # But finding it is hard.
    
    # Given the time constraint and the difficulty, we will return best (which is None here).
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110  # explicit safety margin (< 120s wall-clock)

    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # Baseline: verified rank-17 (Strassen + direct).
    U17, V17, W17 = _rank17(n, m, p)
    if _verify_decomposition(U17, V17, W17, T):
        base = (U17, V17, W17)
    else:
        base = _schoolbook(n, m, p)

    # Bounded refinement: attempt a strictly smaller, EXACTLY verified rank.
    # This never hurts: it only replaces the baseline if verified.
    reduced = _try_reduced(T, base[0], base[1], base[2], base[0].shape[1], deadline)
    if reduced is not None:
        U, V, W = reduced
    else:
        U, V, W = base

    # Final exact verification; fall back to schoolbook on any failure.
    if not _verify_decomposition(U, V, W, T):
        U, V, W = _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

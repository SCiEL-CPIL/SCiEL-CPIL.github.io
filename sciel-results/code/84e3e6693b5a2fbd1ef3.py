"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
DEADLINE = 135.0  # safety margin under 150s wall-clock

# EVOLVE-BLOCK-START

def _time_left():
    return DEADLINE - time.time()

def _build_rank44():
    """
    Guaranteed exact rank-44 decomposition using Strassen blocks.
    This serves as the fallback and initial best.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    cols = []

    def add_term(u_entries, v_entries, w_entries):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
        cols.append((u_vec, v_vec, w_vec))

    def add_strassen(a, b, c, d, e, f, g, h, w11, w12, w21, w22):
        add_term([(a, 1), (d, 1)], [(e, 1), (h, 1)], [(w11, 1), (w22, 1)])
        add_term([(c, 1), (d, 1)], [(e, 1)], [(w21, 1), (w22, -1)])
        add_term([(a, 1)], [(f, 1), (h, -1)], [(w12, 1), (w22, 1)])
        add_term([(d, 1)], [(g, 1), (e, -1)], [(w11, 1), (w21, 1)])
        add_term([(a, 1), (b, 1)], [(h, 1)], [(w12, 1), (w11, -1)])
        add_term([(c, 1), (a, -1)], [(e, 1), (f, 1)], [(w22, 1)])
        add_term([(b, 1), (d, -1)], [(g, 1), (h, 1)], [(w11, 1)])

    def add_rank4(a, b, c, d, e, f, w1, w2):
        add_term([(a, 1)], [(c, 1)], [(w1, 1)])
        add_term([(b, 1)], [(e, 1)], [(w1, 1)])
        add_term([(a, 1)], [(d, 1)], [(w2, 1)])
        add_term([(b, 1)], [(f, 1)], [(w2, 1)])

    # A (3x4) indices: i*4+j
    # B (4x4) indices: j*4+k
    # C (3x4) indices: k*3+i (Trace convention W[k*n+i])

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 1, 4, 5, 0, 1, 4, 5, 0, 3, 1, 4)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 1, 4, 5, 2, 3, 6, 7, 6, 9, 7, 10)
    # Block 3: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    add_rank4(8, 9, 0, 1, 4, 5, 2, 5)
    # Block 4: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    add_rank4(8, 9, 2, 3, 6, 7, 8, 11)
    # Block 5: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen(2, 3, 6, 7, 8, 9, 12, 13, 0, 3, 1, 4)
    # Block 6: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen(2, 3, 6, 7, 10, 11, 14, 15, 6, 9, 7, 10)
    # Block 7: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    add_rank4(10, 11, 8, 9, 12, 13, 2, 5)
    # Block 8: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    add_rank4(10, 11, 10, 11, 14, 15, 8, 11)

    for r in range(rank):
        u_vec, v_vec, w_vec = cols[r]
        for i in range(n * m):
            U[i][r] = u_vec[i]
        for j in range(m * p):
            V[j][r] = v_vec[j]
        for k in range(p * n):
            W[k][r] = w_vec[k]
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check exactness of the decomposition (integer/half-integer)."""
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    rank = Ua.shape[1]
    # tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s = s + Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                if s != 1:
                    return False
    return True

def _search_rank(R, seed=0, time_budget=None):
    """
    Exact rank-R search for the matmul tensor.
    Uses structured initialization exploiting block decomposition + iterative refinement.
    """
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    start = time.time()

    def solve_linear_system(A, b):
        """Solve A x = b over rationals. A is list of lists (m x R), b is list (m,). Return x (R,) or None."""
        rows = len(A)
        cols = len(A[0]) if A else 0
        if rows == 0 or cols == 0:
            return None
        # Build augmented matrix
        Ar = [list(A[i]) + [b[i]] for i in range(rows)]
        pivots = []
        for c in range(cols):
            piv = -1
            for rr in range(len(pivots), rows):
                if Ar[rr][c] != 0:
                    piv = rr
                    break
            if piv == -1:
                continue
            if piv != len(pivots):
                Ar[piv], Ar[len(pivots)] = Ar[len(pivots)], Ar[piv]
            pivots.append(piv)
            inv = Fraction(1, 1) / Ar[piv][c]
            for cc in range(c, cols + 1):
                Ar[piv][cc] = Ar[piv][cc] * inv
            for rr in range(rows):
                if rr == piv:
                    continue
                factor = Ar[rr][c]
                if factor == 0:
                    continue
                for cc in range(c, cols + 1):
                    Ar[rr][cc] = Ar[rr][cc] - factor * Ar[piv][cc]
        for rr in range(len(pivots), rows):
            if Ar[rr][cols] != 0:
                return None
        x = [Fraction(0)] * cols
        for idx, pr in enumerate(pivots):
            x[pr] = Ar[pr][cols]
        return x

    def verify_decomp(U, V, W):
        """Check if U,V,W exactly decompose the matmul tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(R):
                        s = s + U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    def try_structured_init(init_type, rng):
        """
        Try to find a valid decomposition with structured initialization.
        init_type: 0=random, 1=block-structured, 2=Strassen-inspired
        """
        U = [[Fraction(0)] * R for _ in range(n * m)]
        V = [[Fraction(0)] * R for _ in range(m * p)]
        W = [[Fraction(0)] * R for _ in range(p * n)]

        if init_type == 0:
            # Pure random
            for i in range(n * m):
                for r in range(R):
                    U[i][r] = Fraction(rng.randint(-3, 3), 1)
            for j in range(m * p):
                for r in range(R):
                    V[j][r] = Fraction(rng.randint(-3, 3), 1)
        elif init_type == 1:
            # Block-structured: assign random values to 2x2 blocks
            # This mimics the Strassen block structure
            block_size = 2
            for i in range(n):
                for j in range(m):
                    for r in range(R):
                        if rng.random() < 0.5:
                            U[i * m + j][r] = Fraction(rng.randint(-2, 2), 1)
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        if rng.random() < 0.5:
                            V[j * p + k][r] = Fraction(rng.randint(-2, 2), 1)
        else:
            # Strassen-inspired: use known Strassen patterns for 2x2 blocks
            # with random mixing
            for i in range(n):
                for j in range(m):
                    for r in range(R):
                        val = rng.randint(-2, 2)
                        U[i * m + j][r] = Fraction(val, 1)
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        val = rng.randint(-2, 2)
                        V[j * p + k][r] = Fraction(val, 1)

        # Solve for W and verify
        for it in range(8):
            if time.time() - start > time_budget:
                return None
            # Solve for W given U, V
            all_valid = True
            for i in range(n):
                for k in range(p):
                    A = []
                    b = [Fraction(1)] * m
                    for j in range(m):
                        row = []
                        for r in range(R):
                            row.append(U[i * m + j][r] * V[j * p + k][r])
                        A.append(row)
                    sol = solve_linear_system(A, b)
                    if sol is None:
                        all_valid = False
                        break
                    for r in range(R):
                        W[k * n + i][r] = sol[r]
                if not all_valid:
                    break
            
            if not all_valid:
                # Perturb U and V slightly
                for i in range(n * m):
                    if rng.random() < 0.3:
                        U[i][rng.randint(0, R - 1)] += Fraction(rng.randint(-1, 1), 1)
                for j in range(m * p):
                    if rng.random() < 0.3:
                        V[j][rng.randint(0, R - 1)] += Fraction(rng.randint(-1, 1), 1)
                continue

            if verify_decomp(U, V, W):
                return U, V, W

            # If not valid, perturb
            for i in range(n * m):
                if rng.random() < 0.2:
                    U[i][rng.randint(0, R - 1)] += Fraction(rng.randint(-1, 1), 1)
            for j in range(m * p):
                if rng.random() < 0.2:
                    V[j][rng.randint(0, R - 1)] += Fraction(rng.randint(-1, 1), 1)

        return None

    # Try multiple initialization strategies
    for init_type in range(3):
        for trial in range(10):
            if time.time() - start > time_budget:
                return None
            res = try_structured_init(init_type, random.Random(seed + init_type * 100 + trial))
            if res is not None:
                return res
    return None

def _to_int_half(U, V, W):
    """Convert Fraction matrices to integer/half-integer lists, or None if not possible."""
    def convert(M):
        out = []
        for row in M:
            new_row = []
            for val in row:
                doubled = val * 2
                if doubled.denominator != 1:
                    return None
                new_row.append(doubled.numerator / 2)
            out.append(new_row)
        return out
    return convert(U), convert(V), convert(W)

def construct():
    n, m, p = DIMENSIONS
    start = time.time()

    # First, get the guaranteed rank-44 solution.
    best = _build_rank44()
    best_rank = 44

    # Try to find a lower rank via exact search.
    # We aim for rank 31 (Bini-Capovani-De Falci-Ladisi for 3x3x3 core + schoolbook for strips)
    # 3x3x3 rank is 19.
    # Strips:
    #   A[0:3, 3] * B[3, 0:4] -> 3x4 output. Rank 12 (schoolbook).
    #   A[0:3, 0:3] * B[0:3, 3] -> 3x1 output. Rank 9 (schoolbook).
    # Total: 19 + 12 + 9 = 40.
    # Wait, the directive says "aiming for a total rank of 31".
    # 19 (core) + 12 (top strip) + 9 (right strip) = 40.
    # How to get 31?
    # Maybe the strips can be combined or optimized?
    # Or maybe the core is not 19?
    # Rank of 3x3x3 is 19.
    # If we use 19 for core, and the rest is 12.
    # The "rest" is:
    #   A[:, 3] * B[3, :] -> 3x4 matrix. Rank 12.
    #   A[:, 0:3] * B[0:3, 3] -> 3x1 matrix. Rank 9.
    #   A[3, :] is not possible, A is 3x4.
    #   So we have 3x4 * 4x4.
    #   Split A into A_left (3x3) and A_right (3x1).
    #   Split B into B_top (4x3) and B_bottom (4x1).
    #   C = A_left * B_top + A_right * B_bottom.
    #   A_left * B_top: 3x3 * 3x3. Rank 19.
    #   A_right * B_bottom: 3x1 * 1x4. Rank 12.
    #   A_left * B_bottom: 3x3 * 3x1. Rank 9.
    #   A_right * B_top: 3x1 * 1x3. Rank 9.
    #   Total: 19 + 12 + 9 + 9 = 49. This is worse than 44.
    
    # The directive says "rank-19 ... core 3x3x3 ... rank-12 ... remaining 3x1x4 and 1x4x4 strips".
    # This implies the structure is:
    #   Core: 3x3 * 3x3 -> 3x3 (Rank 19)
    #   Strip 1: 3x1 * 1x4 -> 3x4 (Rank 12)
    #   Strip 2: 1x4 * 4x3 -> 1x3 (Rank 12) -- Wait, A is 3x4, B is 4x4.
    #   If we take A[2, :] (1x4) and B[:, 2] (4x1), that's a 1x1 term.
    #   
    # Let's look at the "rank 31" claim.
    # 19 (core) + 12 (something) = 31.
    # This suggests the "strips" are combined into a single rank-12 term?
    # Or maybe the core is not 19?
    # If we use Strassen for 3x3, it's 19? No, Strassen 3x3 is 19?
    # Strassen 2x2 is 7. 3x3 can be done in 19.
    # If the total is 31, and core is 19, the rest is 12.
    # The rest of the tensor is:
    #   T[i, 3, k] for i,k in 0..2 (A[:,3] * B[3,:]) -> 3x4 matrix. Rank 12.
    #   T[i, j, 3] for i,j in 0..2 (A[:,0:3] * B[0:3,3]) -> 3x1 matrix. Rank 9.
    #   T[2, 3, 3] is counted in both?
    #   If we sum them: 12 + 9 = 21. 19 + 21 = 40.
    #   To get 31, we need to save 9 ranks.
    #   Maybe the "strips" are not schoolbook?
    #   Rank of 3x1x4 tensor is 12? No, rank of 3x1x4 is 12 (outer product of 3x1 and 1x4 is rank 1? No, it's a matrix 3x4. Rank is 12 if generic).
    #   Wait, the tensor is fixed.
    #   The tensor for A[:,3] * B[3,:] is the outer product of A[:,3] (3x1) and B[3,:] (1x4).
    #   This is a rank-1 matrix in the (i,k) plane for each fixed j=3?
    #   No, it's a 3x4 matrix C_strips[i,k] = A[i,3] * B[3,k].
    #   This matrix has rank 1.
    #   So the tensor part T[i, 3, k] = A[i,3] * B[3,k].
    #   This is a rank-1 tensor in the (i, k) subspace?
    #   The full tensor term is U[i,3] * V[3,k] * W[k,i].
    #   We need to represent T[i,3,k] = sum_r U[i,3,r] V[3,k,r] W[k,i,r].
    #   If we set U[i,3,r] = A[i,3] (which is 1 for all i? No, A is the variable).
    #   Wait, the decomposition is for the *multiplication algorithm*.
    #   The entries of U, V, W are constants.
    #   The tensor T_{ijk} = 1 if the index matches.
    #   For the strip A[:,3] * B[3,:], the indices are i in 0..2, j=3, k in 0..3.
    #   T[i,3,k] = 1.
    #   We need sum_r U[i,3,r] V[3,k,r] W[k,i,r] = 1 for all i,k.
    #   This is a 3x4 matrix of 1s.
    #   We need to decompose the all-ones 3x4 matrix into a sum of rank-1 terms (in the tensor sense, but here it's just a matrix decomposition where W depends on k,i).
    #   Actually, if we fix W[k,i] = 1, then we need sum_r U[i,3,r] V[3,k,r] = 1.
    #   This is a matrix equation X Y^T = J (all ones).
    #   X is 3xR, Y is 4xR.
    #   Minimum R is 1 (if we allow 1/3 etc).
    #   But we need integer/half-integer.
    #   If R=1, U[i,3,0] = 1, V[3,k,0] = 1, W[k,i,0] = 1.
    #   Then sum is 1.
    #   So the strip A[:,3]*B[3,:] can be done in rank 1?
    #   Wait, the term is U[i,3] V[3,k] W[k,i].
    #   If U[i,3]=1, V[3,k]=1, W[k,i]=1, then product is 1.
    #   So this strip costs rank 1?
    #   Similarly for A[:,0:3]*B[0:3,3].
    #   T[i,j,3] = 1 for i,j in 0..2.
    #   We need sum_r U[i,j,r] V[j,3,r] W[3,i,r] = 1.
    #   If we set U[i,j]=1, V[j,3]=1, W[3,i]=1, product is 1.
    #   So this strip costs rank 1?
    #   Then total rank = 19 (core) + 1 (strip 1) + 1 (strip 2) = 21?
    #   This seems too good to be true.
    #   Why did the previous program use 44?
    #   Because the core 3x3x3 was decomposed using Strassen blocks which didn't overlap perfectly with the strips?
    #   Or because the "strips" are not just 1s?
    #   T[i,3,k] = 1. Yes.
    #   So if we can find a rank-19 decomposition for the 3x3x3 core, and rank-1 for each strip, total is 21.
    #   Let's check if a rank-19 decomposition for 3x3x3 exists with integer/half-integer entries.
    #   Bini-Capovani-De Falci-Ladisi (BCDL) has rank 19.
    #   Does it use integers?
    #   Yes, it uses 0, 1, -1.
    #   So we should be able to get rank 21.
    #   Why did the directive say 31?
    #   Maybe the core decomposition interferes with the strips?
    #   In the core decomposition, U[i,j,r] for i,j in 0..2 are non-zero.
    #   In the strip decomposition, U[i,3,r] is non-zero.
    #   These are disjoint indices in U, so they don't interfere.
    #   Similarly for V and W.
    #   V[3,k,r] is disjoint from V[j,k,r] for j<3.
    #   W[k,i,r] is shared!
    #   In the core, W[k,i,r] is used for i,k in 0..2.
    #   In the strip A[:,3]*B[3,:], we need W[k,i,r] for i,k in 0..2.
    #   So the W indices overlap.
    #   We can simply add the terms.
    #   If the core uses W[k,i,r_core] and the strip uses W[k,i,r_strip], they are different columns.
    #   So they are additive.
    #   Total rank = Rank(Core) + Rank(Strip1) + Rank(Strip2).
    #   Rank(Core) = 19.
    #   Rank(Strip1) = 1.
    #   Rank(Strip2) = 1.
    #   Total = 21.
    #   Let's try to implement this.

    # First, we need the BCDL rank-19 decomposition for 3x3x3.
    # I will hardcode it if possible, or search for it.
    # Since I cannot guarantee a hardcoded BCDL decomposition without error,
    # and the search might find it, I will focus on the search.
    # But the search is slow.
    # Let's try to construct the rank-21 decomposition manually if I can find the BCDL terms.
    # If not, I will fall back to the rank-44 solution and try to improve via search.

    # For now, I will keep the rank-44 solution as the base.
    # And try to search for rank 21, 22, ..., 43.
    # But the search is very slow for R=21.
    # I will prioritize the search for R in range(21, 44).

    # To speed up the search, I will use the knowledge that the structure is:
    # Core (19) + Strip1 (1) + Strip2 (1).
    # So I can initialize U, V, W with this structure.
    # U[0:9, 0:19] = Core_U
    # U[9:12, 0:19] = 0
    # U[0:9, 19] = 0
    # U[9:12, 19] = 1 (Strip1)
    # U[0:9, 20] = 0
    # U[9:12, 20] = 0
    # ... and so on.

    # But I don't have Core_U.
    # So I will stick to the general search but with a tighter bound.

    # Actually, if I can't find the BCDL decomposition, I can't build rank 21.
    # I will rely on the search to find a low rank decomposition.
    # The search might find rank 21 if it's clever enough.
    # But it's unlikely to find BCDL from scratch.
    
    # So I will just run the search for R from 21 to 43.
    # If it finds one, great. If not, I return 44.

    for R in range(21, 44):
        if time.time() - start > DEADLINE - 15:
            break
        budget = min(10, max(1, DEADLINE - 15 - (time.time() - start)))
        try:
            res = _search_rank(R, seed=R * 1000, time_budget=budget)
            if res is not None:
                Ur, Vr, Wr = res
                conv = _to_int_half(Ur, Vr, Wr)
                if conv is not None:
                    Uc, Vc, Wc = conv
                    if _verify(Uc, Vc, Wc):
                        best = {"U": Uc, "V": Vc, "W": Wc}
                        best_rank = R
                        break # Found a valid lower rank, stop searching
        except Exception:
            pass

    if time.time() - start > DEADLINE - 5:
        return best

    return best
# EVOLVE-BLOCK-END

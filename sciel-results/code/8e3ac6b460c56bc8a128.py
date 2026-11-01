"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product as iproduct

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    def time_left():
        return deadline - time.time()

    # Build target tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_rec - T)
        return err < 1e-9

    # ------------------------------------------------------------------
    # Strategy: Constructive search using Strassen block structure.
    #
    # Key insight: A is 2x2. Strassen gives 7 products of A.
    # For B (2x12) and C (12x2), we need to express C = A*B as a sum
    # of rank-1 terms: sum_r U_r * V_r * W_r where U_r depends on A,
    # V_r on B, W_r on C.
    #
    # We partition the 12 columns of B into groups. For each group,
    # we use a local multiplication scheme. Since A is shared, the
    # U-factors are the same Strassen 7 products (possibly permuted/scaled).
    #
    # For a group of g columns:
    #   - If g=1: need 2 terms (direct: a00*b0, a11*b1) -> but actually
    #     multiplying 2x2 by 2x1 needs 4 scalar products, reducible to...
    #     Actually 2x2 * 2x1 = 2x1. C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    #     C[1] = A[1,0]*B[0] + A[1,1]*B[1]. That's 4 terms.
    #     With Strassen on the 2x2 block (treating B column as 2x1):
    #     We can use the 7 Strassen products but only need 2 outputs.
    #     Actually for 2x2 * 2x1, the minimal rank is 4 (it's just 4 scalar mults).
    #     Wait, no. The tensor for 2x2*2x1 has rank 4 (it's [2,2,1] which
    #     is the same as [2,1,2]^T, and rank of [2,2,1] = 4).
    #
    #   - If g=2: 2x2 * 2x2 = 2x2. Strassen gives rank 7.
    #   - If g=3: 2x2 * 2x3 = 2x3. We can split into 2x2 + 2x1 = 7 + 4 = 11.
    #     Or find a better scheme.
    #
    # Total for 12 columns:
    #   6 groups of 2: 6*7 = 42 (the current solution)
    #   4 groups of 3: 4*11 = 44 (worse)
    #   3 groups of 4: 3*rank([2,2,4])
    #   2 groups of 6: 2*rank([2,2,6])
    #   1 group of 12: rank([2,2,12])
    #
    # The key question: can we do better than 42 by finding a smarter
    # decomposition that shares structure across groups?
    #
    # Alternative: use the fact that rank([n,m,p]) <= rank([n,m,p1]) + rank([n,m,p2])
    # but also rank([n,m,p]) <= n*m*ceil(p/2) * 7/2 for even p using Strassen pairs.
    #
    # For p=12: 4 * ceil(12/2) * 7/2 = 4*6*3.5 = 84? No that's wrong.
    # Strassen: rank([2,2,2]) = 7. So rank([2,2,12]) <= 6 * 7 = 42.
    #
    # Can we do better? The lower bound is n*m*p / (n+m+p-2) = 48/12 = 4? No.
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(4, 24, 24) = 24.
    # Also rank >= n*m*p / (n+m+p-2) is not a valid bound.
    # The Lax-Friedrichs bound: rank >= (n*m*p) / (n+m+p-2) = 48/12 = 4.
    # The true lower bound for [2,2,12] is at least 24 (from m*p = 24).
    #
    # Known results: rank([2,2,2]) = 7, rank([2,2,4]) <= 14, rank([2,2,6]) <= 21.
    # Is rank([2,2,12]) = 42 or can it be less?
    #
    # Let's try a different partition: 3 groups of 4.
    # rank([2,2,4]): We can compute 2x2 * 2x4 as two 2x2*2x2 Strassen = 14.
    # Or: 2x2 * 2x4. The output is 2x4 = 8 elements.
    # Using Strassen on the 2x2 block structure:
    #   C[:, 0:2] = A * B[:, 0:2] -> 7 terms
    #   C[:, 2:4] = A * B[:, 2:4] -> 7 terms
    #   Total: 14.
    # So 3 groups of 4 gives 3*14 = 42. Same.
    #
    # What about a non-uniform partition?
    # 2 groups of 2 + 2 groups of 4: 2*7 + 2*14 = 14 + 28 = 42.
    # 1 group of 2 + 5 groups of... no, 12 = 2+2+2+2+2+2 is the only even split into 2s.
    # 12 = 4+4+4, 12 = 2+2+2+2+2+2, 12 = 3+3+3+3, 12 = 4+4+2+2, etc.
    #
    # For groups of 3: rank([2,2,3])? 
    # 2x2 * 2x3. Output 2x3 = 6 elements.
    # Direct: 12 scalar mults.
    # Strassen-like: split 2x3 into 2x2 + 2x1: 7 + 4 = 11.
    # Can we do better for [2,2,3]?
    # Lower bound: max(4, 6, 6) = 6.
    # Known: rank([2,2,3]) is between 6 and 11.
    # If rank([2,2,3]) = 8, then 4 groups of 3 gives 32.
    # If rank([2,2,3]) = 9, then 4 groups of 3 gives 36.
    #
    # Let's try to find a rank-8 or rank-9 decomposition for [2,2,3].
    #
    # Actually, let me try a completely different approach:
    # Use the "Laser method" or "subproduct tree" to get asymptotic bounds,
    # but for small sizes, let's just do a systematic search.
    #
    # NEW STRATEGY: Systematic constructive search with rational arithmetic.
    # We'll try to build decompositions of rank R for R = 24, 26, 28, ..., 42
    # by solving a system of polynomial equations. But that's infeasible.
    #
    # BETTER STRATEGY: Use the known Strassen-42 as a starting point and try
    # to find linear dependencies among the 42 rank-1 tensors. If the 42
    # tensors span a subspace of dimension < 42, we can potentially reduce.
    # But tensor rank is not the same as linear span.
    #
    # MOST PRACTICAL STRATEGY: Try several known constructions and verify:
    # 1. Strassen-42 (baseline)
    # 2. 4 groups of 3 with a rank-9 scheme for [2,2,3] -> 36
    # 3. 3 groups of 4 with a rank-13 scheme for [2,2,4] -> 39
    # 4. 2 groups of 6 with a rank-20 scheme for [2,2,6] -> 40
    #
    # Let me try to construct a rank-9 scheme for [2,2,3].
    # 2x2 * 2x3:
    #   A = [[a00, a01], [a10, a11]]
    #   B = [[b00, b01, b02], [b10, b11, b12]]
    #   C = [[c00, c01, c02], [c10, c11, c12]]
    #
    # Direct: 12 terms.
    # Strassen split: 2x2*2x2 (7 terms) + 2x2*2x1 (4 terms) = 11.
    #
    # Can we share some Strassen products?
    # The 7 Strassen products for the 2x2 block use:
    #   M1 = (a00+a11)(b00+b10)
    #   M2 = (a10+a11)b00
    #   M3 = a00(b01-b11)
    #   M4 = a11(b10-b00)
    #   M5 = (a00+a01)b11
    #   M6 = (a10-a00)(b00+b01)
    #   M7 = (a01-a11)(b10+b11)
    #
    # For the 2x1 block (column 2), we need:
    #   c02 = a00*b02 + a01*b12
    #   c12 = a10*b02 + a11*b12
    #
    # These 4 products (a00*b02, a01*b12, a10*b02, a11*b12) are new.
    # Total: 7 + 4 = 11.
    #
    # To get rank 9, we'd need to save 2 terms. This seems unlikely.
    # Let's try rank 10: save 1 term.
    #
    # Actually, let's just go with the constructive approach and try
    # multiple partitions, verifying each with exact arithmetic.
    #
    # I'll implement a general verifier and try several candidate constructions.
    # ------------------------------------------------------------------

    def make_strassen_42():
        """Create the standard Strassen-42 decomposition."""
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        # Strassen 2x2x2 factors for local indices
        # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)

        # Each entry is (index, coefficient)
        prods = [
            # M1: (a00+a11)(b00+b11) -> c00+=1, c11+=1
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
            # M2: (a10+a11)b00 -> c10+=1
            ([(2,1), (3,1)], [(0,1)], [(2,1)]),
            # M3: a00(b01-b11) -> c01+=1
            ([(0,1)], [(1,1), (3,-1)], [(1,1)]),
            # M4: a11(b10-b00) -> c00+=1, c10+=1
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
            # M5: (a00+a01)b11 -> c00-=1, c01+=1
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
            # M6: (a10-a00)(b00+b01) -> c11+=1
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
            # M7: (a01-a11)(b10+b11) -> c00-=1, c11+=1
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,-1), (3,1)]),
        ]

        for t in range(6):
            for s in range(7):
                col = 7 * t + s
                u_entries, v_entries, w_entries = prods[s]

                for idx, coeff in u_entries:
                    U[idx, col] = coeff

                for idx, coeff in v_entries:
                    j = idx // 2
                    k_loc = idx % 2
                    k = 2 * t + k_loc
                    V[j * p + k, col] = coeff

                for idx, coeff in w_entries:
                    k_loc = idx // 2
                    i = idx % 2
                    k = 2 * t + k_loc
                    W[k * n + i, col] = coeff

        return U, V, W, rank

    # First, get the baseline Strassen-42
    U42, V42, W42, rank42 = make_strassen_42()

    # Verify it
    if not verify(U42, V42, W42):
        # Fallback to schoolbook
        rank_fb = 48
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_fb[a, col] = 1
                    V_fb[b, col] = 1
                    W_fb[c, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_U, best_V, best_W = U42, V42, W42
    best_rank = 42

    # ------------------------------------------------------------------
    # Try to improve by finding linear dependencies or alternative constructions.
    #
    # Approach: Check if any subset of the 42 columns can be replaced by
    # a smaller set. This is equivalent to checking if the tensor T can be
    # decomposed with fewer rank-1 terms.
    #
    # We'll use a greedy approach: try removing each column and see if the
    # remaining 41 can still span T. If not, try adjusting the remaining
    # columns to compensate.
    #
    # Actually, a better approach: use the fact that if we have a valid
    # decomposition with R terms, we can try to find a valid decomposition
    # with R-1 terms by solving for the coefficients.
    #
    # Let's try a different constructive approach:
    # Use 4 groups of 3 columns, with a rank-9 scheme for each group.
    # Total: 36.
    #
    # For [2,2,3], let's try to find a rank-9 decomposition.
    # We'll set up the equations and solve.
    # ------------------------------------------------------------------

    def try_rank_r(R, seed=42):
        """Try to find a rank-R decomposition using random initialization + gradient descent."""
        if R >= 42:
            return None, None, None, R

        np.random.seed(seed)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)

        # Scale to make entries closer to integers
        U /= np.max(np.abs(U)) * 2
        V /= np.max(np.abs(V)) * 2
        W /= np.max(np.abs(W)) * 2

        def compute_error():
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            return np.linalg.norm(T_rec - T)

        current_err = compute_error()

        # Gradient descent with integer rounding
        for it in range(2000):
            if time_left() < 10:
                break

            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_rec - T

            Gu = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
            Gv = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
            Gw = 2 * np.einsum('abc,ar,br->cr', diff, U, V)

            # Update with small step
            lr = 0.01
            U -= lr * Gu
            V -= lr * Gv
            W -= lr * Gw

            new_err = compute_error()
            if new_err > current_err:
                # Revert
                U += lr * Gu
                V += lr * Gv
                W += lr * Gw

            current_err = new_err
            if current_err < 1e-10:
                break

        if current_err < 1e-8:
            # Try to round to integers
            U_int = np.round(U * 2) / 2  # half-integers
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2

            if verify(U_int, V_int, W_int):
                return U_int, V_int, W_int, R

            # Try integers
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if verify(U_int, V_int, W_int):
                return U_int, V_int, W_int, R

        return None, None, None, R

    # Try several ranks
    for R in [36, 38, 40]:
        if time_left() < 15:
            break
        U_try, V_try, W_try, _ = try_rank_r(R)
        if U_try is not None:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = R
            break

    # If we found a better rank, return it
    if best_rank < 42:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }

    # Otherwise, return the Strassen-42 solution
    return {
        "U": U42.tolist(),
        "V": V42.tolist(),
        "W": W42.tolist()
    }
# EVOLVE-BLOCK-END

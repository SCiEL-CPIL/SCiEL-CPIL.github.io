"""
Strategy: Deterministic Block-Structured Integer Search.
The previous approach used continuous optimization (L-BFGS-B) which is prone to
getting stuck in local minima and requires rounding that often fails for exact
tensor decompositions.

This new strategy exploits the block structure of the 2x2 * 2x12 multiplication.
We view the 12 columns of B as 6 blocks of 2 columns each.
For each block, we compute a 2x2 * 2x2 multiplication.
The tensor rank for 2x2 * 2x2 is 7 (Strassen).
A naive composition gives rank 6 * 7 = 42.

However, we suspect that a lower rank might be achievable by sharing structure
across blocks or by using a different decomposition for the 2x2 * 2x2 blocks
if we allow cross-block interactions. But given the difficulty of finding global
non-Strassen decompositions, we first perform an exhaustive search for integer
solutions within the Strassen-like subspace for each block, and then attempt to
find linear dependencies between the 42 terms to reduce the rank.

Actually, a more robust "different" strategy is to perform a systematic search
for integer coefficients for a rank-R decomposition, assuming R is close to 42
but potentially lower due to accidental cancellations or better basis choices.

However, the most promising "different" approach that is computationally feasible
within 120s is to use the **Bini-Capovani** or similar algebraic identities
directly if they exist for this dimension, or to use a **randomized algebraic
verification** of potential rank reductions.

Let's try a different angle: **Exact Integer Search for Rank 40, 38, 36**.
We will parameterize the factors U, V, W with small integers (from -2 to 2)
and use a randomized restart strategy with local refinement (coordinate descent
on integers) to minimize the error, checking for exact zero error.

Given the constraints, we will:
1. Construct the standard rank-42 Strassen solution.
2. Attempt to find a rank-40 solution by randomly merging 2 pairs of Strassen
   terms and solving for the resulting integer coefficients.
3. If that fails, return the rank-42 solution.

Wait, the directive says "COMPUTE BUDGET... KILLS after 150s".
We have plenty of time for a more thorough search.

Let's try a **Simulated Annealing** style search on the space of integer
coefficients for a fixed rank R (e.g., R=36, 38, 40).
We initialize with the rank-42 solution, project it to rank R by randomly
combining columns, and then perform local moves (changing one coefficient by +/-1)
to reduce the squared error. If error reaches 0, we have a solution.

This is a discrete optimization problem, which is structurally different from
continuous L-BFGS-B.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    n, m, p = DIMENSIONS

    def time_left():
        return deadline - time.time()

    # Build target tensor T[a,b,c]
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def compute_error(U, V, W):
        """Compute exact integer error norm."""
        # T_rec = sum_r U[:,r] V[:,r]^T W[:,r]^T
        # We can compute this efficiently using einsum with integer types
        # but einsum with int64 might overflow or be slow.
        # Let's use float64 for speed, but verify with int if close.
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, dtype=np.float64)
        diff = T_rec - T
        return np.sum(diff * diff)

    def verify_exact(U, V, W):
        """Verify exact integer decomposition."""
        # Check if entries are close to integers
        if not np.allclose(U, np.round(U), atol=1e-9): return False
        if not np.allclose(V, np.round(V), atol=1e-9): return False
        if not np.allclose(W, np.round(W), atol=1e-9): return False
        
        U_int = np.round(U).astype(np.int64)
        V_int = np.round(V).astype(np.int64)
        W_int = np.round(W).astype(np.int64)
        
        T_rec = np.einsum('ar,br,cr->abc', U_int, V_int, W_int, dtype=np.int64)
        return np.all(T_rec == T)

    def make_strassen_42():
        rank = 42
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)

        # Strassen 7 terms for 2x2 * 2x2
        # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # Block t corresponds to columns 2t, 2t+1 of B and C.
        
        # Standard Strassen terms (for 2x2):
        # m1 = (a00 - a11) * (b01 + b11)
        # m2 = (a10 + a11) * b00
        # m3 = a00 * (b10 - b00)
        # m4 = a11 * (b11 - b01)
        # m5 = (a00 + a10) * b11
        # m6 = (a11 - a00) * (b00 + b10)
        # m7 = (a01 - a11) * (b11 + b10)
        
        # Output c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        # We map these to the global indices.
        # U index: i*m + j. For 2x2 A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
        # V index: j*p + k. For block t, k is 0 or 1 relative to block.
        #   j=0: k=0 -> 0, k=1 -> 1. j=1: k=0 -> 2, k=1 -> 3.
        #   Global V index: j*p + (2*t + k_loc).
        # W index: k*n + i. For block t, k is 0 or 1 relative to block.
        #   k=0: i=0 -> 0, i=1 -> 1. k=1: i=0 -> 2, i=1 -> 3.
        #   Global W index: (2*t + k_loc)*n + i.

        # Define the 7 Strassen patterns for a single block.
        # Each pattern is a list of (U_idx, coeff), (V_idx, coeff), (W_idx, coeff)
        # Note: U_idx is local to A (0..3). V_idx and W_idx are local to the 2x2 block.
        
        # m1: U: (0,1), (3,-1). V: (1,1), (3,1). W: (1,1), (3,1). 
        # Wait, let's re-verify Strassen terms carefully.
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        
        # m1 = (a00 - a11) * (b01 + b11)
        #   U: a00(0), -a11(3). V: b01(1), b11(3). W: c01(1), c11(3).
        #   Wait, W index is k*n + i.
        #   c01 is i=0, k=1 -> index 1*2 + 0 = 2? No.
        #   C is 2x2. c[i,k].
        #   c00: i=0, k=0 -> idx 0.
        #   c01: i=0, k=1 -> idx 2.
        #   c10: i=1, k=0 -> idx 1.
        #   c11: i=1, k=1 -> idx 3.
        #   So W indices for 2x2 block:
        #   c00 -> 0
        #   c01 -> 2
        #   c10 -> 1
        #   c11 -> 3
        
        # Let's re-map Strassen outputs:
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        # Terms:
        # m1: U(0,1), U(3,-1). V(1,1), V(3,1). W(0,1), W(3,1).
        #      (a00-a11)(b01+b11) -> contributes to c00 and c11?
        #      No, m1 is one product. It contributes to c00 and c11.
        #      So W(0,1) and W(3,1).
        
        # m2: U(2,1), U(3,1). V(0,1). W(1,1).
        #      (a10+a11)b00 -> c10.
        
        # m3: U(0,1). V(2,1), V(0,-1). W(2,1).
        #      a00(b10-b00) -> c01.
        
        # m4: U(3,1). V(3,1), V(1,-1). W(0,1), W(1,1).
        #      a11(b11-b01) -> c00 and c10.
        
        # m5: U(0,1), U(2,1). V(3,1). W(2,-1), W(3,1).
        #      (a00+a10)b11 -> c01 and c11.
        
        # m6: U(3,1), U(0,-1). V(0,1), V(2,1). W(3,1).
        #      (a11-a00)(b00+b10) -> c11.
        
        # m7: U(1,1), U(3,-1). V(2,1), V(3,1). W(0,-1), W(3,1).
        #      (a01-a11)(b11+b10) -> c00 and c11.

        strassen_blocks = [
            # m1
            ([(0,1), (3,-1)], [(1,1), (3,1)], [(0,1), (3,1)]),
            # m2
            ([(2,1), (3,1)], [(0,1)], [(1,1)]),
            # m3
            ([(0,1)], [(2,1), (0,-1)], [(2,1)]),
            # m4
            ([(3,1)], [(3,1), (1,-1)], [(0,1), (1,1)]),
            # m5
            ([(0,1), (2,1)], [(3,1)], [(2,-1), (3,1)]),
            # m6
            ([(3,1), (0,-1)], [(0,1), (2,1)], [(3,1)]),
            # m7
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,-1), (3,1)])
        ]

        for t in range(6):
            for s in range(7):
                col = 7 * t + s
                u_entries, v_entries, w_entries = strassen_blocks[s]

                for idx, coeff in u_entries:
                    U[idx, col] = coeff

                for idx, coeff in v_entries:
                    # idx is local V index (0..3) for 2x2 block.
                    # 0: j=0, k=0
                    # 1: j=0, k=1
                    # 2: j=1, k=0
                    # 3: j=1, k=1
                    j = idx // 2
                    k_loc = idx % 2
                    k = 2 * t + k_loc
                    V[j * p + k, col] = coeff

                for idx, coeff in w_entries:
                    # idx is local W index (0..3) for 2x2 block.
                    # 0: k=0, i=0
                    # 1: k=0, i=1
                    # 2: k=1, i=0
                    # 3: k=1, i=1
                    k_loc = idx // 2
                    i = idx % 2
                    k = 2 * t + k_loc
                    W[k * n + i, col] = coeff

        return U, V, W

    # Baseline
    U42, V42, W42 = make_strassen_42()
    if verify_exact(U42, V42, W42):
        best_U, best_V, best_W = U42, V42, W42
        best_rank = 42
    else:
        # Fallback to schoolbook 48
        best_rank = 48
        U_fb = np.zeros((n * m, best_rank), dtype=np.float64)
        V_fb = np.zeros((m * p, best_rank), dtype=np.float64)
        W_fb = np.zeros((p * n, best_rank), dtype=np.float64)
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
        best_U, best_V, best_W = U_fb, V_fb, W_fb

    # Discrete Search for Lower Rank
    # We try to find rank R < best_rank by perturbing the Strassen solution.
    # We will attempt R = 40, 38, 36.
    
    np.random.seed(42)

    def try_discrete_search(target_rank, U_init, V_init, W_init, max_iters=5000):
        """
        Local search on integer lattice.
        Start from a floating point initialization, round to integers.
        Then perform random moves to reduce error.
        """
        U = np.round(U_init).astype(np.float64)
        V = np.round(V_init).astype(np.float64)
        W = np.round(W_init).astype(np.float64)
        
        current_err = compute_error(U, V, W)
        if current_err < 1e-9:
            return U, V, W, True

        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = current_err

        for it in range(max_iters):
            if time_left() < 10:
                break
                
            # Pick a random matrix and a random element
            mat_choice = np.random.randint(3)
            if mat_choice == 0:
                r, c = np.random.randint(n*m), np.random.randint(target_rank)
                # Change U[r, c] by +/- 1
                delta = np.random.choice([-1, 1])
                U[r, c] += delta
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_err < 1e-9:
                        return best_U, best_V, best_W, True
                else:
                    U[r, c] -= delta # revert
            elif mat_choice == 1:
                r, c = np.random.randint(m*p), np.random.randint(target_rank)
                delta = np.random.choice([-1, 1])
                V[r, c] += delta
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_err < 1e-9:
                        return best_U, best_V, best_W, True
                else:
                    V[r, c] -= delta
            else:
                r, c = np.random.randint(p*n), np.random.randint(target_rank)
                delta = np.random.choice([-1, 1])
                W[r, c] += delta
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_err < 1e-9:
                        return best_U, best_V, best_W, True
                else:
                    W[r, c] -= delta

        return best_U, best_V, best_W, False

    # Attempt to reduce rank from 42
    for target_rank in [40, 38, 36]:
        if time_left() < 20:
            break
        if target_rank >= best_rank:
            continue

        # Initialize by taking random linear combinations of the 42 Strassen columns
        # to form target_rank columns.
        # This preserves the "span" of the Strassen solution.
        for attempt in range(10): # 10 random initializations per rank
            if time_left() < 5:
                break
            
            # Random projection
            proj_matrix = np.random.randn(target_rank, 42)
            U_init = U42 @ proj_matrix.T
            V_init = V42 @ proj_matrix.T
            W_init = W42 @ proj_matrix.T
            
            # Run discrete search
            U_opt, V_opt, W_opt, found = try_discrete_search(target_rank, U_init, V_init, W_init, max_iters=2000)
            
            if found:
                if verify_exact(U_opt, V_opt, W_opt):
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = target_rank
                    break
        
        if target_rank < best_rank:
            continue # if we found it, best_rank is updated, so we break inner loop?
            # Actually, if we found it, we should break outer loop too.
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END

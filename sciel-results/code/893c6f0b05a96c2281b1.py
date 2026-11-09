"""Tensor decomposition search for 3x5 * 5x6 matrix multiplication."""
DIMENSIONS = [3, 5, 6]

import time
import random
import itertools

# EVOLVE-BLOCK-START
def _make_schoolbook(n, m, p):
    """Return the trivial (schoolbook) rank-n*m*p decomposition."""
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


def _make_row_sum_decomposition(n, m, p):
    """
    Construct a rank-reduced decomposition using row-sum elimination.
    
    For the (n,m,p) matmul tensor T[i,j,k]=1, we reduce rank by exploiting
    linear dependencies in the i-fiber for each (j,k).
    
    For n=3: For each (j,k), instead of 3 separate terms, we use 2 specific
    terms plus 1 shared term, saving 1 term per (j,k). Total savings: m*p = 30.
    New rank: 90 - 30 = 60.
    
    Construction:
    For each (j, k):
      - Term for i=0: U[0*m+j, c0]=1, V[j*p+k, c0]=1, W[k*n+0, c0]=1
      - Term for i=1: U[1*m+j, c1]=1, V[j*p+k, c1]=1, W[k*n+1, c1]=1
      - Shared term for i=2: U[2*m+j, c2]=1, V[j*p+k, c2]=1, W[k*n+2, c2]=1
    
    This is still 3 terms. To actually reduce, we need to share across (j,k).
    
    Alternative: Use rank = n*p + (n-1)*m*p = 3*6 + 2*5*6 = 18+60 = 78.
    Or: rank = m*p + (m-1)*n*p = 30 + 4*3*6 = 30+72 = 102 (worse).
    
    Let me try: rank = n*m*p - n = 87.
    For each (j,k), eliminate 1 term by combining across i.
    """
    # For now, this is a placeholder. The actual search will try to find
    # a valid decomposition of rank < 90.
    return None


def _verify(U, V, W, n, m, p):
    """
    Verify whether factor matrices U (n*m x r), V (m*p x r), W (p*n x r)
    exactly realize the matmul tensor under the convention W[k*n + i, r].
    The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all (i,j,k).
    We check all 180 entries.
    """
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                urow = U[ui]
                vrow = V[vj]
                wrow = W[wk]
                for rr in range(r):
                    s += urow[rr] * vrow[rr] * wrow[rr]
                if s != 1:
                    return False
    return True


def _verify_sampled(U, V, W, n, m, p, samples=200):
    """Faster verification by sampling."""
    import random as _rnd
    r = len(U[0])
    for _ in range(samples):
        i = _rnd.randint(0, n - 1)
        j = _rnd.randint(0, m - 1)
        k = _rnd.randint(0, p - 1)
        ui = i * m + j
        vj = j * p + k
        wk = k * n + i
        urow = U[ui]
        vrow = V[vj]
        wrow = W[wk]
        s = 0
        for rr in range(r):
            s += urow[rr] * vrow[rr] * wrow[rr]
        if s != 1:
            return False
    return True


def construct():
    n, m, p = 3, 5, 6
    deadline = time.time() + 110  # safety margin

    # Start with schoolbook as fallback
    fallback = _make_schoolbook(n, m, p)

    # Known lower bound: rank >= max(n*m, m*p, p*n) / max(n,m,p) ... 
    # Actually for 3x5*5x6, the best known bounds:
    #  omega-based: ~20, but exact known? 
    #  Let's try to find anything below 90.
    #  Simple approach: try random rank-r decompositions and solve.

    # Strategy: for a given rank r, we need to find U (15x r), V (30x r), W (18x r)
    # such that for all (i,j,k): sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1.
    #
    # This is a system of equations. We can fix some entries and solve the rest.
    # 
    # Approach 1: Fix U to be in a "standard" form and solve for V, W.
    # Approach 2: Use random initialization and gradient-free local search.
    # Approach 3: Systematic search for rank < 90.
    #
    # Let me try a more structured approach:
    # For rank r, the tensor has 90 nonzero entries (all 1). We have 15*r + 30*r + 18*r = 63*r variables.
    # We need 63*r >= 90 for solvability in principle, so r >= ceil(90/63) = 2 (trivially).
    # But the constraints are nonlinear (trilinear).
    #
    # A practical approach: try to find decompositions of various ranks by:
    # 1. Start with a random or structured rank-r guess
    # 2. Use local optimization to satisfy the equations
    #
    # But this is hard. Let me try something more concrete:
    #
    # For the 3x5x6 case, known results suggest rank is around 20-25.
    # Let me try to find a decomposition by solving the system directly.
    #
    # Approach: Fix a basis for U. Let U have r columns. 
    # The constraint is: for each (i,j,k), <U[i,j,:], V[j,k,:] * W[k,i,:]> = 1
    # where * is elementwise product.
    #
    # This is equivalent to: the trilinear form defined by U,V,W matches the matmul tensor.
    #
    # Let me try a simpler approach: use the fact that we can set W to have a specific form.
    # 
    # Actually, let me try a different tactic: enumerate small-rank candidates by 
    # constraining the structure. For rank r, we can use "border rank" style constructions.
    #
    # Given the time constraint, let me try:
    # 1. Check if rank 20-30 works with random + local refinement
    # 2. If not found in time, return schoolbook

    import math

    def _is_valid(r, U, V, W):
        """Full verification for rank r."""
        return _verify(U, V, W, n, m, p)

    best_result = None

    # Try to find a valid decomposition of rank < 90
    # Use a randomized approach with local search
    # 
    # Key insight: we can parametrize this as finding three matrices.
    # For efficiency, fix some degrees of freedom.
    #
    # Let me try: set U to be (partially) determined, then solve for V and W.
    #
    # Actually, here's a cleaner approach. For a given rank r:
    # - Randomly initialize U, V, W with entries in {0, 0.5, 1} or {-1, 0, 1}
    # - For each (i,j,k), compute the error e_ijk = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1
    # - We want to minimize sum e_ijk^2
    # - Use coordinate descent or random perturbation
    #
    # But this is very hard in high dimension. Let me try something else.
    #
    # Alternative: use the algebraic structure. The matmul tensor for (n,m,p) is
    # a specific 3-tensor. Known algorithms: Strassen for 2x2, etc.
    # For 3x5x6, no simple algebraic formula is known.
    #
    # Given the difficulty, let me try a focused search:
    # For ranks from 18 up to 40, try random initializations with a limited
    # local search. If any valid one is found, return it.

    def _random_init(r, n, m, p, scale=2.0):
        """Random initialization with values in a small set."""
        U = []
        for _ in range(n * m):
            row = [random.uniform(-scale, scale) for _ in range(r)]
            U.append(row)
        V = []
        for _ in range(m * p):
            row = [random.uniform(-scale, scale) for _ in range(r)]
            V.append(row)
        W = []
        for _ in range(p * n):
            row = [random.uniform(-scale, scale) for _ in range(r)]
            W.append(row)
        return U, V, W

    def _compute_errors(U, V, W, n, m, p):
        """Compute all 90 trilinear form values."""
        r = len(U[0])
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    s = 0
                    for c in range(r):
                        s += U[ui][c] * V[vj][c] * W[wk][c]
                    errors.append((i, j, k, s))
        return errors

    def _local_search(U, V, W, n, m, p, max_steps=5000, time_budget=5.0):
        """Simple local search: randomly perturb one entry and accept if it reduces error."""
        r = len(U[0])
        start_time = time.time()
        
        def total_error():
            te = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ui = i * m + j
                        vj = j * p + k
                        wk = k * n + i
                        s = 0
                        for c in range(r):
                            s += U[ui][c] * V[vj][c] * W[wk][c]
                        te += (s - 1) ** 2
            return te

        current_err = total_error()
        if current_err < 1e-6:
            return True

        for step in range(max_steps):
            if time.time() - start_time > time_budget:
                break
            # Pick a random matrix, row, column
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                nrows = n * m
            elif mat == 'V':
                nrows = m * p
            else:
                nrows = p * n
            
            row = random.randint(0, nrows - 1)
            col = random.randint(0, r - 1)
            
            if mat == 'U':
                target = U
            elif mat == 'V':
                target = V
            else:
                target = W
            
            old_val = target[row][col]
            # Small random perturbation
            new_val = old_val + random.uniform(-0.5, 0.5)
            target[row][col] = new_val
            
            new_err = total_error()
            if new_err < current_err:
                current_err = new_err
                if current_err < 1e-6:
                    return True
            else:
                target[row][col] = old_val
        
        return False

    # Try ranks from 18 to 45
    for r in range(18, 46):
        if time.time() > deadline:
            break
        for attempt in range(10):
            if time.time() > deadline:
                break
            U, V, W = _random_init(r, n, m, p)
            
            # Quick check: total sum of products should be nonzero
            # Run local search
            if _local_search(U, V, W, n, m, p, max_steps=3000, time_budget=3.0):
                # Verify fully
                if _verify(U, V, W, n, m, p):
                    # Round to clean values if close to integers/half-integers
                    def _round_vals(mat):
                        rounded = []
                        for row in mat:
                            rrow = []
                            for v in row:
                                rv = round(v * 2) / 2
                                if abs(v - rv) < 1e-4:
                                    rrow.append(rv)
                                else:
                                    rrow.append(rv)  # keep closest half-integer
                            rounded.append(rrow)
                        return rounded
                    U = _round_vals(U)
                    V = _round_vals(V)
                    W = _round_vals(W)
                    if _verify(U, V, W, n, m, p):
                        return {"U": U, "V": V, "W": W}
                    else:
                        # Re-verify with original values
                        if _verify(U, V, W, n, m, p):
                            return {"U": U, "V": V, "W": W}
                        # Try with slightly perturbed values to fix rounding
                        pass
            
            # Alternative: try with integer-valued initialization
            if random.random() < 0.5:
                U, V, W = _random_init(r, n, m, p, scale=1.0)
                # Snap to {-1, 0, 1}
                for mat in [U, V, W]:
                    for row in mat:
                        for idx in range(len(row)):
                            row[idx] = round(row[idx])
                if _local_search(U, V, W, n, m, p, max_steps=2000, time_budget=2.0):
                    if _verify(U, V, W, n, m, p):
                        return {"U": U, "V": V, "W": W}

    # If nothing found, try a structured approach
    # For rank r = n*p + m = 24 (or similar), use a known pattern
    
    # Try: rank = n*p = 18 (might not work for non-square)
    # Try rank = max(n*m, n*p, m*p) = 30
    # Or rank = n*m + m*p - something
    
    # Given time constraints, if we haven't found anything better, return schoolbook
    if time.time() > deadline - 10:
        return fallback

    # Last resort: try a few more ranks
    for r in [17, 16, 15]:
        if time.time() > deadline:
            break
        for attempt in range(20):
            if time.time() > deadline:
                break
            U, V, W = _random_init(r, n, m, p)
            if _local_search(U, V, W, n, m, p, max_steps=5000, time_budget=4.0):
                if _verify(U, V, W, n, m, p):
                    def _round_vals(mat):
                        rounded = []
                        for row in mat:
                            rrow = []
                            for v in row:
                                rv = round(v * 2) / 2
                                rrow.append(rv)
                            rounded.append(rrow)
                        return rounded
                    U2 = _round_vals(U)
                    V2 = _round_vals(V)
                    W2 = _round_vals(W)
                    if _verify(U2, V2, W2, n, m, p):
                        return {"U": U2, "V": V2, "W": W2}
                    return {"U": U, "V": V, "W": W}

    # Return the schoolbook decomposition (rank 90)
    return fallback
# EVOLVE-BLOCK-END

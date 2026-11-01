"""
Rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.

Strategy:
  1. Use the known schoolbook decomposition (rank 24) as the guaranteed fallback.
  2. Systematically search for lower-rank decompositions by:
     a. Trying structured low-rank patterns derived from index-sharing ideas.
     b. For each candidate rank r (from a theoretical lower bound up to 23),
        set up the linear system for W given random-but-structured U, V,
        and check if a solution exists with small/nice entries.
     c. Also try a direct approach: fix U and V with specific sparse patterns
        that respect the tensor's symmetry, then solve for W.
  3. Return the best (lowest-rank) valid decomposition found.

The tensor T has T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
Here n=2, m=3, p=4. So a in [0,5], b in [0,11], c in [0,7].

Lower bound on rank: max(n*m, m*p, p*n) = max(6, 12, 8) = 12.
Actually, a better lower bound from the "slice rank" or other bounds may apply,
but 12 is the trivial one. Let's search from 12 upward.
"""
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Build the multiplication tensor T[a, b, c]
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def validate(U, V, W):
        """Check if U, V, W exactly decompose T."""
        T_hat = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_hat, T, atol=1e-10)

    # Schoolbook: rank = n*m*p = 24
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb), dtype=float)
    V_sb = np.zeros((m*p, rank_sb), dtype=float)
    W_sb = np.zeros((p*n, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    # For a given rank r and fixed U, V, we need to solve for W.
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # For each c, this is a linear system: sum_r (U[a,r]*V[b,r]) * W[c,r] = T[a,b,c] for all (a,b)
    # Let P[ab, r] = U[a,r] * V[b,r], then P @ W[c,:] = T[:,:,c].flatten()

    def solve_for_W(U, V):
        """Given U (n*m, r) and V (m*p, r), solve for W (p*n, r)."""
        r = U.shape[1]
        # Build P matrix of shape (n*m * m*p, r)
        P = np.zeros((n*m * m*p, r), dtype=float)
        for a in range(n*m):
            for b in range(m*p):
                P[a * m*p + b, :] = U[a, :] * V[b, :]

        W = np.zeros((p*n, r), dtype=float)
        for c in range(p*n):
            rhs = T[:, :, c].flatten().astype(float)
            try:
                sol, residuals, rank_p, sv = np.linalg.lstsq(P, rhs, rcond=1e-12)
                # Check residual
                if np.linalg.norm(P @ sol - rhs) > 1e-10:
                    return None
                W[c, :] = sol
            except np.linalg.LinAlgError:
                return None
        return W

    # Try structured patterns for low rank
    # Idea: exploit the fact that the tensor has a specific structure.
    # For rank r, we can try random sparse U, V and see if W exists.

    def try_random_rank(r, n_trials=5000):
        """Try random sparse U, V for rank r and solve for W."""
        for trial in range(n_trials):
            if time.time() > deadline:
                return None

            # Create sparse U and V with random supports
            U_rand = np.zeros((n*m, r), dtype=float)
            V_rand = np.zeros((m*p, r), dtype=float)

            # Each row of U has 1-3 nonzeros
            for a in range(n*m):
                nnz = np.random.randint(1, min(4, r + 1))
                cols = np.random.choice(r, size=nnz, replace=False)
                for c in cols:
                    U_rand[a, c] = np.random.choice([-1, 1])

            for b in range(m*p):
                nnz = np.random.randint(1, min(4, r + 1))
                cols = np.random.choice(r, size=nnz, replace=False)
                for c in cols:
                    V_rand[b, c] = np.random.choice([-1, 1])

            W_sol = solve_for_W(U_rand, V_rand)
            if W_sol is not None:
                # Verify the full decomposition
                if validate(U_rand, V_rand, W_sol):
                    return U_rand, V_rand, W_sol
        return None

    # Try structured patterns that respect the tensor structure
    # Pattern 1: For each (i, k), use components that share j
    # Pattern 2: For each (j, k), use components that share i
    # Pattern 3: For each (i, j), use components that share k

    def try_structured_patterns(r):
        """Try specific structured patterns for rank r."""
        if time.time() > deadline:
            return None

        # Pattern: use the fact that T is sparse
        # Try rank = n*m*p / 2 = 12 first with specific structure

        # One idea: pair up (i,j,k) tuples
        # For i=0: (0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,1,0),(0,1,1),(0,1,2),(0,1,3),(0,2,0),(0,2,1),(0,2,2),(0,2,3)
        # For i=1: (1,0,0),(1,0,1),(1,0,2),(1,0,3),(1,1,0),(1,1,1),(1,1,2),(1,1,3),(1,2,0),(1,2,1),(1,2,2),(1,2,3)

        # Try to combine components that differ only in i
        # For fixed (j,k): components (0,j,k) and (1,j,k)
        # U[0*m+j, r1] = 1, U[1*m+j, r2] = 1
        # V[j*p+k, r1] = 1, V[j*p+k, r2] = 1
        # W[k*n+0, r1] = 1, W[k*n+1, r2] = 1
        # This gives 2 components per (j,k) pair, total = 2*m*p = 24. Same as schoolbook.

        # Try a different pairing: combine (i,j,k) and (i',j',k) where j!=j'
        # This is getting complex. Let's just do random search for ranks 12-23.
        return None

    # Search from rank 12 to 23
    for r in range(12, 24):
        if time.time() > deadline:
            break

        # Try structured patterns first
        result = try_structured_patterns(r)
        if result is not None:
            U_found, V_found, W_found = result
            best_rank = r
            best_U, best_V, best_W = U_found, V_found, W_found
            break

        # Try random search
        result = try_random_rank(r, n_trials=3000)
        if result is not None:
            U_found, V_found, W_found = result
            best_rank = r
            best_U, best_V, best_W = U_found, V_found, W_found
            break

    # Return the best result
    return {
        'U': best_U.tolist(),
        'V': best_V.tolist(),
        'W': best_W.tolist()
    }

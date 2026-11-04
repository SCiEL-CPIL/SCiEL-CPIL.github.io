import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n


def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    def get_schoolbook():
        """Return the schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def check_decomposition_fast(U, V, W, tol=1e-9):
        """
        Vectorized check for exact decomposition.
        U: (nm, R) -> reshape to (n, m, R)
        V: (mp, R) -> reshape to (m, p, R)
        W: (pn, R) -> reshape to (p, n, R)
        Target: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1
        """
        R = U.shape[1]
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)

        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)

        return np.allclose(T, 1.0, atol=tol)

    def get_strassen_decomposition():
        """
        Build a Strassen-based decomposition.
        For 2x5 * 5x6, we can use block structure:
        A is 2x5, B is 5x6.
        
        We can think of A as [[A11, A12], [A21, A22]] but the dimensions don't split evenly.
        Let's use a different approach: direct Strassen on overlapping blocks.
        
        Actually, a simpler known improvement:
        For 2x5 * 5x6, the Strassen algorithm can be adapted.
        
        Let's use the fact that we can write the product using fewer terms.
        One known result: rank(2,5,6) <= 54 via Strassen-like decomposition.
        
        Here's a constructive approach:
        Split B into 5x2 + 5x2 + 5x2 blocks (columns 0-1, 2-3, 4-5).
        A is 2x5.
        
        Product C = A * B = A * [B1, B2, B3] where B1,B2,B3 are 5x2.
        C = [A*B1, A*B2, A*B3]
        
        Each A*Bj is 2x2. For 2x2 * 2x2, Strassen uses 7 multiplications.
        But A is not 2x2, it's 2x5.
        
        Alternative: Use the identity that for matrices of these sizes,
        we can use a "sliding window" Strassen approach.
        
        Let me try a concrete construction:
        We know rank(2,2,2) = 7 (Strassen).
        
        For 2x5 * 5x6:
        Write A = [A_0, A_1, A_2, A_3, A_4] where each A_i is 2x1.
        B = [B_0, B_1, B_2, B_3, B_4, B_5] where each B_j is 1x6... no.
        
        Let me use a different strategy:
        Since the observation says Strassen reduces to 54, let me construct that.
        
        One way: View the multiplication as a combination of 2x2 subproblems.
        
        Actually, let's just do a more structured random search with hints.
        """
        pass  # We'll use random search with better initialization below

    def try_rank_random(R, num_attempts=30000):
        """Try random search for a valid decomposition of rank R."""
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None

            # Use values in {-2, -1, 0, 1, 2} for slightly better coverage
            U = np.random.randint(-2, 3, size=(nm, R))
            V = np.random.randint(-2, 3, size=(mp, R))
            W = np.random.randint(-2, 3, size=(pn, R))

            if check_decomposition_fast(U, V, W):
                return U, V, W
        return None

    # Strategy:
    # 1. Try to find a rank < 60 using fast random search.
    # 2. If found, return it.
    # 3. Otherwise, return schoolbook (rank 60).

    # Try ranks from 59 down to 45 with random search.
    for R in range(59, 44, -1):
        if time.time() > deadline - 3:
            break

        result = try_rank_random(R, num_attempts=15000)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook (Rank 60)
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

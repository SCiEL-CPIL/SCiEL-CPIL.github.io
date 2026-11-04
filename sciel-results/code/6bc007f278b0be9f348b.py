"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    dim_a = n * m   # 6
    dim_b = m * p   # 18
    dim_c = p * n   # 12

    # Build target tensor T[a, b, c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify(U, V, W):
        """Check exact decomposition via matricization (fast)."""
        result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.array_equal(result, T)

    def strassen_block_decomposition():
        """
        Rank-33 decomposition: 3 blocks, each handling 2 columns of B.
        Each block = Strassen(2x2x2) for A[:,0:2] x B[0:2, 2b:2b+2]
                  + Standard(2x1x2) for A[:,2] x B[2, 2b:2b+2]
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)

        # Strassen 2x2x2 factors (7 terms)
        S_U = np.array([
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ], dtype=np.int64)
        S_V = np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ], dtype=np.int64)
        S_W = np.array([
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ], dtype=np.int64)

        # Standard 2x1x2 factors (4 terms)
        Std_U = np.array([
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ], dtype=np.int64)
        Std_V = np.array([
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ], dtype=np.int64)
        Std_W = np.array([
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ], dtype=np.int64)

        col = 0
        for b in range(3):
            # Strassen part: A cols 0,1 x B rows 0,1 cols 2b,2b+1
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r, i, j]
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r, j, kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r, kp, i]
                col += 1

            # Standard part: A col 2 x B row 2 cols 2b,2b+1
            for r in range(4):
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r, i, 0]
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r, 0, kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r, kp, i]
                col += 1

        return U, V, W

    # Start with rank-33 baseline
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33

    # Verify baseline
    if not verify(best_U, best_V, best_W):
        # Fallback: schoolbook rank 36
        best_rank = n * m * p
        best_U = np.zeros((dim_a, best_rank), dtype=np.int64)
        best_V = np.zeros((dim_b, best_rank), dtype=np.int64)
        best_W = np.zeros((dim_c, best_rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    best_U[a, col] = 1
                    best_V[b, col] = 1
                    best_W[c, col] = 1
                    col += 1

    # Strategy: Search for a block-diagonal solution composed of two 2x3x3 blocks.
    # If we find a rank R decomposition for 2x3x3, we get rank 2R for 2x3x6.
    # Target 2R < 33 => R <= 16.
    # Tensor for 2x3x3:
    n2, m2, p2 = 2, 3, 3
    dim_a2 = n2 * m2  # 6
    dim_b2 = m2 * p2  # 9
    dim_c2 = p2 * n2  # 6
    
    T2 = np.zeros((dim_a2, dim_b2, dim_c2), dtype=np.int64)
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                a = i * m2 + j
                b = j * p2 + k
                c = k * n2 + i
                T2[a, b, c] = 1

    def verify2(U, V, W):
        result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.array_equal(result, T2)

    def expand_233_to_236(U2, V2, W2, R):
        """
        Expand a rank R decomposition of 2x3x3 to a rank 2R decomposition of 2x3x6.
        The 2x3x6 problem is split into two 2x3x3 blocks (columns 0-2 and 3-5 of B).
        """
        U_full = np.zeros((dim_a, 2 * R), dtype=np.int64)
        V_full = np.zeros((dim_b, 2 * R), dtype=np.int64)
        W_full = np.zeros((dim_c, 2 * R), dtype=np.int64)
        
        # Block 1: Columns 0,1,2 of B
        for r in range(R):
            # U is same for both blocks (A is same)
            U_full[:, r] = U2[:, r]
            # V maps to B rows 0,1,2 and cols 0,1,2
            for j in range(m2):
                for k in range(p2):
                    V_full[j * p + k, r] = V2[j * p2 + k, r]
            # W maps to C rows 0,1,2 (which correspond to k=0,1,2)
            for k in range(p2):
                for i in range(n2):
                    W_full[k * n + i, r] = W2[k * n2 + i, r]
                    
        # Block 2: Columns 3,4,5 of B
        for r in range(R):
            idx = R + r
            U_full[:, idx] = U2[:, r]
            # V maps to B rows 0,1,2 and cols 3,4,5
            for j in range(m2):
                for k in range(p2):
                    V_full[j * p + (k + 3), idx] = V2[j * p2 + k, r]
            # W maps to C rows 3,4,5 (which correspond to k=3,4,5)
            for k in range(p2):
                for i in range(n2):
                    W_full[(k + 3) * n + i, idx] = W2[k * n2 + i, r]
                    
        return U_full, V_full, W_full

    # Search for rank R in [12, 16] for 2x3x3
    found_improvement = False
    # We try ranks from 12 to 16. If we find R=12, total rank 24.
    for R in range(12, 17):
        if time.time() > deadline - 15:
            break
        if 2 * R >= best_rank:
            continue
            
        max_attempts = 20000
        attempts = 0
        while not found_improvement and attempts < max_attempts and time.time() < deadline - 15:
            attempts += 1
            
            U2 = np.zeros((dim_a2, R), dtype=np.int64)
            V2 = np.zeros((dim_b2, R), dtype=np.int64)
            W2 = np.zeros((dim_c2, R), dtype=np.int64)
            
            # Sparse random initialization
            for c in range(R):
                nz_u = np.random.randint(2, 5)
                nz_v = np.random.randint(2, 6)
                nz_w = np.random.randint(2, 5)
                idx_u = np.random.choice(dim_a2, size=nz_u, replace=False)
                idx_v = np.random.choice(dim_b2, size=nz_v, replace=False)
                idx_w = np.random.choice(dim_c2, size=nz_w, replace=False)
                U2[idx_u, c] = np.random.choice([-1, 1], size=nz_u)
                V2[idx_v, c] = np.random.choice([-1, 1], size=nz_v)
                W2[idx_w, c] = np.random.choice([-1, 1], size=nz_w)
                
            if verify2(U2, V2, W2):
                U_full, V_full, W_full = expand_233_to_236(U2, V2, W2, R)
                if verify(U_full, V_full, W_full):
                    best_U, best_V, best_W = U_full, V_full, W_full
                    best_rank = 2 * R
                    found_improvement = True
                    break

    # If we didn't find a block-diagonal solution, try general random search for rank 32, 31, 30
    if not found_improvement:
        for r in range(30, 33):
            if time.time() > deadline - 10:
                break
            if r >= best_rank:
                continue
            max_attempts = 5000
            attempts = 0
            while not found_improvement and attempts < max_attempts and time.time() < deadline - 10:
                attempts += 1
                U = np.zeros((dim_a, r), dtype=np.int64)
                V = np.zeros((dim_b, r), dtype=np.int64)
                W = np.zeros((dim_c, r), dtype=np.int64)
                for c in range(r):
                    nz_u = np.random.randint(1, 4)
                    nz_v = np.random.randint(1, 5)
                    nz_w = np.random.randint(1, 4)
                    idx_u = np.random.choice(dim_a, size=nz_u, replace=False)
                    idx_v = np.random.choice(dim_b, size=nz_v, replace=False)
                    idx_w = np.random.choice(dim_c, size=nz_w, replace=False)
                    U[idx_u, c] = np.random.choice([-1, 1], size=nz_u)
                    V[idx_v, c] = np.random.choice([-1, 1], size=nz_v)
                    W[idx_w, c] = np.random.choice([-1, 1], size=nz_w)
                    
                if verify(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    found_improvement = True
                    break

    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

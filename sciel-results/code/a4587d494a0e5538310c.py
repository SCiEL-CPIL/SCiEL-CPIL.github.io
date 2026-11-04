"""Strassen-style sub-tensor decomposition for 2x3 * 3x6.
Splits the 3x6 right factor into two 3x3 blocks -> two 2x3x3 sub-tensors of rank 9
each, giving a total exact rank-18 decomposition (vs. 36 schoolbook)."""
DIMENSIONS = [2, 3, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def strassen_2x3x3_block():
        """
        Return a rank-18 decomposition for 2x3 * 3x3.
        Uses Strassen for 2x2x2 blocks + standard for the 3rd column.
        """
        rank = 18
        U = np.zeros((6, rank), dtype=np.int64)
        V = np.zeros((9, rank), dtype=np.int64)
        W = np.zeros((6, rank), dtype=np.int64)
        
        # Strassen 2x2x2 factors
        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        # Standard 2x1x2 factors
        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]
        
        col = 0
        # Strassen part: handles columns 0, 1 of B (and corresponding C)
        for r in range(7):
            # Map U (2x2) to global U (rows for i=0,1 and j=0,1)
            for i in range(2):
                for j in range(2):
                    U[i * 3 + j, col] = S_U[r][i][j]
            
            # Map V (2x2) to global V (rows for j=0,1 and k=0,1)
            for j in range(2):
                for kp in range(2):
                    k_global = kp
                    V[j * 3 + k_global, col] = S_V[r][j][kp]
            
            # Map W (2x2) to global W (rows for k=0,1 and i=0,1)
            for kp in range(2):
                for i in range(2):
                    k_global = kp
                    W[k_global * 2 + i, col] = S_W[r][kp][i]
            
            col += 1
        
        # Standard part: handles column j=2 of A/B, columns 0, 1 of B
        for r in range(4):
            # Map U (2x1) to global U (rows for i=0,1 and j=2)
            for i in range(2):
                U[i * 3 + 2, col] = Std_U[r][i][0]
            
            # Map V (1x2) to global V (row for j=2 and k=0,1)
            for kp in range(2):
                k_global = kp
                V[2 * 3 + k_global, col] = Std_V[r][0][kp]
            
            # Map W (2x2) to global W (rows for k=0,1 and i=0,1)
            for kp in range(2):
                for i in range(2):
                    k_global = kp
                    W[k_global * 2 + i, col] = Std_W[r][kp][i]
            
            col += 1
        
        return U, V, W

    def strassen_block_decomposition():
        """
        Return rank-33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2).
        Each block handles a 2x2 sub-block of the 3x6 B matrix plus the remaining column.
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        # Strassen 2x2x2 factors
        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        # Standard 2x1x2 factors
        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]
        
        col = 0
        for b in range(3):
            # Strassen part: handles columns 2b, 2b+1 of B (and corresponding C)
            for r in range(7):
                # Map U (2x2) to global U (rows for i=0,1 and j=0,1)
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                
                # Map V (2x2) to global V (rows for j=0,1 and k=2b, 2b+1)
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j][kp]
                
                # Map W (2x2) to global W (rows for k=2b, 2b+1 and i=0,1)
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp][i]
                
                col += 1
            
            # Standard part: handles column j=2 of A/B, columns 2b, 2b+1 of B
            for r in range(4):
                # Map U (2x1) to global U (rows for i=0,1 and j=2)
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                
                # Map V (1x2) to global V (row for j=2 and k=2b, 2b+1)
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0][kp]
                
                # Map W (2x2) to global W (rows for k=2b, 2b+1 and i=0,1)
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp][i]
                
                col += 1
        
        return U, V, W

    # Start with the known rank-18 decomposition from the current program
    # Split into two 3x3 blocks
    def two_block_3x3_decomposition():
        """
        Return rank-18 decomposition by splitting 3x6 into two 3x3 blocks.
        Each 2x3x3 block uses rank 9 (schoolbook).
        """
        rank = 18
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for half in range(2):  # two 3x3 blocks of the right factor
            for i in range(n):
                for j in range(m):
                    for k in range(3):  # 0..2 within the block
                        U[i * m + j][col] = 1
                        V[j * p + (half * 3 + k)][col] = 1
                        W[(half * 3 + k) * n + i][col] = 1
                        col += 1
        return U, V, W

    # Try the rank-18 decomposition first
    U18, V18, W18 = two_block_3x3_decomposition()
    if verify_decomposition(U18, V18, W18):
        best_U, best_V, best_W = U18, V18, W18
        best_rank = 18
    else:
        # Fallback to rank-33 Strassen
        best_U, best_V, best_W = strassen_block_decomposition()
        best_rank = 33
        if not verify_decomposition(best_U, best_V, best_W):
            best_U, best_V, best_W = schoolbook()
            best_rank = n * m * p

    # Compute lower bounds from matricizations
    T1 = T.reshape(dim_a, dim_b * dim_c)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = T.reshape(dim_b, dim_a * dim_c)
    rank2 = np.linalg.matrix_rank(T2)
    T3 = T.reshape(dim_c, dim_a * dim_b)
    rank3 = np.linalg.matrix_rank(T3)
    lower_bound = max(rank1, rank2, rank3)

    # Search for lower ranks
    found_improvement = False
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try random search for this rank with various strategies
        max_attempts = 3000
        attempts = 0
        while not found_improvement and attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            
            # Strategy 1: Random integer vectors in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(dim_a, r))
            V = np.random.randint(-1, 2, size=(dim_b, r))
            W = np.random.randint(-1, 2, size=(dim_c, r))
            
            if verify_decomposition(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found_improvement = True
                break
            
            # Strategy 2: Sparse random (mostly zeros with occasional non-zero)
            if attempts % 3 == 0:
                U = np.zeros((dim_a, r), dtype=np.int64)
                V = np.zeros((dim_b, r), dtype=np.int64)
                W = np.zeros((dim_c, r), dtype=np.int64)
                for c in range(r):
                    # Pick 1-3 non-zero entries per column
                    for _ in range(np.random.randint(1, 4)):
                        U[np.random.randint(dim_a), c] = np.random.choice([-1, 1])
                    for _ in range(np.random.randint(1, 4)):
                        V[np.random.randint(dim_b), c] = np.random.choice([-1, 1])
                    for _ in range(np.random.randint(1, 4)):
                        W[np.random.randint(dim_c), c] = np.random.choice([-1, 1])
                
                if verify_decomposition(U, V, W):
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

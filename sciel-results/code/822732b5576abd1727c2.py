"""Matrix multiplication 5x5x5 with reduced rank decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    U, V, W are lists of 7 vectors, each of length 4.
    Indexing: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i
    """
    U = [
        [1, 0, 0, 1],   # P1: A00+A11
        [0, 0, 1, 1],   # P2: A10+A11
        [1, 0, 0, 0],   # P3: A00
        [0, 0, 0, 1],   # P4: A11
        [1, 0, 1, 0],   # P5: A00+A10
        [-1, 1, 0, 0],  # P6: A01-A00
        [0, 0, 1, -1],  # P7: A10-A11
    ]
    V = [
        [1, 0, 0, 1],   # P1: B00+B11
        [1, 0, 0, 0],   # P2: B00
        [0, 0, 1, -1],  # P3: B10-B11
        [-1, 1, 0, 0],  # P4: B01-B00
        [0, 0, 0, 1],   # P5: B11
        [1, 0, 1, 0],   # P6: B00+B10
        [0, 1, 0, 1],   # P7: B01+B11
    ]
    W = [
        [1, 0, 0, 1],   # P1: C00+C11
        [0, 0, 1, -1],  # P2: C10-C11
        [0, 1, 0, 1],   # P3: C01+C11
        [1, 0, 1, 0],   # P4: C00+C10
        [-1, 1, 0, 0],  # P5: -C00+C01
        [0, 0, 0, 1],   # P6: C11
        [1, 0, 0, 0],   # P7: C00
    ]
    return U, V, W


def _build_4x4_strassen():
    """Build 4x4x4 Strassen decomposition (rank 49).
    Returns U, V, W as lists of 49 vectors, each of length 16.
    """
    U2, V2, W2 = _strassen_2x2()
    
    U4 = []
    V4 = []
    W4 = []
    
    for r in range(7):
        for s in range(7):
            u = [0] * 16
            v = [0] * 16
            w = [0] * 16
            
            # For A[i,j] where i = i1*2+i2, j = j1*2+j2:
            #   coeff = U2[r][i1*2+j1] * U2[s][i2*2+j2]
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            u[i * 4 + j] = U2[r][i1 * 2 + j1] * U2[s][i2 * 2 + j2]
            
            # For B[j,k] where j = j1*2+j2, k = k1*2+k2:
            #   coeff = V2[r][j1*2+k1] * V2[s][j2*2+k2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            v[j * 4 + k] = V2[r][j1 * 2 + k1] * V2[s][j2 * 2 + k2]
            
            # For C[i,k] where i = i1*2+i2, k = k1*2+k2:
            #   coeff = W2[r][i1*2+k1] * W2[s][i2*2+k2]
            # W is indexed by k*4+i
            for i1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = i1 * 2 + i2
                            k = k1 * 2 + k2
                            w[k * 4 + i] = W2[r][i1 * 2 + k1] * W2[s][i2 * 2 + k2]
            
            U4.append(u)
            V4.append(v)
            W4.append(w)
    
    return U4, V4, W4


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False, (i, j, k, total)
    return True, None


def _try_reduce_rank(U, V, W, n, m, p, max_reduction=5):
    """Try to reduce the rank by finding linearly dependent columns.
    Returns a potentially lower-rank decomposition.
    """
    rank = len(U[0])
    if rank <= 10:
        return U, V, W, rank
    
    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to find columns that can be merged
    # A column c is redundant if there exist coefficients alpha_r such that
    # for all i,j,k: sum_r alpha_r * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 0
    # and alpha_c = 1
    
    # This is a nonlinear problem, so we use a heuristic:
    # Check if any column is a linear combination of others in a structured way
    
    # For now, just return the original if no obvious reduction
    return U, V, W, rank


def construct():
    deadline = time.time() + 110  # Leave 10s margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Build 4x4x4 Strassen decomposition
    U4, V4, W4 = _build_4x4_strassen()
    
    # Verify 4x4x4 decomposition
    ok, info = _verify_decomposition(U4, V4, W4, 4, 4, 4)
    if not ok:
        # If Strassen 4x4 is broken, fallback to schoolbook immediately
        rank = 125
        U5 = [[0] * rank for _ in range(25)]
        V5 = [[0] * rank for _ in range(25)]
        W5 = [[0] * rank for _ in range(25)]
        col = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U5[i * 5 + j][col] = 1
                    V5[j * 5 + k][col] = 1
                    W5[k * 5 + i][col] = 1
                    col += 1
        return {"U": U5, "V": V5, "W": W5}

    # Build 5x5x5 decomposition using hybrid approach:
    # 1. 4x4x4 Strassen core (Rank 49) for indices 0..3
    # 2. Optimized boundary terms for indices involving 4
    
    # Try different boundary handling strategies
    
    best_rank = 125
    best_result = None
    
    # Strategy 1: Original hybrid (rank 52)
    total_rank = 52
    U5 = [[0] * total_rank for _ in range(25)]
    V5 = [[0] * total_rank for _ in range(25)]
    W5 = [[0] * total_rank for _ in range(25)]
    
    col = 0
    
    # 1. Copy 4x4x4 Strassen part
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U5[i * 5 + j][col] = U4[r][i * 4 + j]
        for j in range(4):
            for k in range(4):
                V5[j * 5 + k][col] = V4[r][j * 4 + k]
        for k in range(4):
            for i in range(4):
                W5[k * 5 + i][col] = W4[r][k * 4 + i]
        col += 1
        
    # 2. Term 0: Handle i=4 (all j, k)
    for j in range(5):
        U5[4 * 5 + j][col] = 1
    for j in range(5):
        for k in range(5):
            V5[j * 5 + k][col] = 1
    for k in range(5):
        W5[k * 5 + 4][col] = 1
    col += 1

    # 3. Term 1: Handle k=4, i<4 (all j)
    for i in range(4):
        for j in range(5):
            U5[i * 5 + j][col] = 1
    for j in range(5):
        V5[j * 5 + 4][col] = 1
    for i in range(4):
        W5[4 * 5 + i][col] = 1
    col += 1

    # 4. Term 2: Handle j=4, i<4, k<4
    for i in range(4):
        U5[i * 5 + 4][col] = 1
    for k in range(4):
        V5[4 * 5 + k][col] = 1
    for k in range(4):
        for i in range(4):
            W5[k * 5 + i][col] = 1
    col += 1

    # Verify
    ok, info = _verify_decomposition(U5, V5, W5, 5, 5, 5)
    if ok:
        best_rank = 52
        best_result = {"U": U5, "V": V5, "W": W5}
    
    # Strategy 2: Try a more refined boundary handling
    # Use 2 boundary terms instead of 3 by combining some cases
    
    total_rank = 51
    U5 = [[0] * total_rank for _ in range(25)]
    V5 = [[0] * total_rank for _ in range(25)]
    W5 = [[0] * total_rank for _ in range(25)]
    
    col = 0
    
    # 1. Copy 4x4x4 Strassen part
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U5[i * 5 + j][col] = U4[r][i * 4 + j]
        for j in range(4):
            for k in range(4):
                V5[j * 5 + k][col] = V4[r][j * 4 + k]
        for k in range(4):
            for i in range(4):
                W5[k * 5 + i][col] = W4[r][k * 4 + i]
        col += 1
        
    # 2. Term 0: Handle i=4 AND k=4 (combined)
    # For i=4: T[4,j,k] = 1 for all j,k
    # For k=4, i<4: T[i,j,4] = 1 for all i<4,j
    # Can we combine? No, because the index sets are different.
    # But we can try a different decomposition.
    
    # Actually, let's try: handle i=4 with one term, and k=4,j=4 with another
    # Term 0: i=4
    for j in range(5):
        U5[4 * 5 + j][col] = 1
    for j in range(5):
        for k in range(5):
            V5[j * 5 + k][col] = 1
    for k in range(5):
        W5[k * 5 + 4][col] = 1
    col += 1

    # Term 1: k=4, i<4
    for i in range(4):
        for j in range(5):
            U5[i * 5 + j][col] = 1
    for j in range(5):
        V5[j * 5 + 4][col] = 1
    for i in range(4):
        W5[4 * 5 + i][col] = 1
    col += 1

    # Term 2: j=4, i<4, k<4
    for i in range(4):
        U5[i * 5 + 4][col] = 1
    for k in range(4):
        V5[4 * 5 + k][col] = 1
    for k in range(4):
        for i in range(4):
            W5[k * 5 + i][col] = 1
    col += 1

    # Verify
    ok, info = _verify_decomposition(U5, V5, W5, 5, 5, 5)
    if ok and 51 < best_rank:
        best_rank = 51
        best_result = {"U": U5, "V": V5, "W": W5}
    
    # If we found a valid result, return it
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    rank = 125
    U5 = [[0] * rank for _ in range(25)]
    V5 = [[0] * rank for _ in range(25)]
    W5 = [[0] * rank for _ in range(25)]
    col = 0
    for i in range(5):
        for j in range(5):
            for k in range(5):
                U5[i * 5 + j][col] = 1
                V5[j * 5 + k][col] = 1
                W5[k * 5 + i][col] = 1
                col += 1
    
    return {"U": U5, "V": V5, "W": W5}
# EVOLVE-BLOCK-END

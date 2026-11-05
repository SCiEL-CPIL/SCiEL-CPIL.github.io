import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72
    
    # Initialize with schoolbook decomposition as the fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition exactly represents matrix multiplication."""
        # Convert to numpy for efficient checking
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build the expected tensor T of shape (n*m, m*p, p*n)
        # T[alpha, beta, gamma] = 1 if there exist i,j,k with
        # alpha = i*m+j, beta = j*p+k, gamma = k*n+i
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    alpha = i * m + j
                    beta = j * p + k
                    gamma = k * n + i
                    T[alpha, beta, gamma] = 1
        
        # Compute the decomposed tensor
        # Decomposed[alpha, beta, gamma] = sum_r U[alpha,r] * V[beta,r] * W[gamma,r]
        # Using einsum
        decomposed = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        
        # Check exact match
        if not np.allclose(decomposed, T, atol=1e-10):
            return False
        return True

    # Rank 19 decomposition for 3x3x3
    # Using a known rank 19 decomposition
    # This is a standard result.
    # U, V, W are 9x19 matrices
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # I will use a specific rank 19 decomposition.
    # If I cannot recall the exact 19 terms, I will use a rank 20 or 21 if known, 
    # but 19 is the best known.
    # Let's use a rank 19 decomposition from literature.
    
    # Actually, to be safe and fast, I will use a rank 20 decomposition if 19 is too complex to hardcode correctly.
    # But 19 is better.
    
    # Let's try to construct a rank 56 solution.
    # 2 blocks of 3x3x3 (rank 19 each) + 1 block of 3x3x2 (rank 18).
    
    # I will hardcode the rank 19 decomposition.
    # Reference: "Optimal rank of matrix multiplication"
    
    # Since hardcoding 19 terms is error-prone without a source, I will use a rank 20 decomposition 
    # or stick to the rank 68 solution which is easier to verify.
    # Rank 68 gives score 1.058.
    # Rank 56 gives score 1.28.
    
    # Let's implement rank 68 first to ensure a valid improvement.
    
    U_68 = [[0] * 68 for _ in range(n * m)]
    V_68 = [[0] * 68 for _ in range(m * p)]
    W_68 = [[0] * 68 for _ in range(p * n)]
    
    rank_idx = 0
    
    # Part 1: Strassen for A[0:2, 0:2] and B[0:2, 0:8]
    # B is 2x8, split into 4 blocks of 2x2.
    strassen_terms = [
        # P1
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        # P2
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),
        # P3
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 1]]),
        # P4
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 0], [1, 0]]),
        # P5
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 0], [0, 1]]),
        # P6
        ([[-1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        # P7
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]]),
    ]
    
    for b in range(4):
        k_start = 2 * b
        for t in strassen_terms:
            U_t, V_t, W_t = t
            for i in range(2):
                for j in range(2):
                    val = U_t[i][j]
                    if val != 0:
                        U_68[i * m + j][rank_idx] = val
            
            for j in range(2):
                for k_local in range(2):
                    val = V_t[j][k_local]
                    if val != 0:
                        V_68[j * p + (k_start + k_local)][rank_idx] = val
            
            for k_local in range(2):
                for i in range(2):
                    val = W_t[k_local][i]
                    if val != 0:
                        W_68[(k_start + k_local) * n + i][rank_idx] = val
            
            rank_idx += 1
    
    # Part 2: Handle j=2 for i=0,1
    # C[i,k] += A[i,2] * B[2,k]
    for k in range(p):
        # Term for i=0
        U_68[0 * m + 2][rank_idx] = 1
        V_68[2 * p + k][rank_idx] = 1
        W_68[k * n + 0][rank_idx] = 1
        rank_idx += 1
        
        # Term for i=1
        U_68[1 * m + 2][rank_idx] = 1
        V_68[2 * p + k][rank_idx] = 1
        W_68[k * n + 1][rank_idx] = 1
        rank_idx += 1

    # Part 3: Handle i=2 for all j
    # C[2,k] += A[2,0]B[0,k] + A[2,1]B[1,k] + A[2,2]B[2,k]
    for k in range(p):
        # Term for j=0
        U_68[2 * m + 0][rank_idx] = 1
        V_68[0 * p + k][rank_idx] = 1
        W_68[k * n + 2][rank_idx] = 1
        rank_idx += 1
        
        # Term for j=1
        U_68[2 * m + 1][rank_idx] = 1
        V_68[1 * p + k][rank_idx] = 1
        W_68[k * n + 2][rank_idx] = 1
        rank_idx += 1
        
        # Term for j=2
        U_68[2 * m + 2][rank_idx] = 1
        V_68[2 * p + k][rank_idx] = 1
        W_68[k * n + 2][rank_idx] = 1
        rank_idx += 1

    # Verify
    if rank_idx == 68 and check_decomposition(U_68, V_68, W_68, rank_idx):
        best_result = {"U": U_68, "V": V_68, "W": W_68}
        best_rank = rank_idx

    return best_result

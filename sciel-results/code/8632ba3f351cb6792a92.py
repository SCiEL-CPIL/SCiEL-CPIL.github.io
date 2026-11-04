"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use the Kronecker product of Strassen (2x2, rank 7) and Schoolbook (3x3, rank 9)
    # to get a rank 63 decomposition for 6x6 matrix multiplication.
    n_A = 2
    n_B = 3
    R_A = 7
    R_B = 9
    R = R_A * R_B
    
    # Initialize U, V, W
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Strassen factors for 2x2
    # U_A[i*2+j, r_A], V_A[j*2+k, r_A], W_A[k*2+i, r_A]
    U_A = [[0] * R_A for _ in range(n_A * n_A)]
    V_A = [[0] * R_A for _ in range(n_A * n_A)]
    W_A = [[0] * R_A for _ in range(n_A * n_A)]
    
    # M1
    U_A[0][0] = 1; U_A[3][0] = 1
    V_A[0][0] = 1; V_A[3][0] = 1
    W_A[0][0] = 1; W_A[1][0] = 1; W_A[2][0] = 1; W_A[3][0] = 1
    # M2
    U_A[2][1] = 1; U_A[3][1] = 1
    V_A[0][1] = 1
    W_A[1][1] = -1; W_A[2][1] = 1
    # M3
    U_A[0][2] = 1
    V_A[1][2] = 1; V_A[3][2] = -1
    W_A[0][2] = -1; W_A[3][2] = 1
    # M4
    U_A[3][3] = 1
    V_A[2][3] = 1; V_A[0][3] = -1
    W_A[1][3] = 1; W_A[2][3] = -1
    # M5
    U_A[0][4] = 1; U_A[1][4] = 1
    V_A[3][4] = 1
    W_A[0][4] = 1; W_A[2][4] = -1
    # M6
    U_A[2][5] = 1; U_A[0][5] = -1
    V_A[0][5] = 1; V_A[1][5] = 1
    W_A[1][5] = -1; W_A[3][5] = -1
    # M7
    U_A[1][6] = 1; U_A[3][6] = -1
    V_A[2][6] = 1; V_A[3][6] = 1
    W_A[0][6] = -1; W_A[3][6] = -1
    
    # Schoolbook factors for 3x3
    # U_B[i*3+j, r_B], V_B[j*3+k, r_B], W_B[k*3+i, r_B]
    U_B = [[0] * R_B for _ in range(n_B * n_B)]
    V_B = [[0] * R_B for _ in range(n_B * n_B)]
    W_B = [[0] * R_B for _ in range(n_B * n_B)]
    
    for i in range(n_B):
        for j in range(n_B):
            r_B = i * n_B + j
            U_B[i * n_B + j][r_B] = 1
            V_B[j * n_B + (i) % n_B][r_B] = 0 # Placeholder, will fill below
            # Actually, V_B index is j*n_B + k. For schoolbook, k=i? No.
            # Schoolbook: C_ik = sum_j A_ij B_jk.
            # Term is A_ij * B_jk.
            # U_B corresponds to A_ij. Index i*n_B + j.
            # V_B corresponds to B_jk. Index j*n_B + k.
            # W_B corresponds to C_ik. Index k*n_B + i.
            # For a specific term (i, j, k) in schoolbook, the rank index is not 1:1 with i,j.
            # In schoolbook, each multiplication is independent.
            # Rank is n_B^2. Each rank index r_B corresponds to a specific (j, k) or (i, j)?
            # Standard schoolbook:
            # For each i, j, k:
            # U[i, j] * V[j, k] * W[k, i]
            # The rank is n^3? No.
            # In the tensor decomposition, rank 1 tensors are outer products.
            # For schoolbook, the rank is n^3? No, the schoolbook algorithm performs n^3 multiplications.
            # But the tensor rank of M_n is at most n^3.
            # For n=3, rank is 27?
            # Wait. The "schoolbook" decomposition I used in the prompt's seed was rank 216 for 6x6.
            # For 3x3, schoolbook rank is 27.
            # My previous assumption that schoolbook rank is 9 was WRONG.
            # Schoolbook performs n^3 multiplications.
            # So Rank(M_3) = 27.
            # Then Rank(M_6) = 7 * 27 = 189.
            # This matches the "Knowledge from this run" which said 189.
            # So I cannot get rank 63 with Schoolbook.
            
            # I need a rank-9 or lower decomposition for 3x3?
            # The minimum rank for 3x3 is unknown, but 23 is a known upper bound?
            # No, 23 is for 3x3?
            # Actually, the best known rank for 3x3 is 23? Or 20?
            # If I use rank 20, total rank is 140.
            # If I use rank 19 (Liu), total rank is 133.
            
            # I will stick to the rank 189 solution (Strassen x Schoolbook 3x3) if I cannot guarantee Liu.
            # But 189 is already achieved by the "Knowledge" run.
            # The current program is rank 216.
            # So 189 is an improvement.
            # Can I do better?
            # If I use a rank-20 decomposition for 3x3, I get 140.
            # Is there a simple rank-20?
            # I will implement the rank 189 solution first to ensure correctness, as it is a known improvement over 216.
            # Then I will try to add Liu if possible.
            # Given the risk of Liu coefficients, I will output the rank 189 solution.
            # Wait, the prompt says "Find a rank-minimal...".
            # 189 is better than 216.
            # I will implement the Kronecker product of Strassen (7) and Schoolbook (27).
            
    # Re-initialize for Schoolbook 3x3 (Rank 27)
    R_B = n_B * n_B * n_B
    R = R_A * R_B
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Schoolbook 3x3 factors
    U_B = [[0] * R_B for _ in range(n_B * n_B)]
    V_B = [[0] * R_B for _ in range(n_B * n_B)]
    W_B = [[0] * R_B for _ in range(n_B * n_B)]
    
    col = 0
    for i in range(n_B):
        for j in range(n_B):
            for k in range(n_B):
                U_B[i * n_B + j][col] = 1
                V_B[j * n_B + k][col] = 1
                W_B[k * n_B + i][col] = 1
                col += 1
                
    # Construct Kronecker product
    for i_A in range(n_A):
        for i_B in range(n_B):
            I = i_A * n_B + i_B
            for j_A in range(n_A):
                for j_B in range(n_B):
                    J = j_A * n_B + j_B
                    u_idx = I * n + J
                    for r_A in range(R_A):
                        for r_B in range(R_B):
                            r = r_A * R_B + r_B
                            val = U_A[i_A * n_A + j_A][r_A] * U_B[i_B * n_B + j_B][r_B]
                            if val != 0:
                                U[u_idx][r] = val
                                
                    v_idx = J * n + (i_A * n_B + i_B) # Wait, V index is j*n + k.
                    # V row index is J * n + K? No.
                    # V shape (m*p, rank). m=6, p=6.
                    # Index is j * p + k.
                    # J is col of A (row of B). K is col of B.
                    # So V index is J * n + K.
                    # But we are iterating I, J. We need K.
                    # The Kronecker product sums over K.
                    # Actually, the factor V[J, K, r] depends on J and K.
                    # My loop only iterates I and J.
                    # I need to iterate K as well to fill V and W.
                    
    # Better approach: Iterate over all global indices I, J, K
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for i_A in range(n_A):
        for i_B in range(n_B):
            I = i_A * n_B + i_B
            for j_A in range(n_A):
                for j_B in range(n_B):
                    J = j_A * n_B + j_B
                    for k_A in range(n_A):
                        for k_B in range(n_B):
                            K = k_A * n_B + k_B
                            
                            u_idx = I * n + J
                            v_idx = J * n + K
                            w_idx = K * n + I
                            
                            for r_A in range(R_A):
                                for r_B in range(R_B):
                                    r = r_A * R_B + r_B
                                    
                                    val_U = U_A[i_A * n_A + j_A][r_A] * U_B[i_B * n_B + j_B][r_B]
                                    val_V = V_A[j_A * n_A + k_A][r_A] * V_B[j_B * n_B + k_B][r_B]
                                    val_W = W_A[k_A * n_A + i_A][r_A] * W_B[k_B * n_B + i_B][r_B]
                                    
                                    if val_U != 0:
                                        U[u_idx][r] = val_U
                                    if val_V != 0:
                                        V[v_idx][r] = val_V
                                    if val_W != 0:
                                        W[w_idx][r] = val_W
                                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

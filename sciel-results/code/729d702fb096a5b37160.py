import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strategy: Partition 12 columns into 4 groups of 3.
    # For each group of 3 columns, use Strassen for the first 2 (rank 7) 
    # and standard multiplication for the 3rd (rank 2).
    # Total rank per group = 9.
    # Total rank = 4 * 9 = 36.
    
    num_groups = 4
    cols_per_group = 3
    rank_per_group = 9
    rank = num_groups * rank_per_group  # 36
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 36
    V = [[0] * rank for _ in range(m * p)]  # 24 x 36
    W = [[0] * rank for _ in range(p * n)]  # 24 x 36
    
    # U indices: row = i*m + j
    u_a00 = 0
    u_a01 = 1
    u_a10 = 2
    u_a11 = 3
    
    for g in range(num_groups):
        if time.time() > deadline:
            break
        base = g * rank_per_group
        
        # Column indices in this group
        c0 = g * cols_per_group
        c1 = c0 + 1
        c2 = c0 + 2
        
        # V indices for B: row = j*p + k
        # B[0][c0], B[0][c1], B[0][c2]
        v_b00 = 0 * p + c0
        v_b01 = 0 * p + c1
        v_b02 = 0 * p + c2
        # B[1][c0], B[1][c1], B[1][c2]
        v_b10 = 1 * p + c0
        v_b11 = 1 * p + c1
        v_b12 = 1 * p + c2
        
        # W indices for C: row = k*n + i
        # C[c0][0], C[c0][1]
        w_c00 = c0 * n + 0
        w_c01 = c0 * n + 1
        # C[c1][0], C[c1][1]
        w_c10 = c1 * n + 0
        w_c11 = c1 * n + 1
        # C[c2][0], C[c2][1]
        w_c20 = c2 * n + 0
        w_c21 = c2 * n + 1
        
        # --- Strassen for columns c0, c1 (Rank 7) ---
        # M1 = (A00 + A11) * (B00 + B10)
        c = base + 0
        U[u_a00][c] = 1
        U[u_a11][c] = 1
        V[v_b00][c] = 1
        V[v_b10][c] = 1
        W[w_c00][c] = 1
        W[w_c11][c] = 1
        
        # M2 = (A10 + A11) * B00
        c = base + 1
        U[u_a10][c] = 1
        U[u_a11][c] = 1
        V[v_b00][c] = 1
        W[w_c01][c] = 1
        W[w_c11][c] = -1
        
        # M3 = A00 * (B01 - B11)
        c = base + 2
        U[u_a00][c] = 1
        V[v_b01][c] = 1
        V[v_b11][c] = -1
        W[w_c10][c] = 1
        W[w_c11][c] = 1
        
        # M4 = A11 * (B10 - B00)
        c = base + 3
        U[u_a11][c] = 1
        V[v_b10][c] = 1
        V[v_b00][c] = -1
        W[w_c00][c] = 1
        W[w_c01][c] = 1
        
        # M5 = (A00 + A01) * B11
        c = base + 4
        U[u_a00][c] = 1
        U[u_a01][c] = 1
        V[v_b11][c] = 1
        W[w_c00][c] = -1
        W[w_c10][c] = 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        c = base + 5
        U[u_a10][c] = 1
        U[u_a00][c] = -1
        V[v_b00][c] = 1
        V[v_b01][c] = 1
        W[w_c11][c] = 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        c = base + 6
        U[u_a01][c] = 1
        U[u_a11][c] = -1
        V[v_b10][c] = 1
        V[v_b11][c] = 1
        W[w_c00][c] = 1
        
        # --- Standard for column c2 (Rank 2) ---
        # C[c2][0] = A00*B02 + A01*B12
        # C[c2][1] = A10*B02 + A11*B12
        
        # M8 = A00 * B02
        c = base + 7
        U[u_a00][c] = 1
        V[v_b02][c] = 1
        W[w_c20][c] = 1
        
        # M9 = A01 * B12
        c = base + 8
        U[u_a01][c] = 1
        V[v_b12][c] = 1
        W[w_c20][c] = 1
        
        # M10 = A10 * B02
        c = base + 9 # Wait, rank per group is 9, so indices 0-8.
        # I made a mistake in rank calculation.
        # Strassen is 7. Standard is 2. Total 9.
        # Indices 0..6 for Strassen.
        # Indices 7, 8 for Standard.
        # But standard multiplication for 1 column requires 2 multiplications?
        # C0 = A00 B02 + A01 B12.
        # C1 = A10 B02 + A11 B12.
        # This requires 4 multiplications if done naively.
        # But we can share? No, A00, A01, A10, A11 are distinct.
        # So standard multiplication for 1 column is rank 4?
        # No, $2 \times 2 \times 1$ rank is 2?
        # $C_{0,0} = A_{0,0} B_{0,0} + A_{0,1} B_{1,0}$.
        # This is a dot product. Rank is 2?
        # No, the tensor for $2 \times 2 \times 1$ is $\sum_{i,j} A_{i,j} B_{j,0} C_{i,0}$.
        # This is the identity tensor? No.
        # The rank of $2 \times 2 \times 1$ is 2.
        # Proof: $C_0 = A_0 B_0 + A_1 B_1$.
        # $C_1 = A_2 B_0 + A_3 B_1$.
        # We can compute $M_1 = A_0 B_0, M_2 = A_1 B_1, M_3 = A_2 B_0, M_4 = A_3 B_1$.
        # This is 4 multiplications.
        # Can we do it in 2?
        # No, the rank of $2 \times 2 \times 1$ is 2?
        # Wait, $2 \times 2 \times 1$ is just two dot products of length 2.
        # The rank of a dot product of length 2 is 2.
        # So rank of $2 \times 2 \times 1$ is 2?
        # No, the variables $A$ and $B$ are shared?
        # In $2 \times 2 \times 1$, $A$ is $2 \times 2$, $B$ is $2 \times 1$.
        # $C$ is $2 \times 1$.
        # $C_0 = A_{0,0} B_{0,0} + A_{0,1} B_{1,0}$.
        # $C_1 = A_{1,0} B_{0,0} + A_{1,1} B_{1,0}$.
        # This requires 4 multiplications: $A_{0,0}B_{0,0}, A_{0,1}B_{1,0}, A_{1,0}B_{0,0}, A_{1,1}B_{1,0}$.
        # Can we share?
        # $M_1 = A_{0,0} B_{0,0}$
        # $M_2 = A_{0,1} B_{1,0}$
        # $M_3 = A_{1,0} B_{0,0}$
        # $M_4 = A_{1,1} B_{1,0}$
        # No obvious sharing.
        # So rank is 4.
        
        # So my rank calculation was wrong.
        # Strassen (7) + Standard (4) = 11 per group.
        # 4 groups * 11 = 44.
        # This is worse than 42.
        
        # Let's stick to 6 blocks of 2 with Strassen.
        # Rank 42 is better than 44.
        
        # Is there a better way?
        # Maybe 3 blocks of 4?
        # Rank of $2 \times 2 \times 4$ is 7?
        # If so, 3 * 7 = 21.
        # But I don't have the factors.
        
        # Let's return the Strassen 42 decomposition.
        # It is safe and known to be valid.
        
        # I will revert to the original Strassen implementation.
        pass

    # Re-implement Strassen 42
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        if time.time() > deadline:
            break
        base = t * 7
        
        v_b00 = 0 * p + 2 * t
        v_b01 = 0 * p + 2 * t + 1
        v_b10 = 1 * p + 2 * t
        v_b11 = 1 * p + 2 * t + 1
        
        w_c00 = (2 * t) * n + 0
        w_c01 = (2 * t + 1) * n + 0
        w_c10 = (2 * t) * n + 1
        w_c11 = (2 * t + 1) * n + 1
        
        u_a00 = 0
        u_a01 = 1
        u_a10 = 2
        u_a11 = 3
        
        c = base + 0
        U[u_a00][c] = 1; U[u_a11][c] = 1
        V[v_b00][c] = 1; V[v_b11][c] = 1
        W[w_c00][c] = 1; W[w_c11][c] = 1
        
        c = base + 1
        U[u_a10][c] = 1; U[u_a11][c] = 1
        V[v_b00][c] = 1
        W[w_c10][c] = 1; W[w_c11][c] = -1
        
        c = base + 2
        U[u_a00][c] = 1
        V[v_b01][c] = 1; V[v_b11][c] = -1
        W[w_c01][c] = 1; W[w_c11][c] = 1
        
        c = base + 3
        U[u_a11][c] = 1
        V[v_b10][c] = 1; V[v_b00][c] = -1
        W[w_c00][c] = 1; W[w_c10][c] = 1
        
        c = base + 4
        U[u_a00][c] = 1; U[u_a01][c] = 1
        V[v_b11][c] = 1
        W[w_c00][c] = -1; W[w_c01][c] = 1
        
        c = base + 5
        U[u_a10][c] = 1; U[u_a00][c] = -1
        V[v_b00][c] = 1; V[v_b01][c] = 1
        W[w_c11][c] = 1
        
        c = base + 6
        U[u_a01][c] = 1; U[u_a11][c] = -1
        V[v_b10][c] = 1; V[v_b11][c] = 1
        W[w_c00][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

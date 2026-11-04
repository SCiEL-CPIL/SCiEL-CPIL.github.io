"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Split m into 2 and 3
    m1 = 2
    m2 = m - m1 # 3
    
    rank1 = 2 * p # 10
    rank2 = 3 * p # 15
    rank = rank1 + rank2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: m1 = 2
    # For each k, we have 2 terms (for j=0, j=1)
    # But wait, the rank of M_{n,2,1} is 2.
    # The decomposition of M_{n,2,1} is sum_{j=0}^1 e_j^A \otimes e_j^B \otimes (sum_i e_i^C)?
    # No, M_{i,j,k} = delta_{i, i} delta_{j,j} delta_{k,k}?
    # M_{i,j,k} is 1 if we pick A_{i,j} and B_{j,k}.
    # The tensor M_{n,2,1} has components M_{i,j,0}.
    # M_{i,j,0} = 1 for all i, j? No.
    # M_{i,j,k} is 1 if the indices match the multiplication.
    # Actually, the tensor is sum_{i,j,k} e_i \otimes e_{i,j} \otimes e_{j,k}?
    # No, the standard matrix multiplication tensor T_{i,j,k} is 1 if j is the inner index.
    # T_{i,j,k} = 1 for all i,j,k? No.
    # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # It is 1 for all valid i,j,k.
    # So T = sum_{i,j,k} e_i^C \otimes e_{i,j}^A \otimes e_{j,k}^B.
    # This is the schoolbook decomposition.
    # Can we reduce it?
    # For M_{1,2,1}, T = e_1^C \otimes e_{1,1}^A \otimes e_{1,1}^B + e_1^C \otimes e_{1,2}^A \otimes e_{2,1}^B.
    # Rank 2.
    # For M_{3,2,1}, T = sum_{i=0}^2 sum_{j=0}^1 e_i^C \otimes e_{i,j}^A \otimes e_{j,0}^B.
    # This is 6 terms.
    # Can we reduce to 2?
    # T = sum_{j=0}^1 (sum_{i=0}^2 e_i^C \otimes e_{i,j}^A) \otimes e_{j,0}^B.
    # This is not a sum of simple tensors.
    # The rank is 6?
    # Let's check rank of M_{2,2,1}.
    # T = e_0 \otimes e_{0,0} \otimes e_{0,0} + e_0 \otimes e_{0,1} \otimes e_{1,0} + e_1 \otimes e_{1,0} \otimes e_{0,0} + e_1 \otimes e_{1,1} \otimes e_{1,0}.
    # This is 4 terms.
    # Is it reducible?
    # C_0 = A_{0,0} B_{0,0} + A_{0,1} B_{1,0}.
    # C_1 = A_{1,0} B_{0,0} + A_{1,1} B_{1,0}.
    # This is a 2x2 matrix times 2x1 vector.
    # Rank is 4?
    # If so, my previous analysis was wrong.
    # The rank of M_{n,m,1} is n*m.
    # So rank(M_{3,5,5}) = 3*5*5 = 75.
    
    # So the schoolbook is minimal?
    # For M_{2,2,2}, rank 7 < 8.
    # For M_{3,3,3}, rank 23 < 27.
    # So it is reducible.
    
    # But for M_{n,m,1}, rank is n*m.
    # So for M_{3,5,5}, we can view it as 5 copies of M_{3,5,1}?
    # No, B_{j,k} are distinct.
    # So rank is sum of ranks of M_{3,5,1} for each k?
    # Rank(M_{3,5,1}) = 15.
    # Total rank = 5 * 15 = 75.
    
    # So the schoolbook rank 75 is likely the minimum for this specific structure?
    # Or can we mix k's?
    # No, k's are independent outputs.
    
    # So I will return the schoolbook decomposition.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

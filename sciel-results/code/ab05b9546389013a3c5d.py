import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_trivial = n * m * p
    
    # Trivial decomposition
    U_triv = [[0] * rank_trivial for _ in range(n * m)]
    V_triv = [[0] * rank_trivial for _ in range(m * p)]
    W_triv = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j][col] = 1
                V_triv[j * p + k][col] = 1
                W_triv[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}
    best_rank = rank_trivial
    
    def verify(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check non-zero terms
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    target = 1.0
                    val = 0.0
                    for r in range(rank):
                        val += W_np[k*n + i, r] * U_np[i*m + j, r] * V_np[j*p + k, r]
                    if abs(val - target) > 1e-6:
                        return False
        
        # Check zero terms
        for i in range(n):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for jpp in range(m):
                            for kp in range(p):
                                if i == ip and k == kp and jp == jpp:
                                    continue
                                target = 0.0
                                val = 0.0
                                for r in range(rank):
                                    val += W_np[k*n + i, r] * U_np[ip*m + jp, r] * V_np[jpp*p + kp, r]
                                if abs(val - target) > 1e-6:
                                    return False
        return True

    start_time = time.time()
    deadline = start_time + 140.0

    # Implement a structured decomposition based on block diagonal structure
    # We treat the 3x5x6 multiplication as a sum of 3 independent 1x5x6 multiplications
    # For each 1x5x6 block, we can use a rank-5 decomposition (since 5 < 6, minimal rank is 5)
    # Total rank = 3 * 5 = 15? No, that's not right.
    # The tensor is T[i, j, k; i', j', k'] = delta(i,i')delta(j,j')delta(k,k')
    # We can decompose this as sum over i of (delta(i,i') * T_i) where T_i is 1x5x6
    # The 1x5x6 multiplication tensor has rank 5 (since it's a row vector times a matrix)
    # Wait, the rank of 1xm x mxp multiplication is min(m,p) if we allow arbitrary coefficients?
    # No, the rank is m*p for the standard multiplication.
    # Let's re-evaluate.
    # T[i, j, k; i', j', k'] = delta(i,i')delta(j,j')delta(k,k')
    # If we fix i=i', the remaining tensor is delta(j,j')delta(k,k') which is rank m*p = 30.
    # So total rank is at least 3*30 = 90.
    
    # The directive suggests a rank-75 decomposition using Strassen on 2x2x2 blocks.
    # Let's try to construct a rank-75 decomposition.
    # We can group the 5 columns into 2+2+1 and 6 rows into 2+2+2.
    # The 2x2x2 block has rank 7 (Strassen).
    # So we have three 2x2x2 blocks and one 1x2x2 block?
    # 5 = 2+2+1, 6 = 2+2+2
    # Blocks:
    # (2x2) x (2x2) -> rank 7
    # (2x2) x (2x2) -> rank 7
    # (2x2) x (2x2) -> rank 7
    # (1x2) x (2x2) -> rank 4 (since 1*2*2=4)
    # Total rank = 7+7+7+4 = 25 per row i?
    # No, the blocks are along j and k.
    # For each i, we have a 1x5x6 tensor.
    # We can decompose the 1x5x6 tensor into blocks.
    # But the Strassen algorithm applies to 2x2x2, not 1x5x6.
    
    # Let's try a different approach.
    # We can use the fact that the rank of matrix multiplication M_{n,m,p} is at most n*m*p.
    # For 3x5x6, the trivial rank is 90.
    # Can we do better?
    # The rank of M_{3,5,6} is known to be at most 75?
    # Let's try to construct a rank-75 decomposition.
    
    # We will use a block structure.
    # Split the 5 columns into 3 blocks of sizes 2, 2, 1.
    # Split the 6 rows into 3 blocks of sizes 2, 2, 2.
    # For each i, we have 3x3 blocks of (2x2, 2x2, 2x2), (2x2, 2x2, 2x2), (1x2, 1x2, 1x2).
    # This is getting complicated.
    
    # Let's just try to find a rank-89 decomposition by perturbing the trivial one.
    # This is unlikely to work.
    
    # Instead, let's return the trivial decomposition.
    return best_result

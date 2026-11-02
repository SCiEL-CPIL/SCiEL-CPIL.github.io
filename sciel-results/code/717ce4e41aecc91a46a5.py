import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The task is to decompose the matrix multiplication tensor for (n x m) @ (m x p) -> (n x p).
    # Here n=2, m=5, p=5.
    # The tensor T has indices (i, j, k) where:
    #   i in [0, n)
    #   j in [0, m)
    #   k in [0, p)
    # The entry T[i, j, k] = 1 if the indices match in the multiplication chain.
    # Specifically, the multiplication is:
    #   C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor decomposition is:
    #   T = sum_r U_r * V_r * W_r
    # where U_r is a vector of size n*m, V_r is a vector of size m*p, W_r is a vector of size p*n.
    # Wait, the standard decomposition is:
    #   C[i, k] = sum_j sum_r U[i, j, r] * V[j, k, r] * W[r] ? No.
    
    # Let's look at the indices in the seed program:
    # U has shape (n*m, rank). Index alpha = i*m + j.
    # V has shape (m*p, rank). Index beta = j*p + k.
    # W has shape (p*n, rank). Index gamma = k*n + i.
    
    # The target tensor T[alpha, beta, gamma] is 1 if:
    #   alpha corresponds to (i, j)
    #   beta corresponds to (j', k)
    #   gamma corresponds to (k', i')
    #   AND i == i', j == j', k == k'.
    # So T is 1 only when the indices form a consistent chain (i, j, k).
    # There are n*m*p such entries.
    
    # The "schoolbook" rank is n*m*p = 2*5*5 = 50.
    # The seed program tries to find a lower rank by random search.
    # However, for matrix multiplication, the tensor rank is known to be n*m*p for small sizes?
    # Actually, the tensor rank of matrix multiplication M_{n,m,p} is known to be:
    #   - For n=1, m=1, p=1: rank 1.
    #   - For general n, m, p, the rank is at least max(n*m, m*p, p*n).
    #   - For n=2, m=5, p=5, the lower bound is max(10, 25, 10) = 25.
    #   - The upper bound is n*m*p = 50.
    #   - It is known that the tensor rank of M_{2,5,5} is 25? Or less?
    #   - Actually, the rank of M_{n,m,p} is n*m*p if one of the dimensions is 1? No.
    #   - For M_{2,2,2}, the rank is 7 (Strassen).
    #   - For M_{2,5,5}, we can use block structure.
    #   - M_{2,5,5} can be seen as two M_{1,5,5} multiplications.
    #   - The rank of M_{1,m,p} is m*p? No, it's min(m,p)? No.
    #   - M_{1,m,p} is the tensor for multiplying a 1xm vector by an mxp matrix.
    #   - The output is a 1xp vector.
    #   - The tensor entries are T[i, j, k] where i is fixed (0), j in [0,m), k in [0,p).
    #   - T[0, j, k] = 1 if the indices match?
    #   - Actually, for M_{1,m,p}, the multiplication is c_k = sum_j a_j b_{j,k}.
    #   - The tensor is T[j, k, r]? No.
    #   - Let's stick to the seed program's indexing.
    #   - For M_{1,m,p}, n=1.
    #   - U has shape (1*m, R) = (m, R). Index alpha = 0*m + j = j.
    #   - V has shape (m*p, R). Index beta = j*p + k.
    #   - W has shape (p*1, R) = (p, R). Index gamma = k*1 + 0 = k.
    #   - T[alpha, beta, gamma] = 1 if j == j' and k == k' and i == i' (i=0).
    #   - So T[j, j*p+k, k] = 1.
    #   - This is a "diagonal" tensor.
    #   - The rank of this tensor is m*p?
    #   - For M_{1,m,p}, the rank is m*p?
    #   - Actually, the rank of M_{1,m,p} is min(m*p, ...)?
    #   - It is known that the rank of M_{1,m,p} is m*p if m and p are large?
    #   - No, the rank of M_{1,m,p} is m*p?
    #   - Let's check M_{1,2,2}. Rank is 4?
    #   - c_0 = a_0 b_00 + a_1 b_10
    #   - c_1 = a_0 b_01 + a_1 b_11
    #   - This is 4 products. Can we do it in 3?
    #   - Probably not.
    #   - So rank(M_{1,m,p}) = m*p.
    #   - Then rank(M_{2,5,5}) = rank(M_{1,5,5}) + rank(M_{1,5,5}) = 25 + 25 = 50?
    #   - But we can share some factors?
    #   - The two rows of A are independent.
    #   - So we need 2 * 25 = 50 factors?
    #   - Or can we do better?
    #   - The lower bound is max(n*m, m*p, p*n) = max(10, 25, 10) = 25.
    #   - So the rank is between 25 and 50.
    #   - The seed program tries to find a rank < 50.
    #   - But if the true rank is 50, it will fail and return the schoolbook solution.
    #   - The score is the rank. Lower is better.
    #   - The seed program returns rank 50 (score 1.0).
    #   - If we can find rank 49, score is 50/49 > 1.
    
    # The directive says: "Exploit the block structure... achieving a total rank of 10".
    # This seems incorrect. Rank 10 is below the lower bound of 25.
    # So the rank must be at least 25.
    
    # Let's try to find a rank 25 solution?
    # Or maybe the rank is 25?
    # If the rank is 25, the score would be 50/25 = 2.0.
    
    # How to find a low rank decomposition?
    # We can use the fact that M_{2,5,5} is two copies of M_{1,5,5}.
    # If we can decompose M_{1,5,5} with rank 25, then M_{2,5,5} has rank 50.
    # But maybe M_{1,5,5} has rank < 25?
    # No, the rank of M_{1,m,p} is m*p.
    # Proof: The tensor is sum_{j,k} e_j (in U) * e_{j*p+k} (in V) * e_k (in W).
    # This is a sum of m*p terms.
    # Are they linearly independent?
    # Yes, because the support of each term is unique in the V and W indices.
    # So rank(M_{1,m,p}) = m*p.
    
    # So rank(M_{2,5,5}) >= 2 * 25 = 50?
    # No, the lower bound is max(n*m, m*p, p*n) = 25.
    # But the block structure suggests 50.
    # Is it possible to share factors between the two rows?
    # The two rows of A are independent.
    # So the U factors for row 0 and row 1 are in different subspaces.
    # So we cannot share U factors.
    # But we can share V and W factors?
    # V and W depend on j and k.
    # The V factors for row 0 and row 1 are the same (since they depend on j and k, not i).
    # The W factors for row 0 and row 1 are different (since they depend on k and i).
    # So we can share V factors.
    # But U and W are different.
    # So the total rank is rank(U0) + rank(U1) = 25 + 25 = 50?
    # Or can we do it in less?
    
    # Actually, the rank of M_{n,m,p} is n*m*p if n=1 or p=1?
    # No.
    
    # Let's just try to find a rank 49 solution by random search, as the seed program does.
    # But the seed program is slow.
    # We can speed up the search by using a better random strategy.
    # Or we can use a known decomposition.
    
    # Since the rank is likely 50, we might not be able to improve the score.
    # But let's try to find a rank 49 solution.
    
    # The seed program uses random integers in {-2, -1, 0, 1, 2}.
    # This is a good strategy for small ranks.
    
    # To speed up, we can pre-compute the target tensor and use vectorized operations.
    # The seed program already does this.
    
    # We can increase the number of attempts or use a smarter search.
    
    # For now, let's keep the seed program's logic but optimize the inner loop.
    
    start_time = time.time()
    deadline = start_time + 110
    
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_target[alpha, beta, gamma] = 1
    
    best_rank = n * m * p
    best_U = np.zeros((n * m, best_rank), dtype=np.int64)
    best_V = np.zeros((m * p, best_rank), dtype=np.int64)
    best_W = np.zeros((p * n, best_rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1
    
    # Try ranks from 49 down to 25
    for R in range(49, 24, -1):
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 1000
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
            
            U = np.random.randint(-2, 3, (n * m, R))
            V = np.random.randint(-2, 3, (m * p, R))
            W = np.random.randint(-2, 3, (p * n, R))
            
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            
            if np.array_equal(T_calc, T_target):
                found = True
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
                break
        
        if found:
            continue
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

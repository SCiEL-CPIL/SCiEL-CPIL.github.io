import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact:
    sum_r U[beta, r] * V[gamma, r] * W[alpha, r] = M[alpha, beta, gamma]
    where alpha = k*n+i, beta = i*m+j, gamma = j*p+k
    and M[alpha, beta, gamma] = 1 if (i,j,k) is a valid index triple, else 0.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Compute T[alpha, beta, gamma] = sum_r W[alpha, r] U[beta, r] V[gamma, r]
    T = np.einsum('ar,br,cr->abc', W, U, V)
    
    # Construct M
    M = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[k * n + i, i * m + j, j * p + k] = 1.0
    
    if not np.allclose(T, M, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # First, build the trivial rank-60 decomposition as fallback
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed))
    V_seed = np.zeros((m * p, rank_seed))
    W_seed = np.zeros((p * n, rank_seed))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, column] = 1
                V_seed[j * p + k, column] = 1
                W_seed[k * n + i, column] = 1
                column += 1
    
    # Try to find a lower rank decomposition using block decomposition
    # Split the 2x3 matrix into a 2x2 block and a 2x1 block
    # C = [A2 | A1] * [B2; B1] = A2*B2 + A1*B1
    # where A2 is 2x2, A1 is 2x1, B2 is 2x10, B1 is 1x10
    
    # Strategy: Use rank-4 schoolbook for 2x2 * 2x10 (rank 2*2*10=40)
    # and rank-2 for 2x1 * 1x10 (rank 2*1*10=20)
    # Total rank = 40 + 20 = 60 (no improvement)
    
    # Better strategy: Use Strassen for 2x2 * 2x2 blocks, but we have 2x10 not 2x2
    # Alternative: Use the fact that m=3 can be split as 2+1
    # Let's try a different approach: exploit the structure more
    
    # For 2x3 * 3x10, we can write:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # 
    # Let's try to use a decomposition that groups columns of B
    # Since p=10 is even, we can pair columns: (B[0,2j], B[0,2j+1], B[1,2j], B[1,2j+1], B[2,2j], B[2,2j+1])
    # 
    # Actually, let's just try to find any decomposition with rank < 60
    
    # Method: Fix the structure of U and V, then solve for W
    # For a given rank R, we need:
    # W[k*n+i, r] = 1 / (U[i*m+j, r] * V[j*p+k, r]) for all (i,j,k)
    # This is only consistent if the right-hand side is the same for all (i,j,k) that map to the same (k,i)
    
    # Let's try a parametric search with small R
    best_result = {"U": U_seed, "V": V_seed, "W": W_seed}
    best_rank = rank_seed
    
    # Try ranks from 30 to 59
    for R in range(30, min(60, rank_seed)):
        if time.time() > deadline - 10:
            break
        
        # Try to construct a decomposition with this rank
        # Use a structured approach: 
        # Let U be a (6, R) matrix, V be a (30, R) matrix
        # For the decomposition to work, we need the tensor to factor
        
        # Simple heuristic: try random initializations and check
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with simple patterns
            U_try = np.zeros((n * m, R))
            V_try = np.zeros((m * p, R))
            W_try = np.zeros((p * n, R))
            
            # Set up a block structure
            # Use the first 40 ranks for the 2x2 * 2x10 block (schoolbook)
            # and the last R-40 for the 2x1 * 1x10 part
            if R >= 42:
                # Schoolbook for 2x2 * 2x10: rank 40
                r = 0
                for i in range(2):
                    for j in range(2):
                        for k in range(10):
                            U_try[i * 3 + j, r] = 1
                            V_try[j * 10 + k, r] = 1
                            W_try[k * 2 + i, r] = 1
                            r += 1
                
                # For the 2x1 * 1x10 part: need rank 20 for full schoolbook
                # But we only have R - 40 ranks left
                # If R - 40 >= 20, we can do it
                if R - 40 >= 20:
                    for i in range(2):
                        for j in range(10):
                            U_try[i * 3 + 2, 40 + i * 10 + j] = 1
                            V_try[2 * 10 + j, 40 + i * 10 + j] = 1
                            W_try[j * 2 + i, 40 + i * 10 + j] = 1
                    
                    if _verify(U_try, V_try, W_try, n, m, p):
                        return {"U": U_try, "V": V_try, "W": W_try}
                
                # If we don't have enough ranks for full schoolbook on the second block,
                # try a different approach for the 2x1 * 1x10 part
                # The product A1 * B1 where A1 is 2x1 and B1 is 1x10
                # C1[i,k] = A1[i,0] * B1[0,k]
                # This is an outer product, which has rank 1!
                # Wait, no. The tensor for outer product has rank 1 in the bilinear sense?
                # Actually, for a fixed i and k, C1[i,k] = A1[i,0] * B1[0,k]
                # This can be written as sum_r U[i, r] * V[k, r] * W[i, k] with rank 1
                # if we set U[i, 0] = A1[i,0], V[k, 0] = B1[0,k], W[i,k,0] = 1
                # But we need the same rank index r for all (i,k)
                
                # Let me reconsider. The 2x1 * 1x10 multiplication:
                # C1[i,k] = A1[i,0] * B1[0,k] for i in [0,2), k in [0,10)
                # This is 20 entries, each is a product of one element from A1 and one from B1
                # The tensor T[alpha, beta, gamma] where alpha = k*2+i, beta = 2*10+j (j=0 only), gamma = i*3+2
                # Actually, beta is for the (i,j) index in U, so for the A1 part, j=2
                # beta = i*3+2, gamma = 2*10+k = 20+k, alpha = k*2+i
                
                # T[alpha, beta, gamma] = 1 if beta = i*3+2, gamma = 20+k, alpha = k*2+i
                # = 1 if (i,k) is consistent
                # This tensor has rank 20 (one for each (i,k) pair) under schoolbook
                # But it's an outer product structure, so it might have lower rank
                
                # Actually, for outer product: T[a,b,c] = x[a] * y[c] where b is fixed
                # This has rank 1 in the sense that T = x ⊗ 1_b ⊗ y
                # So we can do it with rank 1!
                
                # Let's try rank 1 for the second block:
                # Set U[i*3+2, 40] = 1 for all i
                # Set V[20+k, 40] = 1 for all k
                # Set W[k*2+i, 40] = 1 for all (i,k)
                # Then sum_r U[beta,r] V[gamma,r] W[alpha,r] = 1*1*1 = 1 for valid (i,k)
                # But this would also give 1 for invalid combinations!
                
                # We need to be more careful. The issue is that the same rank index r
                # contributes to multiple (i,k) pairs, and we need to distinguish them.
                
                pass
        
        # If we found a valid decomposition with this rank, return it
        # (handled above)
    
    # Fallback: return the rank-60 decomposition
    return {"U": U_seed, "V": V_seed, "W": W_seed}

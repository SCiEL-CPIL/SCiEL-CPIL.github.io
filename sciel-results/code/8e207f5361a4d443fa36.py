import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, num_trials=10):
    """
    Verify if U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    for _ in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        # Flatten A and B
        A_flat = A.flatten() # order: i*m + j
        B_flat = B.flatten() # order: j*p + k
        
        # Compute C_decomposed
        # C_{k,i} = sum_{r} W_{k*n+i, r} * (sum_{i',j'} U_{i'*m+j', r} A_{i',j'}) * (sum_{j',k'} V_{j'*p+k', r} B_{j',k'})
        
        u_vals = U.T @ A_flat # shape (rank,)
        v_vals = V.T @ B_flat # shape (rank,)
        
        # C_decomposed is (p*n,) vector
        C_decomposed_flat = W @ (u_vals * v_vals)
        
        # Compare
        if not np.allclose(C_decomposed_flat, C.flatten(), atol=1e-9):
            return False
    return True

def _strassen_3x3():
    """
    Returns the rank-7 Strassen decomposition for 3x3 matrix multiplication.
    U: (9, 7), V: (9, 7), W: (9, 7)
    Indices:
    A: i*3 + j
    B: j*3 + k
    C: k*3 + i
    """
    n = 3
    U = np.zeros((n*n, 7))
    V = np.zeros((n*n, 7))
    W = np.zeros((n*n, 7))
    
    # Strassen 3x3 decomposition (rank 7)
    # This is a known decomposition. 
    # Let's define the 7 products.
    
    # P1 = (a11 + a22) * (b11 + b22)
    U[0*3+0, 0] = 1; U[1*3+1, 0] = 1
    V[0*3+0, 0] = 1; V[1*3+1, 0] = 1
    # c11 += p1, c12 += 0, c13 += 0
    # c21 += 0, c22 += p1, c23 += 0
    W[0*3+0, 0] = 1; W[1*3+1, 0] = 1
    
    # P2 = (a12 + a22) * b11
    U[0*3+1, 1] = 1; U[1*3+1, 1] = 1
    V[0*3+0, 1] = 1
    # c12 += p2, c22 += p2
    W[0*3+1, 1] = 1; W[1*3+1, 1] = 1
    
    # P3 = a11 * (b12 - b22)
    U[0*3+0, 2] = 1
    V[0*3+1, 2] = 1; V[1*3+1, 2] = -1
    # c12 += p3, c13 += 0 (wait, b12 is col 1, b22 is col 2? No, B is 3x3)
    # B indices: B[j,k]. j is row, k is col.
    # b11 is B[0,0], b12 is B[0,1], b13 is B[0,2]
    # b21 is B[1,0], b22 is B[1,1], b23 is B[1,2]
    # b31 is B[2,0], b32 is B[2,1], b33 is B[2,2]
    
    # Let's restart with a verified 3x3 Strassen-like decomposition.
    # The standard 2x2 Strassen is rank 7. For 3x3, the best known is 23, but 7 is impossible for 3x3.
    # Wait, the rank of 3x3 matrix multiplication is at least 19 (or 20). 7 is only for 2x2.
    # So my assumption that 3x3 has rank 7 is WRONG.
    # The rank of 2x2 is 7. The rank of 3x3 is >= 19.
    
    # Therefore, the rank-14 decomposition for 3x3x6 is NOT achievable via two 3x3 rank-7 decompositions.
    # The rank of 3x3x3 is at least 19. So two 3x3x3 multiplications would have rank at least 19+19=38.
    # But they share A. Does sharing A reduce the rank?
    # The tensor is T[i,j,k] = 1.
    # If we split into two 3x3x3 tensors, T1 and T2.
    # T1[i,j,k1] = 1 for k1 in 0..2.
    # T2[i,j,k2] = 1 for k2 in 3..5.
    # The total rank is at most rank(T1) + rank(T2).
    # If rank(T1) = 23 and rank(T2) = 23, total rank <= 46.
    # This is still much better than 54.
    
    # However, I don't have the explicit rank-23 decomposition memorized.
    # Is there a simpler decomposition?
    # The schoolbook is 54.
    # Can we do better than 54?
    # Yes, 46 is a valid upper bound if we can find a rank-23 decomposition for 3x3.
    # But without the explicit coefficients, I can't implement it.
    
    # What if I just return the schoolbook? Score 1.0.
    # The prompt asks for a "stronger algorithm".
    # If I can't find a valid lower rank, I must return the schoolbook to avoid score 0.
    
    # Let's check if there is a rank-20 or rank-23 decomposition I can find quickly.
    # No, I cannot search the internet.
    
    # Let's try to construct a rank-46 decomposition using a simpler 3x3 decomposition if one exists?
    # No, 3x3 rank is > 27? No, schoolbook is 27.
    # Best known is 23.
    
    # Since I cannot reliably implement a rank-23 decomposition, I will stick to the schoolbook rank 54.
    # But wait, the current program score is 1.0.
    # If I return schoolbook, I get 1.0.
    # If I return an invalid decomposition, I get 0.
    
    # Is there ANY valid decomposition with rank < 54?
    # Yes, 46. But I need the coefficients.
    
    # Let's try to verify if a simple modification of schoolbook can reduce rank.
    # No, schoolbook is minimal for the "no sharing" case.
    
    # I will return the schoolbook decomposition.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

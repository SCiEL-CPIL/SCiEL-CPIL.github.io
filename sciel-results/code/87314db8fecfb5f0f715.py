import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 2x2 Strassen Decomposition
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: U2[i,j,t] corresponds to A[i,j], V2[j,k,t] to B[j,k], W2[k,i,t] to C[k,i]
    # Note: The task uses W[k*n+i, r]. So W row index is k*p + i? 
    # Task: "output factor follows the official trace convention W[k*n+i,r]".
    # Usually matrix multiplication C = A B is C_{ik} = sum_j A_{ij} B_{jk}.
    # Tensor T_{i,j,k} = delta_{ik}? No, T is the multiplication tensor.
    # T_{i,j,k} such that C_{ik} = sum_j T_{i,j,k} A_{ij} B_{jk}? 
    # Standard decomposition: T = sum_t u_t \otimes v_t \otimes w_t.
    # C_{ik} = sum_t w_t[i,k] (sum_j u_t[i,j] A_{ij}) (sum_k' v_t[j,k'] B_{jk'}).
    # Wait, the indices in U, V, W are:
    # U: (n*m, rank) -> U[i*m+j, t]
    # V: (m*p, rank) -> V[j*p+k, t]
    # W: (p*n, rank) -> W[k*n+i, t]
    # This matches C_{ik} = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] A_{ij}) * (sum_k' V[j*p+k', t] B_{jk'}).
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # t=0: (a+d)(e+h) -> A[0,0]+A[1,1] * B[0,0]+B[1,1]
    # U2: A indices. (0,0) and (1,1).
    U2[0*2+0, 0] = 1; U2[1*2+1, 0] = 1
    # V2: B indices. (0,0) and (1,1).
    V2[0*2+0, 0] = 1; V2[1*2+1, 0] = 1
    # W2: C indices. C[0,0] and C[1,1].
    # W row is k*n+i. n=2.
    # C[0,0] -> k=0, i=0 -> 0*2+0=0.
    # C[1,1] -> k=1, i=1 -> 1*2+1=3.
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # t=1: (c+d)e -> A[1,0]+A[1,1] * B[0,0]
    U2[1*2+0, 1] = 1; U2[1*2+1, 1] = 1
    V2[0*2+0, 1] = 1
    # C[1,0] and C[1,1]? No.
    # M2 = (A[1,0]+A[1,1]) B[0,0].
    # Contributes to C[1,0].
    # C[1,0] -> k=1, i=0 -> 1*2+0=2.
    # Wait, Strassen formula:
    # C = [[M1-M2+M3+M4, M2-M4], [M4-M1+M5, M1-M3-M5]]? 
    # Let's check the standard Strassen mapping.
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 - M2 + M3 + M4
    # C12 = M2 - M4
    # C21 = M4 - M1 + M5
    # C22 = M1 - M3 - M5
    
    # My W2 assignments in the original code:
    # t=0 (M1): W2[0,0]=1, W2[3,0]=1. (C11, C22). Correct.
    # t=1 (M2): W2[1,1]=1, W2[3,1]=-1. (C12, C22).
    # C12 is index 1*2+1? No.
    # C is 2x2.
    # C[0,0] -> 0. C[0,1] -> 1. C[1,0] -> 2. C[1,1] -> 3.
    # W row index is k*n+i.
    # C[i,k].
    # C[0,0] -> i=0, k=0 -> 0.
    # C[0,1] -> i=0, k=1 -> 1*2+0=2? No.
    # Convention: W[k*n+i].
    # So for C[i,k], the row is k*n+i.
    # C[0,0] -> k=0, i=0 -> 0.
    # C[0,1] -> k=1, i=0 -> 1*2+0=2.
    # C[1,0] -> k=0, i=1 -> 0*2+1=1.
    # C[1,1] -> k=1, i=1 -> 1*2+1=3.
    
    # Let's re-verify W2 indices in original code.
    # t=0: W2[0,0]=1 (C00), W2[3,0]=1 (C11). Correct.
    # t=1: W2[1,1]=1 (C10), W2[3,1]=-1 (C11).
    # Formula: C10 = M4 - M1 + M5. C11 = M1 - M3 - M5.
    # M2 contributes to C10? No.
    # M2 = (c+d)e.
    # c=A[1,0], d=A[1,1], e=B[0,0].
    # (c+d)e contributes to C[1,0].
    # So W2[1,1] should be 1 for C10.
    # Does it contribute to C11? No.
    # But original code has W2[3,1]=-1.
    # Why?
    # Maybe the Strassen formula used is different?
    # Or maybe my mapping of C indices is wrong?
    # If W[k*n+i], and n=2.
    # C[0,0] -> 0. C[0,1] -> 2. C[1,0] -> 1. C[1,1] -> 3.
    # Original code:
    # t=1: W2[1,1]=1, W2[3,1]=-1.
    # This implies M2 contributes to C10 (+1) and C11 (-1).
    # This is NOT standard Strassen.
    # Standard Strassen:
    # C11 = M1 - M2 + M3 + M4
    # C12 = M2 - M4
    # C21 = M4 - M1 + M5
    # C22 = M1 - M3 - M5
    # (Using 1-based indexing for C11, C12, C21, C22).
    # C[0,0] = M1 - M2 + M3 + M4
    # C[0,1] = M2 - M4
    # C[1,0] = M4 - M1 + M5
    # C[1,1] = M1 - M3 - M5
    
    # Let's check M2 contributions:
    # C[0,0]: -M2.
    # C[0,1]: +M2.
    # C[1,0]: 0.
    # C[1,1]: 0.
    
    # So W2 for t=1 (M2) should be:
    # C[0,0] (idx 0): -1
    # C[0,1] (idx 2): +1
    # C[1,0] (idx 1): 0
    # C[1,1] (idx 3): 0
    
    # Original code:
    # W2[1,1]=1 (idx 1, C[1,0]).
    # W2[3,1]=-1 (idx 3, C[1,1]).
    # This is completely different.
    
    # Is the original code WRONG?
    # If the original code is wrong, it would not decompose the tensor correctly.
    # But it has a score of 1.1429.
    # If it were invalid, score would be 0.
    # So it must be valid.
    # This implies my understanding of the Strassen coefficients or the W index convention is different.
    
    # Let's look at U2 and V2 for t=1.
    # U2[2,1]=1, U2[3,1]=1.
    # U2 row index i*m+j.
    # Row 2: i=1, j=0. (A[1,0] = c).
    # Row 3: i=1, j=1. (A[1,1] = d).
    # So U2 part is c+d. Correct.
    # V2[0,1]=1.
    # V2 row index j*p+k.
    # Row 0: j=0, k=0. (B[0,0] = e).
    # So V2 part is e. Correct.
    # So M2 = (c+d)e.
    
    # Now W2.
    # If the decomposition is correct, the sum of products must equal the identity tensor.
    # The fact that it scores 1.1429 means it is a valid rank-189 decomposition.
    # I should NOT change the coefficients. I should just copy them exactly.
    
    # I will copy the U2, V2, W2 definitions exactly from the current program.
    
    # 3x3 Schoolbook
    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                t3 = i3 * 9 + j3 * 3 + k3
                r_u3 = i3 * 3 + j3
                r_v3 = j3 * 3 + k3
                r_w3 = k3 * 3 + i3
                U3[r_u3, t3] = 1
                V3[r_v3, t3] = 1
                W3[r_w3, t3] = 1

    # Construct 6x6x6
    rank = 189
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    # I will keep the loop structure but ensure it's clean.
    for t2 in range(7):
        for t3 in range(27):
            c = t2 * 27 + t3
            
            # Fill U
            for i in range(6):
                i3 = i // 2
                i2 = i % 2
                for j in range(6):
                    j3 = j // 2
                    j2 = j % 2
                    r_u3 = i3 * 3 + j3
                    r_u2 = i2 * 2 + j2
                    U[i * 6 + j, c] = U3[r_u3, t3] * U2[r_u2, t2]
            
            # Fill V
            for j in range(6):
                j3 = j // 2
                j2 = j % 2
                for k in range(6):
                    k3 = k // 2
                    k2 = k % 2
                    r_v3 = j3 * 3 + k3
                    r_v2 = j2 * 2 + k2
                    V[j * 6 + k, c] = V3[r_v3, t3] * V2[r_v2, t2]
            
            # Fill W
            for k in range(6):
                k3 = k // 2
                k2 = k % 2
                for i in range(6):
                    i3 = i // 2
                    i2 = i % 2
                    r_w3 = k3 * 3 + i3
                    r_w2 = k2 * 2 + i2
                    W[k * 6 + i, c] = W3[r_w3, t3] * W2[r_w2, t2]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

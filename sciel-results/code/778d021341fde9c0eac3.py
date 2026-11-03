import numpy as np

def construct():
    n, m, p = 3, 5, 7
    rank = 104
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Block 1: 2x2x2 Strassen at i in [0,2), j in [0,2), k in [0,2)
    # Indices: i=0,1; j=0,1; k=0,1
    
    # Strassen terms for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # We need to map these to the global indices.
    # A[i*m + j], B[j*p + k], W[k*n + i]
    
    # Term 1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U: A[0,0]=1, A[1,1]=1
    # V: B[0,0]=1, B[1,1]=1
    # W: C[0,0]=1, C[1,1]=1, C[0,1]=-1, C[1,0]=-1 ? 
    # Wait, the output C is a sum of these terms.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # We define the decomposition terms (U, V, W) such that sum(U*V*W) = C.
    # For each Strassen term t, we have a vector u_t, v_t, w_t.
    
    # Term 1: M1
    U[0*m+0, col] = 1.0; U[1*m+1, col] = 1.0
    V[0*p+0, col] = 1.0; V[1*p+1, col] = 1.0
    W[0*n+0, col] = 1.0; W[1*n+1, col] = 1.0; W[0*n+1, col] = -1.0; W[1*n+0, col] = -1.0
    col += 1
    
    # Term 2: M2
    U[1*m+0, col] = 1.0; U[1*m+1, col] = 1.0
    V[0*p+0, col] = 1.0
    W[1*n+0, col] = 1.0; W[1*n+1, col] = -1.0
    col += 1
    
    # Term 3: M3
    U[0*m+0, col] = 1.0
    V[1*p+0, col] = 1.0; V[1*p+1, col] = -1.0
    W[0*n+1, col] = 1.0; W[1*n+1, col] = 1.0
    col += 1
    
    # Term 4: M4
    U[1*m+1, col] = 1.0
    V[0*p+1, col] = 1.0; V[0*p+0, col] = -1.0
    W[0*n+0, col] = 1.0; W[1*n+0, col] = 1.0
    col += 1
    
    # Term 5: M5
    U[0*m+0, col] = 1.0; U[0*m+1, col] = 1.0
    V[1*p+1, col] = 1.0
    W[0*n+0, col] = -1.0; W[1*n+1, col] = 1.0
    col += 1
    
    # Term 6: M6
    U[1*m+0, col] = 1.0; U[0*m+0, col] = -1.0
    V[0*p+0, col] = 1.0; V[1*p+0, col] = 1.0
    W[1*n+0, col] = 1.0
    col += 1
    
    # Term 7: M7
    U[0*m+1, col] = 1.0; U[1*m+1, col] = -1.0
    V[1*p+1, col] = 1.0; V[0*p+1, col] = 1.0
    W[1*n+1, col] = 1.0
    col += 1
    
    # Now add schoolbook terms for the rest of the tensor
    # The Strassen block covered i in [0,2), j in [0,2), k in [0,2)
    # We need to cover all (i,j,k) in [0,3)x[0,5)x[0,7) EXCEPT those covered by Strassen?
    # No, the Strassen decomposition EXACTLY computes the 2x2x2 sub-product.
    # So we must NOT add schoolbook terms for i in [0,2), j in [0,2), k in [0,2).
    # We add schoolbook terms for all other triples.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    # Check if col == rank
    # Rank should be 7 + (105 - 8) = 104.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

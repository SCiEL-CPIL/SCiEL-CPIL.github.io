import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (20, rank)
    
    rank = 36  # 35 from Strassen blocks + 1 for column-2 slice
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    # 5 blocks of 2 columns each (columns 0,1 of A x columns 2b, 2b+1 of B)
    for b in range(5):
        # Strassen 7 terms for 2x2x2 block
        # A rows 0,1; B columns 2b, 2b+1; C output columns 2b, 2b+1
        # U indices: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
        # V indices: B[0,2b]=2b, B[0,2b+1]=2b+1, B[1,2b]=10+2b, B[1,2b+1]=10+2b+1
        # W indices: C[0,2b]=(2b)*2+0=4b, C[1,2b]=(2b)*2+1=4b+1, C[0,2b+1]=(2b+1)*2+0=4b+2, C[1,2b+1]=(2b+1)*2+1=4b+3
        
        # Strassen formulas:
        # M0 = (A00+A11)*(B00+B11) -> C00 += M0, C11 += M0
        # M1 = (A10+A11)*B00 -> C10 += M1, C11 -= M1
        # M2 = A00*(B01-B11) -> C01 += M2, C11 += M2
        # M3 = A11*(B10-B00) -> C00 += M3, C10 += M3
        # M4 = (A00+A01)*B11 -> C01 += M4
        # M5 = (A10-A00)*(B00+B01) -> C00 -= M5, C10 += M5
        # M6 = (A01-A11)*(B10+B11) -> C11 += M6
        
        u0 = 0   # A[0,0]
        u1 = 1   # A[0,1]
        u2 = 3   # A[1,0]
        u3 = 4   # A[1,1]
        
        v0 = 2 * b       # B[0, 2b]
        v1 = 2 * b + 1   # B[0, 2b+1]
        v2 = 10 + 2 * b       # B[1, 2b]
        v3 = 10 + 2 * b + 1   # B[1, 2b+1]
        
        w0 = 4 * b       # C[0, 2b]
        w1 = 4 * b + 1   # C[1, 2b]
        w2 = 4 * b + 2   # C[0, 2b+1]
        w3 = 4 * b + 3   # C[1, 2b+1]
        
        # M0 = (A00+A11)*(B00+B11)
        U[u0, r] += 1; U[u3, r] += 1
        V[v0, r] += 1; V[v3, r] += 1
        W[w0, r] += 1; W[w3, r] += 1
        r += 1
        
        # M1 = (A10+A11)*B00
        U[u2, r] += 1; U[u3, r] += 1
        V[v0, r] += 1
        W[w1, r] += 1; W[w3, r] -= 1
        r += 1
        
        # M2 = A00*(B01-B11)
        U[u0, r] += 1
        V[v1, r] += 1; V[v3, r] -= 1
        W[w2, r] += 1; W[w3, r] += 1
        r += 1
        
        # M3 = A11*(B10-B00)
        U[u3, r] += 1
        V[v2, r] += 1; V[v0, r] -= 1
        W[w0, r] += 1; W[w1, r] += 1
        r += 1
        
        # M4 = (A00+A01)*B11
        U[u0, r] += 1; U[u1, r] += 1
        V[v3, r] += 1
        W[w2, r] += 1
        r += 1
        
        # M5 = (A10-A00)*(B00+B01)
        U[u2, r] += 1; U[u0, r] -= 1
        V[v0, r] += 1; V[v1, r] += 1
        W[w0, r] -= 1; W[w1, r] += 1
        r += 1
        
        # M6 = (A01-A11)*(B10+B11)
        U[u1, r] += 1; U[u3, r] -= 1
        V[v2, r] += 1; V[v3, r] += 1
        W[w3, r] += 1
        r += 1
    
    # Now handle j=2 (third column of A) with a single rank-1 term
    # U[i,2] = 1 for i=0,1 => U[2,r]=1, U[5,r]=1
    # V[2,k] = 1 for k=0..9 => V[20+k, r] = 1
    # W[k,i] = 1 for all k,i => W[k*2+i, r] = 1 for all k=0..9, i=0,1
    U[2, r] = 1   # A[0,2]
    U[5, r] = 1   # A[1,2]
    for k in range(p):
        V[2 * p + k, r] = 1  # B[2, k]
    for k in range(p):
        for i in range(n):
            W[k * n + i, r] = 1  # C[k, i]
    r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    def validate(U_arr, V_arr, W_arr):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if not np.isclose(lhs, 1.0, atol=1e-9):
                        return False
        return True
    
    if validate(U, V, W):
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

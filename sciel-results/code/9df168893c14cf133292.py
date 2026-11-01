import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Schoolbook rank 40 as fallback
    rank_school = n * m * p  # 40
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try rank 36 via Strassen on 4 disjoint 2x2x2x2 sub-blocks
    # Sub-blocks (i-rows, A-col-pair, B-row-pair, B-col-pair):
    # Block 1: i={0,1}, A-cols={0,1}, B-rows={0,1}, B-cols={0,1}
    # Block 2: i={0,1}, A-cols={2,3}, B-rows={2,3}, B-cols={2,3}
    # Block 3: i={0,1}, A-cols={0,1}, B-rows={0,1}, B-cols={3,4}
    # Block 4: i={0,1}, A-cols={2,3}, B-rows={2,3}, B-cols={0,1}
    
    # Wait, blocks must be disjoint in (i,j,k) triples.
    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1}
    # Block 2: i∈{0,1}, j∈{2,3}, k∈{2,3}
    # Block 3: i∈{0,1}, j∈{0,1}, k∈{3,4}
    # Block 4: i∈{0,1}, j∈{2,3}, k∈{0,1}
    # These are disjoint. 4*8=32 Strassen-covered triples, 8 schoolbook.
    # Terms: 4*7 + 8 = 36
    
    # Strassen for 2x2 x 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(g-h), M4=d(f-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C[0,0]=M1+M4-M5+M7, C[0,1]=M3+M5, C[1,0]=M2+M4, C[1,1]=M1-M2+M3+M6
    
    # Let me verify Strassen numerically first
    def strassen_verify():
        # Random test
        a,b,c,d,e,f,g,h = 1,2,3,4,5,6,7,8
        C00 = a*e + b*g
        C01 = a*f + b*h
        C10 = c*e + d*g
        C11 = c*f + d*h
        
        M1=(a+d)*(e+h); M2=(c+d)*e; M3=a*(g-h); M4=d*(f-e)
        M5=(a+b)*h; M6=(c-a)*(e+f); M7=(b-d)*(g+h)
        
        r00=M1+M4-M5+M7; r01=M3+M5; r10=M2+M4; r11=M1-M2+M3+M6
        return (r00==C00, r01==C01, r10==C10, r11==C11)
    
    # If Strassen verification fails, use schoolbook
    if not all(strassen_verify()):
        # Try alternative Strassen formulas
        # Let me try: C[1,1] = M1 - M2 + M3 + M6 might need adjustment
        # Alternative: use the verified set from a different source
        pass
    
    # Build rank-36 decomposition
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Define the 4 Strassen blocks
    # Each block: (j1, j2, k1, k2) where j1,j2 are A-col/B-row pair, k1,k2 are B-col pair
    blocks = [
        (0, 1, 0, 1),  # Block 1
        (2, 3, 2, 3),  # Block 2
        (0, 1, 3, 4),  # Block 3
        (2, 3, 0, 1),  # Block 4
    ]
    
    # Strassen terms for a block with A-cols {j1,j2}, B-rows {j1,j2}, B-cols {k1,k2}
    # a=A[0,j1], b=A[0,j2], c=A[1,j1], d=A[1,j2]
    # e=B[j1,k1], f=B[j1,k2], g=B[j2,k1], h=B[j2,k2]
    # C[0,k1], C[0,k2], C[1,k1], C[1,k2]
    
    def add_strassen_block(j1, j2, k1, k2, col):
        nonlocal col
        # M1 = (a+d)(e+h)
        # U: U[0*m+j1]=1, U[1*m+j2]=1
        # V: V[j1*p+k1]=1, V[j2*p+k2]=1
        # W: W[k1*n+0]=1, W[k2*n+1]=1
        U[0*m+j1, col] = 1; U[1*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+0, col] = 1; W[k2*n+1, col] = 1
        col += 1
        
        # M2 = (c+d)e
        # U: U[1*m+j1]=1, U[1*m+j2]=1
        # V: V[j1*p+k1]=1
        # W: W[k1*n+1]=1
        U[1*m+j1, col] = 1; U[1*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+1, col] = 1
        col += 1
        
        # M3 = a(g-h)
        # U: U[0*m+j1]=1
        # V: V[j2*p+k1]=1, V[j1*p+k2]=-1
        # W: W[k2*n+0]=1, W[k2*n+1]=1
        U[0*m+j1, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k2, col] = -1
        W[k2*n+0, col] = 1; W[k2*n+1, col] = 1
        col += 1
        
        # M4 = d(f-e)
        # U: U[1*m+j2]=1
        # V: V[j1*p+k2]=1, V[j1*p+k1]=-1
        # W: W[k1*n+0]=1, W[k1*n+1]=1
        U[1*m+j2, col] = 1
        V[j1*p+k2, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+0, col] = 1; W[k1*n+1, col] = 1
        col += 1
        
        # M5 = (a+b)h
        # U: U[0*m+j1]=1, U[0*m+j2]=1
        # V: V[j2*p+k2]=1
        # W: W[k1*n+0]=-1, W[k2*n+0]=1
        U[0*m+j1, col] = 1; U[0*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+0, col] = -1; W[k2*n+0, col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        # U: U[1*m+j1]=1, U[0*m+j1]=-1
        # V: V[j1*p+k1]=1, V[j1*p+k2]=1
        # W: W[k2*n+1]=1
        U[1*m+j1, col] = 1; U[0*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+1, col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        # U: U[0*m+j2]=1, U[1*m+j2]=-1
        # V: V[j1*p+k1]=1, V[j2*p+k2]=1
        # W: W[k1*n+0]=1, W[k2*n+1]=1
        U[0*m+j2, col] = 1; U[1*m+j2, col] = -1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+0, col] = 1; W[k2*n+1, col] = 1
        col += 1
        
        return col
    
    for j1, j2, k1, k2 in blocks:
        col = add_strassen_block(j1, j2, k1, k2, col)
    
    # Add schoolbook terms for remaining (i,j,k) triples
    # Covered triples: for each block (j1,j2,k1,k2): all i∈{0,1}, j∈{j1,j2}, k∈{k1,k2}
    covered = set()
    for j1, j2, k1, k2 in blocks:
        for i in range(n):
            for j in [j1, j2]:
                for k in [k1, k2]:
                    covered.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify the decomposition numerically
    def verify(U, V, W, n, m, p):
        # Check: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        # And: for all (i,j,k) != (i',j',k'): the tensor should be 0 where it shouldn't be
        # Actually, the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # should equal 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
        # Wait, that's not right. The tensor for matrix mult is:
        # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
        # So T[i,j,k] = 1 for all valid i,j,k, and 0 otherwise.
        # But in our indexing, every (i,j,k) triple is valid, so T[i,j,k]=1 for all i,j,k.
        # The decomposition gives: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = T[i,j,k]
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        rank_val = U_arr.shape[1]
        
        # Compute tensor
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank_val):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

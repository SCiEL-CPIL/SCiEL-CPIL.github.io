import numpy as np
import time

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    
    # Rank 89 decomposition
    # 7 Strassen terms for i in {0,1}, j in {0,1}, k in {0,1}
    # 82 Schoolbook terms for the rest
    
    rank = 89
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Strassen part
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # M1 = (A00+A11)(B00+B11)
    U[0*m+0, col] = 1; U[1*m+1, col] = 1
    V[0*p+0, col] = 1; V[1*p+1, col] = 1
    W[0*n+0, col] = 1; W[1*n+1, col] = 1
    # Wait, W index is k*n+i.
    # C00 gets +M1, C01 gets 0, C10 gets 0, C11 gets +M1?
    # Let's check Strassen formulas again.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M4
    
    # M1
    U[0, col] = 1; U[1*m+1, col] = 1
    V[0, col] = 1; V[1*p+1, col] = 1
    W[0*n+0, col] = 1 # C00
    W[1*n+1, col] = 1 # C11
    col += 1
    
    # M2 = (A10+A11)B00
    U[1*m+0, col] = 1; U[1*m+1, col] = 1
    V[0*p+0, col] = 1
    W[0*n+1, col] = 1 # C10
    W[1*n+1, col] = -1 # C11
    col += 1
    
    # M3 = A00(B10-B11)
    U[0*m+0, col] = 1
    V[1*p+0, col] = 1; V[1*p+1, col] = -1
    W[0*n+0, col] = 1 # C01? No, C01 = M3+M5. So W for k=1, i=0.
    W[1*n+0, col] = 1 # C11? No, C11 = M1-M2+M3+M4. So W for k=1, i=1.
    # Wait, W index is k*n+i.
    # C01 is i=0, k=1 -> index 1*n+0 = 1.
    # C11 is i=1, k=1 -> index 1*n+1 = 4.
    W[1*n+0, col] = 1
    W[1*n+1, col] = 1
    col += 1
    
    # M4 = A11(B01+B11)
    U[1*m+1, col] = 1
    V[0*p+1, col] = 1; V[1*p+1, col] = 1
    W[0*n+0, col] = 1 # C00
    W[1*n+1, col] = 1 # C11
    col += 1
    
    # M5 = (A00+A01)B11
    U[0*m+0, col] = 1; U[0*m+1, col] = 1
    V[1*p+1, col] = 1
    W[0*n+0, col] = -1 # C00
    W[1*n+0, col] = 1 # C01
    col += 1
    
    # M6 is not used in standard Strassen for 2x2?
    # Standard Strassen uses 7 mults: M1..M7.
    # M6 = (A10-A00)(B01+B11) is NOT in the 7?
    # Let's re-verify Strassen 7 mults.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01+B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B01+B11) -> This is often M6 in some derivations, but let's check the 7.
    # Actually, the 7th is M7 = (A01-A11)(B00+B10).
    # And M6 is not needed?
    # Let's check C10 = M2 + M7.
    # C11 = M1 - M2 + M3 + M4.
    # C01 = M3 + M5.
    # C00 = M1 + M4 - M5 - M7.
    # Yes, M6 is not used.
    
    # M7 = (A01-A11)(B00+B10)
    U[0*m+1, col] = 1; U[1*m+1, col] = -1
    V[0*p+0, col] = 1; V[1*p+0, col] = 1
    W[0*n+0, col] = -1 # C00
    W[0*n+1, col] = 1 # C10
    col += 1
    
    # Now Schoolbook for the rest
    # We need to iterate all (i,j,k) and skip the ones covered by Strassen
    # Strassen covers i in {0,1}, j in {0,1}, k in {0,1}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in [0, 1] and j in [0, 1] and k in [0, 1]:
                    continue
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    # Verify col == rank
    assert col == rank, f"Col mismatch: {col} vs {rank}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

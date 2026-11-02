import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_blocks = p // 2  # 6
    rank = num_blocks * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen 7-product decomposition for 2x2x2:
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[C00,C01],[C10,C11]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e)
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    #
    # Local A indices: 0=a(A[0,0]), 1=b(A[0,1]), 2=c(A[1,0]), 3=d(A[1,1])
    # Local B indices: 0=e(B[0,0]), 1=f(B[0,1]), 2=g(B[1,0]), 3=h(B[1,1])
    # Local C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # For each block t (columns 2t and 2t+1):
    for t in range(num_blocks):
        if time.time() > deadline:
            break
        base = t * 7
        k0 = 2 * t      # first column in pair
        k1 = 2 * t + 1  # second column in pair
        
        # M1 = (A00+A11)*(B00+B11)
        c = base + 0
        U[0, c] += 1   # A[0,0]
        U[3, c] += 1   # A[1,1]
        V[0*p + k0, c] += 1   # B[0, k0] = B00
        V[1*p + k1, c] += 1   # B[1, k1] = B11
        # C00 += M1, C11 += M1
        W[k0*n + 0, c] += 1   # C[0, k0]
        W[k1*n + 1, c] += 1   # C[1, k1]
        
        # M2 = (A10+A11)*B00
        c = base + 1
        U[2, c] += 1   # A[1,0]
        U[3, c] += 1   # A[1,1]
        V[0*p + k0, c] += 1   # B[0, k0]
        # C10 += M2, C11 -= M2
        W[k0*n + 1, c] += 1   # C[1, k0]
        W[k1*n + 1, c] += -1  # C[1, k1]
        
        # M3 = A00*(B01-B11)
        c = base + 2
        U[0, c] += 1   # A[0,0]
        V[0*p + k1, c] += 1    # B[0, k1] = B01
        V[1*p + k1, c] += -1   # B[1, k1] = B11
        # C01 += M3, C11 += M3
        W[k1*n + 0, c] += 1   # C[0, k1]
        W[k1*n + 1, c] += 1   # C[1, k1]
        
        # M4 = A11*(B10-B00)
        c = base + 3
        U[3, c] += 1   # A[1,1]
        V[1*p + k0, c] += 1    # B[1, k0] = B10
        V[0*p + k0, c] += -1   # B[0, k0] = B00
        # C00 += M4, C10 += M4
        W[k0*n + 0, c] += 1   # C[0, k0]
        W[k0*n + 1, c] += 1   # C[1, k0]
        
        # M5 = (A00+A01)*B11
        c = base + 4
        U[0, c] += 1   # A[0,0]
        U[1, c] += 1   # A[0,1]
        V[1*p + k1, c] += 1   # B[1, k1] = B11
        # C00 -= M5, C01 += M5
        W[k0*n + 0, c] += -1  # C[0, k0]
        W[k1*n + 0, c] += 1   # C[0, k1]
        
        # M6 = (A10-A00)*(B00+B01)
        c = base + 5
        U[2, c] += 1   # A[1,0]
        U[0, c] += -1  # A[0,0]
        V[0*p + k0, c] += 1    # B[0, k0] = B00
        V[0*p + k1, c] += 1    # B[0, k1] = B01
        # C11 += M6
        W[k1*n + 1, c] += 1   # C[1, k1]
        
        # M7 = (A01-A11)*(B10+B11)
        c = base + 6
        U[1, c] += 1   # A[0,1]
        U[3, c] += -1  # A[1,1]
        V[1*p + k0, c] += 1    # B[1, k0] = B10
        V[1*p + k1, c] += 1    # B[1, k1] = B11
        # C00 += M7
        W[k0*n + 0, c] += 1   # C[0, k0]
    
    # Convert to lists as expected by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

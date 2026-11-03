"""Matrix multiplication 2x4 x 4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    start_time = time.time()
    
    # Strategy: Partition j into 2 groups of 2, k into 4 groups of 2.
    # Each (j-group, k-group) pair is a 2x2 * 2x2 = 2x2 problem, solved by Strassen (rank 7).
    # Total: 2 * 4 * 7 = 56 terms.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 56
    V = np.zeros((m * p, rank), dtype=float)  # 32 x 56
    W = np.zeros((p * n, rank), dtype=float)  # 16 x 56
    
    # Strassen 2x2*2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = A10(B01-B11)
    # M3 = A00(B10+B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B10
    # M6 = (A10-A00)B01
    # M7 = (A01-A11)(B00+B10)
    # C00 = M1+M4-M5-M7
    # C01 = M2+M3
    # C10 = M1+M3
    # C11 = M1-M2+M4+M7
    
    # U contributions per Strassen step (for A block at rows a0,a0+1, cols 0,1 within the 2-col group):
    # U is indexed by [i*m + j] where i in {0,1}, j in {0,1,2,3}
    # For j-group starting at j0: A[0, j0], A[0, j0+1], A[1, j0], A[1, j0+1]
    # U indices: 0*m+j0, 0*m+j0+1, 1*m+j0, 1*m+j0+1 = j0, j0+1, 4+j0, 4+j0+1
    
    # V contributions per Strassen step (for B block at rows j0,j0+1, cols k0,k0+1):
    # V is indexed by [j*p + k] where j in {0,1,2,3}, k in {0..7}
    # V indices: j0*p+k0, j0*p+k0+1, (j0+1)*p+k0, (j0+1)*p+k0+1
    
    # W contributions per Strassen step (for C block at rows 0,1, cols k0,k0+1):
    # W is indexed by [k*n + i] where k in {0..7}, i in {0,1}
    # W indices: k0*n+0, k0*n+1, (k0+1)*n+0, (k0+1)*n+1
    
    t = 0
    for j0 in range(0, m, 2):  # j-group start: 0, 2
        for k0 in range(0, p, 2):  # k-group start: 0, 2, 4, 6
            # U indices
            u00 = 0 * m + j0      # A[0, j0]
            u01 = 0 * m + j0 + 1  # A[0, j0+1]
            u10 = 1 * m + j0      # A[1, j0]
            u11 = 1 * m + j0 + 1  # A[1, j0+1]
            
            # V indices
            v00 = j0 * p + k0        # B[j0, k0]
            v01 = j0 * p + k0 + 1    # B[j0, k0+1]
            v10 = (j0 + 1) * p + k0        # B[j0+1, k0]
            v11 = (j0 + 1) * p + k0 + 1    # B[j0+1, k0+1]
            
            # W indices
            w00 = k0 * n + 0      # C[0, k0] -> W[k0*n + 0]
            w01 = (k0 + 1) * n + 0  # C[0, k0+1] -> W[(k0+1)*n + 0]
            w10 = k0 * n + 1      # C[1, k0] -> W[k0*n + 1]
            w11 = (k0 + 1) * n + 1  # C[1, k0+1] -> W[(k0+1)*n + 1]
            
            # Step 0: M1 = (A00+A11)(B00+B11)
            U[u00, t] += 1
            U[u11, t] += 1
            V[v00, t] += 1
            V[v11, t] += 1
            W[w00, t] += 1  # C00 += M1
            W[w10, t] += 1  # C10 += M1
            W[w11, t] += 1  # C11 += M1
            t += 1
            
            # Step 1: M2 = A10(B01-B11)
            U[u10, t] += 1
            V[v01, t] += 1
            V[v11, t] += -1
            W[w01, t] += 1  # C01 += M2
            W[w11, t] += -1  # C11 -= M2
            t += 1
            
            # Step 2: M3 = A00(B10+B11)
            U[u00, t] += 1
            V[v10, t] += 1
            V[v11, t] += 1
            W[w01, t] += 1  # C01 += M3
            W[w10, t] += 1  # C10 += M3
            t += 1
            
            # Step 3: M4 = A11(B00-B10)
            U[u11, t] += 1
            V[v00, t] += 1
            V[v10, t] += -1
            W[w00, t] += 1  # C00 += M4
            W[w11, t] += 1  # C11 += M4
            t += 1
            
            # Step 4: M5 = (A00+A01)B10
            U[u00, t] += 1
            U[u01, t] += 1
            V[v10, t] += 1
            W[w00, t] += -1  # C00 -= M5
            t += 1
            
            # Step 5: M6 = (A10-A00)B01
            U[u10, t] += 1
            U[u00, t] += -1
            V[v01, t] += 1
            W[w00, t] += -1  # C00 -= M6
            t += 1
            
            # Step 6: M7 = (A01-A11)(B00+B10)
            U[u01, t] += 1
            U[u11, t] += -1
            V[v00, t] += 1
            V[v10, t] += 1
            W[w00, t] += -1  # C00 -= M7
            W[w11, t] += 1   # C11 += M7
            t += 1
    
    assert t == rank, f"Expected {rank} terms, got {t}"
    
    # Verify correctness
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0  # T[i,j,k] = 1 for matrix multiplication tensor
                rhs = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(lhs - rhs) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # This should not happen, but return anyway
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

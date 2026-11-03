import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # We use the Kronecker product of the rank-7 Strassen algorithm.
    # The base 2x2x2 Strassen tensor has rank 7.
    # The 4x4x4 tensor (Kronecker product with itself) has rank 7 * 7 = 49.
    # Entries are integers (products of -1, 0, 1).
    
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen rank-7 tensors for 2x2 matrix multiplication
    # U[t, i, j] = A[i, j] coefficient in factor t
    # V[t, j, k] = B[j, k] coefficient in factor t
    # W[t, i, k] = C[i, k] coefficient in factor t (trace convention: W[k*n+i, t])
    
    # Strassen factors:
    # t=0: A[0,0]*(B[0,0]-B[1,1]) -> C[0,0] += 1, C[1,1] += 1
    # t=1: A[1,0]*(B[0,0]+B[0,1]) -> C[1,0] += 1, C[1,1] += 1
    # t=2: A[0,1]*(B[1,0]+B[1,1]) -> C[0,0] += 1, C[0,1] += 1
    # t=3: (A[0,0]+A[1,1])*(B[0,0]+B[0,1]) -> C[0,0] += 1, C[0,1] += 1, C[1,0] += 1, C[1,1] += 1
    # t=4: (A[0,1]+A[1,1])*(B[1,0]-B[0,0]) -> C[1,0] += 1
    # t=5: (A[0,0]-A[1,0])*(B[0,1]+B[1,1]) -> C[0,1] += 1
    # t=6: (A[0,0]+A[1,1])*(B[1,0]-B[1,1]) -> C[1,0] += 1, C[1,1] += 1
    
    # Let's use the standard Strassen formulation:
    # M1 = A00*(B00+B11)
    # M2 = A10*(B00+B01)
    # M3 = A01*(B10+B11)
    # M4 = A00*(B00-B11)
    # M5 = A11*(B01-B00)
    # M6 = (A00+A11)*(B10+B11)
    # M7 = (A10+A11)*(B00-B10)
    # C00 = M1 + M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M4 + M5 - M7
    
    # U (A coefficients) for each factor t, indexed by (i, j)
    # i is row of A (0 or 1), j is col of A (0 or 1)
    U2 = np.zeros((7, 2, 2), dtype=int)
    V2 = np.zeros((7, 2, 2), dtype=int)
    W2 = np.zeros((7, 2, 2), dtype=int)
    
    # t=0: M1 = A00*(B00+B11)
    U2[0, 0, 0] = 1
    V2[0, 0, 0] = 1
    V2[0, 1, 1] = 1
    W2[0, 0, 0] = 1
    W2[0, 1, 1] = 1
    
    # t=1: M2 = A10*(B00+B01)
    U2[1, 1, 0] = 1
    V2[1, 0, 0] = 1
    V2[1, 0, 1] = 1
    W2[1, 1, 0] = 1
    W2[1, 1, 1] = 1
    
    # t=2: M3 = A01*(B10+B11)
    U2[2, 0, 1] = 1
    V2[2, 1, 0] = 1
    V2[2, 1, 1] = 1
    W2[2, 0, 0] = 1
    W2[2, 0, 1] = 1
    
    # t=3: M4 = A00*(B00-B11)
    U2[3, 0, 0] = 1
    V2[3, 0, 0] = 1
    V2[3, 1, 1] = -1
    W2[3, 0, 0] = 1
    W2[3, 1, 1] = -1
    
    # t=4: M5 = A11*(B01-B00)
    U2[4, 1, 1] = 1
    V2[4, 0, 0] = -1
    V2[4, 0, 1] = 1
    W2[4, 0, 1] = 1
    W2[4, 1, 1] = -1
    
    # t=5: M6 = (A00+A11)*(B10+B11)
    U2[5, 0, 0] = 1
    U2[5, 1, 1] = 1
    V2[5, 1, 0] = 1
    V2[5, 1, 1] = 1
    W2[5, 0, 0] = -1
    W2[5, 1, 0] = 1
    
    # t=6: M7 = (A10+A11)*(B00-B10)
    U2[6, 1, 0] = 1
    U2[6, 1, 1] = 1
    V2[6, 0, 0] = 1
    V2[6, 1, 0] = -1
    W2[6, 0, 0] = 1
    W2[6, 1, 0] = -1
    
    # Now build the 4x4x4 Kronecker product
    # U4[i, j, r] where r = t1*7 + t0
    # i = i1*2 + i0, j = j1*2 + j0
    # U4[i1*2+i0, j1*2+j0, t1*7+t0] = U2[t1, i1, j1] * U2[t0, i0, j0]
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for t1 in range(7):
        for t0 in range(7):
            r = t1 * 7 + t0
            for i1 in range(2):
                for i0 in range(2):
                    i = i1 * 2 + i0
                    for j1 in range(2):
                        for j0 in range(2):
                            j = j1 * 2 + j0
                            ij = i * 4 + j
                            
                            u_val = U2[t1, i1, j1] * U2[t0, i0, j0]
                            v_val = V2[t1, i1, j1] * V2[t0, i0, j0]
                            
                            # For V: V[j, k, r]
                            # j = j1*2 + j0, k = k1*2 + k0
                            # V[j1*2+j0, k1*2+k0, t1*7+t0] = V2[t1, j1, k1] * V2[t0, j0, k0]
                            for k1 in range(2):
                                for k0 in range(2):
                                    k = k1 * 2 + k0
                                    jk = j * 4 + k
                                    v_val = V2[t1, j1, k1] * V2[t0, j0, k0]
                                    V[jk, r] = v_val
                                    
                                    # For W: W[k, i, r] (trace convention: W[k*n+i, r] where n=4)
                                    # W[k1*2+k0, i1*2+i0, t1*7+t0] = W2[t1, i1, k1] * W2[t0, i0, k0]
                                    w_val = W2[t1, i1, k1] * W2[t0, i0, k0]
                                    ki = k * 4 + i
                                    W[ki, r] = w_val
                            
                            U[ij, r] = u_val
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

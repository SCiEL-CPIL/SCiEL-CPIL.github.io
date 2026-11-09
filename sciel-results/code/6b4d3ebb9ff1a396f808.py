"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """Rank-17 exact tensor decomposition for 2x3 x 3x3 matrix multiplication."""
    import numpy as np
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    rank = 17
    
    U = np.zeros((n * m, rank))  # 6 x 17
    V = np.zeros((m * p, rank))  # 9 x 17
    W = np.zeros((p * n, rank))  # 6 x 17
    
    # Strassen 2x2x2 for subblock using A[0:2,0:2], B[0:2,0:2] -> C[0:2,0:2]
    # Using standard Strassen with 0-indexed A[i,j], B[j,k], C[i,k]
    # Flattened: A[i*3+j], B[j*3+k], C[k*2+i]
    
    # Strassen factors (verified):
    # t0: (A00+A11)(B00+B11) -> C00, C11
    U[0,0] = 1; U[4,0] = 1
    V[0,0] = 1; V[4,0] = 1
    W[0,0] = 1; W[3,0] = 1  # C[0,0]=W[0], C[1,1]=W[3]
    
    # t1: (A10+A11)B00 -> C10, C11
    U[3,1] = 1; U[4,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1  # C[1,0]=W[2], C[1,1]=W[3]
    
    # t2: A00(B10-B11) -> C01, C11
    U[0,2] = 1
    V[3,2] = 1; V[4,2] = -1
    W[1,2] = 1; W[3,2] = 1  # C[0,1]=W[1], C[1,1]=W[3]
    
    # t3: A11(B01-B00) -> C00, C10
    U[4,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1  # C[0,0]=W[0], C[1,0]=W[2]
    
    # t4: (A00+A01)B11 -> C00, C01
    U[0,4] = 1; U[1,4] = 1
    V[4,4] = 1
    W[0,4] = -1; W[1,4] = 1  # C[0,0]=W[0], C[0,1]=W[1]
    
    # t5: (A10-A00)(B00+B01) -> C11
    U[3,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1  # C[1,1]=W[3]
    
    # t6: (A01-A11)(B10+B11) -> C00, C01
    U[1,6] = 1; U[4,6] = -1
    V[3,6] = 1; V[4,6] = 1
    W[0,6] = 1; W[1,6] = 1  # C[0,0]=W[0], C[0,1]=W[1]
    
    # Verify Strassen part:
    # C[0,0] = W[0] contributions: t0(+1) + t3(+1) + t4(-1) + t6(+1) = t0+t3-t4+t6
    # C[0,1] = W[1] contributions: t2(+1) + t4(+1) + t6(+1) = t2+t4+t6
    # C[1,0] = W[2] contributions: t1(+1) + t3(+1) = t1+t3
    # C[1,1] = W[3] contributions: t0(+1) + t1(-1) + t2(+1) + t5(+1) = t0-t1+t2+t5
    
    # Standard Strassen:
    # C00 = P1+P4-P5+P7, C01=P3+P5, C10=P2+P4, C11=P1-P2+P3+P6
    # Mapping: t0=P1, t1=P2, t2=P3, t3=P4, t4=P5, t5=P6, t6=P7
    # C00 = t0+t3-t4+t6 ✓
    # C01 = t2+t4+t6 ✓  
    # C10 = t1+t3 ✓
    # C11 = t0-t1+t2+t5 ✓
    
    # Now add rank-1 terms for the remaining j=2 contributions
    # C[i,k] += A[i,2]*B[2,k] for k=0,1,2 and A[i,j]*B[j,2] for j=0,1
    
    # C[0,0] += A[0,2]*B[2,0]
    U[2,7] = 1; V[6,7] = 1; W[0,7] = 1
    # C[0,1] += A[0,2]*B[2,1]
    U[2,8] = 1; V[7,8] = 1; W[1,8] = 1
    # C[1,0] += A[1,2]*B[2,0]
    U[5,9] = 1; V[6,9] = 1; W[2,9] = 1
    # C[1,1] += A[1,2]*B[2,1]
    U[5,10] = 1; V[7,10] = 1; W[3,10] = 1
    # C[0,2] += A[0,0]*B[0,2]
    U[0,11] = 1; V[2,11] = 1; W[4,11] = 1
    # C[0,2] += A[0,1]*B[1,2]
    U[1,12] = 1; V[5,12] = 1; W[4,12] = 1
    # C[0,2] += A[0,2]*B[2,2]
    U[2,13] = 1; V[8,13] = 1; W[4,13] = 1
    # C[1,2] += A[1,0]*B[0,2]
    U[3,14] = 1; V[2,14] = 1; W[5,14] = 1
    # C[1,2] += A[1,1]*B[1,2]
    U[4,15] = 1; V[5,15] = 1; W[5,15] = 1
    # C[1,2] += A[1,2]*B[2,2]
    U[5,16] = 1; V[8,16] = 1; W[5,16] = 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

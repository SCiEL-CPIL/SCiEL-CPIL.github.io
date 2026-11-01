import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal tensor decomposition for 2x2 matrix multiplication.
    This is Strassen's algorithm with rank 7.
    
    Interface:
    U: (n*m, rank) = (4, 7)
    V: (m*p, rank) = (4, 7)
    W: (p*n, rank) = (4, 7)
    
    Convention:
    A is 2x2, B is 2x2, C is 2x2.
    U[i*m + j, r] corresponds to coefficient of A[i,j] in r-th product.
    V[j*p + k, r] corresponds to coefficient of B[j,k] in r-th product.
    W[k*n + i, r] corresponds to coefficient of C[i,k] in r-th product.
    Note: W index is k*n + i. Since n=2, index is 2*k + i.
    C[i,k] = sum_r W[2*k + i, r] * (sum_{i',j'} U[2*i'+j', r] A[i',j']) * (sum_{j',k'} V[2*j'+k', r] B[j',k'])
    
    Wait, the standard bilinear form is:
    C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) ... No.
    
    The correct decomposition form for matrix multiplication tensor T_{ikj} (where T_{ikj}=1 if the product A_{ij}B_{jk} contributes to C_{ik})
    is:
    C[i,k] = sum_r W_{ik,r} * (sum_{i',j'} U_{i'j',r} A_{i'j'}) * (sum_{j',k'} V_{j'k',r} B_{j'k'})
    
    Actually, the standard definition is:
    Let M_r = (sum_{i,j} u_{ij,r} A_{ij}) (sum_{j,k} v_{jk,r} B_{jk})
    Then C_{ik} = sum_r w_{ik,r} M_r
    
    However, the indices in U, V, W in the prompt are flattened.
    U has shape (n*m, rank). Index is i*m + j.
    V has shape (m*p, rank). Index is j*p + k.
    W has shape (p*n, rank). Index is k*n + i.
    
    Let's verify Strassen's factors with this convention.
    n=m=p=2.
    
    M1 = (A00 + A11) * (B00 + B11)
    U1: A00=1, A11=1 => U[0,0]=1, U[3,0]=1 (indices 0*2+0=0, 1*2+1=3)
    V1: B00=1, B11=1 => V[0,0]=1, V[3,0]=1 (indices 0*2+0=0, 1*2+1=3)
    W1: C00=1, C11=1 => W[0,0]=1, W[3,0]=1 (indices 0*2+0=0, 1*2+1=3)
    
    M2 = (A10 + A11) * B00
    U2: A10=1, A11=1 => U[2,1]=1, U[3,1]=1 (indices 1*2+0=2, 1*2+1=3)
    V2: B00=1 => V[0,1]=1
    W2: C10=1, C11=-1 => W[1,1]=1, W[3,1]=-1 (indices 0*2+1=1, 1*2+1=3)
    
    M3 = A00 * (B01 - B11)
    U3: A00=1 => U[0,2]=1
    V3: B01=1, B11=-1 => V[1,2]=1, V[3,2]=-1 (indices 0*2+1=1, 1*2+1=3)
    W3: C01=1, C11=1 => W[2,2]=1, W[3,2]=1 (indices 1*2+0=2, 1*2+1=3)
    
    M4 = A11 * (B10 - B00)
    U4: A11=1 => U[3,3]=1
    V4: B10=1, B00=-1 => V[2,3]=1, V[0,3]=-1 (indices 1*2+0=2, 0*2+0=0)
    W4: C00=1, C10=1 => W[0,3]=1, W[1,3]=1 (indices 0*2+0=0, 0*2+1=1)
    
    M5 = (A00 + A01) * B11
    U5: A00=1, A01=1 => U[0,4]=1, U[1,4]=1 (indices 0*2+0=0, 0*2+1=1)
    V5: B11=1 => V[3,4]=1
    W5: C00=-1, C01=1 => W[0,4]=-1, W[2,4]=1 (indices 0*2+0=0, 1*2+0=2)
    
    M6 = (A10 - A00) * (B00 + B01)
    U6: A10=1, A00=-1 => U[2,5]=1, U[0,5]=-1 (indices 1*2+0=2, 0*2+0=0)
    V6: B00=1, B01=1 => V[0,5]=1, V[1,5]=1 (indices 0*2+0=0, 0*2+1=1)
    W6: C11=1 => W[3,5]=1 (indices 1*2+1=3)
    
    M7 = (A01 - A11) * (B10 + B11)
    U7: A01=1, A11=-1 => U[1,6]=1, U[3,6]=-1 (indices 0*2+1=1, 1*2+1=3)
    V7: B10=1, B11=1 => V[2,6]=1, V[3,6]=1 (indices 1*2+0=2, 1*2+1=3)
    W7: C00=1 => W[0,6]=1 (indices 0*2+0=0)
    """
    start_time = time.time()
    deadline = 10.0
    
    n, m, p = DIMENSIONS
    rank = 7
    
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    # Define factors (r, u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # Indices are flat indices into the matrices
    factors = [
        # r=0: M1
        (0, [0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # r=1: M2
        (1, [2, 3], [1, 1], [0], [1], [1, 3], [1, -1]),
        # r=2: M3
        (2, [0], [1], [1, 3], [1, -1], [2, 3], [1, 1]),
        # r=3: M4
        (3, [3], [1], [2, 0], [1, -1], [0, 1], [1, 1]),
        # r=4: M5
        (4, [0, 1], [1, 1], [3], [1], [0, 2], [-1, 1]),
        # r=5: M6
        (5, [2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # r=6: M7
        (6, [1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]
    
    for r, u_idx, u_val, v_idx, v_val, w_idx, w_val in factors:
        for idx, val in zip(u_idx, u_val):
            U[idx, r] = val
        for idx, val in zip(v_idx, v_val):
            V[idx, r] = val
        for idx, val in zip(w_idx, w_val):
            W[idx, r] = val
            
    if time.time() > deadline - 0.1:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

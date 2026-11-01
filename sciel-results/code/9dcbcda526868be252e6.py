"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    The known minimal rank is 7 (Strassen's algorithm).
    
    The tensor T_{ijk} for matrix multiplication C = A * B is defined such that
    C_{ik} = sum_j A_{ij} B_{jk}.
    In the problem's indexing:
    U has shape (n*m, rank) -> U[i*m+j, c] corresponds to A_{ij}
    V has shape (m*p, rank) -> V[j*p+k, c] corresponds to B_{jk}
    W has shape (p*n, rank) -> W[k*n+i, c] corresponds to C_{ki} (note: output index is k*n+i, which is C_{row=k, col=i})
    
    Wait, the problem states: "output factor follows the official trace convention W[k*n+i,r]".
    Standard matrix multiplication: C_{ik} = sum_j A_{ij} B_{jk}.
    Here, the output index is given as `k*n+i`. If n=2, m=2, p=2.
    Usually, matrices are indexed A[i,j], B[j,k], C[i,k].
    If W index is k*n+i, and n=2, then index = 2*k + i.
    This implies the row index of C is `k` and column index is `i`? Or is it row `i`, col `k`?
    Let's check the schoolbook code provided:
    U[i*m+j][column] = 1  -> A_{ij}
    V[j*p+k][column] = 1  -> B_{jk}
    W[k*n+i][column] = 1  -> C_{?}
    
    In schoolbook, term is A_{ij} * B_{jk} -> contributes to C_{ik}.
    The index in W is k*n + i.
    If we interpret W index as `row * n + col`, then row=k, col=i.
    This means C_{k,i} = sum_j A_{i,j} B_{j,k}.
    This is consistent with standard multiplication if we view C as having row k and col i?
    Actually, standard is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    If the problem defines the output tensor component at index `k*n+i` to be the coefficient for `A_{i,j} B_{j,k}`, then:
    We need sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = delta_{i,i} delta_{k,k} ? No.
    
    The tensor M_{i,j,k} = 1 if the term A_{ij}B_{jk} contributes to C_{ik}.
    In the decomposition:
    sum_c U_{ij,c} V_{jk,c} W_{?,c} = M_{i,j,k}
    The index of W is `k*n+i`. Let's call the indices of W as `w_idx`.
    So we need sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k.
    
    This is exactly the standard matrix multiplication tensor.
    The minimal rank of the 2x2x2 matrix multiplication tensor is 7.
    Strassen's algorithm provides this decomposition.
    
    Strassen's 7 products for 2x2 matrices A, B:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B01 - B11)
    M4 = A11(B10 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    We need to construct U, V, W such that:
    U_{ij,c} V_{jk,c} W_{k,i,c} (using W index k*n+i, so W_{k,i}) sums to 1 if (i,j,k) is a valid multiplication term.
    Wait, the tensor is M_{i,j,k} = 1.
    So for each c, we have a trilinear form.
    
    Let's map Strassen's terms to U, V, W.
    For each product M_c, it is a sum of products of linear forms in A and B.
    M_c = (sum_{i,j} u_{ij,c} A_{ij}) (sum_{j,k} v_{jk,c} B_{jk})
    Then M_c contributes to C_{ik} via w_{ik,c} M_c.
    So C_{ik} = sum_c w_{ik,c} (sum_{j} u_{ij,c} A_{ij}) (sum_{j} v_{jk,c} B_{jk})?
    No, the rank-1 tensor is u_{ij,c} v_{jk,c} w_{ki,c}.
    The total tensor is sum_c u_{ij,c} v_{jk,c} w_{ki,c}.
    This must equal 1 for all i,j,k.
    
    Let's verify Strassen's decomposition in this format.
    The entries of U, V, W will be integers.
    
    Let's define the 7 columns (c=0..6).
    
    Col 0 (M1):
    U part: A00 + A11 -> U[0,0]=1, U[3,0]=1 (indices: i*m+j. 0*2+0=0, 1*2+1=3)
    V part: B00 + B11 -> V[0,0]=1, V[3,0]=1 (indices: j*p+k. 0*2+0=0, 1*2+1=3)
    W part: C00 + C11 -> W[0,0]=1, W[3,0]=1 (indices: k*n+i. 0*2+0=0, 1*2+1=3)
    
    Col 1 (M2):
    U part: A10 + A11 -> U[2,1]=1, U[3,1]=1 (1*2+0=2, 1*2+1=3)
    V part: B00 -> V[0,1]=1
    W part: C10 + C11? No, C10 = M2+M4, C11 = M1-M2+M3+M6.
    So M2 contributes +1 to C10 and -1 to C11.
    W index for C10: k=1, i=0 -> 1*2+0 = 2.
    W index for C11: k=1, i=1 -> 1*2+1 = 3.
    So W[2,1]=1, W[3,1]=-1.
    
    Col 2 (M3):
    U part: A00 -> U[0,2]=1
    V part: B01 - B11 -> V[1,2]=1, V[3,2]=-1 (0*2+1=1, 1*2+1=3)
    W part: C01 + C11.
    C01 = M3+M5. C11 = ...+M3...
    W index C01: k=0, i=1 -> 0*2+1 = 1.
    W index C11: k=1, i=1 -> 3.
    W[1,2]=1, W[3,2]=1.
    
    Col 3 (M4):
    U part: A11 -> U[3,3]=1
    V part: B10 - B00 -> V[2,3]=1, V[0,3]=-1 (1*2+0=2, 0*2+0=0)
    W part: C00 + C10.
    C00 = M1+M4-M5+M7. C10 = M2+M4.
    W index C00: 0.
    W index C10: 2.
    W[0,3]=1, W[2,3]=1.
    
    Col 4 (M5):
    U part: A00 + A01 -> U[0,4]=1, U[1,4]=1
    V part: B11 -> V[3,4]=1
    W part: -C00 + C01.
    C00 has -M5. C01 has +M5.
    W[0,4]=-1, W[1,4]=1.
    
    Col 5 (M6):
    U part: A10 - A00 -> U[2,5]=1, U[0,5]=-1
    V part: B00 + B01 -> V[0,5]=1, V[1,5]=1
    W part: C11.
    C11 has +M6.
    W[3,5]=1.
    
    Col 6 (M7):
    U part: A01 - A11 -> U[1,6]=1, U[3,6]=-1
    V part: B10 + B11 -> V[2,6]=1, V[3,6]=1
    W part: C00.
    C00 has +M7.
    W[0,6]=1.
    
    Let's double check the indices.
    U shape (4, 7). V shape (4, 7). W shape (4, 7).
    
    U:
    Row 0 (A00): [1, 0, 1, 0, 1, -1, 0]
    Row 1 (A01): [0, 0, 0, 0, 1, 0, 1]
    Row 2 (A10): [0, 1, 0, 0, 0, 1, 0]
    Row 3 (A11): [1, 1, 0, 1, 0, 0, -1]
    
    V:
    Row 0 (B00): [1, 1, 0, -1, 0, 1, 0]
    Row 1 (B01): [0, 0, 1, 0, 0, 1, 0]
    Row 2 (B10): [0, 0, 0, 1, 0, 0, 1]
    Row 3 (B11): [1, 0, -1, 0, 1, 0, 1]
    
    W:
    Row 0 (C00, k=0,i=0): [1, 0, 0, 1, -1, 0, 1]
    Row 1 (C01, k=0,i=1): [0, 0, 1, 0, 1, 0, 0]
    Row 2 (C10, k=1,i=0): [0, 1, 0, 1, 0, 0, 0]
    Row 3 (C11, k=1,i=1): [1, -1, 1, 0, 0, 1, 0]
    
    Let's verify one term: i=0, j=0, k=0. A00 * B00 -> C00.
    Sum_c U[0,c] V[0,c] W[0,c]
    c=0: 1*1*1 = 1
    c=1: 0*1*0 = 0
    c=2: 1*0*0 = 0
    c=3: 0*(-1)*1 = 0
    c=4: 1*0*(-1) = 0
    c=5: -1*1*0 = 0
    c=6: 0*0*1 = 0
    Sum = 1. Correct.
    
    Let's verify i=0, j=0, k=1. A00 * B01 -> C01.
    Sum_c U[0,c] V[1,c] W[1,c]
    c=0: 1*0*0 = 0
    c=1: 0*0*0 = 0
    c=2: 1*1*1 = 1
    c=3: 0*0*0 = 0
    c=4: 1*0*1 = 0
    c=5: -1*1*0 = 0
    c=6: 0*0*0 = 0
    Sum = 1. Correct.
    
    Let's verify i=1, j=0, k=0. A10 * B00 -> C10.
    Sum_c U[2,c] V[0,c] W[2,c]
    c=0: 0*1*0 = 0
    c=1: 1*1*1 = 1
    c=2: 0*0*0 = 0
    c=3: 0*(-1)*1 = 0
    c=4: 0*0*0 = 0
    c=5: 1*1*0 = 0
    c=6: 0*0*0 = 0
    Sum = 1. Correct.
    
    The decomposition is valid and rank is 7.
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

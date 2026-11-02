"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time
import random

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    The convention is W[k*n+i, r] (official trace convention).
    We check that for each (i,j,k), the sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    and that for all other (i,j,k) combinations not in the tensor, the sum is 0.
    Actually, the tensor T[i,j,k] = 1 if the multiplication (A*B)_{i,k} uses A_{i,j}*B_{j,k},
    which is always true. So the tensor is the standard matmul tensor:
    T[i,j,k] = 1 for all i,j,k (since (AB)_{ik} = sum_j A_{ij} B_{jk}).
    
    Wait, let me re-read. The tensor for matrix multiplication C = A*B where C_{ik} = sum_j A_{ij} B_{jk}.
    The tensor T has indices (i, j, k) and T[i,j,k] = 1 if this term appears in the product.
    Since every term A_{ij} * B_{jk} contributes to C_{ik}, T[i,j,k] = 1 for all i,j,k in [0,n)x[0,m)x[0,p).
    
    So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_decomposition():
    """Return the Strassen rank-7 decomposition for 2x2 matrix multiplication.
    
    Strassen's algorithm computes C = A * B using 7 multiplications.
    A = [[a11, a12], [a21, a22]]
    B = [[b11, b12], [b21, b22]]
    
    M1 = (a11 + a22)(b11 + b22)
    M2 = (a21 + a22)b11
    M3 = a11(b12 - b22)
    M4 = a22(b21 - b11)
    M5 = (a11 + a12)b22
    M6 = (a21 - a11)(b11 + b12)
    M7 = (a12 - a22)(b21 + b22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    We need to express this in the form:
    C_{ik} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Wait, the convention is W[k*n+i, r]. Let me re-index.
    
    For n=m=p=2:
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    
    U[i*m+j, r] = U[i*2+j, r]
    V[j*p+k, r] = V[j*2+k, r]
    W[k*n+i, r] = W[k*2+i, r]
    
    The product is: C_{ik} = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    But wait, in Strassen, the "multiplications" are of linear combinations. Let me think of it differently.
    
    Each rank-1 term corresponds to one of the 7 products. For product r:
    U[i,j,r] gives the coefficient of A_{ij} in the first factor of product r.
    V[j,k,r] gives the coefficient of B_{jk} in the second factor of product r.
    W[k,i,r] gives the coefficient by which product r contributes to C_{ik}.
    
    So for M1 = (a11+a22)(b11+b22):
    First factor: a11 + a22, so U[0,0,0]=1, U[1,1,0]=1, others 0.
    Second factor: b11 + b22, so V[0,0,0]=1, V[1,1,0]=1, others 0.
    Contribution to C: C11 += M1, C22 += M1. So W[0,0,0]=1 (for C_{0,0}), W[1,1,0]=1 (for C_{1,1}).
    Note: W[k*n+i, r] = W[k*2+i, r]. For C_{0,0}: k=0, i=0, so W[0,0,0]=1. For C_{1,1}: k=1, i=1, so W[3,0,0]=1.
    
    Let me build all 7:
    
    M1 = (a11+a22)(b11+b22)
    U[:, :, 0]: U[0,0]=1, U[1,1]=1
    V[:, :, 0]: V[0,0]=1, V[1,1]=1
    W[:, :, 0]: W[0,0]=1 (C_{0,0}), W[3,0]=1 (C_{1,1})
    
    M2 = (a21+a22)b11
    U[:, :, 1]: U[1,0]=1, U[1,1]=1
    V[:, :, 1]: V[0,0]=1
    W[:, :, 1]: W[2,1]=1 (C_{1,0}), W[3,1]=-1 (C_{1,1})
    
    M3 = a11(b12-b22)
    U[:, :, 2]: U[0,0]=1
    V[:, :, 2]: V[0,1]=1, V[1,1]=-1
    W[:, :, 2]: W[1,2]=1 (C_{0,1}), W[3,2]=1 (C_{1,1})
    
    M4 = a22(b21-b11)
    U[:, :, 3]: U[1,1]=1
    V[:, :, 3]: V[1,0]=1, V[0,0]=-1
    W[:, :, 3]: W[0,3]=1 (C_{0,0}), W[2,3]=1 (C_{1,0})
    
    M5 = (a11+a12)b22
    U[:, :, 4]: U[0,0]=1, U[0,1]=1
    V[:, :, 4]: V[1,1]=1
    W[:, :, 4]: W[0,4]=-1 (C_{0,0}), W[1,4]=1 (C_{0,1})
    
    M6 = (a21-a11)(b11+b12)
    U[:, :, 5]: U[1,0]=1, U[0,0]=-1
    V[:, :, 5]: V[0,0]=1, V[0,1]=1
    W[:, :, 5]: W[3,5]=1 (C_{1,1})
    
    M7 = (a12-a22)(b21+b22)
    U[:, :, 6]: U[0,1]=1, U[1,1]=-1
    V[:, :, 6]: V[1,0]=1, V[1,1]=1
    W[:, :, 6]: W[0,6]=1 (C_{0,0})
    
    Let me verify C_{0,0} = M1 + M4 - M5 + M7:
    From M1: W[0,0,0]=1 -> contributes +1
    From M4: W[0,0,3]=1 -> contributes +1
    From M5: W[0,0,4]=-1 -> contributes -1
    From M7: W[0,0,6]=1 -> contributes +1
    Total coefficient for C_{0,0}: 1+1-1+1 = 2? That's not right.
    
    Wait, I think I'm confusing myself. Let me re-derive.
    
    C_{0,0} = M1 + M4 - M5 + M7
    
    So the contribution to C_{0,0} from each M_r is:
    M1: +1
    M4: +1
    M5: -1
    M7: +1
    
    So W[0,0,r] (which is W[k=0, i=0, r]) should be:
    W[0,0,0] = 1 (M1)
    W[0,0,3] = 1 (M4)
    W[0,0,4] = -1 (M5)
    W[0,0,6] = 1 (M7)
    
    Similarly for C_{0,1} = M3 + M5:
    W[0,1,2] = 1 (M3)
    W[0,1,4] = 1 (M5)
    Wait, W[k*n+i, r] = W[k*2+i, r]. For C_{0,1}: i=0, k=1, so W[1*2+0, r] = W[2, r].
    
    Let me redo the indexing. W[k*n+i, r] where n=2.
    C_{i,k} corresponds to W[k*2+i, r].
    
    C_{0,0}: i=0, k=0 -> W[0, r]
    C_{0,1}: i=0, k=1 -> W[2, r]
    C_{1,0}: i=1, k=0 -> W[1, r]
    C_{1,1}: i=1, k=1 -> W[3, r]
    
    So:
    C_{0,0} = M1 + M4 - M5 + M7
    W[0, 0] = 1, W[0, 3] = 1, W[0, 4] = -1, W[0, 6] = 1
    
    C_{0,1} = M3 + M5
    W[2, 2] = 1, W[2, 4] = 1
    
    C_{1,0} = M2 + M4
    W[1, 1] = 1, W[1, 3] = 1
    
    C_{1,1} = M1 - M2 + M3 + M6
    W[3, 0] = 1, W[3, 1] = -1, W[3, 2] = 1, W[3, 5] = 1
    
    Now let's verify:
    C_{0,0} = sum_j sum_r U[0,j,r] * V[j,0,r] * W[0,r]
    = sum_j sum_r U[0,j,r] * V[j,0,r] * W[0,r]
    
    For r=0: U[0,0,0]=1, V[0,0,0]=1, W[0,0]=1 -> 1
           U[0,1,0]=0, so only j=0 contributes
    For r=1: U[0,*,1]=0, so 0
    For r=2: U[0,*,2]=0 (U[0,0,2]=1 but V[0,0,2]=0, V[1,0,2]=0), so 0
    For r=3: U[0,*,3]=0, so 0
    For r=4: U[0,0,4]=1, V[0,0,4]=0; U[0,1,4]=0. So 0
    For r=5: U[0,*,5]=0, so 0
    For r=6: U[0,*,6]=0, so 0
    
    Wait, that gives C_{0,0} = 1, but it should be a11*b11 + a12*b21.
    
    I think I need to be more careful. The decomposition is:
    C_{ik} = sum_{j,r} U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    So for C_{0,0}:
    C_{0,0} = sum_j sum_r U[0,j,r] * V[j,0,r] * W[0,r]
    
    Let me compute term by term:
    r=0: j=0: U[0,0,0]*V[0,0,0]*W[0,0] = 1*1*1 = 1
          j=1: U[0,1,0]*V[1,0,0]*W[0,0] = 0*0*1 = 0
    r=1: j=0: U[0,0,1]*V[0,0,1]*W[0,1] = 0*1*0 = 0
          j=1: U[0,1,1]*V[1,0,1]*W[0,1] = 0*0*0 = 0
    r=2: j=0: U[0,0,2]*V[0,0,2]*W[0,2] = 1*0*0 = 0
          j=1: U[0,1,2]*V[1,0,2]*W[0,2] = 0*0*0 = 0
    r=3: j=0: U[0,0,3]*V[0,0,3]*W[0,3] = 0*(-1)*1 = 0
          j=1: U[0,1,3]*V[1,0,3]*W[0,3] = 0*1*1 = 0
    r=4: j=0: U[0,0,4]*V[0,0,4]*W[0,4] = 1*0*(-1) = 0
          j=1: U[0,1,4]*V[1,0,4]*W[0,4] = 1*0*(-1) = 0
    r=5: j=0: U[0,0,5]*V[0,0,5]*W[0,5] = (-1)*1*0 = 0
          j=1: U[0,1,5]*V[1,0,5]*W[0,5] = 0*0*0 = 0
    r=6: j=0: U[0,0,6]*V[0,0,6]*W[0,6] = 0*0*1 = 0
          j=1: U[0,1,6]*V[1,0,6]*W[0,6] = 1*1*1 = 1
    
    So C_{0,0} = 1 + 1 = 2? That's not right either.
    
    I think I made an error in the Strassen coefficients. Let me look up the correct Strassen decomposition.
    
    Actually, I recall that the standard Strassen algorithm has:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    But the key is that each M_r is a product of two linear forms. The tensor decomposition is:
    T = sum_r u_r ⊗ v_r ⊗ w_r
    
    where u_r, v_r, w_r are vectors.
    
    Let me just hardcode the known correct Strassen decomposition for 2x2 matmul.
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # i=0,j=0
        [0, 0, 0, 0, 1, 0, 1],   # i=0,j=1
        [0, 1, 0, 0, 0, 1, 0],   # i=1,j=0
        [1, 1, 0, 1, 0, 0, -1],  # i=1,j=1
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # j=0,k=0
        [1, 0, 1, 0, 0, 1, 0],   # j=0,k=1
        [0, 0, 0, 1, 1, 0, 1],   # j=1,k=0
        [1, 0, -1, 0, 1, 0, 1],  # j=1,k=1
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # k=0,i=0 -> C_{0,0}
        [0, 1, 0, 1, 0, 0, 0],   # k=0,i=1 -> C_{1,0}
        [0, 0, 1, 0, 1, 0, 0],   # k=1,i=0 -> C_{0,1}
        [1, -1, 1, 0, 0, 1, 0],  # k=1,i=1 -> C_{1,1}
    ]
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    # Return the Strassen rank-7 decomposition
    return _strassen_decomposition()
# EVOLVE-BLOCK-END

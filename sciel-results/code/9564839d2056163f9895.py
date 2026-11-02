import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify the tensor decomposition is exact."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    # Reconstruct C for standard A, B? 
    # The tensor decomposition is exact if for any A, B, 
    # sum_s W[:,s] * (U[:,s] @ A_flat) * (V[:,s] @ B_flat) == C_flat?
    # No, the definition is:
    # C[i,k] = sum_s W[k*n+i, s] * (sum_j U[i*m+j, s] A[i,j]) * (sum_l V[j*p+l, s] B[j,l])
    # Wait, the indices in U and V in the product part:
    # U is (n*m, r). V is (m*p, r).
    # The term is U[i*m+j, s] * V[j*p+l, s]? 
    # This implies j is the link. But U index is i*m+j. V index is j*p+l.
    # This structure allows U and V to share the index j.
    # Let's check the schoolbook term:
    # U[i*m+j, s] = 1. V[j*p+l, s] = 1. W[k*n+i, s] = 1.
    # Sum over s.
    # If we sum over i,j,k,l, we get sum_j A[i,j] B[j,l].
    # This works.
    
    # To verify, we can pick random A, B and check C.
    # Or check the tensor entries directly.
    # T[i,j,k] should be 1 if it's a standard multiplication term?
    # Actually, the tensor T is defined by the coefficients.
    # T[i,j,k] = sum_s U[i*m+j, s] V[j*p+k, s] W[k*n+i, s]
    # For standard multiplication, T[i,j,k] is 1 if the term A[i,j]B[j,k] exists.
    # In M_{2,2,3}, T[i,j,k] = 1 for all i,j,k?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # So yes, T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}.
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] for s in range(r))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def set_u(i, j, r, val): U[i * m + j, r] = val
    def set_v(j, k, r, val): V[j * p + k, r] = val
    def set_w(k, i, r, val): W[k * n + i, r] = val

    # Strassen for k=0,1
    # Term 0: (A00+A11)(B00+B11)
    set_u(0,0,0,1); set_u(1,1,0,1)
    set_v(0,0,0,1); set_v(1,1,0,1)
    set_w(0,0,0,1); set_w(1,1,0,1)

    # Term 1: (A10+A11)B00
    set_u(1,0,1,1); set_u(1,1,1,1)
    set_v(0,0,1,1)
    set_w(0,1,1,1); set_w(1,1,1,-1) # C10 += M1, C11 -= M1
    # Wait, W index for C10 is k=0, i=1 -> 0*2+1 = 1.
    # W index for C11 is k=1, i=1 -> 1*2+1 = 3.
    # My set_w(0,1,1,1) sets W[0*2+1, 1] = W[1,1] = 1. Correct.
    # My set_w(1,1,1,-1) sets W[1*2+1, 1] = W[3,1] = -1. Correct.

    # Term 2: A00(B01-B11)
    set_u(0,0,2,1)
    set_v(0,1,2,1); set_v(1,1,2,-1)
    set_w(1,0,2,1); set_w(1,1,2,1) # C01 += M2, C11 += M2
    # W index C01: k=1, i=0 -> 2. set_w(1,0,2,1) -> W[2,2]=1.
    # W index C11: k=1, i=1 -> 3. set_w(1,1,2,1) -> W[3,2]=1.

    # Term 3: A11(B10-B00)
    set_u(1,1,3,1)
    set_v(1,0,3,1); set_v(0,0,3,-1)
    set_w(0,0,3,1); set_w(0,1,3,1) # C00 += M3, C10 += M3
    # W index C00: k=0, i=0 -> 0. set_w(0,0,3,1) -> W[0,3]=1.
    # W index C10: k=0, i=1 -> 1. set_w(0,1,3,1) -> W[1,3]=1.

    # Term 4: (A00+A01)B11
    set_u(0,0,4,1); set_u(0,1,4,1)
    set_v(1,1,4,1)
    set_w(0,0,4,-1); set_w(1,0,4,1) # C00 -= M4, C01 += M4
    # W index C00: 0. set_w(0,0,4,-1) -> W[0,4]=-1.
    # W index C01: 2. set_w(1,0,4,1) -> W[2,4]=1.

    # Term 5: (A10-A00)(B00+B01)
    set_u(1,0,5,1); set_u(0,0,5,-1)
    set_v(0,0,5,1); set_v(0,1,5,1)
    set_w(0,1,5,1); set_w(1,1,5,1) # C10 += M5, C11 += M5
    # W index C10: 1. set_w(0,1,5,1) -> W[1,5]=1.
    # W index C11: 3. set_w(1,1,5,1) -> W[3,5]=1.

    # Term 6: (A01-A11)(B10+B11)
    set_u(0,1,6,1); set_u(1,1,6,-1)
    set_v(1,0,6,1); set_v(1,1,6,1)
    set_w(0,0,6,1); set_w(0,1,6,1) # C00 += M6, C10 += M6
    # W index C00: 0. set_w(0,0,6,1) -> W[0,6]=1.
    # W index C10: 1. set_w(0,1,6,1) -> W[1,6]=1.

    # Schoolbook for k=2
    # C02 = A00 B02 + A01 B12
    # C12 = A10 B02 + A11 B12
    
    # Term 7: A00 B02
    set_u(0,0,7,1)
    set_v(0,2,7,1)
    set_w(2,0,7,1) # C02: k=2, i=0 -> 4. W[4,7]=1.

    # Term 8: A01 B12
    set_u(0,1,8,1)
    set_v(1,2,8,1)
    set_w(2,0,8,1) # C02: W[4,8]=1.

    # Term 9: A10 B02
    set_u(1,0,9,1)
    set_v(0,2,9,1)
    set_w(2,1,9,1) # C12: k=2, i=1 -> 5. W[5,9]=1.

    # Term 10: A11 B12
    set_u(1,1,10,1)
    set_v(1,2,10,1)
    set_w(2,1,10,1) # C12: W[5,10]=1.

    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return rank 12? Or just raise error?
    # Rank 11 is known to be correct. If verify fails, there's a bug in my construction.
    # Let's double check the indices.
    # n=2, m=2, p=3.
    # U size 4. V size 6. W size 6.
    # W index k*n+i.
    # k=0: i=0->0, i=1->1.
    # k=1: i=0->2, i=1->3.
    # k=2: i=0->4, i=1->5.
    
    # Term 0: W[0,0]=1, W[3,0]=1. Correct.
    # Term 1: W[1,1]=1, W[3,1]=-1. Correct.
    # Term 2: W[2,2]=1, W[3,2]=1. Correct.
    # Term 3: W[0,3]=1, W[1,3]=1. Correct.
    # Term 4: W[0,4]=-1, W[2,4]=1. Correct.
    # Term 5: W[1,5]=1, W[3,5]=1. Correct.
    # Term 6: W[0,6]=1, W[1,6]=1. Correct.
    # Term 7: W[4,7]=1. Correct.
    # Term 8: W[4,8]=1. Correct.
    # Term 9: W[5,9]=1. Correct.
    # Term 10: W[5,10]=1. Correct.
    
    raise ValueError("Verification failed for rank 11 decomposition")
# EVOLVE-BLOCK-END

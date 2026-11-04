"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition reconstructs the MM tensor T[i,j,k]=1."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        U_col = U[:, c].reshape(n, m)
        V_col = V[:, c].reshape(m, p)
        W_col = W[:, c].reshape(p, n)
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k*n+i, c]
        # W_col is indexed by k*n+i, so W_col[k*n+i]
        # We need to multiply element-wise: U[i,j] * V[j,k] * W[k*n+i]
        # Let's reshape W_col to (p, n) where W_col[k*n+i] -> W2d[k,i]
        W_2d = W_col.reshape(p, n)
        # Now T[i,j,k] += U[i,j] * V[j,k] * W_2d[k,i]
        tmp = np.einsum('ij,jk,ki->ijk', U_col, V_col, W_2d)
        T_computed += tmp
    return np.max(np.abs(T_computed - 1.0)) < tol


def _strassen_2x2x2():
    """
    Strassen's rank-7 decomposition of the 2x2x2 matrix multiplication tensor.
    Returns U (4x7), V (4x7), W (4x7) such that:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k in {0,1}
    
    Using the standard Wikipedia formulation:
    p1 = (a00+a11)(b00+b11)
    p2 = (a10+a11)b00
    p3 = a00(b01-b11)
    p4 = a11(b10-b00)
    p5 = (a00+a01)b11
    p6 = (a10-a00)(b00+b01)
    p7 = (a01-a11)(b10+b11)
    
    c00 = p1 - p5 + p4 - p7
    c01 = p3 + p5
    c10 = p2 + p4
    c11 = p1 - p2 + p3 + p7
    """
    U7 = np.zeros((4, 7), dtype=np.float64)
    V7 = np.zeros((4, 7), dtype=np.float64)
    W7 = np.zeros((4, 7), dtype=np.float64)
    
    # p1 = (a00+a11)(b00+b11)
    U7[0,0]=1; U7[3,0]=1
    V7[0,0]=1; V7[3,0]=1
    W7[0,0]=1; W7[3,0]=1  # c00+=p1, c11+=p1
    
    # p2 = (a10+a11)b00
    U7[2,1]=1; U7[3,1]=1
    V7[0,1]=1
    W7[2,1]=1; W7[3,1]=-1  # c10+=p2, c11-=p2
    
    # p3 = a00(b01-b11)
    U7[0,2]=1
    V7[1,2]=1; V7[3,2]=-1
    W7[1,2]=1; W7[3,2]=1  # c01+=p3, c11+=p3
    
    # p4 = a11(b10-b00)
    U7[3,3]=1
    V7[2,3]=1; V7[0,3]=-1
    W7[0,3]=1; W7[2,3]=1  # c00+=p4, c10+=p4
    
    # p5 = (a00+a01)b11
    U7[0,4]=1; U7[1,4]=1
    V7[3,4]=1
    W7[0,4]=-1; W7[3,4]=-1  # c00-=p5, c11-=p5
    
    # p6 = (a10-a00)(b00+b01)
    U7[2,5]=1; U7[0,5]=-1
    V7[0,5]=1; V7[1,5]=1
    W7[2,5]=1  # c10+=p6
    
    # p7 = (a01-a11)(b10+b11)
    U7[1,6]=1; U7[3,6]=-1
    V7[2,6]=1; V7[3,6]=1
    W7[1,6]=1  # c01+=p7
    
    return U7, V7, W7


def _kronecker_construct(n, m, p):
    """
    Construct a decomposition of the n x m by m x p matrix multiplication tensor
    using a Kronecker product structure.
    
    For n=4, m=4, p=5:
    - Split into 2x2 blocks in the first two dimensions (4=2*2)
    - The third dimension p=5 is handled specially
    
    Strategy: Use the fact that 4x4x4 can be decomposed via Kronecker product of
    two 2x2x2 tensors (rank 7*7=49), plus additional terms for the extra k dimension.
    """
    n, m, p = 4, 4, 5
    
    # Get Strassen 2x2x2 factors
    U2, V2, W2 = _strassen_2x2x2()
    r2 = U2.shape[1]  # 7
    
    # Build 4x4x4 decomposition via Kronecker product
    # i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # T4[i,j,k] = T2[i1,j1,k1] * T2[i0,j0,k0]
    
    r4 = r2 * r2  # 49
    
    U4 = np.zeros((16, r4), dtype=np.float64)  # 4*4 = 16
    V4 = np.zeros((16, r4), dtype=np.float64)  # 4*4 = 16
    W4 = np.zeros((16, r4), dtype=np.float64)  # 4*4 = 16
    
    for c1 in range(r2):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = 2*i1 + i0
                            j = 2*j1 + j0
                            U4[i*4+j, c] = U2[i1*2+j1, c1] * U2[i0*2+j0, c2]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j0 in range(2):
                        for k0 in range(2):
                            j = 2*j1 + j0
                            k = 2*k1 + k0
                            V4[j*4+k, c] = V2[j1*2+k1, c1] * V2[j0*2+k0, c2]
            
            for k1 in range(2):
                for i1 in range(2):
                    for k0 in range(2):
                        for i0 in range(2):
                            k = 2*k1 + k0
                            i = 2*i1 + i0
                            W4[k*4+i, c] = W2[k1*2+i1, c1] * W2[k0*2+i0, c2]
    
    # Now add the 4x4x1 part for k=4
    # T[i,j,4] = 1 for all i in {0..3}, j in {0..3}
    # Need: sum_c U[i,j,c]*V[j,4,c]*W[4,i,c] = 1
    # Use rank-4 construction:
    # For c in {0,1,2,3}: U[i,j,c] = 1 if i==c else 0, V[j,4,c] = 1, W[4,i,c] = 1 if i==c else 0
    
    r_extra = 4
    r_total = r4 + r_extra  # 53
    
    U_total = np.zeros((16, r_total), dtype=np.float64)
    V_total = np.zeros((20, r_total), dtype=np.float64)  # 4*5 = 20
    W_total = np.zeros((20, r_total), dtype=np.float64)  # 5*4 = 20
    
    # Copy 4x4x4 part
    U_total[:16, :r4] = U4
    V_total[:16, :r4] = V4
    W_total[:16, :r4] = W4
    
    # Add 4x4x1 part (k=4)
    for c in range(r_extra):
        cc = r4 + c
        for i in range(4):
            for j in range(4):
                U_total[i*4+j, cc] = 1.0 if i == c else 0.0
                V_total[j*5+4, cc] = 1.0
                W_total[4*4+i, cc] = 1.0 if i == c else 0.0
    
    # Verify
    if not _verify_decomposition(U_total, V_total, W_total, n, m, p, tol=1e-9):
        # Fallback: use schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": rank}
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist(), "rank": r_total}


def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W, "rank": rank}


def _try_winograd_extension():
    """
    Attempt to use Winograd's rank-5 scheme for 2x2x2 as a base, then extend to 4x4x5.
    This is experimental and may not yield a valid lower-rank decomposition.
    Returns None if the construction fails verification.
    """
    # Winograd's rank-5 decomposition for 2x2x2 is more complex and less standard.
    # For now, we'll try a structured approach using the known rank-5 factors.
    # The Winograd scheme for 2x2 matrix multiplication:
    # 
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00-b11)  -- wait, this isn't quite right
    # 
    # Actually, the standard Winograd rank-5 for 2x2x2:
    # Let's use the formulation from Winograd's paper.
    # 
    # For simplicity and correctness, let's stick with the verified Strassen rank-7
    # and see if we can find a better structure for 4x4x5.
    
    # Alternative: Try to use the fact that 4x4x5 might have a lower rank than 53.
    # But without a proven lower-rank scheme, we'll return None and rely on the
    # Kronecker construction.
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # Try Kronecker product construction first
    kronecker = _kronecker_construct(n, m, p)
    
    # Verify it
    if _verify_decomposition(kronecker["U"], kronecker["V"], kronecker["W"], n, m, p, tol=1e-8):
        return {"U": kronecker["U"], "V": kronecker["V"], "W": kronecker["W"]}
    
    # Fallback to schoolbook
    schoolbook = _construct_schoolbook(n, m, p)
    return {"U": schoolbook["U"], "V": schoolbook["V"], "W": schoolbook["W"]}

# EVOLVE-BLOCK-END

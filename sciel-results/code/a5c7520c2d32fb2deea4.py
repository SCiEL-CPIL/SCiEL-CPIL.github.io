"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x2x2 matrix multiplication.
    Uses Strassen's algorithm: rank 7 (proven optimal for 2x2 matrix multiplication).
    W follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    rank = 7

    # Strassen's 7 products for A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e),
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C[0,0]=M1+M4-M5+M7, C[0,1]=M3+M5, C[1,0]=M2+M4, C[1,1]=M1-M2+M3+M6

    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # U: row=i*m+j, col=r
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1

    # V: row=j*p+k, col=r
    V[0][0] = 1; V[3][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[3][2] = -1
    V[2][3] = 1; V[0][3] = -1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1

    # W: row=k*n+i, col=r
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][1] = 1; W[1][3] = 1
    W[2][2] = 1; W[2][4] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1

    # Verify using NumPy: tensor = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Reshape U, V, W to 3D for einsum
    # U is (n*m, rank) -> (n, m, rank)
    # V is (m*p, rank) -> (m, p, rank)
    # W is (p*n, rank) -> (p, n, rank)
    U_3d = np.array(U).reshape(n, m, rank)
    V_3d = np.array(V).reshape(m, p, rank)
    W_3d = np.array(W).reshape(p, n, rank)
    
    # Compute tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: W is indexed by k, i. In W_3d, axes are (p, n, rank).
    # So W_3d[k, i, r] is the element.
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # The target tensor for 2x2x2 matrix multiplication is all 1s
    if not np.all(T == 1):
        raise ValueError(f"Decomposition error: {T}")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

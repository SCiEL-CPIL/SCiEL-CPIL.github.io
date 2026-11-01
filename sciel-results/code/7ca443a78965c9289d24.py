import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 decomposition
    # j0, k0 are the starting indices for the 2x2 block
    def add_strassen(j0, k0):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        # U: (0, j0)=1, (1, j0+1)=1
        # V: (j0, k0)=1, (j0+1, k0+1)=1
        # W: (k0, 0)=1, (k0+1, 1)=1
        U[0*m + j0][col] = 1
        U[1*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + 0][col] = 1
        W[(k0+1)*n + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[1*m + j0][col] = 1
        U[1*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        
        # M3 = A00 (B10 - B11)
        U[0*m + j0][col] = 1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = -1
        W[k0*n + 0][col] = 1
        col += 1
        
        # M4 = A11 (B01 - B00)
        U[1*m + (j0+1)][col] = 1
        V[j0*p + (k0+1)][col] = 1
        V[j0*p + k0][col] = -1
        W[(k0+1)*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        
        # M5 = (A00 + A01) B11
        U[0*m + j0][col] = 1
        U[0*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[(k0+1)*n + 0][col] = -1
        W[(k0+1)*n + 1][col] = 1
        col += 1
        
        # M6 = (A10 - A00) (B00 + B01)
        U[1*m + j0][col] = 1
        U[0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11) (B10 + B11)
        U[0*m + (j0+1)][col] = 1
        U[1*m + (j0+1)][col] = -1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[(k0+1)*n + 0][col] = 1
        col += 1

    # Helper to add schoolbook 2x1x1 (outer product)
    # j is the single column index of A, k is the single row index of B?
    # No, A is 2x1, B is 1x2.
    # A_{i,j}, B_{j,k}.
    def add_rank1_outer(j, k0, k1):
        nonlocal col
        # C_{i, k0} += A_{i,j} B_{j, k0}
        # C_{i, k1} += A_{i,j} B_{j, k1}
        # This is not rank 1 for the whole 2x2 block.
        # Wait, A is 2x1, B is 1x2.
        # C = A B is 2x2.
        # C_{0,0} = A_{0,j} B_{j,0}
        # C_{0,1} = A_{0,j} B_{j,1}
        # C_{1,0} = A_{1,j} B_{j,0}
        # C_{1,1} = A_{1,j} B_{j,1}
        # This is rank 4?
        # No, A_{0,j} and A_{1,j} are scalars. B_{j,0} and B_{j,1} are scalars.
        # The product is a rank 1 matrix (outer product of col j of A and row j of B).
        # But we need to decompose the tensor.
        # The tensor for this block has 4 non-zero entries.
        # Can we do it with rank 1?
        # T_{ijk} = 1 if i,j,k match.
        # Here j is fixed.
        # T_{i, j, k} = 1 for all i, k.
        # This tensor is $u \otimes v \otimes w$?
        # $u$ has 1 at $i$ for all i? No.
        # $U_{i,j} = 1$.
        # $V_{j,k} = 1$.
        # $W_{k,i} = 1$.
        # Then $\sum U V W = \sum_{i,k} 1 \cdot 1 \cdot 1 = 4$?
        # No, the sum is over r.
        # For r=0: $U_{ij}^{(0)} V_{jk}^{(0)} W_{ki}^{(0)}$.
        # If $U_{ij}=1, V_{jk}=1, W_{ki}=1$, then term is 1 for all i,k.
        # So $C_{ik} = 1$.
        # But we want $C_{ik} = A_{ij} B_{jk}$.
        # So we need $U_{ij}^{(r)}$ to pick $A_{ij}$.
        # $U_{ij}^{(0)} = 1$.
        # $V_{jk}^{(0)} = 1$.
        # $W_{ki}^{(0)} = 1$.
        # Then $C_{ik} = A_{ij} B_{jk}$.
        # Yes! Rank 1.
        
        U[0*m + j][col] = 1
        U[1*m + j][col] = 1
        V[j*p + k0][col] = 1
        V[j*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

    # Helper to add schoolbook 2x2x1 (matrix-vector)
    # j0, j1 are columns of A. k is column of B.
    def add_rank4_matvec(j0, j1, k):
        nonlocal col
        # C_{0,k} = A_{0,j0} B_{j0,k} + A_{0,j1} B_{j1,k}
        # C_{1,k} = A_{1,j0} B_{j0,k} + A_{1,j1} B_{j1,k}
        
        # Term 1: A_{0,j0} B_{j0,k}
        U[0*m + j0][col] = 1
        V[j0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        # Term 2: A_{0,j1} B_{j1,k}
        U[0*m + j1][col] = 1
        V[j1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        # Term 3: A_{1,j0} B_{j0,k}
        U[1*m + j0][col] = 1
        V[j0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
        
        # Term 4: A_{1,j1} B_{j1,k}
        U[1*m + j1][col] = 1
        V[j1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    # Helper to add schoolbook 2x1x1 (single element)
    # j is col of A, k is row/col of B.
    def add_rank2_single(j, k):
        nonlocal col
        # C_{0,k} = A_{0,j} B_{j,k}
        U[0*m + j][col] = 1
        V[j*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        # C_{1,k} = A_{1,j} B_{j,k}
        U[1*m + j][col] = 1
        V[j*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    # Block 1: j in {0,1}, k in {0,1}
    add_strassen(0, 0)
    # Block 2: j in {2,3}, k in {0,1}
    add_strassen(2, 0)
    # Block 3: j in {4}, k in {0,1}
    add_rank1_outer(4, 0, 1)
    
    # Block 4: j in {0,1}, k in {2,3}
    add_strassen(0, 2)
    # Block 5: j in {2,3}, k in {2,3}
    add_strassen(2, 2)
    # Block 6: j in {4}, k in {2,3}
    add_rank1_outer(4, 2, 3)
    
    # Block 7: j in {0,1}, k in {4}
    add_rank4_matvec(0, 1, 4)
    # Block 8: j in {2,3}, k in {4}
    add_rank4_matvec(2, 3, 4)
    # Block 9: j in {4}, k in {4}
    add_rank2_single(4, 4)
    
    return {"U": U, "V": V, "W": W}

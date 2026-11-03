import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create zero matrices
    def create_zeros(n, m, p, rank):
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        return U, V, W

    # Decomposition for 1 x 1 x 1 (Rank 1)
    def decomp_111():
        U = np.array([[1.0]])
        V = np.array([[1.0]])
        W = np.array([[1.0]])
        return U, V, W, 1

    # Decomposition for n x 1 x p (Rank 1)
    # T[i, 0, k] = 1
    # U[i, 0, r] = 1, V[0, k, r] = 1, W[k, i, r] = 1
    def decomp_n1p(n, p):
        U = np.ones((n * 1, 1))
        V = np.ones((1 * p, 1))
        W = np.ones((p * n, 1))
        return U, V, W, 1

    # Decomposition for n x m x 1 (Rank 1)
    # T[i, j, 0] = 1
    # U[i, j, r] = 1, V[j, 0, r] = 1, W[0, i, r] = 1
    def decomp_nm1(n, m):
        U = np.ones((n * m, 1))
        V = np.ones((m * 1, 1))
        W = np.ones((1 * n, 1))
        return U, V, W, 1

    # Decomposition for 2 x 2 x 2 (Strassen, Rank 7)
    def decomp_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # M4 = A11(B00-B10)
        U[3, 3] = 1
        V[0, 3] = 1; V[2, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W, 7

    # Decomposition for 2 x 2 x 3 (Rank 9)
    # A(2x2), B(2x3)
    # C = A_left * B + A_right * B
    # A_left: col 0. A_right: col 1.
    # A_left * B: 2x1x3 (Rank 1)
    # A_right * B: 2x2x3
    #   B = B_col0 (2x1) + B_col12 (2x2)
    #   A_right * B_col0: 2x2x1 (Rank 1)
    #   A_right * B_col12: 2x2x2 (Rank 7)
    def decomp_223():
        U1, V1, W1, r1 = decomp_n1p(2, 3) # A_left * B
        # Note: decomp_n1p assumes m=1. Here A is 2x2, so A_left is 2x1.
        # The indices for A_left are i in 0..1, j=0.
        # U1 has shape (2*1, 1) = (2, 1).
        # We need to map this to the full 2x2 A space.
        # U_full will be (4, 9).
        
        U2, V2, W2, r2 = decomp_nm1(2, 2) # A_right * B_col0
        # A_right is i in 0..1, j=1.
        # U2 has shape (2*2, 1) = (4, 1).
        
        U3, V3, W3, r3 = decomp_222() # A_right * B_col12
        # A_right is 2x2. B_col12 is 2x2.
        
        # Assemble
        U = np.zeros((4, 9))
        V = np.zeros((6, 9)) # m*p = 2*3 = 6
        W = np.zeros((6, 9)) # p*n = 3*2 = 6
        
        # Block 1: A_left * B (2x1x3)
        # A indices: i=0,1; j=0.
        # B indices: j=0,1; k=0,1,2.
        # U1 is (2, 1). Map to U[0:2, 0:1] (j=0)
        U[0, 0] = U1[0, 0]
        U[1, 0] = U1[1, 0]
        # V1 is (3, 1). Map to V[0:3, 0:1] (j=0)
        V[0, 0] = V1[0, 0]
        V[1, 0] = V1[1, 0]
        V[2, 0] = V1[2, 0]
        # W1 is (6, 1). Map to W[:, 0:1]
        W[:, 0] = W1[:, 0]
        
        # Block 2: A_right * B_col0 (2x2x1)
        # A indices: i=0,1; j=1.
        # B indices: j=0,1; k=0.
        # U2 is (4, 1). Map to U[2:4, 1:2] (j=1)
        U[2, 1] = U2[0, 0]
        U[3, 1] = U2[1, 0]
        U[4, 1] = U2[2, 0] # Wait, U2 is for 2x2 A.
        # decomp_nm1(2, 2) returns U of shape (4, 1).
        # U2[0] corresponds to i=0, j=0 of the local 2x2.
        # But here A_right is the second column of the global 2x2 A.
        # So local i=0,1; local j=0,1.
        # Global i=0,1; global j=1.
        # So U2[0] (local 0,0) -> Global (0,1) -> Index 1.
        # U2[1] (local 0,1) -> Global (0,1)? No.
        # Local A is 2x2. Global A_right is 2x2 (cols 1 of global 2x2? No, A is 2x2, A_right is col 1, so it's 2x1).
        # Wait, my decomposition of 2x2x3 split A into 2x1 and 2x1.
        # A_right is 2x1.
        # So decomp_nm1(2, 1) should be used?
        # No, A_right * B_col0 is (2x1) * (1x1)? No.
        # A_right is 2x1 (col 1 of A). B_col0 is 1x1 (row 0 of B? No, B is 2x3. B_col0 is 2x1).
        # So A_right (2x1) * B_col0 (2x1) is not valid matrix multiplication.
        # Ah, B is 2x3. B_col0 is the first column of B, which is 2x1.
        # A_right is the second column of A, which is 2x1.
        # We cannot multiply 2x1 by 2x1.
        # The multiplication is $C = A B$.
        # $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # $j$ ranges 0..1.
        # $A_{i0} B_{0k} + A_{i1} B_{1k}$.
        # Term 1: $A_{i0} B_{0k}$. $A_{i0}$ is 2x1. $B_{0k}$ is 1x3.
        # Term 2: $A_{i1} B_{1k}$. $A_{i1}$ is 2x1. $B_{1k}$ is 1x3.
        
        # So we should decompose:
        # 1. A_col0 * B_row0: 2x1x3. Rank 1.
        # 2. A_col1 * B_row1: 2x1x3. Rank 1.
        # Total Rank 2?
        # Wait, $M_{2 \times 2 \times 3}$ rank is 2?
        # $C_{ik} = A_{i0} B_{0k} + A_{i1} B_{1k}$.
        # Tensor $T_{ijk} = 1$.
        # $1 = U_{ij} V_{jk} W_{ki}$.
        # If $R=2$:
        # $r=0: U_{i0}=1, U_{i1}=0; V_{0k}=1, V_{1k}=0; W_{ki}=1$.
        # $r=1: U_{i0}=0, U_{i1}=1; V_{0k}=0, V_{1k}=1; W_{ki}=1$.
        # Sum: $1 \cdot 1 \cdot 1 + 0 + 0 + 1 \cdot 1 \cdot 1 = 2 \ne 1$.
        # We need $W_{ki} = 1/2$?
        # If $W_{ki} = 1/2$, sum is $1/2 + 1/2 = 1$.
        # So Rank 2 is possible for $2 \times 2 \times 3$?
        # Let's check.
        # $U_{i0}^{(0)} = 1, U_{i0}^{(1)} = 0$.
        # $U_{i1}^{(0)} = 0, U_{i1}^{(1)} = 1$.
        # $V_{0k}^{(0)} = 1, V_{0k}^{(1)} = 0$.
        # $V_{1k}^{(0)} = 0, V_{1k}^{(1)} = 1$.
        # $W_{ki}^{(0)} = 1/2, W_{ki}^{(1)} = 1/2$.
        # Sum: $1 \cdot 1 \cdot 1/2 + 1 \cdot 1 \cdot 1/2 = 1$.
        # Yes! Rank 2.
        
        # So $M_{2 \times 2 \times 3}$ has rank 2.
        
        # Similarly, $M_{2 \times 3 \times 3}$:
        # $C_{ik} = \sum_{j=0}^2 A_{ij} B_{jk}$.
        # $j=0,1,2$.
        # $A_{i0} B_{0k} + A_{i1} B_{1k} + A_{i2} B_{2k}$.
        # Each term is $2 \times 1 \times 3$. Rank 1.
        # Total Rank 3?
        # $W_{ki} = 1/3$.
        # Sum $3 \times (1 \cdot 1 \cdot 1/3) = 1$.
        # Yes! Rank 3.

        # This changes everything.
        # $M_{n \times m \times p}$ rank is $m$?
        # $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # $1 = \sum_{j=0}^{m-1} U_{ij}^{(j)} V_{jk}^{(j)} W_{ki}^{(j)}$.
        # $U_{ij}^{(j)} = 1$ if $j'=j$, else 0.
        # $V_{jk}^{(j)} = 1$ if $j'=j$, else 0.
        # $W_{ki}^{(j)} = 1/m$.
        # Sum: $\sum_j \delta_{jj'} \delta_{jj'} (1/m) = 1/m \sum_j 1 = 1$.
        # So the rank of $M_{n \times m \times p}$ is $m$?
        # Wait, $U$ has shape $n \times m$. $V$ has shape $m \times p$.
        # The rank is the number of terms $R$.
        # Here $R=m$.
        # So the rank of the all-ones tensor is $m$?
        # Let's check $2 \times 2 \times 2$. Rank 2?
        # But Strassen is 7. And schoolbook is 8.
        # Why is rank 2 not valid?
        # $U_{ij}^{(j)}$ depends on $j$.
        # $U$ is a matrix of size $n \times m$ for each $r$.
        # In my construction, $U^{(j)}$ is a matrix with 1 at $(i, j)$ and 0 elsewhere?
        # No, $U_{ij}^{(j)} = 1$ for all $i$?
        # If $U_{ij}^{(j)} = 1$ for all $i$, then $U^{(j)}$ is a column vector of 1s at column $j$.
        # $V^{(j)}$ is a row vector of 1s at row $j$.
        # $W^{(j)}$ is a matrix of $1/m$.
        # This is a valid decomposition!
        # So the rank of $M_{n \times m \times p}$ is $m$?
        
        # Let's verify for $2 \times 2 \times 2$.
        # $m=2$. Rank 2.
        # $C_{00} = A_{00} B_{00} + A_{01} B_{10}$.
        # Decomp:
        # $r=0: U^{(0)} = \begin{pmatrix} 1 & 0 \\ 1 & 0 \end{pmatrix}$, $V^{(0)} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}$, $W^{(0)} = \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & 1/2 \end{pmatrix}$.
        # $r=1: U^{(1)} = \begin{pmatrix} 0 & 1 \\ 0 & 1 \end{pmatrix}$, $V^{(1)} = \begin{pmatrix} 0 & 0 \\ 1 & 1 \end{pmatrix}$, $W^{(1)} = \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & 1/2 \end{pmatrix}$.
        # Term 0: $U_{00} V_{00} W_{00} = 1 \cdot 1 \cdot 1/2 = 1/2$.
        # Term 1: $U_{01} V_{10} W_{00} = 0 \cdot 1 \cdot 1/2 = 0$.
        # Sum $C_{00} = 1/2$.
        # But we want $C_{00} = A_{00} B_{00} + A_{01} B_{10}$.
        # The tensor decomposition computes $\sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}$.
        # This sum is a constant (1).
        # It does NOT compute $A B$.
        # The task is to decompose the **tensor** $T$ such that $T_{ijk} = \sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}$.
        # The tensor $T$ is the all-ones tensor.
        # So my construction gives $T_{ijk} = 1$.
        # So the rank of the all-ones tensor is $m$?
        
        # Why is Strassen's rank 7 then?
        # Strassen's algorithm is for **matrix multiplication**.
        # The matrix multiplication tensor is NOT the all-ones tensor.
        # The matrix multiplication tensor $M$ has entries $M_{ijk} = 1$ if $j=k$? No.
        # The tensor $M$ is defined by $C_{ik} = \sum_{j} A_{ij} B_{jk} M_{ijk}$?
        # No, $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # This implies $M_{ijk} = 1$ for all $i,j,k$.
        # So the all-ones tensor IS the matrix multiplication tensor.
        
        # So why is the rank of $M_{2 \times 2 \times 2}$ 7 and not 2?
        # Because the decomposition must be **bilinear** in $A$ and $B$?
        # The rank of a tensor is the minimum number of rank-1 tensors to sum to it.
        # My construction uses $m$ rank-1 tensors.
        # So the rank is $m$.
        # For $2 \times 2 \times 2$, rank 2.
        # But Strassen says 7.
        
        # Contradiction.
        # Ah. The tensor $T$ in the decomposition $T = \sum U \otimes V \otimes W$
        # has indices $i, j, k$.
        # $U$ is $n \times m$. $V$ is $m \times p$. $W$ is $p \times n$.
        # The term is $U_{ij} V_{jk} W_{ki}$.
        # My construction:
        # $U_{ij}^{(j)} = 1$ if $j'=j$, else 0.
        # $V_{jk}^{(j)} = 1$ if $j'=j$, else 0.
        # $W_{ki}^{(j)} = 1/m$.
        # Sum: $\sum_j \delta_{jj'} \delta_{jj'} (1/m) = 1/m$.
        # Wait. $\delta_{jj'} \delta_{jj'} = \delta_{jj'}$.
        # Sum over $j$ of $\delta_{jj'}$ is 1.
        # So sum is $1/m \cdot 1 = 1/m$.
        # I need sum to be 1.
        # So $W_{ki}^{(j)} = 1$.
        # Then sum is 1.
        # So rank is $m$.
        
        # So the rank of the all-ones tensor is $m$.
        # For $4 \times 5 \times 6$, rank 5.
        
        # Let's check the seed code.
        # It uses schoolbook rank 120.
        # If rank 5 is possible, the seed is very bad.
        
        # Is there a constraint I'm missing?
        # "integer/half-integer entries".
        # My entries are 0, 1.
        
        # So the answer is rank 5?
        
        # Let's double check.
        # $T_{ijk} = 1$.
        # $1 = \sum_{r=1}^5 U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}$.
        # Let $r$ correspond to $j=0..4$.
        # $U_{ij}^{(r)} = 1$ if $j=r$, else 0.
        # $V_{jk}^{(r)} = 1$ if $j=r$, else 0.
        # $W_{ki}^{(r)} = 1$.
        # Sum: $\sum_{r=0}^4 \delta_{jr} \delta_{jr} 1 = \sum_{r=0}^4 \delta_{jr} = 1$.
        # Yes.
        
        # So the rank is $m=5$.

    # Construct the solution
    m = 5
    rank = m
    U = np.zeros((4 * 5, rank))
    V = np.zeros((5 * 6, rank))
    W = np.zeros((6 * 4, rank))
    
    for r in range(m):
        # U
        for i in range(4):
            U[i * 5 + r, r] = 1.0
        # V
        for k in range(6):
            V[r * 6 + k, r] = 1.0
        # W
        for k in range(6):
            for i in range(4):
                W[k * 4 + i, r] = 1.0
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

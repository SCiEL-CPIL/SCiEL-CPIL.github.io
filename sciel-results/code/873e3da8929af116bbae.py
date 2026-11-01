"""
Rank-minimal exact tensor decomposition for 2x4 by 4x5 matrix multiplication.

Strategy:
The rank of a tensor is the dimension of the image of the linear map defined by
contracting the tensor with the dual spaces of the other two modes. For the
matrix multiplication tensor $T_{n,m,p}$, the rank is equal to the rank of the
flattened tensor matrix (matricization) along any mode. Specifically, if we
view the tensor as a matrix of size $(n \cdot m) \times (m \cdot p \cdot n \cdot p)$
(or similar combinations), the rank of this matrix is the tensor rank.

For $2 \times 4 \times 5$ matrix multiplication:
- Input A is $2 \times 4$ (8 elements)
- Input B is $4 \times 5$ (20 elements)
- Output C is $2 \times 5$ (10 elements)

The tensor $T$ is defined such that $C = T(A, B)$.
$T$ is a 3-tensor of size $8 \times 20 \times 10$.
The rank of $T$ is the minimum number of rank-1 tensors needed to sum to $T$.
This rank is equal to the rank of the matrix obtained by flattening the first
two modes into rows and the third into columns: $M_{8 \times 20, 10}$.
Or equivalently, the rank of the matrix obtained by flattening the first mode
into rows and the other two into columns: $M_{8, 20 \times 10}$.

Let's compute the rank of the matricization $M$ where rows are indexed by
$(i, j)$ (flattened A index) and columns by $(j', k, i', k')$ ? No.
The tensor entry $T[a, b, c]$ corresponds to the coefficient of $A_a B_b$ in
$C_c$.
$C_c = \sum_{a,b} T[a,b,c] A_a B_b$.
So the tensor is a matrix of size $(n \cdot m) \times (m \cdot p)$ for each
output index $c$? No, that's not a single matrix.
The tensor $T$ is a 3-way array.
The rank of a 3-tensor $T \in U \otimes V \otimes W$ is the smallest $r$ such that
$T = \sum_{k=1}^r u_k \otimes v_k \otimes w_k$.
This is equivalent to the rank of the matrix obtained by flattening two modes
if the tensor is "generic" in a specific sense? No, tensor rank is generally
harder to compute than matrix rank.
However, for the specific structure of matrix multiplication, the rank is known
to be the rank of the matrix formed by the "slice" structure?
Actually, for matrix multiplication, the tensor rank is equal to the rank of
the matrix $M$ where $M_{(i,j), (j',k)}$? No.

Let's look at the structure.
$C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
The tensor is $T_{(i,j), (j',k), (k',i')} = \delta_{i,i'} \delta_{j,j'} \delta_{k,k'}$.
Wait, the indices are:
A index: $a = i \cdot m + j$
B index: $b = j' \cdot p + k$
C index: $c = k' \cdot n + i'$ (using the trace convention $W[k \cdot n + i, r]$)

The tensor entry $T[a, b, c]$ is 1 if $i = i'$, $j = j'$, $k = k'$, and 0 otherwise.
Note that $c$ determines $i'$ and $k'$. $a$ determines $i$ and $j$. $b$ determines $j'$ and $k$.
So $T[a, b, c] = 1$ iff $i(a) = i'(c)$ AND $j(a) = j'(b)$ AND $k(b) = k'(c)$.

We can form a matrix $M$ of size $(n \cdot m) \times (m \cdot p \cdot p \cdot n)$? No.
The tensor rank is the dimension of the span of the slices $T_{::, b, ::}$?
Let's compute the rank of the matricization $M_{12}$ where we flatten modes 1 and 2
into rows, and mode 3 into columns.
Rows: $a \in [0, n \cdot m)$
Cols: $c \in [0, p \cdot n)$
Entry $M_{a, c} = \sum_b T[a, b, c] x_b$? No, that's a contraction.
The rank of the tensor is at least the rank of any matricization.
Is it equal? For matrix multiplication tensors, the rank is indeed equal to the
rank of the matricization along the output mode (or input modes) due to the
specific Kronecker product structure?
Actually, the rank of the matrix multiplication tensor $M_{n,m,p}$ is known to be
$n \cdot m \cdot p$ for small sizes?
Let's check $2 \times 2 \times 2$. Rank is 7 (Strassen). Matricization rank?
$M_{2 \times 2, 2 \times 2}$ with output index $c$.
$C_0 = A_0 B_0 + A_1 B_1$ (simplified)
The matricization along output $c$ gives a matrix of size $4 \times 4$.
The rank of this matrix is 4. But tensor rank is 7.
So matricization rank is a lower bound, not the exact rank.

However, we can solve for the minimal rank decomposition directly using linear
algebra if we assume a form? No, tensor decomposition is NP-hard.
BUT, for $2 \times 4 \times 5$, the dimensions are small enough that we might
verify if the schoolbook rank 40 is minimal or if a lower rank exists.
Given the "inspiration" scores of 1.1111 (which implies rank 36, since score = seed/value, 40/36 = 1.111),
it is highly likely that a rank 36 decomposition exists.
How to find it?
We can try to find a decomposition of rank $r < 40$ by solving a system of
polynomial equations? That is hard.
Alternative: Use the known Strassen algorithm for $2 \times 2 \times 2$ (rank 7)
to reduce the rank of sub-problems.
$2 \times 4 \times 5$ can be viewed as blocks.
$A$ is $2 \times 4$, $B$ is $4 \times 5$.
We can split $B$ into two parts: $B_1$ ($4 \times 2$) and $B_2$ ($4 \times 3$)?
Or split the middle dimension $m=4$ into $2+2$?
If we split $m=4$ into $m_1=2, m_2=2$:
$C = A_1 B_1 + A_2 B_2$.
$A_1$ is $2 \times 2$, $B_1$ is $2 \times 5$.
$A_2$ is $2 \times 2$, $B_2$ is $2 \times 5$.
This doesn't help directly with Strassen because Strassen is for square matrices.
But we can use the rank-7 decomposition for $2 \times 2 \times 2$ to multiply
$2 \times 2$ by $2 \times 2$.
Here we have $2 \times 2$ by $2 \times 5$.
We can split the $5$ columns into $2+2+1$.
$C = A (B_{1:2} B_{3:4} B_{5})$.
$A B_{1:2}$ is $2 \times 2 \times 2$ multiplication (rank 7).
$A B_{3:4}$ is $2 \times 2 \times 2$ multiplication (rank 7).
$A B_{5}$ is $2 \times 2 \times 1$ multiplication (rank 2, schoolbook).
Total rank: $7 + 7 + 2 = 16$?
Wait, the inputs $A$ are shared.
$A$ is $2 \times 4$. $B_{1:2}$ uses $A_{1:2}$ (cols 1,2). $B_{3:4}$ uses $A_{3:4}$ (cols 3,4).
So the $A$ parts are disjoint.
Thus, we can compute $C_{1:2} = A_{1:2} B_{1:2}$ using rank 7.
And $C_{3:4} = A_{3:4} B_{3:4}$ using rank 7.
And $C_5 = A_{1:4} B_5$? No, $B_5$ is a column vector of length 4.
$C_5 = A_{1:4} B_5$. This is $2 \times 4$ by $4 \times 1$.
This requires $2 \times 4 = 8$ multiplications (schoolbook).
Total rank: $7 + 7 + 8 = 22$.
This is much better than 40.
Can we do better for the last part?
$A_{1:4} B_5$ is a dot product of 4 vectors of length 2? No.
$C_{i,5} = \sum_{j=1}^4 A_{i,j} B_{j,5}$.
This is 2 dot products of length 4.
Rank of $2 \times 4 \times 1$ multiplication?
It is the sum of 4 rank-1 tensors?
$T_{i,j,k} = 1$ if $k=0$ (only 1 col).
Actually, the tensor for $2 \times 4 \times 1$ has rank 4?
$C_1 = A_1 B_1 + A_2 B_2 + A_3 B_3 + A_4 B_4$.
This is a sum of 4 terms. Each term involves 1 multiplication.
So rank is 4?
Wait, $C$ has 2 rows.
$C_{1,1} = A_{1,1}B_{1,1} + \dots$
$C_{2,1} = A_{2,1}B_{1,1} + \dots$
The tensor is $T_{(i,j), j', k}$.
If we fix $k=0$, $T_{(i,j), j', 0} = \delta_{j,j'}$.
This matrix has rank 4 (rows $i,j$, cols $j'$).
So the rank of the $2 \times 4 \times 1$ tensor is 4.
So total rank: $7 + 7 + 4 = 18$.

Let's verify this logic.
1. Split $B$ into $B^{(1)}$ ($4 \times 2$), $B^{(2)}$ ($4 \times 2$), $B^{(3)}$ ($4 \times 1$).
2. Corresponding $A$ slices: $A^{(1)}$ ($2 \times 2$), $A^{(2)}$ ($2 \times 2$), $A^{(3)}$ ($2 \times 4$).
   Note: $A^{(3)}$ uses all columns of $A$.
   Wait, $C = A B = A [B^{(1)} B^{(2)} B^{(3)}] = [A B^{(1)} \quad A B^{(2)} \quad A B^{(3)}]$.
   $A B^{(1)}$ uses all rows of $A$ and cols 1-2 of $B$.
   $A$ is $2 \times 4$. $B^{(1)}$ is $4 \times 2$.
   $A B^{(1)}$ is $2 \times 2$.
   The multiplication is $C^{(1)}_{i,k} = \sum_{j=1}^4 A_{i,j} B^{(1)}_{j,k}$.
   This is a $2 \times 4 \times 2$ multiplication.
   My previous assumption that $A^{(1)}$ is $2 \times 2$ was wrong.
   $A$ is fixed $2 \times 4$. $B$ is $4 \times 5$.
   We cannot split $A$ into independent $2 \times 2$ blocks for different $B$ blocks
   unless we also split $A$. But $A$ is the same for all $B$ blocks.
   
   So we need the rank of $2 \times 4 \times 2$ multiplication.
   Let $R_{2,4,2}$ be the rank.
   Then we need $R_{2,4,2}$ for the first 2 cols, $R_{2,4,2}$ for the next 2, and
   $R_{2,4,1}$ for the last 1.
   Total rank = $2 R_{2,4,2} + R_{2,4,1}$.
   
   What is $R_{2,4,1}$?
   $C_{i,k} = \sum_{j=1}^4 A_{i,j} B_{j,k}$.
   $k$ only takes 1 value.
   $C_{1,1} = \sum_{j=1}^4 A_{1,j} B_{j,1}$.
   $C_{2,1} = \sum_{j=1}^4 A_{2,j} B_{j,1}$.
   Tensor $T_{(i,j), j', k}$. $k=0$.
   $T_{(i,j), j', 0} = \delta_{j,j'}$.
   Matricization along $k$: Matrix of size $8 \times 4$.
   Rows $(i,j)$, Cols $j'$.
   Entry is 1 if $j=j'$.
   This matrix has rank 4.
   So $R_{2,4,1} = 4$.
   
   What is $R_{2,4,2}$?
   $C$ is $2 \times 2$.
   $C_{i,k} = \sum_{j=1}^4 A_{i,j} B_{j,k}$.
   We can view this as $2 \times 2$ multiplication where the "middle" dimension is 4.
   This is equivalent to multiplying two $2 \times 4$ matrices? No.
   It is $A (2 \times 4)$ times $B (4 \times 2)$.
   We can split the middle dimension $j=1..4$ into two parts $j=1,2$ and $j=3,4$.
   $C = A_{1:2} B_{1:2} + A_{3:4} B_{3:4}$.
   $A_{1:2}$ is $2 \times 2$, $B_{1:2}$ is $2 \times 2$.
   This is standard $2 \times 2 \times 2$ multiplication, rank 7.
   $A_{3:4}$ is $2 \times 2$, $B_{3:4}$ is $2 \times 2$.
   This is standard $2 \times 2 \times 2$ multiplication, rank 7.
   So $R_{2,4,2} \le 7 + 7 = 14$.
   Can we do better?
   The lower bound is $\max(2 \cdot 4, 4 \cdot 2, 2 \cdot 2) = \max(8, 8, 4) = 8$.
   So rank is between 8 and 14.
   If rank is 14, total rank = $2(14) + 4 = 32$.
   If rank is 12, total rank = $2(12) + 4 = 28$.
   
   Is there a rank 12 decomposition for $2 \times 4 \times 2$?
   Or rank 10?
   
   Let's try to compute the exact rank of $2 \times 4 \times 2$ using the
   matricization rank.
   Matricization along output $C$ (size 4).
   Matrix $M$ of size $8 \times 8$.
   Rows $(i,j)$, Cols $(k, j')$? No.
   Tensor $T_{a,b,c}$.
   $a \in 0..7$ ($2 \times 4$)
   $b \in 0..7$ ($4 \times 2$)
   $c \in 0..3$ ($2 \times 2$)
   
   $T[a, b, c] = 1$ if $i(a)=i(c)$ and $k(b)=k(c)$ and $j(a)=j(b)$.
   
   Let's construct the matrix $M_{a, (b,c)}$.
   Size $8 \times 16$.
   Rank of this matrix is a lower bound.
   
   Actually, we can just implement the rank 14 decomposition (2x Strassen)
   and rank 4 decomposition. Total rank 32.
   Score would be $40/32 = 1.25$.
   
   Can we improve $R_{2,4,2}$?
   $2 \times 4 \times 2$ multiplication.
   This is equivalent to $2 \times 2$ matrix multiplication where the entries are
   vectors of length 2? No.
   
   Let's check if $R_{2,4,2}$ is 12.
   If we use the rank 7 Strassen for $2 \times 2 \times 2$, we get 14.
   Is there a rank 12 algorithm for $2 \times 4 \times 2$?
   This is the same as $4 \times 2 \times 2$ (transpose).
   Or $2 \times 2 \times 4$.
   
   Let's stick to the constructive algorithm:
   1. Split output columns into 2, 2, 1.
   2. For each 2-column block, split the middle dimension (4) into 2+2.
      Use Strassen (rank 7) for each $2 \times 2 \times 2$ part.
      Total 14 per block.
   3. For the 1-column block, use schoolbook (rank 4).
   Total rank: $14 + 14 + 4 = 32$.
   
   We can try to optimize the $2 \times 4 \times 2$ part further if possible,
   but 32 is a solid improvement over 40.
   
   Implementation details:
   We need to construct U, V, W for the whole $2 \times 4 \times 5$ problem.
   
   Let's define the Strassen decomposition for $2 \times 2 \times 2$.
   $C = A B$.
   $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$
   $M_2 = (A_{10}+A_{11})B_{00}$
   $M_3 = A_{00}(B_{10}-B_{11})$
   $M_4 = A_{11}(B_{01}-B_{00})$
   $M_5 = (A_{00}+A_{10})B_{11}$
   $M_6 = (A_{11}-A_{00})(B_{00}+B_{10})$
   $M_7 = (A_{01}-A_{11})(B_{11}+B_{01})$
   
   $C_{00} = M_1 + M_4 - M_5 + M_7$
   $C_{01} = M_3 + M_5$
   $C_{10} = M_2 + M_4$
   $C_{11} = M_1 - M_2 + M_3 + M_6$
   
   We need to map this to the global indices.
   
   Global Problem:
   $A$ is $2 \times 4$. Indices $i \in \{0,1\}, j \in \{0,1,2,3\}$.
   $B$ is $4 \times 5$. Indices $j \in \{0,1,2,3\}, k \in \{0,1,2,3,4\}$.
   $C$ is $2 \times 5$. Indices $i \in \{0,1\}, k \in \{0,1,2,3,4\}$.
   
   Decomposition Plan:
   Block 1: $C[:, 0:2] = A[:, :] B[:, 0:2]$.
     Split $j$ into $0,1$ and $2,3$.
     $C[:, 0:2] = A[:, 0:2] B[:, 0:2] + A[:, 2:4] B[:, 0:2]$.
     Term 1.1: $A_{0:2, 0:2} B_{0:2, 0:2}$. ($2 \times 2 \times 2$). Rank 7.
     Term 1.2: $A_{0:2, 2:4} B_{2:4, 0:2}$. ($2 \times 2 \times 2$). Rank 7.
     
   Block 2: $C[:, 2:4] = A[:, :] B[:, 2:4]$.
     Term 2.1: $A_{0:2, 0:2} B_{0:2, 2:4}$. ($2 \times 2 \times 2$). Rank 7.
     Term 2.2: $A_{0:2, 2:4} B_{2:4, 2:4}$. ($2 \times 2 \times 2$). Rank 7.
     
   Block 3: $C[:, 4:5] = A[:, :] B[:, 4:5]$.
     Term 3.1: $A_{0:2, 0:2} B_{0:2, 4:5}$. ($2 \times 2 \times 1$). Rank 2.
     Term 3.2: $A_{0:2, 2:4} B_{2:4, 4:5}$. ($2 \times 2 \times 1$). Rank 2.
     
   Total Rank: $7+7+7+7+2+2 = 32$.
   
   We need to build U, V, W for each term and concatenate.
   
   Helper: Build U, V, W for a $2 \times 2 \times 2$ Strassen term.
   Inputs:
   $A$ slice: rows $i_0, i_1$, cols $j_0, j_1$.
   $B$ slice: rows $j_0, j_1$, cols $k_0, k_1$.
   $C$ slice: rows $i_0, i_1$, cols $k_0, k_1$.
   
   Strassen terms $M_1..M_7$.
   For each $M_r$, we define coefficients for $A, B, C$.
   
   $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$
   $U_{A_{00}} = 1, U_{A_{11}} = 1$.
   $V_{B_{00}} = 1, V_{B_{11}} = 1$.
   $W_{C_{00}} = 1, W_{C_{11}} = 1$.
   
   $M_2 = (A_{10}+A_{11})B_{00}$
   $U_{A_{10}} = 1, U_{A_{11}} = 1$.
   $V_{B_{00}} = 1$.
   $W_{C_{10}} = 1, W_{C_{11}} = -1$.
   
   $M_3 = A_{00}(B_{10}-B_{11})$
   $U_{A_{00}} = 1$.
   $V_{B_{10}} = 1, V_{B_{11}} = -1$.
   $W_{C_{01}} = 1, W_{C_{11}} = 1$.
   
   $M_4 = A_{11}(B_{01}-B_{00})$
   $U_{A_{11}} = 1$.
   $V_{B_{01}} = 1, V_{B_{00}} = -1$.
   $W_{C_{00}} = 1, W_{C_{10}} = 1$.
   
   $M_5 = (A_{00}+A_{10})B_{11}$
   $U_{A_{00}} = 1, U_{A_{10}} = 1$.
   $V_{B_{11}} = 1$.
   $W_{C_{00}} = -1, W_{C_{01}} = 1$.
   
   $M_6 = (A_{11}-A_{00})(B_{00}+B_{10})$
   $U_{A_{11}} = 1, U_{A_{00}} = -1$.
   $V_{B_{00}} = 1, V_{B_{10}} = 1$.
   $W_{C_{11}} = 1$.
   
   $M_7 = (A_{01}-A_{11})(B_{11}+B_{01})$
   $U_{A_{01}} = 1, U_{A_{11}} = -1$.
   $V_{B_{11}} = 1, V_{B_{01}} = 1$.
   $W_{C_{00}} = 1, W_{C_{01}} = -1$.
   
   Wait, I must be careful with the indices in the global matrices.
   
   Helper: Build U, V, W for a $2 \times 2 \times 1$ schoolbook term.
   $A$ slice: $2 \times 2$. $B$ slice: $2 \times 1$. $C$ slice: $2 \times 1$.
   $C_{i,0} = A_{i,0} B_{0,0} + A_{i,1} B_{1,0}$.
   Term 1: $A_{0,0} B_{0,0} \to C_{0,0}$.
   Term 2: $A_{1,0} B_{0,0} \to C_{1,0}$.
   Term 3: $A_{0,1} B_{1,0} \to C_{0,0}$.
   Term 4: $A_{1,1} B_{1,0} \to C_{1,0}$.
   Rank 4.
   
   Let's implement this.
"""

import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict {(i,j): val}
        v_coeffs: dict {(j,k): val}
        w_coeffs: dict {(k,i): val}
        """
        u_row = [0.0] * (n * m)
        v_row = [0.0] * (m * p)
        w_row = [0.0] * (p * n)
        
        for (i, j), val in u_coeffs.items():
            u_row[i * m + j] = val
        for (j, k), val in v_coeffs.items():
            v_row[j * p + k] = val
        for (k, i), val in w_coeffs.items():
            w_row[k * n + i] = val
            
        U_list.append(u_row)
        V_list.append(v_row)
        W_list.append(w_row)
        
    def strassen_2x2x2(i_base, j_base, k_base):
        """
        Add rank-7 Strassen terms for A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2]
        contributing to C[i_base:i_base+2, k_base:k_base+2]
        """
        # M1 = (A00 + A11) * (B00 + B11)
        add_term({(i_base, j_base): 1, (i_base+1, j_base+1): 1},
                 {(j_base, k_base): 1, (j_base+1, k_base+1): 1},
                 {(k_base, i_base): 1, (k_base+1, i_base+1): 1})
                 
        # M2 = (A10 + A11) * B00
        add_term({(i_base+1, j_base): 1, (i_base+1, j_base+1): 1},
                 {(j_base, k_base): 1},
                 {(k_base, i_base+1): 1, (k_base+1, i_base+1): -1})
                 
        # M3 = A00 * (B10 - B11)
        add_term({(i_base, j_base): 1},
                 {(j_base+1, k_base): 1, (j_base+1, k_base+1): -1},
                 {(k_base+1, i_base): 1, (k_base+1, i_base+1): 1})
                 
        # M4 = A11 * (B01 - B00)
        add_term({(i_base+1, j_base+1): 1},
                 {(j_base, k_base+1): 1, (j_base, k_base): -1},
                 {(k_base, i_base): 1, (k_base, i_base+1): 1})
                 
        # M5 = (A00 + A10) * B11
        add_term({(i_base, j_base): 1, (i_base+1, j_base): 1},
                 {(j_base+1, k_base+1): 1},
                 {(k_base, i_base): -1, (k_base+1, i_base): 1})
                 
        # M6 = (A11 - A00) * (B00 + B10)
        add_term({(i_base+1, j_base+1): 1, (i_base, j_base): -1},
                 {(j_base, k_base): 1, (j_base+1, k_base): 1},
                 {(k_base+1, i_base+1): 1})
                 
        # M7 = (A01 - A11) * (B11 + B01)
        add_term({(i_base, j_base+1): 1, (i_base+1, j_base+1): -1},
                 {(j_base+1, k_base+1): 1, (j_base, k_base+1): 1},
                 {(k_base, i_base): 1, (k_base+1, i_base): -1})

    def schoolbook_2x2x1(i_base, j_base, k_base):
        """
        Add rank-4 schoolbook terms for A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+1]
        contributing to C[i_base:i_base+2, k_base:k_base+1]
        """
        # C[i, k] = A[i, j0] B[j0, k] + A[i, j1] B[j1, k]
        # Term 1: A[i0, j0] B[j0, k0] -> C[i0, k0]
        add_term({(i_base, j_base): 1},
                 {(j_base, k_base): 1},
                 {(k_base, i_base): 1})
                 
        # Term 2: A[i1, j0] B[j0, k0] -> C[i1, k0]
        add_term({(i_base+1, j_base): 1},
                 {(j_base, k_base): 1},
                 {(k_base, i_base+1): 1})
                 
        # Term 3: A[i0, j1] B[j1, k0] -> C[i0, k0]
        add_term({(i_base, j_base+1): 1},
                 {(j_base+1, k_base): 1},
                 {(k_base, i_base): 1})
                 
        # Term 4: A[i1, j1] B[j1, k0] -> C[i1, k0]
        add_term({(i_base+1, j_base+1): 1},
                 {(j_base+1, k_base): 1},
                 {(k_base, i_base+1): 1})

    # Block 1: C[:, 0:2]
    # A[:, 0:2] * B[0:2, 0:2]
    strassen_2x2x2(0, 0, 0)
    # A[:, 2:4] * B[2:4, 0:2]
    strassen_2x2x2(0, 2, 0)
    
    # Block 2: C[:, 2:4]
    # A[:, 0:2] * B[0:2, 2:4]
    strassen_2x2x2(0, 0, 2)
    # A[:, 2:4] * B[2:4, 2:4]
    strassen_2x2x2(0, 2, 2)
    
    # Block 3: C[:, 4:5]
    # A[:, 0:2] * B[0:2, 4:5]
    schoolbook_2x2x1(0, 0, 4)
    # A[:, 2:4] * B[2:4, 4:5]
    schoolbook_2x2x1(0, 2, 4)
    
    # Convert to lists of ints if possible
    U_final = []
    for row in U_list:
        U_final.append([int(val) if val == int(val) else val for val in row])
    V_final = []
    for row in V_list:
        V_final.append([int(val) if val == int(val) else val for val in row])
    W_final = []
    for row in W_list:
        W_final.append([int(val) if val == int(val) else val for val in row])
        
    return {"U": U_final, "V": V_final, "W": W_final}

if __name__ == "__main__":
    res = construct()
    rank = len(res["U"])
    print(f"Rank: {rank}")
    # Verification
    n, m, p = 2, 4, 5
    U = np.array(res["U"]).T # (rank, n*m)
    V = np.array(res["V"]).T # (rank, m*p)
    W = np.array(res["W"]).T # (rank, p*n)
    
    # Tensor T[a, b, c] = sum_r U[r, a] V[r, b] W[r, c]
    # We want T to match the matrix multiplication tensor.
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i=i', j=j', k=k'
    
    # Check a few entries
    errors = 0
    for i in range(n):
        for j in range(m):
            for j_prime in range(m):
                for k in range(p):
                    for k_prime in range(p):
                        for i_prime in range(n):
                            if i_prime == i and j_prime == j and k_prime == k:
                                expected = 1.0
                            else:
                                expected = 0.0
                            
                            a = i * m + j
                            b = j_prime * p + k
                            c = k_prime * n + i_prime
                            
                            # Sum over r
                            val = 0.0
                            for r in range(rank):
                                val += U[r, a] * V[r, b] * W[r, c]
                                
                            if abs(val - expected) > 1e-9:
                                errors += 1
                                if errors < 5:
                                    print(f"Error at {i},{j},{j_prime},{k},{k_prime},{i_prime}: got {val}, expected {expected}")
    
    print(f"Errors: {errors}")


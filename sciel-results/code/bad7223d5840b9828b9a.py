import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication.
    For n x m * m x p, the rank is n * m * p.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The decomposition satisfies: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,i'} delta_{j,j'} delta_{k,k'} 
    Wait, the tensor T_{i,j,k}^{i',j',k'} for matrix mult is 1 if i=i', j=j', k=k' is not quite right.
    The matrix multiplication tensor M_{n,m,p} is defined such that:
    (A * B)_{i,k} = sum_j A_{i,j} B_{j,k}
    The bilinear form is sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} ? No.
    The tensor is T(i,j,k) such that the multiplication is <T, A \otimes B \otimes C> ?
    Standard convention:
    We want to compute C = A B.
    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    The tensor decomposition of the multiplication map is given by:
    C_{i,k} = sum_{c=1}^r U_{i,j,c} V_{j,k,c} W_{k,i,c} ? No, the indices don't match.
    
    The standard tensor for matrix multiplication $M_{n,m,p}$ has entries:
    $T_{i,j,k} = 1$ if the indices correspond to a valid multiplication term?
    Actually, the tensor is defined on the space $F^n \otimes F^m \otimes F^p$?
    No, it's a map $F^{n \times m} \times F^{m \times p} \to F^{n \times p}$.
    This is a trilinear form if we include the output.
    $T(A, B, C) = \sum_{i,j,k} C_{i,k} A_{i,j} B_{j,k}$.
    The rank of this tensor is the minimum number of rank-1 tensors needed to sum to T.
    A rank-1 tensor is $u \otimes v \otimes w$ where $u \in F^{n \times m}$, $v \in F^{m \times p}$, $w \in F^{n \times p}$.
    Wait, the indices in the prompt are:
    U: (n*m, rank) -> indexed by (i,j) and c
    V: (m*p, rank) -> indexed by (j,k) and c
    W: (p*n, rank) -> indexed by (k,i) and c
    
    The condition for exact decomposition is:
    $\sum_{c=1}^r U_{i,j,c} V_{j,k,c} W_{k,i,c} = \delta_{i,i'} \delta_{j,j'} \delta_{k,k'}$?
    No, the tensor T is such that $T_{i,j,k} = 1$ if it's a valid term?
    Actually, the multiplication tensor is:
    $T = \sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}$?
    Let's check the schoolbook.
    Schoolbook: $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    This corresponds to terms $A_{i,j} B_{j,k}$.
    The output is $C_{i,k}$.
    So the trilinear form is $\sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}$? No, $C$ is the result.
    The tensor representing the multiplication is $T = \sum_{i=1}^n \sum_{j=1}^m \sum_{k=1}^p E_{i,j} \otimes E_{j,k} \otimes E_{k,i}$?
    Wait, $E_{k,i}$ is the matrix unit with 1 at $(k,i)$.
    If we take the trace of $C A^T B^T$?
    Let's look at the seed code:
    U[i*m+j][column] = 1
    V[j*p+k][column] = 1
    W[k*n+i][column] = 1
    
    This corresponds to the term where index i, j, k are fixed.
    The product $U_{i,j,c} V_{j,k,c} W_{k,i,c}$ for a fixed c (which corresponds to a specific i,j,k) is 1 only when the indices match that specific i,j,k.
    Summing over c, we get 1 if (i,j,k) is the one selected, 0 otherwise.
    This reconstructs the tensor $T = \sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{k,i}$?
    Actually, the standard multiplication tensor is $T_{mult} = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$ in the appropriate bases?
    Yes, the rank is the number of terms in the sum, which is $n \cdot m \cdot p$.
    """
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, leaving 20s margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists
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
                
                # Simple guard, though this loop is very fast
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

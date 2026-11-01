"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x14 matrix.
    
    Strategy: We model the tensor as a direct sum of two independent 2x1x14
    row-vector multiplications. For each j in {0,1}, we need to find a 
    rank-14 decomposition of the 2x1x14 tensor T_{i,j,k} = delta_{i,j} (actually, 
    let's re-examine the tensor structure).
    
    The matrix multiplication tensor T_{i,j,k} for (n,m,p) = (2,2,14) is defined by:
    T_{i,j,k} = 1 if the term appears in the product.
    Wait, the standard tensor decomposition for matrix multiplication is:
    C = A * B
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    So the tensor T_{i,j,k} is 1 if we are computing C_{i,k} using A_{i,j} and B_{j,k}.
    Actually, the tensor is T_{i,j,k} = 1 for all i, j, k? No.
    The tensor is defined such that T_{i,j,k} = 1 if and only if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    In standard matrix multiplication, every term A_{i,j} B_{j,k} contributes to C_{i,k} exactly once for a fixed j.
    So T_{i,j,k} = 1 for all i, j, k?
    No, the tensor indices are (i, j, k) corresponding to A_{i,j}, B_{j,k}, and C_{i,k}.
    The value of the tensor at (i, j, k) is 1 if A_{i,j} * B_{j,k} contributes to C_{i,k}.
    In standard multiplication, it does. So T_{i,j,k} = 1 for all i, j, k?
    That would make it a rank 1 tensor? No.
    
    Let's look at the schoolbook decomposition in the seed program:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][column] = 1
                V[j*p + k][column] = 1
                W[k*n + i][column] = 1
                column += 1
    This sets T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}.
    In the schoolbook case, for each (i,j,k), there is exactly one r where U_{i,j}^{(r)}=1, V_{j,k}^{(r)}=1, W_{k,i}^{(r)}=1.
    So T_{i,j,k} = 1.
    This implies the tensor we are decomposing is the one with T_{i,j,k} = 1 for all i, j, k?
    No, that's not right. The tensor of matrix multiplication is not all 1s.
    
    Let's clarify the tensor definition.
    The tensor T for matrix multiplication of n x m and m x p matrices is defined by:
    T_{i,j,k} = 1 if the entry (i,j) of A, (j,k) of B, and (i,k) of C are related.
    Actually, the standard definition is that the tensor T has components T_{i,j,k} such that:
    <T, A \otimes B \otimes C> = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k}.
    And we want this to be equal to <A B, C> = sum_{i,k} (sum_j A_{i,j} B_{j,k}) C_{i,k} = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}.
    So T_{i,j,k} = 1 for all i, j, k?
    Yes, for standard matrix multiplication, the tensor T is indeed the all-1s tensor of shape (n*m, m*p, p*n)?
    No, the indices are (i, j, k) for A, B, C respectively.
    A has indices (i, j), B has (j, k), C has (i, k).
    So the tensor is of shape (n, m, m, p, n, p)? No, it's a 3-tensor with indices (i, j, k) where i is from A and C, j is from A and B, k is from B and C.
    Wait, the tensor is usually indexed by the three matrices.
    Let's stick to the convention in the seed program.
    U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    The tensor T is a 3-tensor of shape (n*m, m*p, p*n).
    The condition is that for all a, b, c (indices of U, V, W),
    T_{a,b,c} = sum_{r} U_{a,r} V_{b,r} W_{c,r}.
    
    In the schoolbook decomposition, T_{a,b,c} = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k.
    And T_{a,b,c} = 0 otherwise.
    So the tensor is not all 1s. It is the permutation tensor that picks out the valid (i,j,k) triples.
    
    So we need to find a low-rank decomposition of this specific tensor.
    
    The directive suggests splitting the intermediate dimension m into two 1x14 sub-tensors.
    For m=2, we have two blocks: j=0 and j=1.
    For j=0, the tensor slice is T_{i, 0, k} = 1 if a=i*2+0, b=0*14+k, c=k*2+i.
    For j=1, the tensor slice is T_{i, 1, k} = 1 if a=i*2+1, b=1*14+k, c=k*2+i.
    
    These two slices are independent (they have disjoint supports in the a and b indices).
    So the total rank is the sum of the ranks of the two slices.
    
    Each slice is a tensor of shape (2, 14, 28) (since n=2, p=14, and c has 28 values).
    Wait, c = k*n + i, so c has 2*14=28 values.
    But for a fixed j, the slice only involves specific a, b, c values.
    Specifically, for j=0:
    a in {0, 2} (i=0,1)
    b in {0, 1, ..., 13} (k=0..13)
    c in {0, 1, 2, ..., 27} (c=k*2+i)
    
    The tensor slice for j=0 is T^{(0)}_{i,k,c} = 1 if c = k*2 + i, else 0.
    This is a tensor of shape (2, 14, 28).
    We need to find its rank.
    
    Note that c = k*2 + i. So for each (i,k), there is exactly one c.
    This looks like a rank 14 tensor? Or rank 28?
    Let's see. We can write T^{(0)}_{i,k,c} = sum_r U^{(0)}_{i,r} V^{(0)}_{k,r} W^{(0)}_{c,r}.
    We want to minimize the rank r.
    
    Consider the following decomposition:
    For each k in 0..13, we can set:
    U^{(0)}_{0, r=k} = 1
    U^{(0)}_{1, r=k} = 0
    V^{(0)}_{k, r=k} = 1
    W^{(0)}_{2k, r=k} = 1
    And for i=1:
    U^{(0)}_{1, r=k+14} = 1
    U^{(0)}_{0, r=k+14} = 0
    V^{(0)}_{k, r=k+14} = 1
    W^{(0)}_{2k+1, r=k+14} = 1
    
    This uses rank 28 for the j=0 slice.
    Can we do better?
    The tensor T^{(0)}_{i,k,c} is 1 if c = 2k + i.
    This is a linear function of k and i.
    It is a "slice" of a larger tensor.
    The rank of this tensor is known to be 28? Or 14?
    
    If we use Strassen-like tricks, maybe we can reduce the rank.
    However, the directive says "each of which has a minimal rank of 14".
    Let's trust the directive and try to find a rank 14 decomposition for each slice.
    
    How to get rank 14?
    We need to represent T^{(0)}_{i,k,c} = delta_{c, 2k+i} as a sum of 14 rank-1 tensors.
    This seems difficult because the support has 28 non-zero entries (one for each i,k).
    A rank-1 tensor has at most 1 non-zero entry per slice? No.
    
    Wait, maybe the tensor is not the permutation tensor?
    Let's re-read the task description.
    "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix by a 2 by 14 matrix."
    
    Standard matrix multiplication tensor is indeed the one with T_{i,j,k} = 1 for all i,j,k in the appropriate index sets, but with the constraint that the indices must match (j in A and B, i in A and C, k in B and C).
    So the tensor is sparse.
    
    If the directive is correct that the rank is 28, then we should output a rank 28 decomposition.
    If I can't verify the rank 14 per slice, I will use the schoolbook rank 56 as a fallback, or try to implement the rank 28 decomposition if I can construct it.
    
    Let's try to construct the rank 28 decomposition as described in the previous attempt:
    For j=0:
    For each k in 0..13:
        Term 2k: U_{0, 2k}=1, V_{k, 2k}=1, W_{2k, 2k}=1 (i=0, c=2k)
        Term 2k+1: U_{1, 2k+1}=1, V_{k, 2k+1}=1, W_{2k+1, 2k+1}=1 (i=1, c=2k+1)
    For j=1:
    For each k in 0..13:
        Term 28+2k: U_{2, 28+2k}=1, V_{14+k, 28+2k}=1, W_{2k, 28+2k}=1 (i=0, c=2k)
        Term 28+2k+1: U_{3, 28+2k+1}=1, V_{14+k, 28+2k+1}=1, W_{2k+1, 28+2k+1}=1 (i=1, c=2k+1)
    
    This gives rank 56. This is the schoolbook decomposition.
    So the "split" doesn't reduce the rank if we use the same structure.
    
    The directive claims rank 28. This implies that the rank of each 2x1x14 tensor is 14, not 28.
    How is that possible?
    Maybe the tensor is not the permutation tensor?
    Or maybe there is a Strassen-like algorithm for 2x1x14?
    
    Given the uncertainty and the previous rejection, I will output the schoolbook decomposition (rank 56) which is known to be valid.
    """
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

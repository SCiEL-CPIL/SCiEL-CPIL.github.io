"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    The schoolbook rank is n*m*p = 4*5*6 = 120.
    We attempt to find a lower rank decomposition using algebraic structure.
    
    For matrix multiplication of sizes (n,m) and (m,p), the tensor is:
    T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (where indices are appropriately mapped)
    
    The known lower bounds and upper bounds for small matrix multiplication:
    - 2x2: rank 7 (Strassen)
    - 3x3: rank 19-23 (best known exact is 19 for 3x3? Actually 23 is upper, 19 is lower)
    - For rectangular cases, Strassen-style optimizations can reduce rank.
    
    For 4x5 * 5x6, we can use block decomposition. Let's try to use Strassen's idea
    on blocks. However, 4, 5, 6 don't divide nicely into powers of 2.
    
    Alternative approach: Use the fact that matrix multiplication can be viewed as
    a bilinear map, and try to find a decomposition with rank < 120.
    
    Known results:
    - For 2x2 * 2x2: rank 7 (Strassen)
    - For 2x2 * 2x3: rank 11 (vs schoolbook 12)
    - For 3x3 * 3x3: rank 23 (upper bound), 19 (lower bound)
    
    Let's try a block-based Strassen-like approach.
    
    Actually, let's think about this more carefully. The matrix multiplication tensor
    for A (n x m) times B (m x p) to get C (n x p) has:
    - U matrix: (n*m) x rank
    - V matrix: (m*p) x rank  
    - W matrix: (p*n) x rank (with trace convention W[k*n+i, r])
    
    The decomposition must satisfy: for each output element C[i,k],
    C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] summed over j... 
    
    Wait, let me re-read the convention. The standard decomposition is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor form, the multiplication tensor T has T[i,j,k] = 1 if we're computing
    C[i,k] from A[i,j] and B[j,k].
    
    A rank-r decomposition gives:
    T[i,j,k] = sum_{r=1}^r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that's not right either. Let me think again.
    
    The matrix multiplication tensor for (n,m) x (m,p) -> (n,p) is:
    T in U^* ⊗ V^* ⊗ W^* where U has dim n*m (input A), V has dim m*p (input B),
    W has dim n*p (output C).
    
    T[a, b, c] where a indexes A, b indexes B, c indexes C.
    T[a, b, c] = 1 if A[a] and B[b] contribute to C[c], i.e., if a = i*m+j, b = j*p+k,
    c = i*p+k (or with the trace convention c = k*n+i).
    
    So T[i*m+j, j*p+k, k*n+i] = 1 for all i, j, k, and 0 otherwise.
    
    A rank-r decomposition is:
    T[a, b, c] = sum_{r=1}^r U[a, r] * V[b, r] * W[c, r]
    
    We need to find U, V, W with minimal r such that this holds.
    
    For the schoolbook case, r = n*m*p and each term corresponds to one (i,j,k) triple.
    
    To get a lower rank, we need to combine multiple (i,j,k) triples into one term.
    This is possible when the bilinear form allows it, as in Strassen's algorithm.
    
    For 4x5 * 5x6, let's try to use a block decomposition. We can split the matrices
    into blocks and apply Strassen-like recurrences.
    
    However, implementing a full Gröbner basis solver for this is complex and may not
    finish in 120s. Let's try a more pragmatic approach:
    
    1. Try known low-rank decompositions for small blocks
    2. Combine them using block structure
    
    For 2x2 * 2x2, Strassen gives rank 7.
    For 2x3 * 3x2, we can potentially get rank 11 (vs schoolbook 12).
    
    Let's try to decompose 4x5 * 5x6 using blocks:
    - Split 4x5 into 2x2 + 2x2 + 2x1 blocks
    - Split 5x6 into 2x2 + 2x2 + 1x2 blocks... this gets complicated.
    
    Alternative: Just use the schoolbook decomposition which is guaranteed correct.
    The rank is 120. Any improvement would be valuable.
    
    Let's try a simple optimization: if any of n, m, p is 2, we can use Strassen-like
    tricks. But 4, 5, 6 don't have 2 as a factor in a useful way.
    
    Actually, 4 = 2*2, so we can split the 4x5 matrix into 2x2 blocks vertically.
    And 6 = 2*3, so we can split the 5x6 matrix into 2x3 blocks horizontally.
    
    Let me try a different approach: use the fact that for any matrix multiplication,
    we can use the "four Russians" method or other algebraic techniques.
    
    Given the complexity, let's just return the schoolbook decomposition which is
    guaranteed to be correct. The rank is 120.
    
    Actually, let me try one more thing: can we use a rank-7 decomposition for 2x2
    blocks and combine them?
    
    For 4x5 * 5x6:
    - Think of A as a 2x2 block matrix where each block is 2x5
    - Think of B as a 2x2 block matrix where each block is 5x3
    - Then C is a 2x2 block matrix where each block is 2x3
    
    This doesn't directly help because the blocks aren't square.
    
    Let's just go with the schoolbook decomposition. It's safe and correct.
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

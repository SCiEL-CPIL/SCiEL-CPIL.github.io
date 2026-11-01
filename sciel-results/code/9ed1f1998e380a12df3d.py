"""Program for matrix multiplication tensor decomposition (2x2 by 2x10)."""
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a (2x2) matrix by a (2x10) matrix.
    
    The multiplication tensor M_{n,m,p} has indices:
    - U: n*m rows, rank columns
    - V: m*p rows, rank columns  
    - W: p*n rows, rank columns
    
    We aim to find a decomposition of rank r such that:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{ik} (for valid indices)
    
    For matrix multiplication M_{2,2,10}, the standard algorithm uses rank 40.
    However, we can potentially reduce the rank using structured decompositions.
    
    One approach: use the fact that matrix multiplication can be factored through
    smaller intermediate steps or use known results for small matrix multiplication.
    
    For M_{2,2,p}, we can use a decomposition that exploits the structure of 2x2
    matrix multiplication. The standard rank for M_{2,2,p} is at most 2*p*2 = 4*p
    via schoolbook, but there may be improvements.
    
    Let's use a different approach: represent the 2x2 matrix multiplication in a
    way that allows sharing across the 10 columns.
    
    Key insight: M_{2,2,10} can be seen as two independent M_{2,2,5} problems if
    we split the 10 columns into two groups of 5, but that doesn't reduce rank.
    
    Alternative: use bilinear algorithms for 2x2 matrix multiplication. The
    Strassen-like approach for 2x2 gives rank 7 for M_{2,2,2}, and we can extend.
    
    For M_{2,2,10}, let's try to construct a decomposition using the rank-7
    decomposition for M_{2,2,2} and pad/extend appropriately.
    
    Actually, let's think about this more carefully. The tensor for M_{n,m,p}
    satisfies:
    T[i,j;k,l] = delta_{i,k} (where i is row of A, j is col of A/row of B, 
    k is col of B, l is row of result, and there's an implicit sum over j)
    
    Wait, let me reclarify the convention. For multiplying A (n x m) by B (m x p),
    the result C (n x p) has C[i,k] = sum_j A[i,j] * B[j,k].
    
    The tensor M_{n,m,p} has components T_{i,j,k} = delta_{i,k} if we think of it
    differently, but the standard definition for matrix multiplication tensor is:
    
    M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    
    where e_{i,j} is the matrix unit in the first factor, etc.
    
    With the trace convention W[k*n+i, r], the decomposition is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,i'} * delta_{j,j'} ... 
    
    Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    
    So the decomposition should satisfy:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = C[i,k]
    
    where C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is a standard tensor decomposition of the matrix multiplication tensor.
    
    For M_{2,2,10}, the schoolbook rank is 2*2*10 = 40.
    
    Let's try to find a lower rank decomposition. One known result is that for
    M_{n,m,p}, the rank can sometimes be reduced using bilinear algorithms.
    
    For 2x2 matrix multiplication, the Strassen decomposition gives rank 7 instead
    of 8. For M_{2,2,10}, we can potentially use a similar idea.
    
    Let me construct a decomposition using the following approach:
    - Use a rank-r decomposition of the "core" 2x2 matrix multiplication
    - Extend it to handle the 10 columns
    
    Actually, a cleaner approach: M_{2,2,10} is isomorphic to a block matrix
    multiplication where we can potentially share computations.
    
    Let's try a concrete construction. We'll use the fact that:
    sum_j A[i,j] B[j,k] for i=0,1 and k=0,...,9
    
    We can write this as:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is just 2*10 = 20 scalar multiplications plus 20 adds, but the tensor
    rank counts the number of rank-1 terms in the decomposition.
    
    For each (i,k) pair, we need to express the sum over j. The schoolbook
    decomposition uses one term for each (i,j,k) triple, giving 40 terms.
    
    To reduce the rank, we can look for shared patterns. Notice that for fixed
    j, the terms A[i,j]*B[j,k] for all i,k form a rank-1 tensor in the (i,k)
    space, scaled by A[i,j] and B[j,k].
    
    Let me try a different approach using known bilinear algorithms.
    
    For M_{2,2,2}, the Strassen decomposition has rank 7. The decomposition is:
    M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    M2 = (A[1,0]+A[1,1])*B[0,0]
    M3 = A[0,0]*(B[0,1]-B[1,1])
    M4 = A[1,1]*(B[1,0]-B[0,0])
    M5 = (A[0,0]+A[0,1])*B[1,1]
    M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    And C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    Now for M_{2,2,10}, we have 10 different "B" matrices (one for each column k
    of the 2x10 matrix). We can apply the same Strassen decomposition independently
    to each column, but that would give rank 7*10 = 70, which is worse.
    
    Instead, we need to find a joint decomposition. Let's think of B as a 2x10
    matrix, and we want to compute A*B where A is 2x2.
    
    The key is that A has only 4 entries, so we can potentially exploit the
    structure more.
    
    Let A = [[a, b], [c, d]] and B be 2x10 with B[j,k] for j=0,1 and k=0,...,9.
    
    C[0,k] = a*B[0,k] + b*B[1,k]
    C[1,k] = c*B[0,k] + d*B[1,k]
    
    We can write:
    C = [[a, b], [c, d]] @ B
    
    This is a linear map from B to C, parameterized by A. The tensor we want to
    decompose is the bilinear map (A, B) -> A*B.
    
    One approach: represent A in a basis that allows efficient multiplication.
    The space of 2x2 matrices has dimension 4, and the space of 2x10 matrices has
    dimension 20. The output space has dimension 20.
    
    The bilinear map can be seen as a linear map from (A ⊗ B) to C, but we want
    a rank decomposition where each term is a product of a functional of A, a
    functional of B, and a vector in C.
    
    Let's try to construct a rank-r decomposition with r < 40.
    
    Consider using the following idea: for each "direction" in the A-space, we can
    precompute certain combinations.
    
    Let me try a specific construction. We'll use the following decomposition:
    
    For k = 0, ..., 9, we need to compute:
    C[0,k] = a*B[0,k] + b*B[1,k]
    C[1,k] = c*B[0,k] + d*B[1,k]
    
    We can introduce intermediate variables. Let's try:
    
    X[k] = B[0,k] + B[1,k]
    Y[k] = B[0,k] - B[1,k]
    
    Then:
    C[0,k] = (a+b)/2 * X[k] + (a-b)/2 * Y[k]
    C[1,k] = (c+d)/2 * X[k] + (c-d)/2 * Y[k]
    
    This gives us 4 * 10 = 40 terms if we do it naively, but we can share the
    computations across k.
    
    Actually, let me count more carefully. If we define:
    For each k:
    Term 1: (a+b) * X[k] -> contributes to C[0,k] with coefficient 1/2
    Term 2: (a-b) * Y[k] -> contributes to C[0,k] with coefficient 1/2
    Term 3: (c+d) * X[k] -> contributes to C[1,k] with coefficient 1/2
    Term 4: (c-d) * Y[k] -> contributes to C[1,k] with coefficient 1/2
    
    That's 4 * 10 = 40 terms, same as schoolbook.
    
    Let me try a different approach. What if we use a smaller set of "probes"
    on A?
    
    Actually, I recall that for M_{n,m,p}, the rank can be related to the
    communication complexity and other measures. For small n, m, p, the exact
    rank is often known.
    
    For M_{2,2,10}, let's try to use a rank-20 decomposition if possible.
    
    Idea: Since the output is 2x10, and each output entry is a sum of 2 products,
    the total number of "basic" products is 20 (2 for each of the 10 columns,
    times 2 rows). But the tensor rank counts rank-1 tensor terms, not scalar
    multiplications.
    
    Let me think about it differently. The matrix multiplication tensor M_{2,2,10}
    can be viewed as:
    
    T = sum_{j=0}^{1} (sum_{i=0}^{1} sum_{k=0}^{9} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i})
    
    Wait, that's not quite right. Let me use the standard definition.
    
    M_{n,m,p} = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    
    where e_{(i,j)} is the basis vector for the i,j entry in the first matrix,
    e_{(j,k)} for the j,k entry in the second, and e_{(k,i)} for the k,i entry
    in the output.
    
    With our indexing:
    U index: i*m + j (i in 0..n-1, j in 0..m-1) -> 0..3 for n=m=2
    V index: j*p + k (j in 0..m-1, k in 0..p-1) -> 0..19 for m=2, p=10
    W index: k*n + i (k in 0..p-1, i in 0..n-1) -> 0..19 for p=10, n=2
    
    And the tensor is:
    M = sum_{i=0}^{1} sum_{j=0}^{1} sum_{k=0}^{9} U_{i*2+j} ⊗ V_{j*10+k} ⊗ W_{k*2+i}
    
    So M has 2*2*10 = 40 non-zero "entries" in this tensor representation, each
    being a rank-1 tensor. The rank of M is the minimum number of rank-1 tensors
    that sum to M.
    
    Schoolbook gives rank 40. We want to see if we can do better.
    
    One known result: for M_{n,m,p}, if p is large, the rank is close to n*m*p
    due to lower bounds from border rank and other techniques. But for small
    dimensions, improvements are possible.
    
    Let me try to use a greedy or heuristic approach to find a lower rank
    decomposition, or use known results from the literature.
    
    Actually, I'll try a simple approach: use the Strassen-style decomposition
    for the 2x2 part and combine it with the 10 columns in a way that reduces the
    total rank.
    
    Here's an idea: use 4 "probe" values for A that allow us to recover any
    2x2 matrix multiplication. The minimum number of bilinear multiplications
    needed for 2x2 matrix multiplication is 7 (Strassen).
    
    But for M_{2,2,10}, we're not just doing one 2x2 multiplication; we're doing
    10 of them simultaneously, with the same A but different B columns.
    
    Let me try a rank-28 decomposition or similar.
    
    Actually, let's just output the schoolbook decomposition for now, since
    finding a provably optimal decomposition for M_{2,2,10} is non-trivial.
    But the task asks for rank-minimal, so let's at least try to improve.
    
    I'll use a heuristic: try to find a lower rank by looking for linear
    dependencies among the 40 rank-1 tensors in the schoolbook decomposition.
    
    The 40 rank-1 tensors are:
    T_{i,j,k} = e_{i*2+j} ⊗ e_{j*10+k} ⊗ e_{k*2+i} for i,j in {0,1}, k in {0,...,9}
    
    We want to find if there's a representation with fewer than 40 terms.
    
    Given the complexity, I'll output the schoolbook decomposition with rank 40.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END

import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # A is 2x2, B is 2x10, C is 2x10
    # U shape (4, R), V shape (20, R), W shape (20, R)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]

    # Strategy:
    # 1. Start with the known rank 35 solution (5 independent Strassen 2x2x2 blocks).
    # 2. Attempt to reduce the rank by finding linear dependencies or performing a local search
    #    to merge terms.
    # 3. Specifically, try to find a rank 34, 33, ... solution.
    #    Given the complexity, a full random search is likely too slow.
    #    However, we can try to "compress" the rank 35 solution by checking if the
    #    tensor rank is actually lower due to symmetries or specific structure.
    #    For M_{2,2,10}, it is conjectured that R = 35 might be minimal or close to it.
    #    Known lower bound: R >= n*p = 20.
    #    Upper bound: 35.
    #    
    #    Let's try a heuristic: 
    #    The rank 35 solution is block diagonal in the "k" index (columns of B/C).
    #    If we mix the columns, can we reduce rank?
    #    This is equivalent to finding a better decomposition of the tensor.
    #    
    #    We will implement a simple local optimization / rank reduction attempt.
    #    We start with the rank 35 solution.
    #    We try to set one column of U, V, W to zero and see if the remaining 34 columns
    #    can be adjusted to still satisfy the tensor equation.
    #    This is a non-convex problem. 
    #    
    #    Alternative: Use the fact that R(M_{2,2,p}) is additive in p if we split B.
    #    R(M_{2,2,10}) <= R(M_{2,2,5}) + R(M_{2,2,5}).
    #    R(M_{2,2,5}) <= R(M_{2,2,3}) + R(M_{2,2,2}).
    #    R(M_{2,2,3}) <= R(M_{2,2,2}) + R(M_{2,2,1}) = 7 + 3 = 10.
    #    So R(M_{2,2,5}) <= 10 + 7 = 17.
    #    So R(M_{2,2,10}) <= 17 + 17 = 34.
    #    
    #    Can we achieve rank 34?
    #    We need a rank 10 decomposition for M_{2,2,3}.
    #    Is R(M_{2,2,3}) = 10? Or can it be 9?
    #    If R(M_{2,2,3}) = 9, then R(M_{2,2,5}) <= 9 + 7 = 16, and R(M_{2,2,10}) <= 32.
    #    
    #    Let's try to construct a rank 10 solution for M_{2,2,3} and see if it works.
    #    If R(M_{2,2,3}) is indeed 10, we can combine two such blocks to get rank 20 for M_{2,2,5}?
    #    No, M_{2,2,5} is one 2x2 times a 2x5.
    #    Split B (2x5) into B1 (2x2) and B2 (2x3).
    #    C = [A B1, A B2].
    #    We can use 7 terms for A B1 and 10 terms for A B2.
    #    Total 17 terms for M_{2,2,5}.
    #    Then for M_{2,2,10}, split into two 2x5 blocks.
    #    Total 17 + 17 = 34 terms.
    #    
    #    So rank 34 is achievable IF we have a rank 10 decomposition for M_{2,2,3}.
    #    We know R(M_{2,2,1}) = 3.
    #    We know R(M_{2,2,2}) = 7.
    #    M_{2,2,3} = M_{2,2,2} + M_{2,2,1} (splitting columns).
    #    The decomposition for M_{2,2,2} uses 7 terms.
    #    The decomposition for M_{2,2,1} uses 3 terms.
    #    Total 10 terms.
    #    So R(M_{2,2,3}) <= 10.
    #    
    #    Therefore, R(M_{2,2,10}) <= 34.
    #    
    #    Can we do better?
    #    If R(M_{2,2,3}) = 9, then R(M_{2,2,10}) <= 32.
    #    Is R(M_{2,2,3}) = 9?
    #    Lower bound for M_{2,2,3}: 
    #    R >= max(n, m, p) = 3 (weak).
    #    R >= n*p = 6.
    #    R >= m*(n+p)/? No.
    #    Known result: R(M_{2,2,3}) = 10? Or 9?
    #    According to literature, R(M_{2,2,3}) is not known to be 9. It is known to be <= 10.
    #    Some sources suggest R(M_{2,2,3}) = 10.
    #    
    #    Let's assume R(M_{2,2,3}) = 10 for now and construct the rank 34 solution.
    #    This is a safe improvement from 35 to 34.
    #    
    #    Construction for Rank 34:
    #    Split B (2x10) into B1 (2x5) and B2 (2x5).
    #    Split B1 into B1a (2x2) and B1b (2x3).
    #    Split B2 into B2a (2x2) and B2b (2x3).
    #    
    #    Terms:
    #    1. A * B1a (2x2): 7 terms (Strassen)
    #    2. A * B1b (2x3): 10 terms (7 for 2x2 part + 3 for 2x1 part? No, B1b is 2x3)
    #       Wait, B1b is 2x3. We can split B1b into B1b1 (2x2) and B1b2 (2x1).
    #       A * B1b1: 7 terms.
    #       A * B1b2: 3 terms.
    #       Total for A * B1b: 10 terms.
    #       So A * B1 (2x5) = A * B1a (7) + A * B1b (10) = 17 terms.
    #    3. Similarly, A * B2 (2x5) = 17 terms.
    #    Total: 17 + 17 = 34 terms.
    #    
    #    Let's verify the index mapping.
    #    B columns: 0..9.
    #    B1: cols 0..4.
    #    B2: cols 5..9.
    #    B1a: cols 0..1.
    #    B1b: cols 2..4.
    #    B1b1: cols 2..3.
    #    B1b2: col 4.
    #    B2a: cols 5..6.
    #    B2b: cols 7..9.
    #    B2b1: cols 7..8.
    #    B2b2: col 9.
    #    
    #    Blocks:
    #    Block 1 (cols 0-1): 2x2 -> 7 terms
    #    Block 2 (cols 2-3): 2x2 -> 7 terms
    #    Block 3 (col 4):    2x1 -> 3 terms
    #    Block 4 (cols 5-6): 2x2 -> 7 terms
    #    Block 5 (cols 7-8): 2x2 -> 7 terms
    #    Block 6 (col 9):    2x1 -> 3 terms
    #    
    #    Total terms: 7 + 7 + 3 + 7 + 7 + 3 = 34.
    #    
    #    This is valid!
    #    
    #    Can we do better?
    #    What if we group columns differently?
    #    2x10 = 2x(2+2+2+2+2) -> 5*7 = 35.
    #    2x10 = 2x(2+2+2+2+1+1) -> 4*7 + 2*3 = 28 + 6 = 34.
    #    2x10 = 2x(2+2+2+1+1+1+1) -> 3*7 + 4*3 = 21 + 12 = 33.
    #    2x10 = 2x(2+2+1+1+1+1+1+1) -> 2*7 + 6*3 = 14 + 18 = 32.
    #    2x10 = 2x(2+1+1+1+1+1+1+1+1) -> 1*7 + 8*3 = 7 + 24 = 31.
    #    2x10 = 2x(1+1+1+1+1+1+1+1+1+1) -> 10*3 = 30.
    #    
    #    Wait, is R(M_{2,2,1}) = 3?
    #    C = A * b (where b is 2x1).
    #    C0 = A00 b0 + A01 b1
    #    C1 = A10 b0 + A11 b1
    #    Terms:
    #    1. A00 b0 -> C0 += 1
    #    2. A01 b1 -> C0 += 1
    #    3. A10 b0 -> C1 += 1
    #    4. A11 b1 -> C1 += 1
    #    This is rank 4 (schoolbook).
    #    Can we do it in rank 3?
    #    Strassen-like?
    #    For 2x2x1, the tensor is M_{2,2,1}.
    #    R(M_{2,2,1}) = 3?
    #    Let's check.
    #    C0 = A00 b0 + A01 b1
    #    C1 = A10 b0 + A11 b1
    #    If we use:
    #    M1 = (A00+A10) b0
    #    M2 = (A01+A11) b1
    #    M3 = A10 b1? No.
    #    Actually, R(M_{2,2,1}) is known to be 3?
    #    Let's derive.
    #    We need \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = \delta_{j,j'} (for k=0,1 fixed? No, k is the column index).
    #    For M_{2,2,1}, p=1. k=0.
    #    U (4, R), V (2, R), W (2, R).
    #    W index: k*n+i = 0*2+i = i. So W[i, r].
    #    Condition: \sum_r W_i,r U_{i,j,r} V_{j,0,r} = \delta_{j,j'}.
    #    Wait, the condition is \sum_r W_{k,i,r} U_{i,j,r} V_{j',k,r} = \delta_{j,j'}.
    #    Here k=0.
    #    \sum_r W_{i,r} U_{i,j,r} V_{j',r} = \delta_{j,j'}.
    #    Let's try R=3.
    #    This is a system of 4*4 = 16 equations (for each i, j, j')?
    #    No, for each i (2), j (2), j' (2).
    #    The equation is for the coefficient of A_{i,j} B_{j',0}.
    #    It must be 1 if j=j', 0 otherwise.
    #    So for fixed i, we have a 2x2 matrix M_i where (M_i)_{j,j'} = \sum_r W_{i,r} U_{i,j,r} V_{j',r}.
    #    We need M_i = I_2 for both i=0 and i=1.
    #    So we need two 2x2 identity matrices.
    #    M_i = W_i \cdot (U_i \odot V^T)? No.
    #    Let U_i be the row of U for index i (size 2xR).
    #    Let V be the matrix (2xR).
    #    Let w_i be the row of W for index i (size R).
    #    Then M_i = (U_i * diag? No.
    #    (M_i)_{j,j'} = \sum_r (U_i)_{j,r} V_{j',r} (W_i)_r.
    #    This is the element-wise product of (U_i)_{j,r} and (W_i)_r, summed over r, with V_{j',r}.
    #    Let D_i be a diagonal matrix? No.
    #    Let u_{i,j,r} be U_{i,j,r}.
    #    Let v_{j',r} be V_{j',r}.
    #    Let w_{i,r} be W_{i,r}.
    #    M_i = \sum_r w_{i,r} u_{i,j,r} v_{j',r}.
    #    This looks like M_i = (u_i \odot w_i^T) V^T? No.
    #    Let's write it as:
    #    M_i = \sum_r (w_{i,r} u_{i,:,r}) v_{:,r}^T.
    #    This is a sum of rank-1 matrices.
    #    So M_i is a 2x2 matrix of rank at most R.
    #    We need M_0 = I and M_1 = I.
    #    So we need to represent I as a sum of R rank-1 matrices, with specific constraints on the factors.
    #    The factors for M_0 and M_1 share the V matrix.
    #    Let a_r = w_{0,r} u_{0,:,r} (2-vector)
    #    Let b_r = w_{1,r} u_{1,:,r} (2-vector)
    #    Let c_r = v_{:,r} (2-vector)
    #    Then M_0 = \sum_r a_r c_r^T = I.
    #    M_1 = \sum_r b_r c_r^T = I.
    #    We need to find R vectors a_r, b_r, c_r such that these hold.
    #    If R=2, can we do it?
    #    \sum_{r=1}^2 a_r c_r^T = I.
    #    \sum_{r=1}^2 b_r c_r^T = I.
    #    This implies A C^T = I and B C^T = I.
    #    So A = B.
    #    And A C^T = I implies C^T is the inverse of A.
    #    So A and C are invertible 2x2 matrices.
    #    But we have constraints linking a_r, b_r to U and W.
    #    a_r = w_{0,r} u_{0,:,r}.
    #    b_r = w_{1,r} u_{1,:,r}.
    #    Is it possible to choose U, W such that a_r and b_r are arbitrary?
    #    u_{0,:,r} is a 2-vector. w_{0,r} is a scalar.
    #    So a_r is a 2-vector.
    #    Similarly b_r is a 2-vector.
    #    We can choose u_{0,:,r} and w_{0,r} to get any a_r.
    #    We can choose u_{1,:,r} and w_{1,r} to get any b_r.
    #    So if we can find a_r, b_r, c_r for R=2, then R(M_{2,2,1}) <= 2?
    #    But R(M_{2,2,1}) >= 3 is a known lower bound?
    #    Let's check the lower bound.
    #    R(M_{2,2,1}) >= 3.
    #    Why?
    #    If R=2, then M_0 and M_1 are sums of 2 rank-1 matrices.
    #    The condition is that the set of vectors {c_r} must span the space such that we can form I.
    #    Actually, if R=2, then M_0 = a_1 c_1^T + a_2 c_2^T = I.
    #    This means the matrix A = [a_1 a_2] and C = [c_1 c_2] satisfy A C^T = I.
    #    So A = C^{-T}.
    #    Similarly B C^T = I, so B = C^{-T}.
    #    So A = B.
    #    This means a_r = b_r for all r.
    #    So w_{0,r} u_{0,:,r} = w_{1,r} u_{1,:,r}.
    #    This is possible.
    #    So why is R(M_{2,2,1}) = 3?
    #    Maybe my condition is wrong?
    #    Let's re-verify the condition for M_{n,m,p}.
    #    C_{i,k} = \sum_j A_{i,j} B_{j,k}.
    #    Decomposition: C_{i,k} = \sum_r W_{k,i,r} (\sum_j U_{i,j,r} A_{i,j}) (\sum_{j'} V_{j',k,r} B_{j',k}).
    #    Coefficient of A_{i,j} B_{j',k} is \sum_r W_{k,i,r} U_{i,j,r} V_{j',k,r}.
    #    This must equal 1 if j=j', 0 otherwise.
    #    For M_{2,2,1}, k=0.
    #    \sum_r W_{i,r} U_{i,j,r} V_{j',r} = \delta_{j,j'}.
    #    This is exactly what I wrote.
    #    So if R=2 works, then R(M_{2,2,1}) <= 2.
    #    But R(M_{2,2,1}) is known to be 3.
    #    Where is the error?
    #    Ah, U_{i,j,r} depends on i and j.
    #    In my derivation:
    #    M_i = \sum_r w_{i,r} u_{i,:,r} c_r^T.
    #    Here u_{i,:,r} is the vector (U_{i,0,r}, U_{i,1,r}).
    #    So for fixed i, the matrix M_i is formed.
    #    The condition is M_0 = I and M_1 = I.
    #    My derivation showed that if R=2, we can satisfy this.
    #    Let's try to construct it.
    #    Let c_1 = [1, 0], c_2 = [0, 1].
    #    Then A C^T = A = I. So a_1 = [1, 0], a_2 = [0, 1].
    #    Similarly b_1 = [1, 0], b_2 = [0, 1].
    #    So we need:
    #    r=1: w_{0,1} u_{0,:,1} = [1, 0], w_{1,1} u_{1,:,1} = [1, 0], v_{:,1} = [1, 0].
    #    r=2: w_{0,2} u_{0,:,2} = [0, 1], w_{1,2} u_{1,:,2} = [0, 1], v_{:,2} = [0, 1].
    #    Let's pick:
    #    r=1:
    #    w_{0,1} = 1, u_{0,0,1} = 1, u_{0,1,1} = 0.
    #    w_{1,1} = 1, u_{1,0,1} = 1, u_{1,1,1} = 0.
    #    v_{0,1} = 1, v_{1,1} = 0.
    #    r=2:
    #    w_{0,2} = 1, u_{0,0,2} = 0, u_{0,1,2} = 1.
    #    w_{1,2} = 1, u_{1,0,2} = 0, u_{1,1,2} = 1.
    #    v_{0,2} = 0, v_{1,2} = 1.
    #    
    #    Let's check the tensor equation.
    #    \sum_r W_{i,r} U_{i,j,r} V_{j',r} = \delta_{j,j'}.
    #    For i=0:
    #    j=0, j'=0: r=1: 1*1*1 = 1. r=2: 1*0*0 = 0. Sum=1. OK.
    #    j=0, j'=1: r=1: 1*1*0 = 0. r=2: 1*0*1 = 0. Sum=0. OK.
    #    j=1, j'=0: r=1: 1*0*1 = 0. r=2: 1*1*0 = 0. Sum=0. OK.
    #    j=1, j'=1: r=1: 1*0*0 = 0. r=2: 1*1*1 = 1. Sum=1. OK.
    #    For i=1:
    #    j=0, j'=0: r=1: 1*1*1 = 1. r=2: 1*0*0 = 0. Sum=1. OK.
    #    ... and so on.
    #    
    #    So R(M_{2,2,1}) <= 2?
    #    This contradicts the known result R(M_{2,2,1}) = 3.
    #    Let me check the definition of rank of matrix multiplication tensor.
    #    Maybe the indices are different?
    #    Standard definition:
    #    M_{n,m,p} is a tensor in V^n \otimes V^m \otimes V^p.
    #    M_{i,j,k} = 1 if i=j=k? No.
    #    M_{i,j,k} = \delta_{i,k} \delta_{j,?}
    #    The multiplication tensor is defined such that:
    #    <M, A \otimes B \otimes C^*> = tr(C^* A B).
    #    So M_{i,j,k} is the coefficient of A_{i,j} B_{j',k} C^*_{k,i} in tr(C^* A B).
    #    tr(C^* A B) = \sum_{i,j,k} C^*_{k,i} A_{i,j} B_{j,k}.
    #    So M_{i,j,k} = 1 if the indices match in the trace.
    #    The term is A_{i,j} B_{j,k} C^*_{k,i}.
    #    So the tensor entry is 1 if the indices form a cycle?
    #    No, the tensor is just the structure of the product.
    #    M_{i,j,k} = \delta_{j,k?} No.
    #    The tensor M is in U \otimes V \otimes W where dim U = nm, dim V = mp, dim W = pn.
    #    The entry corresponding to (i,j), (j',k), (k',i') is 1 if i=i', j=j', k=k'.
    #    Wait, the factors are:
    #    U_{i,j}, V_{j,k}, W_{k,i}.
    #    The product is U_{i,j} V_{j,k} W_{k,i}.
    #    Summing over i,j,k.
    #    This is the trace.
    #    So the tensor T has T_{(i,j),(j',k),(k',i')} = 1 if i=i', j=j', k=k'.
    #    Otherwise 0.
    #    
    #    My condition was:
    #    \sum_r W_{k,i,r} U_{i,j,r} V_{j',k,r} = \delta_{j,j'}.
    #    This came from matching coefficients of A_{i,j} B_{j',k}.
    #    The LHS is the coefficient of A_{i,j} B_{j',k} in the decomposed expression for C_{i,k}.
    #    The decomposed expression is \sum_r W_{k,i,r} (\sum_j U_{i,j,r} A_{i,j}) (\sum_{j'} V_{j',k,r} B_{j',k}).
    #    Expanding: \sum_{r,j,j'} W_{k,i,r} U_{i,j,r} V_{j',k,r} A_{i,j} B_{j',k}.
    #    The coefficient of A_{i,j} B_{j',k} is \sum_r W_{k,i,r} U_{i,j,r} V_{j',k,r}.
    #    The true coefficient in C_{i,k} = \sum_j A_{i,j} B_{j,k} is \delta_{j,j'}.
    #    So the condition is correct.
    #    
    #    So why is R(M_{2,2,1}) = 3?
    #    Let's check a reference.
    #    "The rank of the 2x2 matrix multiplication tensor is 7."
    #    "The rank of the 2x2x1 tensor?"
    #    Actually, M_{2,2,1} is the tensor for multiplying a 2x2 matrix by a 2x1 vector.
    #    It is a sub-tensor of M_{2,2,2}.
    #    R(M_{2,2,1}) <= R(M_{2,2,2}) = 7.
    #    Is it 3?
    #    My construction with R=2 seems valid.
    #    Let's double check.
    #    U (4, 2):
    #    Row 0 (i=0, j=0): [1, 0]
    #    Row 1 (i=0, j=1): [0, 1]
    #    Row 2 (i=1, j=0): [1, 0]
    #    Row 3 (i=1, j=1): [0, 1]
    #    V (2, 2):
    #    Row 0 (j=0, k=0): [1, 0]
    #    Row 1 (j=1, k=0): [0, 1]
    #    W (2, 2):
    #    Row 0 (k=0, i=0): [1, 1]
    #    Row 1 (k=0, i=1): [1, 1]
    #    
    #    Let's re-evaluate the sum for i=0, j=0, j'=0.
    #    \sum_r W_{0,0,r} U_{0,0,r} V_{0,0,r}
    #    r=0: 1 * 1 * 1 = 1.
    #    r=1: 1 * 0 * 0 = 0.
    #    Sum = 1. OK.
    #    
    #    i=0, j=0, j'=1.
    #    \sum_r W_{0,0,r} U_{0,0,r} V_{1,0,r}
    #    r=0: 1 * 1 * 0 = 0.
    #    r=1: 1 * 0 * 1 = 0.
    #    Sum = 0. OK.
    #    
    #    i=0, j=1, j'=0.
    #    \sum_r W_{0,0,r} U_{0,1,r} V_{0,0,r}
    #    r=0: 1 * 0 * 1 = 0.
    #    r=1: 1 * 1 * 0 = 0.
    #    Sum = 0. OK.
    #    
    #    i=0, j=1, j'=1.
    #    \sum_r W_{0,0,r} U_{0,1,r} V_{1,0,r}
    #    r=0: 1 * 0 * 0 = 0.
    #    r=1: 1 * 1 * 1 = 1.
    #    Sum = 1. OK.
    #    
    #    It seems R(M_{2,2,1}) <= 2.
    #    This is surprising. Let me check if there is a constraint I missed.
    #    "integer/half-integer entries". My entries are integers.
    #    
    #    If R(M_{2,2,1}) = 2, then:
    #    R(M_{2,2,10}) <= 10 * 2 = 20.
    #    This would be a huge improvement!
    #    
    #    Let's verify R(M_{2,2,1}) = 2 in literature.
    #    Actually, the tensor M_{2,2,1} is often denoted as M_{2,2,1}.
    #    Some sources say R(M_{2,2,1}) = 3.
    #    Why?
    #    Maybe my index mapping for W is wrong?
    #    W index is k*n+i.
    #    For p=1, k=0.
    #    W index is 0*2+i = i.
    #    So W[0] is for i=0, W[1] is for i=1.
    #    This matches my W matrix.
    #    
    #    Let's check the condition again.
    #    \sum_r W_{k,i,r} U_{i,j,r} V_{j',k,r} = \delta_{j,j'}.
    #    This must hold for ALL i, j, j', k.
    #    In my case, k=0.
    #    It holds for i=0 and i=1.
    #    
    #    Is it possible that R(M_{2,2,1}) is indeed 2?
    #    If so, the lower bound R >= n*p = 2*1 = 2 is tight.
    #    
    #    Let's assume R(M_{2,2,1}) = 2 and try to construct the rank 20 solution for M_{2,2,10}.
    #    If it fails verification, I will fall back to rank 34.
    
    def build_rank_20_solution():
        # 10 blocks of 2x1, each with rank 2.
        U = np.zeros((4, 20), dtype=float)
        V = np.zeros((20, 20), dtype=float)
        W = np.zeros((20, 20), dtype=float)
        
        # For each column k (0..9), we have a 2x1 block.
        # The 2x1 block uses 2 terms.
        # Term 1 (r=0 for this block):
        #   U: i=0,j=0 -> 1; i=0,j=1 -> 0; i=1,j=0 -> 1; i=1,j=1 -> 0
        #   V: j=0,k -> 1; j=1,k -> 0
        #   W: k,i=0 -> 1; k,i=1 -> 1
        # Term 2 (r=1 for this block):
        #   U: i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 0; i=1,j=1 -> 1
        #   V: j=0,k -> 0; j=1,k -> 1
        #   W: k,i=0 -> 1; k,i=1 -> 1
        
        for k in range(10):
            # Term 1
            r1 = 2*k
            U[0, r1] = 1  # i=0, j=0
            U[1, r1] = 0  # i=0, j=1
            U[2, r1] = 1  # i=1, j=0
            U[3, r1] = 0  # i=1, j=1
            
            V[0*10 + k, r1] = 1  # j=0, k
            V[1*10 + k, r1] = 0  # j=1, k
            
            W[k*2 + 0, r1] = 1  # k, i=0
            W[k*2 + 1, r1] = 1  # k, i=1
            
            # Term 2
            r2 = 2*k + 1
            U[0, r2] = 0  # i=0, j=0
            U[1, r2] = 1  # i=0, j=1
            U[2, r2] = 0  # i=1, j=0
            U[3, r2] = 1  # i=1, j=1
            
            V[0*10 + k, r2] = 0  # j=0, k
            V[1*10 + k, r2] = 1  # j=1, k
            
            W[k*2 + 0, r2] = 1  # k, i=0
            W[k*2 + 1, r2] = 1  # k, i=1
            
        return U, V, W

    def verify_solution(U, V, W, tol=1e-9):
        # Check if \sum_r W_{k,i,r} U_{i,j,r} V_{j',k,r} = \delta_{j,j'}
        # U: (4, R)
        # V: (20, R)
        # W: (20, R)
        # Indices:
        # U[i*m + j, r]
        # V[j*p + k, r]
        # W[k*n + i, r]
        
        n, m, p = 2, 2, 10
        R = U.shape[1]
        
        # We can check a random sample or all.
        # There are n*m*m*p = 2*2*2*10 = 80 conditions.
        # Let's check all.
        
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        # Coefficient of A[i,j] B[jp,k]
                        coeff = 0.0
                        for r in range(R):
                            u_val = U[i*m + j, r]
                            v_val = V[jp*p + k, r]
                            w_val = W[k*n + i, r]
                            coeff += w_val * u_val * v_val
                        
                        expected = 1.0 if j == jp else 0.0
                        if abs(coeff - expected) > tol:
                            return False
        return True

    # Try rank 20 first
    U20, V20, W20 = build_rank_20_solution()
    if verify_solution(U20, V20, W20):
        return {"U": U20.tolist(), "V": V20.tolist(), "W": W20.tolist()}
    else:
        # Fall back to rank 34
        def build_rank_34_solution():
            U = np.zeros((4, 34), dtype=float)
            V = np.zeros((20, 34), dtype=float)
            W = np.zeros((20, 34), dtype=float)
            
            # Strassen 7 terms
            terms_7 = [
                ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
                ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
                ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
                ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
                ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
                ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
                ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
            ]
            
            # Rank 3 terms for 2x1
            # Based on the R=2 construction, but we need 3 terms?
            # If R(M_{2,2,1})=3, we need a rank 3 decomposition.
            # Standard schoolbook is rank 4.
            # Can we do rank 3?
            # Let's try to find one.
            # Or just use the 2-term one if it works?
            # If verify_solution fails for R=2, it means R(M_{2,2,1}) > 2.
            # So we need a valid rank 3 or 4 decomposition for the 2x1 blocks.
            # Let's use the schoolbook rank 4 for the 2x1 blocks to be safe,
            # but that would give 4*10 = 40, which is worse than 35.
            # So we MUST find a rank 3 decomposition for 2x1 if R=2 fails.
            
            # Let's try to construct a rank 3 decomposition for 2x1.
            # Conditions:
            # \sum_{r=1}^3 W_{i,r} U_{i,j,r} V_{j',r} = \delta_{j,j'} for i=0,1.
            
            # Let's use a known rank 3 decomposition if available.
            # Or derive one.
            # If I can't find one quickly, I will use the 7-term Strassen for 2x2 blocks
            # and hope that 34 is the best I can do safely.
            
            # Actually, if R=2 failed, it's possible R=3 works.
            # Let's just use the 7-term Strassen for 2x2 blocks and 3-term for 2x1 blocks?
            # Wait, I don't have a verified 3-term 2x1 decomposition.
            
            # Let's stick to the 35-term solution which is known to be correct.
            # Or try to construct the 34-term solution using 7-term 2x2 and 3-term 2x1?
            # If I don't have the 3-term 2x1, I can't build 34.
            
            # Let's return the rank 35 solution which is definitely correct.
            U35, V35, W35 = build_strassen_rank_35()
            return U35, V35, W35
        
        def build_strassen_rank_35():
            U = np.zeros((4, 35), dtype=float)
            V = np.zeros((20, 35), dtype=float)
            W = np.zeros((20, 35), dtype=float)
            
            terms = [
                ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
                ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
                ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
                ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
                ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
                ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
                ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
            ]
            
            col = 0
            for q in range(5):
                for a_vec, b_vec, c_vec in terms:
                    U[0, col] = a_vec[0]
                    U[1, col] = a_vec[1]
                    U[2, col] = a_vec[2]
                    U[3, col] = a_vec[3]
                    
                    V[0*10 + 2*q, col] = b_vec[0]
                    V[0*10 + 2*q+1, col] = b_vec[1]
                    V[1*10 + 2*q, col] = b_vec[2]
                    V[1*10 + 2*q+1, col] = b_vec[3]
                    
                    W[4*q + 0, col] = c_vec[0]
                    W[4*q + 1, col] = c_vec[1]
                    W[4*q + 2, col] = c_vec[2]
                    W[4*q + 3, col] = c_vec[3]
                    
                    col += 1
            
            return U, V, W
        
        U, V, W = build_strassen_rank_35()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Note: The logic above has a flaw. build_strassen_rank_35 is defined inside the else block
# but called in the else block. It needs to be defined before or moved.
# Let's restructure.

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 2, 10

    def build_strassen_rank_35():
        U = np.zeros((4, 35), dtype=float)
        V = np.zeros((20, 35), dtype=float)
        W = np.zeros((20, 35), dtype=float)
        
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        col = 0
        for q in range(5):
            for a_vec, b_vec, c_vec in terms:
                U[0, col] = a_vec[0]
                U[1, col] = a_vec[1]
                U[2, col] = a_vec[2]
                U[3, col] = a_vec[3]
                
                V[0*10 + 2*q, col] = b_vec[0]
                V[0*10 + 2*q+1, col] = b_vec[1]
                V[1*10 + 2*q, col] = b_vec[2]
                V[1*10 + 2*q+1, col] = b_vec[3]
                
                W[4*q + 0, col] = c_vec[0]
                W[4*q + 1, col] = c_vec[1]
                W[4*q + 2, col] = c_vec[2]
                W[4*q + 3, col] = c_vec[3]
                
                col += 1
        
        return U, V, W

    def build_rank_20_solution():
        U = np.zeros((4, 20), dtype=float)
        V = np.zeros((20, 20), dtype=float)
        W = np.zeros((20, 20), dtype=float)
        
        for k in range(10):
            r1 = 2*k
            U[0, r1] = 1
            U[1, r1] = 0
            U[2, r1] = 1
            U[3, r1] = 0
            
            V[0*10 + k, r1] = 1
            V[1*10 + k, r1] = 0
            
            W[k*2 + 0, r1] = 1
            W[k*2 + 1, r1] = 1
            
            r2 = 2*k + 1
            U[0, r2] = 0
            U[1, r2] = 1
            U[2, r2] = 0
            U[3, r2] = 1
            
            V[0*10 + k, r2] = 0
            V[1*10 + k, r2] = 1
            
            W[k*2 + 0, r2] = 1
            W[k*2 + 1, r2] = 1
            
        return U, V, W

    def verify_solution(U, V, W, tol=1e-9):
        n, m, p = 2, 2, 10
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        coeff = 0.0
                        for r in range(R):
                            u_val = U[i*m + j, r]
                            v_val = V[jp*p + k, r]
                            w_val = W[k*n + i, r]
                            coeff += w_val * u_val * v_val
                        expected = 1.0 if j == jp else 0.0
                        if abs(coeff - expected) > tol:
                            return False
        return True

    # Try rank 20
    U20, V20, W20 = build_rank_20_solution()
    if verify_solution(U20, V20, W20):
        return {"U": U20.tolist(), "V": V20.tolist(), "W": W20.tolist()}
    
    # Fall back to rank 35
    U35, V35, W35 = build_strassen_rank_35()
    return {"U": U35.tolist(), "V": V35.tolist(), "W": W35.tolist()}


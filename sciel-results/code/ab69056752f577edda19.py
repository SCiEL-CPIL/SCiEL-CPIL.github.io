"""Matrix multiplication 3x4x5 - explicit rank-minimal decomposition attempt."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_expected_tensor(n, m, p):
    """Build the expected matmul tensor T[x,y,z] where:
    x = i*m+j (A index), y = j*p+k (B index), z = k*n+i (C index, column-major)
    T[x,y,z] = 1 if x=i*m+j, y=j*p+k, z=k*n+i for some consistent (i,j,k), else 0.
    """
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _compute_decomposed_tensor(U, V, W):
    """Compute D[x,y,z] = sum_r U[x,r] * V[y,r] * W[z,r]"""
    return np.einsum('xr,yr,zr->xyz', U, V, W)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matmul tensor."""
    expected = _build_expected_tensor(n, m, p)
    D = _compute_decomposed_tensor(U, V, W)
    return np.allclose(D, expected, atol=tol)

def _construct_rank_decomposition(n, m, p):
    """
    Construct a decomposition for M_{n,m,p}.
    
    Strategy: Split the p dimension into groups to leverage known low-rank decompositions.
    For 3x4x5:
      - Split k=0..3 into a 3x4x4 block.
      - Split k=4 into a 3x4x1 block.
    
    For 3x4x4:
      - Use the fact that 4x4x4 has rank 16 (Strassen-like or just schoolbook is 64, but we can do better).
      - Actually, the schoolbook rank for 4x4x4 is 64. But we can use a specific structure.
      - Wait, the rank of 4x4x4 is known to be 16? No, it's 16 for the bilinear complexity? 
      - Actually, the bilinear complexity of 4x4 matrix multiplication is 16 (Strassen's algorithm generalization or specific construction).
      - Let's use a known rank-16 decomposition for 4x4x4.
      - Or simpler: Just use schoolbook for the sub-blocks if we don't have a specific low-rank one handy, 
        but the task asks for rank-minimal.
        Schoolbook for 3x4x4 is 3*4*4 = 48.
        Schoolbook for 3x4x1 is 3*4*1 = 12.
        Total = 60. This is the same as the original.
        
    We need to find a better decomposition.
    
    Let's look at the structure of M_{3,4,5}.
    T[x,y,z] = 1 if x=i*4+j, y=j*5+k, z=k*3+i.
    
    We can try to find a rank < 60 decomposition.
    Known results:
    M_{3,3,3} rank <= 23 (Strassen 7, but 3x3 is 23?). Actually 3x3x3 rank is 23 (Landsberg).
    M_{4,4,4} rank <= 49? No, 4x4x4 rank is 16? No.
    The rank of M_{n,n,n} is at most n^3.
    For 4x4x4, the rank is 16? No, that's for 2x2x2.
    Strassen's algorithm gives rank 7 for 2x2x2.
    For 4x4x4, we can use Strassen's algorithm recursively?
    4x4 can be split into 2x2 blocks.
    Rank(4x4x4) <= 7 * Rank(2x2x2)? No, the recursion is on the size.
    If we split 4x4 into four 2x2 blocks, we need 7 multiplications of 2x2 matrices.
    Each 2x2 multiplication has rank 7.
    So Rank(4x4x4) <= 7 * 7 = 49?
    Actually, the standard Strassen recursion for 2^k x 2^k matrices gives rank 7^k.
    For 4x4 (k=2), rank is 7^2 = 49.
    This is better than 64.
    
    So for M_{3,4,4}, we can use the 4x4x4 decomposition (rank 49) and project it to 3x4x4.
    How to project?
    The 4x4x4 tensor has indices i,j,k in 0..3.
    We want i in 0..2, j in 0..3, k in 0..3.
    We can just take the 4x4x4 decomposition and zero out the rows of U corresponding to i=3, and the rows of W corresponding to i=3 (since z=k*4+i, wait, indices change).
    
    Let's be precise.
    M_{4,4,4}: T4[x,y,z] where x=i*4+j, y=j*4+k, z=k*4+i, i,j,k in 0..3.
    M_{3,4,4}: T3[x,y,z] where x=i*4+j, y=j*4+k, z=k*3+i, i in 0..2, j in 0..3, k in 0..3.
    
    The indices z are different! k*4+i vs k*3+i.
    So we cannot directly reuse the factors. We need to reconstruct the decomposition for the specific tensor M_{3,4,4}.
    
    However, we can use the same *algorithmic* structure.
    The Strassen decomposition for 4x4x4 involves 49 terms.
    Each term r has U4[x,r], V4[y,r], W4[z,r].
    For M_{3,4,4}, we need U3[x,r], V3[y,r], W3[z,r] such that sum U3[x,r]V3[y,r]W3[z,r] = T3[x,y,z].
    
    If we take the 49 terms from 4x4x4, they satisfy:
    sum_r U4[x,r] V4[y,r] W4[z4,r] = T4[x,y,z4]
    where z4 = k*4+i.
    We need z = k*3+i.
    This is a different tensor. The factors W4 are indexed by z4.
    We cannot just relabel.
    
    So we must find a decomposition for M_{3,4,4} directly.
    
    Let's try a different split.
    M_{3,4,5}.
    Split j (the middle dimension) instead?
    j=0,1,2,3.
    If we split j into two groups of 2: j=0,1 and j=2,3.
    Then we have two blocks:
    Block 1: i in 0..2, j in 0..1, k in 0..4. (M_{3,2,5})
    Block 2: i in 0..2, j in 2..3, k in 0..4. (M_{3,2,5})
    
    Rank(M_{3,2,5})?
    Schoolbook: 3*2*5 = 30.
    Total rank = 30 + 30 = 60. No gain.
    
    What if we use Strassen for the 2x2 blocks?
    M_{3,2,5} can be seen as multiplying a 3x2 matrix by a 2x5 matrix.
    This is not a square matrix multiplication, so Strassen's 2x2x2 doesn't apply directly to the 2x2 block of the middle dimension.
    
    Let's go back to the 4x4x4 idea.
    Is there a known low-rank decomposition for M_{3,4,4}?
    Maybe we can just use the schoolbook for the whole thing and try to optimize with a solver?
    The current program does random search which is unlikely to find rank < 60.
    
    Let's check if rank 59 or lower is possible.
    Lower bound for M_{n,m,p} is max(n*m, m*p, p*n).
    For 3,4,5: max(12, 20, 15) = 20.
    So rank could be as low as 20.
    Schoolbook is 60.
    There is a large gap.
    
    However, finding a specific decomposition is hard.
    The prompt says "Find a rank-minimal exact tensor decomposition".
    If we can't find a better one, we should return the schoolbook (rank 60).
    But maybe we can find a slightly better one.
    
    Let's try to construct a decomposition of rank 59?
    Or maybe use the fact that 3x4x5 can be embedded in 4x4x4?
    No, the dimensions are different.
    
    Let's try a simple optimization:
    Start with schoolbook rank 60.
    Try to merge terms?
    
    Actually, there is a known result for M_{3,4,4}.
    Rank(M_{3,4,4}) <= 3 * 4 * 4 = 48.
    Rank(M_{3,4,1}) <= 3 * 4 * 1 = 12.
    Total 60.
    
    What if we use a rank-16 decomposition for the 4x4 part of M_{3,4,4}?
    If we could decompose M_{4,4,4} with rank 16, then M_{3,4,4} would have rank <= 16 (by projection).
    But rank(4x4x4) is not 16. It is 49 (via Strassen).
    So M_{3,4,4} rank <= 49.
    M_{3,4,1} rank <= 12.
    Total rank <= 49 + 12 = 61. This is worse than 60!
    
    So the Strassen decomposition for 4x4x4 is not helpful if we add the 12 for the last column.
    We need a decomposition of M_{3,4,4} with rank <= 48 to beat 60.
    Schoolbook is 48. So we need to improve the 3x4x4 part to rank < 48.
    
    Is there a known decomposition for M_{3,4,4} with rank < 48?
    Maybe using the structure of 3x4?
    
    Let's try to use the rank-7 decomposition for 2x2x2 to reduce the rank of M_{3,4,4}.
    Split i=0,1,2 and j=0,1,2,3 and k=0,1,2,3.
    This is complex.
    
    Given the time constraint and the difficulty of finding a provably lower rank decomposition without a solver,
    the safest bet is to return the schoolbook decomposition (rank 60).
    However, the task asks for "rank-minimal".
    If we can't find a lower one, 60 is the best we can guarantee.
    
    BUT, the current program already returns 60.
    To improve, we MUST find a rank < 60.
    
    Let's try a random search with a better initialization or a solver.
    The current program uses random search with 50 trials and 2s limit. This is very unlikely to find rank 59.
    
    Maybe we can use `scipy.optimize` to find a rank-59 decomposition?
    We have 60*3*59 variables? No, U is 12x59, V is 20x59, W is 15x59.
    Total variables: 12*59 + 20*59 + 15*59 = (12+20+15)*59 = 47*59 = 2773 variables.
    Constraints: T[x,y,z] = 0 or 1.
    This is a non-convex problem.
    
    Let's try a different approach.
    What if we use the rank-23 decomposition for 3x3x3?
    M_{3,4,5} contains M_{3,3,3} as a sub-tensor?
    i=0,1,2; j=0,1,2; k=0,1,2.
    If we use a rank-23 decomposition for the 3x3x3 part, and schoolbook for the rest?
    The 3x3x3 part involves i,j,k in 0..2.
    The rest involves j=3 or k=3,4.
    This is not a simple direct sum because the indices are interleaved.
    
    Let's stick to the schoolbook decomposition but try to optimize the constants?
    No, the rank is the number of terms.
    
    Given the difficulty, I will keep the schoolbook decomposition but add a check for a potential rank-59 decomposition using a simple heuristic?
    No, that's unlikely to work.
    
    I will output the schoolbook decomposition. It is correct and safe.
    The score will be 60.
    
    Wait, the score is `seed/value`. If the seed program has score 1.0, and value is 60, then seed is 60.
    If I find rank 59, score is 60/59 = 1.016.
    
    Is there ANY way to get rank < 60?
    Yes, if there exists a decomposition.
    But finding it is the hard part.
    
    I will implement the schoolbook decomposition clearly and efficiently.
    
    """
    n, m, p = 3, 4, 5
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    U, V, W = _construct_rank_decomposition(n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to explicit loop (should be identical)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

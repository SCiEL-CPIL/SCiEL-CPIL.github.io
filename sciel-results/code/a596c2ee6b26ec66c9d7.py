"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook decomposition of rank n*m*p for C = A*B."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _strassen_2x2_decomposition():
    """
    Strassen's algorithm for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) such that
    for each (i,j,k) in {0,1}^3, sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta[i,j,k]
    where delta is 1 when the matrix product entry matches.
    
    The tensor T_{i,j,k} for 2x2 matrix mult: (C)_{i,k} = sum_j A_{i,j} B_{j,k}
    So T_{i,j,k} = 1 for all i,j,k in {0,1}.
    """
    # Strassen's 7 products
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    #
    # T_{i,j,k} = 1 for all i,j,k in {0,1} (it's the all-ones 2x2x2 tensor)
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    # Using Strassen's construction:
    # For each Strassen product p_r, we have U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: p1 = (a+d)(e+h)
    # U[0,0]=1 (a), U[1,0]=1 (d), U[2,0]=0 (c not used in p1... wait)
    # Actually, p1 involves a and d from A, e and h from B
    # U[i,j] for p1: a is (0,0), d is (1,1)
    # V[j,k] for p1: e is (0,0), h is (1,1)
    # W[k,i] for p1: contributes to C[0,0] and C[1,1]
    # This is getting complicated. Let me use a known exact decomposition.
    
    # The 2x2x2 tensor of rank 7 (Strassen):
    # I'll use the standard formulation.
    # Let me define it directly.
    
    # Actually, let me just verify: the all-ones 2x2x2 tensor has rank 7 over reals.
    # A known rank-7 decomposition:
    # Using the Strassen scheme, the factor matrices are:
    
    # U (A factors), indexed by (i,j) where i,j in {0,1}:
    # r: 0  1  2  3  4  5  6
    # (0,0): 1  0  1  0  1  1  0
    # (0,1): 0  0  0  0  1  0  1
    # (1,0): 0  1  0  0  0  0  0
    # (1,1): 1  1  0  1  0  0  0
    
    U[0, 0] = 1; U[0, 1] = 0; U[0, 2] = 1; U[0, 3] = 0; U[0, 4] = 1; U[0, 5] = 1; U[0, 6] = 0
    U[1, 0] = 0; U[1, 1] = 0; U[1, 2] = 0; U[1, 3] = 0; U[1, 4] = 1; U[1, 5] = 0; U[1, 6] = 1
    U[2, 0] = 0; U[2, 1] = 1; U[2, 2] = 0; U[2, 3] = 0; U[2, 4] = 0; U[2, 5] = 0; U[2, 6] = 0
    U[3, 0] = 1; U[3, 1] = 1; U[3, 2] = 0; U[3, 3] = 1; U[3, 4] = 0; U[3, 5] = 0; U[3, 6] = 0
    
    # V (B factors), indexed by (j,k):
    # r: 0  1  2  3  4  5  6
    # (0,0): 1  1  0  1  0  1  1
    # (0,1): 1  0  0  0  0  1  1
    # (1,0): 0  0  1  0  1  0  0
    # (1,1): 1  0  1  0  1  0  0
    
    V[0, 0] = 1; V[0, 1] = 1; V[0, 2] = 0; V[0, 3] = 1; V[0, 4] = 0; V[0, 5] = 1; V[0, 6] = 1
    V[1, 0] = 1; V[1, 1] = 0; V[1, 2] = 0; V[1, 3] = 0; V[1, 4] = 0; V[1, 5] = 1; V[1, 6] = 1
    V[2, 0] = 0; V[2, 1] = 0; V[2, 2] = 1; V[2, 3] = 0; V[2, 4] = 1; V[2, 5] = 0; V[2, 6] = 0
    V[3, 0] = 1; V[3, 1] = 0; V[3, 2] = 1; V[3, 3] = 0; V[3, 4] = 1; V[3, 5] = 0; V[3, 6] = 0
    
    # W (C factors), indexed by (k,i):
    # r: 0  1  2  3  4  5  6
    # (0,0): 1  0  0  0  0  0  1
    # (0,1): 0  0  1  0  1  0  0
    # (1,0): 0  1  0  0  0  1  0
    # (1,1): 1  0  0  1  0  0  0
    
    W[0, 0] = 1; W[0, 1] = 0; W[0, 2] = 0; W[0, 3] = 0; W[0, 4] = 0; W[0, 5] = 0; W[0, 6] = 1
    W[1, 0] = 0; W[1, 1] = 0; W[1, 2] = 1; W[1, 3] = 0; W[1, 4] = 1; W[1, 5] = 0; W[1, 6] = 0
    W[2, 0] = 0; W[2, 1] = 1; W[2, 2] = 0; W[2, 3] = 0; W[2, 4] = 0; W[2, 5] = 1; W[2, 6] = 0
    W[3, 0] = 1; W[3, 1] = 0; W[3, 2] = 0; W[3, 3] = 1; W[3, 4] = 0; W[3, 5] = 0; W[3, 6] = 0
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # For each (i,j,k), compute sum over r
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False, (i, j, k, s)
    return True, None


def _combine_decompositions(decomp_list, n, m, p):
    """
    Combine multiple decompositions by block-diagonal stacking.
    Each decomposition is for a sub-problem.
    This is a simplified approach - for now, just return schoolbook.
    """
    # For a proper block decomposition, we'd need to handle the tensor structure carefully.
    # For 2x5x6, we can split the 5 dimension into 2+3.
    # But the tensor coupling makes simple block combination non-trivial.
    # Let's just return the best single decomposition we can construct.
    raise NotImplementedError("Block combination not fully implemented")


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 60)
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_decomp = (U_sb.tolist(), V_sb.tolist(), W_sb.tolist())
    
    # Check time
    if time.time() > deadline:
        return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}
    
    # Try Strassen-based approaches
    # For 2x2x2, Strassen gives rank 7 vs schoolbook 8.
    # For 2x5x6, we can try to use a mixed approach.
    
    # One approach: split 5 = 2 + 3
    # Use Strassen for the 2x2 part (rows 0-1, cols 0-1 of the 5-dim)
    # Use schoolbook for the rest
    
    # Actually, let me try a different strategy:
    # The tensor T for 2x5x6 has T[i,j,k]=1 for all i in {0,1}, j in {0..4}, k in {0..5}
    # This is the all-ones tensor of shape (2,5,6).
    # Its rank is known to be at most 30 (from bilinear complexity bounds) but exact value unclear.
    
    # Let me try to construct a rank-reduced decomposition by exploiting structure.
    
    # Approach: For fixed i (i=0 or 1), the slice T[i,:,:] is a 5x6 all-ones matrix.
    # The rank of a 5x6 all-ones matrix is 1.
    # But the tensor rank couples all three indices.
    
    # Let me try a greedy/heuristic approach:
    # Start with rank 60, try to eliminate some terms by finding linear dependencies.
    
    # Actually, let me try a concrete construction:
    # For the 2x2x6 sub-tensor (using columns 0-1 of the middle dimension):
    # We can use Strassen-like methods.
    # For 2x2x6: schoolbook is 24, can we do better?
    # The 2x2x2 part can be done in rank 7 (Strassen).
    # The remaining 2x2x4 part... 
    
    # Let me just try to verify the Strassen 2x2x2 first.
    U_s, V_s, W_s = _strassen_2x2_decomposition()
    valid, info = _verify_decomposition(U_s, V_s, W_s, 2, 2, 2)
    if not valid:
        # Strassen construction might be wrong; fall back to schoolbook
        pass
    else:
        # Strassen works for 2x2x2. Now try to extend.
        # For 2x5x6, we can think of it as:
        # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
        # = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{4} A[i,j] * B[j,k]
        # The first sum is a 2x2x6 tensor, the second is 2x3x6.
        # This doesn't directly give a tensor decomposition of the whole thing
        # because the factors need to be consistent across the split.
        
        pass
    
    # Given the complexity, let me try a simpler improvement:
    # Use the fact that for the all-ones tensor, there might be structure.
    # 
    # Actually, a known result: the rank of the 2 x m x p all-ones tensor
    # is at most 2*m + 2*p - 2 (from a construction).
    # For m=5, p=6: 2*5 + 2*6 - 2 = 10 + 12 - 2 = 20.
    # Wait, that seems too good. Let me think again.
    # 
    # The matrix multiplication tensor for C = A*B where A is 2xm, B is mxp
    # has T[i,j,k] = 1 for all i,j,k.
    # A known upper bound is min(m*p, n*m*p, ...) but better constructions exist.
    # 
    # For the all-ones tensor of format (n,m,p), the tensor rank is at most
    # n*m + m*p + n*p - n - m - p + 1? No, that's not right either.
    # 
    # Let me just try a concrete construction with rank < 60.
    # 
    # Construction idea:
    # For each i in {0,1}, define U vectors that depend on i.
    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    
    # Let me try rank 30: for each (j,k) pair, have one term, and use W to handle i.
    # If we set W[k*n+i, r] to separate the i dimension:
    # Let r index (j,k) pairs, so rank = m*p = 30.
    # U[i*m+j, r=(j',k')] = delta[i,0] * 1 + ... 
    # This doesn't work simply.
    
    # Alternative: Let me just return the schoolbook if I can't find a better one quickly.
    # But let me try one more thing: a rank-40 construction.
    
    # For now, return schoolbook as fallback.
    if time.time() > deadline - 5:
        return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}
    
    # Try to find a better decomposition using a simple optimization
    # This is a placeholder - in practice, tensor decomposition is hard.
    
    return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}

# EVOLVE-BLOCK-END

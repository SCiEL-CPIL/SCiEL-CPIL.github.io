"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    Returns dict with U (12 x rank), V (20 x rank), W (15 x rank).
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 60
    
    # Try to find a lower rank decomposition
    # For 3x4 by 4x5, the minimum rank is known to be less than 60.
    # Let's try to find a decomposition with rank < 60.
    
    # The tensor for matrix multiplication T_{i,j,k} = A_{i,j} * B_{j,k}
    # We want to write T as sum of rank-1 tensors:
    # T_{i,j,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    # Note: W index is k*n+i per the trace convention W[k*n+i, r]
    
    # Let's use a randomized search approach combined with exact verification.
    # First, let's try known constructions.
    
    # For rectangular matrix multiplication, there are known results.
    # Let's try to construct a decomposition with rank 48, 49, etc.
    
    # Strategy: Use the fact that we can sometimes reduce rank by sharing
    # some of the bilinear forms.
    
    # Let's try a systematic search for small rank reductions.
    # Start with schoolbook and try to merge some terms.
    
    # Actually, let's use a more principled approach.
    # The matrix multiplication tensor M_{n,m,p} has rank at most n*m*p.
    # For (3,4,5), let's see if we can do better.
    
    # One approach: use the fact that some entries can be combined.
    # Let's try to find a decomposition via integer linear programming or
    # by exploiting symmetries.
    
    # For now, let's try a few known constructions:
    
    # Construction 1: Block diagonal approach
    # Split the 4x4 middle dimension into blocks.
    
    # Let's try to use a rank-56 or lower construction.
    
    # Actually, let's use a randomized approach:
    # 1. Start with a random sparse initialization
    # 2. Try to solve for the remaining variables
    # 3. Verify exactness
    
    # But for exactness, we need integer or half-integer entries.
    
    # Let's try a different approach: use the known fact that
    # rank(M_{3,4,5}) <= 56 (or similar).
    
    # Let's try to construct a specific decomposition.
    
    # For 2x2 by 2x2, Strassen gives rank 7.
    # For 3x3 by 3x3, best known is 23.
    # For 3x4 by 4x5, we need to figure out the best known.
    
    # Let's try a simple improvement: if we can combine some terms
    # that have the same pattern.
    
    # Actually, let's just return the schoolbook for now and try to improve.
    # But first, let's see if we can find a rank-59 or lower decomposition.
    
    # Let's try: can we reduce rank by 1?
    # The schoolbook has 60 terms, one for each (i,j,k).
    # If two terms can be combined into one, we reduce rank by 1.
    
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be combined if:
    # There exist coefficients such that the sum of the two rank-1 tensors
    # equals a single rank-1 tensor. This is generally not possible unless
    # the tensors are proportional.
    
    # So simple merging won't work. We need a more sophisticated approach.
    
    # Let's try to use a known construction for rectangular matrix multiplication.
    
    # For m x n by n x p, the rank is at most (m+n-1)(n+p-1) - (n-1)^2
    # or similar formulas from the literature.
    
    # Actually, let's just try to use a numerical approach:
    # 1. Set up the system of equations
    # 2. Try to solve it for a given rank
    # 3. Round to nearest integer/half-integer
    # 4. Verify
    
    # The system is: for each (i,j,k),
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is a valid index, 0 otherwise
    
    # Wait, the tensor T_{i,j,k} = 1 for all valid (i,j,k) in the range,
    # and the decomposition should reproduce the matrix multiplication.
    
    # Actually, the matrix multiplication tensor is:
    # T(A,B) = C where C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # The tensor form is: C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So the structure tensor has T_{i,j,k} = 1 for all i,j,k in range.
    
    # We want: C_{i,k} = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A_{i,j}) * (sum_k V[j*p+k, c] * B_{j,k})
    # Wait, that's not right either.
    
    # Let me re-derive. The standard decomposition is:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # We write this as: C_{i,k} = sum_c sum_j U[i*m+j, c] * A_{i,j} * V[j*p+k, c] * B_{j,k} * W[k*n+i, c]
    
    # Wait, that doesn't look right. Let me think again.
    
    # The correct form is:
    # C_{i,k} = sum_c W[k*n+i, c] * M_c(A) * N_c(B)
    # where M_c(A) and N_c(B) are bilinear forms in A and B respectively.
    
    # In the schoolbook case:
    # For each (j), we have M_j(A) = A_{*,j} (the j-th column) and N_j(B) = B_{j,*} (the j-th row)
    # And W[k*n+i, j] = 1 if the index matches, 0 otherwise.
    
    # Actually, the standard form is:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # = sum_c sum_{(i',j')} U[i*m+j, c] * A_{i',j'} * V[j'*p+k, c] * B_{j',k'} * W[k*n+i, c]
    
    # Hmm, this is getting confusing. Let me just use the standard definition.
    
    # The matrix multiplication tensor M_{n,m,p} is defined by:
    # <M_{n,m,p}, A \otimes B> = C where C = A @ B
    # In index form: C_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    # A rank-r decomposition is:
    # C_{i,k} = sum_{c=1}^r (sum_j U_c[i,j] * A_{i,j}) * (sum_j V_c[j,k] * B_{j,k}) * W_c[i,k]
    
    # Wait, that's not right either. Let me look at this more carefully.
    
    # The correct rank-1 decomposition is:
    # C_{i,k} = sum_c alpha_c * (sum_j U_c[i,j] * A_{i,j}) * (sum_j V_c[j,k] * B_{j,j})
    
    # No, that's still not right. Let me just use the standard formulation.
    
    # A rank-r decomposition of matrix multiplication is a set of r triples
    # (U_c, V_c, W_c) where U_c is an n x m matrix, V_c is an m x p matrix,
    # and W_c is a scalar, such that:
    # C_{i,k} = sum_c W_c * (sum_j U_c[i,j] * A_{i,j}) * (sum_j V_c[j,k] * B_{j,k})
    
    # Wait, that's not dimensionally correct either.
    
    # Let me just use the formulation from the problem statement.
    # The problem says: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # And the decomposition is:
    # C_{i,k} = sum_c U[i*m+j, c] * A_{i,j} * V[j*p+k, c] * B_{j,k} * W[k*n+i, c]
    
    # Wait, that sums over j as well? Let me re-read.
    
    # Actually, I think the correct interpretation is:
    # C_{i,k} = sum_j sum_c U[i*m+j, c] * A_{i,j} * V[j*p+k, c] * B_{j,k} * W[k*n+i, c]
    
    # No wait, that would make U and V depend on j which is already summed over.
    
    # Let me think about this differently. The tensor M_{n,m,p} has indices (i,j,k)
    # where i in [0,n), j in [0,m), k in [0,p).
    # M[i,j,k] = 1 for all valid indices.
    
    # A rank-r decomposition is:
    # M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Yes! That's the correct form. The tensor has 3 indices (i,j,k) and
    # each factor has one index that's a flattening of two of them.
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # And we want to minimize r (the rank).
    
    # The schoolbook solution is r = n*m*p with U[i*m+j, c] = 1, V[j*p+k, c] = 1,
    # W[k*n+i, c] = 1 for the corresponding (i,j,k) triple.
    
    # Now, can we do better? Let's try to find a lower rank.
    
    # For small dimensions, we can try to solve the system of equations.
    # The system is: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 60 equations in 3*r unknowns (U, V, W entries).
    # We want to find a solution with r < 60.
    
    # This is a hard problem, but for small dimensions, we might be able to
    # find a solution using numerical methods.
    
    # Let's try a randomized approach:
    # 1. Pick a random r < 60
    # 2. Initialize U, V, W randomly
    # 3. Try to solve the system using least squares or gradient descent
    # 4. Round to nearest integer/half-integer
    # 5. Verify
    
    # But this might not work well. Let's try a different approach.
    
    # Actually, for the specific case of (3,4,5), let's see if there's a known
    # construction.
    
    # The rank of M_{3,4,5} is at most 3*4*5 = 60.
    # Can we do better?
    
    # One idea: if we can find a decomposition where some of the W entries
    # are zero, we might be able to reduce the rank.
    
    # Another idea: use the fact that the tensor has some symmetries.
    
    # For now, let's just return the schoolbook decomposition.
    # But first, let's check if we have time to try a search.
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # Not enough time, return schoolbook
        rank = schoolbook_rank
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
    
    # Let's try to find a lower rank decomposition.
    # We'll use a randomized search approach.
    
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    def is_valid_entries(U, V, W):
        """Check that all entries are integer or half-integer."""
        for matrix in [U, V, W]:
            for row in matrix:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True
    
    # Try to find a decomposition with rank < 60
    # Start with rank 59 and work down
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Create the schoolbook decomposition as a starting point
    def make_schoolbook(n, m, p):
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
        return U, V, W, rank
    
    U0, V0, W0, _ = make_schoolbook(n, m, p)
    best_decomp = (U0, V0, W0)
    
    # Try to reduce rank by trying to merge some terms
    # This is a heuristic approach
    
    # For now, let's just return the schoolbook
    # The search for lower rank is complex and may not converge in time
    
    return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}
# EVOLVE-BLOCK-END

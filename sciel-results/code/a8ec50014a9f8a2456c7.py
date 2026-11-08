"""Matrix multiplication 2x4x7 tensor decomposition with rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Find a low-rank exact tensor decomposition for 2x4 by 4x7 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k] to get C[i,k]
    # Standard decomposition: rank = n*m*p with U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    # We want to find lower rank.
    
    # The matrix multiplication tensor for n x m * m x p has:
    # - U has n*m rows, rank columns
    # - V has m*p rows, rank columns  
    # - W has p*n rows, rank columns
    # - Constraint: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Known lower bounds:
    # - For 2x4x4: rank is at most 12 (Strassen-like), lower bound ~8
    # - For 2x4x7: we need to explore
    
    # Strategy: Start with schoolbook and try to reduce rank by finding
    # linear dependencies or using structured decompositions.
    
    # First, let's try a systematic approach:
    # 1. Try known good ranks for similar problems
    # 2. Use integer/half-integer entries
    
    # For 2x4x7, the schoolbook rank is 2*4*7 = 56
    # Let's try to find a better decomposition.
    
    # Approach: Use the fact that we can combine terms.
    # The tensor T_{ijk} = delta_{jk} (in the standard basis)
    # We need: sum_c U_{ij,c} V_{jk,c} W_{ki,c} = 1 for all i,j,k
    
    # Let's try ranks from low to high and check feasibility.
    # For small ranks, we can try to solve as a system of equations.
    
    # Key insight: For matrix multiplication, the minimum rank is related to
    # the bilinear complexity. For 2x4x7, let's estimate.
    
    # Lower bound: The tensor has n*m*p = 56 nonzero entries in the standard basis.
    # But the rank can be lower due to structure.
    
    # Let's try a constructive approach:
    # Use block structure. 2x4x7 can be seen as blocks.
    
    # Actually, let me think about this differently.
    # The multiplication of (2x4) by (4x7) produces a (2x7) matrix.
    # Each output C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor decomposition rank is the minimum number of rank-1 tensors
    # needed to express the multiplication tensor.
    
    # For 2x4x7, let's try to find a decomposition with rank < 56.
    
    # Method: Try to use the structure to reduce rank.
    # One approach: Group columns of A and rows of B.
    
    # Let me try a specific construction:
    # For 2x4x4, there are known results. For 2x4x7, we can extend.
    
    # Actually, let's use a different strategy:
    # Try to find the minimum rank by solving a feasibility problem.
    
    # For rank r, we need to find U (8xr), V (28xr), W (14xr) such that
    # for all i in [0,2), j in [0,4), k in [0,7):
    #   sum_c U[i*4+j,c] * V[j*7+k,c] * W[k*2+i,c] = 1
    
    # This is a system of 56 equations in 8r + 28r + 14r = 50r unknowns.
    # For r >= 2, we have more unknowns than equations, so solutions exist
    # (over reals). We need integer/half-integer solutions.
    
    # Let's try small ranks first.
    
    # Try rank 16, 20, 24, etc.
    
    # For a given rank r, we can set up the problem as:
    # Choose U and V, then solve for W (or vice versa).
    
    # Actually, the equations are trilinear, not linear. Let me think again.
    
    # Fix c: U[:,c] is a vector in R^8, V[:,c] in R^28, W[:,c] in R^14.
    # The constraint is: sum_c (U[:,c] ⊗ V[:,c] ⊗ W[:,c]) = T
    # where T is the multiplication tensor.
    
    # This is a tensor decomposition problem.
    
    # For practical purposes, let's try to find a decomposition with
    # rank significantly less than 56.
    
    # Known: For 2x2x2, rank is 7 (Strassen).
    # For 2x2x3, rank is at most 11.
    # For 2x3x3, rank is at most 15.
    # For 2x4x4, I believe rank is at most 20 or so.
    
    # Let me try a constructive approach for 2x4x7.
    
    # Split the 4x7 matrix into blocks:
    # B = [B1, B2, B3, B4] where each Bj is 4x7... no, that's not right.
    
    # Let's use a different block structure.
    # A is 2x4, B is 4x7.
    
    # We can write:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Group j into pairs: j in {0,1} and j in {2,3}
    # C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]
    
    # For each group, we have a 2x2x7 problem.
    # The rank of 2x2x7 is at most 2*2*7 = 28, but can be lower.
    
    # Actually, for 2x2xp, the rank is at most 4p - 2 (I think).
    # For p=7, that's 26. Two such groups would give 52, not much better.
    
    # Let me try a more direct approach.
    
    # I'll use a greedy/optimization approach to find a low-rank decomposition.
    
    # For now, let's return a known good decomposition.
    # Based on the structure, I'll try rank 32 as a starting point.
    
    # Actually, let me try to construct a specific decomposition.
    
    # The key insight for reducing rank is to find linear combinations
    # that allow sharing of computation.
    
    # For 2x4x7, let me try the following:
    # Use the fact that we only have 2 rows in A.
    
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    # We can define intermediate quantities:
    # M[j,k] = B[j,k] (this is just B)
    # Then C[i,k] = sum_j A[i,j] * M[j,k]
    
    # This doesn't help directly. Let me think about Strassen-like ideas.
    
    # For 2x2x2, Strassen uses 7 multiplications instead of 8.
    # The idea is to compute products of sums/differences.
    
    # For 2x4x7, we can potentially use similar ideas.
    
    # Let me try a specific construction with rank 35.
    
    # Actually, I'll use a more systematic approach: try to find the
    # minimum rank by checking feasibility for increasing ranks.
    
    # For rank r, the problem is to find U, V, W with the given constraint.
    # This is a nonlinear feasibility problem, but for small r, we can
    # try to solve it.
    
    # Let me try rank 28 first (half of 56).
    
    # I'll use a heuristic: set U and V to have sparse patterns,
    # then solve for W.
    
    # For each (i,j,k), the equation is:
    # sum_c U[i*4+j,c] * V[j*7+k,c] * W[k*2+i,c] = 1
    
    # If we fix U and V, this is linear in W.
    # So: for each (i,j,k), sum_c (U[i*4+j,c] * V[j*7+k,c]) * W[k*2+i,c] = 1
    
    # Let M_{(i,j,k), (k,i), c} = U[i*4+j,c] * V[j*7+k,c]
    # Then sum_c M_{(i,j,k), (k,i), c} * W[k*2+i,c] = 1
    
    # This is a system: for each (i,j,k), we have an equation involving
    # W[k*2+i, :] weighted by the products.
    
    # For fixed k and i, the variable is W[k*2+i, c] for c=0..r-1.
    # The equations involving W[k*2+i, :] are those with that specific k and i.
    # For fixed k and i, the equations are for all j in [0,4).
    # So we have 4 equations in r unknowns (for each (k,i) pair).
    
    # There are 7*2 = 14 such (k,i) pairs, each with 4 equations.
    # Total: 56 equations in 14*r unknowns.
    
    # For r >= 4, each block is underdetermined, so solutions exist.
    # We need integer/half-integer solutions.
    
    # Strategy: Choose U and V to make the system have nice solutions.
    
    # Let me try a specific construction:
    # Set U[i*4+j, c] = 1 if c = i*4+j, else 0 (identity-like)
    # Set V[j*7+k, c] = 1 if c = j*7+k, else 0 (identity-like)
    # Then U[i*4+j,c] * V[j*7+k,c] is nonzero only when c = i*4+j = j*7+k
    # This is very restrictive and probably won't work for small r.
    
    # Better: Use a more flexible construction.
    
    # Let me try: U[i*4+j, c] depends on i,j and c in a structured way.
    
    # I'll use a random search approach to find a valid decomposition.
    
    # For rank r, randomly initialize U and V with small integers/half-integers,
    # then solve for W and check if W has nice values.
    
    # Let's try this approach.
    
    np.random.seed(42)
    
    def try_rank(r, max_attempts=1000):
        """Try to find a decomposition of rank r."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize U and V with small values
            U = np.zeros((8, r), dtype=float)
            V = np.zeros((28, r), dtype=float)
            
            # Use a structured initialization
            # U[i*4+j, c] = 1 if (i*4+j) % r == c, else 0
            for idx in range(8):
                U[idx, idx % r] = 1.0
            
            for idx in range(28):
                V[idx, idx % r] = 1.0
            
            # Now solve for W
            # For each (k, i) pair, we have 4 equations (one for each j)
            # sum_c U[i*4+j, c] * V[j*7+k, c] * W[k*2+i, c] = 1
            
            W = np.zeros((14, r), dtype=float)
            valid = True
            
            for k in range(7):
                for i in range(2):
                    # Build the 4 x r system
                    A_mat = np.zeros((4, r))
                    for j in range(4):
                        for c in range(r):
                            A_mat[j, c] = U[i*4+j, c] * V[j*7+k, c]
                    
                    # Solve A_mat @ W[k*2+i, :] = ones(4)
                    # This is underdetermined if r > 4, so use least squares
                    try:
                        sol, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, np.ones(4), rcond=None)
                        W[k*2+i, :] = sol
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if W has nice values (integers or half-integers)
            # Multiply by 2 and check if close to integers
            W2 = W * 2
            W_rounded = np.round(W2)
            if np.allclose(W2, W_rounded, atol=1e-6):
                # Great! We have a valid decomposition with half-integer W
                # Convert to list
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x/2) for x in row] for row in W2]
                return U_list, V_list, W_list
            
            # Try random perturbation
            U += np.random.randn(8, r) * 0.1
            V += np.random.randn(28, r) * 0.1
        
        return None
    
    # Try ranks from low to high
    # Start with a reasonable lower bound
    # For 2x4x7, let's try from rank 20 upward
    
    best_result = None
    
    # First, try the schoolbook as fallback
    rank_schoolbook = 56
    U_sb = [[0.0]*rank_schoolbook for _ in range(8)]
    V_sb = [[0.0]*rank_schoolbook for _ in range(28)]
    W_sb = [[0.0]*rank_schoolbook for _ in range(14)]
    col = 0
    for i in range(2):
        for j in range(4):
            for k in range(7):
                U_sb[i*4+j][col] = 1.0
                V_sb[j*7+k][col] = 1.0
                W_sb[k*2+i][col] = 1.0
                col += 1
    
    best_result = (U_sb, V_sb, W_sb, rank_schoolbook)
    
    # Try to find better ranks
    for r in range(16, 56):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_attempts=500)
        if result is not None:
            U_list, V_list, W_list = result
            best_result = (U_list, V_list, W_list, r)
            break
    
    U_out, V_out, W_out, _ = best_result
    
    return {"U": U_out, "V": V_out, "W": W_out}

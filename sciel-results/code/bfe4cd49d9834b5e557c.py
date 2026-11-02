"""Matrix multiplication 5,5,5 - attempt to find lower rank than schoolbook (125)."""

import numpy as np
import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Build the full multiplication tensor T[i,j,k] which is 1 if (i,j,k) is a valid
    # multiplication term. In the standard form, T is a 3D tensor of shape (n, m, p)
    # where T[i,j,k] = 1 for all valid (i,j,k) triples.
    # Actually, the multiplication tensor is defined as:
    # M(A,B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) with T[i,j,k] = 1.
    # We want to decompose: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k] for all i,j,k
    # Wait, let me re-read the convention.
    # U has shape (n*m, rank), indexed by (i*m+j, r)
    # V has shape (m*p, rank), indexed by (j*p+k, r)  
    # W has shape (p*n, rank), indexed by (k*n+i, r)
    # The decomposition says: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 (the multiplication tensor for 5x5x5).
    
    # The schoolbook solution has rank 125 with U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1.
    # This gives sum_r = 1 for the matching (i,j,k) term.
    
    # Known best rank for 5x5 matrix multiplication is 100 (Laderman-like or similar constructions).
    # Actually, the best known for 5x5 is 100, proven by various authors.
    # Let me try to find a rank < 125 decomposition using optimization.
    
    # Strategy: Try to find a decomposition of rank R where R < 125.
    # The constraint is: for all (i,j,k) in [0,5)^3:
    #   sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And we want integer or half-integer entries.
    
    # This is a highly nonlinear constraint satisfaction problem.
    # Let me try a different approach: use the known structure.
    
    # For matrix multiplication, there's a known technique using bilinear forms.
    # The multiplication tensor M_{n,m,p} can be decomposed using the "Laderman" construction
    # or other algebraic methods.
    
    # Actually, let me try a practical approach: 
    # 1. Start with the schoolbook rank-125 solution
    # 2. Try to find linear dependencies among the 125 rank-1 terms
    # 3. If we find that some combination of terms sums to zero, we can reduce the rank
    
    # The 125 rank-1 terms are: for each (i,j,k), term (i,j,k) has:
    #   U[i*m+j] = e_{i*m+j} (standard basis vector)
    #   V[j*p+k] = e_{j*p+k}
    #   W[k*n+i] = e_{k*n+i}
    
    # These are all distinct rank-1 tensors. The question is whether any linear combination
    # of these 125 rank-1 tensors sums to zero (in the tensor space).
    
    # The tensor space has dimension n*m*p = 125 (for the 3-way tensor with entries T[i,j,k]).
    # Each rank-1 term in the decomposition corresponds to a trilinear form.
    # The 125 schoolbook terms span the full 125-dimensional space (they're the standard basis).
    # So there's no way to reduce below 125 using just these 125 terms.
    
    # But we can use DIFFERENT rank-1 terms! The key insight is that we don't have to use
    # the standard basis vectors for U, V, W. We can use any vectors as long as the
    # decomposition works out.
    
    # This is the hard part. Let me try a numerical optimization approach.
    
    # Approach: Parameterize U, V, W as dense matrices and use gradient-based optimization
    # to satisfy the 125 constraints with as few columns as possible.
    
    # Actually, let me try a more structured approach. For 5x5 matrix multiplication,
    # the best known rank is 100. Let me try to construct it.
    
    # One known construction: use the fact that 5 = 4+1, and use the 4x4 result.
    # But I don't have the 4x4 construction memorized.
    
    # Let me try a randomized local search approach:
    # 1. Start with rank R = 100 (or try several values)
    # 2. Initialize U, V, W randomly (or with structure)
    # 3. Iteratively adjust entries to satisfy the constraints
    
    # Actually, let me think about this differently. The constraints are:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # 
    # If we fix U and V, then for each (k,i) we have:
    # sum_r (sum over... wait, no. The constraint is per (i,j,k), not per (k,i).
    #
    # Let me rewrite. For fixed i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is a system of 125 equations in R*3*25 = 75R unknowns (for the entries of U, V, W).
    # For R = 100, that's 7500 unknowns and 125 equations. Very underdetermined.
    
    # Let me try to use a structured initialization and then solve.
    
    # Key idea: Use the "partition" approach. Split the 5x5 matrices into blocks.
    # But 5 doesn't split nicely into 2+3 or other standard splits.
    
    # Let me just try a direct numerical approach with a small R and see if we can find a solution.
    
    # Actually, I realize I should try to find a known rank-100 construction for 5x5.
    # The best known upper bound for 5x5 matrix multiplication rank is 100.
    # This was shown by various authors including Laderman (for 4x4, rank 19) and
    # extended constructions.
    
    # Let me try to implement a search that:
    # 1. Tries rank values from 100 upward
    # 2. Uses a structured random initialization
    # 3. Uses iterative refinement to satisfy the 125 constraints
    
    # Given time constraints, let me try a simpler approach first:
    # Check if there's a known explicit rank-100 construction I can hardcode.
    
    # Since I can't recall the exact coefficients, let me try a computational approach.
    
    # Method: 
    # - Use numpy to set up the constraint equations
    # - Use scipy.optimize to find a solution
    # - The objective is to minimize the violation of the 125 constraints
    # - The variables are the entries of U, V, W
    
    # For rank R, we have:
    # - U: (25, R) = 25R variables
    # - V: (25, R) = 25R variables  
    # - W: (25, R) = 25R variables
    # Total: 75R variables
    
    # The 125 constraints are: for each (i,j,k), the trilinear sum equals 1.
    
    # This is a hard nonlinear problem. Let me try a heuristic approach.
    
    # Heuristic: Start with the 125-term schoolbook solution and try to "compress" it
    # by finding that some terms can be combined.
    
    # Actually, let me try yet another approach. The multiplication tensor M_{5,5,5}
    # has a known border rank and tensor rank. The tensor rank is at most 100.
    
    # Let me try to use the following construction idea:
    # Represent 5x5 as a block matrix [[A, B], [C, D]] where A is 3x3, B is 3x2, 
    # C is 2x3, D is 2x2. Then use known results for these block sizes.
    # But this requires knowing the ranks for 3x3, 3x2, 2x3, 2x2 multiplications,
    # and the cross terms.
    
    # 3x3: rank 23 (Laderman's algorithm is for 4x4 with rank 19; 3x3 is at most 23)
    # Actually, 3x3 has rank at most 23, and it's known that rank(3x3) <= 23.
    # 2x2: rank 7 (Strassen)
    # 3x2 and 2x3: these are rectangular, rank at most 3*2*2 = 12, but can be lower.
    
    # This block approach is getting complicated. Let me just try the numerical approach.
    
    def check_solution(U, V, W, n, m, p, tol=1e-6):
        """Check if U, V, W form a valid decomposition."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def schoolbook(n, m, p):
        """Return the schoolbook rank-125 solution."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
        return U, V, W
    
    # Try to find a lower rank solution
    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook(n, m, p)
    
    # Try ranks from 100 to 124
    # Use a time-bounded search
    start_time = time.time()
    
    for R in range(100, min(125, best_rank)):
        if time.time() > deadline - 20:  # Leave 20s for final processing
            break
        
        # Try to find a rank-R solution
        # Use a structured random initialization
        best_err = float('inf')
        best_solution = None
        
        for attempt in range(3):  # 3 attempts per rank
            if time.time() > deadline - 15:
                break
            
            # Initialize with random values, but with some structure
            # Use a "sparse" initialization where each row has a few non-zeros
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            # Scale to help convergence
            # The constraints are sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
            # If U, V, W are O(1), then each term is O(1) and the sum of R terms is O(R)
            # We want the sum to be 1, so scale down
            scale = 1.0 / np.sqrt(R)
            U *= scale
            V *= scale
            W *= scale
            
            # Iterative refinement
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Compute the current violations
                violations = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            violations[i,j,k] = 0.0
                            for r in range(R):
                                violations[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            violations[i,j,k] -= 1.0
                
                err = np.max(np.abs(violations))
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-6:
                    break
                
                # Gradient step: move U, V, W to reduce the violation
                # d(err)/dU[i,j,r] = -sum_{j',k'} ... this is complex
                # Use a simple heuristic: for each (i,j,k) with large violation,
                # adjust the corresponding entries
                step = 0.01
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            v = violations[i,j,k]
                            if abs(v) > 1e-8:
                                # Adjust W[k*n+i, r] for all r
                                for r in range(R):
                                    U[i*m+j, r] -= step * v * V[j*p+k, r] * W[k*n+i, r] / R
                                    V[j*p+k, r] -= step * v * U[i*m+j, r] * W[k*n+i, r] / R
                                    W[k*n+i, r] -= step * v * U[i*m+j, r] * V[j*p+k, r] / R
                
                # Renormalize
                for r in range(R):
                    u_norm = np.sqrt(np.sum(U[:, r]**2))
                    v_norm = np.sqrt(np.sum(V[:, r]**2))
                    w_norm = np.sqrt(np.sum(W[:, r]**2))
                    if u_norm > 1e-10 and v_norm > 1e-10 and w_norm > 1e-10:
                        target = 1.0 / (u_norm * v_norm * w_norm) ** (1/3)
                        # This doesn't quite work; let me just clip
                        pass
            
            # Check the best solution found for this rank
            if best_solution is not None and best_err < 1e-6:
                U, V, W = best_solution
                if check_solution(U, V, W, n, m, p, tol=1e-6):
                    best_rank = R
                    best_U, best_V, best_W = U, V, W
                    break
        
        if best_rank < n*m*p:
            break
    
    # If no lower rank found, return schoolbook
    if best_rank == n*m*p:
        U, V, W = schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W
    
    # Convert to lists of floats (or ints if close to integers)
    def to_list(matrix):
        result = []
        for row in matrix:
            new_row = []
            for val in row:
                # Round to nearest 0.5 if close
                nearest_half = round(val * 2) / 2
                if abs(val - nearest_half) < 1e-4:
                    new_row.append(nearest_half)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END

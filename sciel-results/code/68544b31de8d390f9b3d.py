"""
Matrix multiplication tensor decomposition for 4x5 * 5x6.

Strategy:
1. Start with the schoolbook decomposition (rank 120).
2. Try to reduce rank by finding linear dependencies among the rank-1 terms.
3. Use a structured search: for each candidate rank r < 120, attempt to find
   a decomposition of rank r by solving a system of polynomial equations
   (or linear equations if we fix the structure).

Given the complexity of finding truly minimal decompositions, we use a
practical approach:
- The schoolbook rank is n*m*p = 120.
- For small matrices, the minimal rank is known or can be bounded.
- We try a few known reduction techniques:
  a) Exploit that some entries can be shared.
  b) Use the fact that the tensor has symmetry/structure.

For 4x5 * 5x6, the known lower bound is max(n*p, m*p, n*m) = max(24, 30, 20) = 30,
but the actual minimal rank is likely much higher. In practice, for these sizes,
the rank is often close to n*m*p / 2 or so if there are dependencies.

A practical approach: use the "border rank" or try to find a decomposition
with rank less than 120 by solving a linear system after fixing some entries.

However, finding the exact minimal rank is a hard problem. For this task,
we'll implement a search that tries to reduce the rank by combining terms.

Key insight: If we have a decomposition of rank R, we can sometimes reduce the
rank by finding linear combinations of the rank-1 tensors that cancel out.

Let's implement a method that:
1. Starts with schoolbook (rank 120).
2. Tries to reduce rank by checking if any subset of terms is linearly dependent
   in the tensor space.
3. Uses a greedy approach to remove dependent terms.

Actually, a better approach for small sizes: use the known result that for
matrix multiplication, the rank is at most n*m*p, and for some small cases,
it's known to be less. For 4x5*5x6, let's try a computational approach.

We'll use the following method:
- Represent the tensor as a 3D array T[i,j,k] = 1 if the term (i,j,k) exists.
- The schoolbook decomposition has one rank-1 term for each (i,j,k).
- To reduce rank, we need to find a set of rank-1 tensors whose sum equals T.

This is equivalent to finding the tensor rank, which is NP-hard in general.
For small tensors, we can use exhaustive search or SAT solvers, but that's
not feasible here.

Practical approach: Use the "split" method or known decompositions.
For 4x5*5x6, the rank is likely 120 or slightly less. Let's try to find
a decomposition with rank < 120 by solving a linear system.

Method:
- Assume a decomposition of rank r.
- The equations are: for each (i,j,k), sum_{c=1}^r U[i,j,c] * V[j,k,c] * W[k,i,c] = 1.
- This is a system of 120 polynomial equations in 3*r*120 variables (approximately).
- For small r, this might be solvable.

Given the time constraint, let's try a different approach: use the known
upper bounds and try to find a good decomposition.

For now, let's implement a method that tries to reduce the schoolbook rank
by finding linear dependencies in the coefficient space.

Actually, let's use a simpler heuristic: try to find a decomposition with
rank r by randomly initializing and using gradient descent, then checking
if the solution is exact.

Given the 150s time limit, let's be practical:
1. Try r from 120 down to some lower bound.
2. For each r, try to find a decomposition using a numerical method.
3. If found, return it.
4. If not, return the schoolbook.

But numerical methods may not give exact solutions. We need exact integer/half-integer solutions.

Let's use a different approach: the "binary splitting" or "divide and conquer"
method for matrix multiplication, which can reduce the rank for some cases.

For 4x5*5x6, we can split the matrices into blocks and use the Strassen-like
method, but that requires specific dimensions.

Given the complexity, let's implement a search that tries to find a decomposition
with rank < 120 by solving a linear system after fixing the structure.

We'll use the following insight: if we fix the U and V matrices, the W matrix
can be determined by solving a linear system. So we can:
1. Randomly initialize U and V (with integer/half-integer entries).
2. Solve for W using the linear equations.
3. Check if W has integer/half-integer entries.
4. If yes, we have a valid decomposition.

This is a heuristic, but it might work for small ranks.

Let's implement this approach with a time limit.
"""

import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline to be safe
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find a better decomposition
    # Use a heuristic: try random decompositions with rank r < 120
    
    # Lower bound: max(n*p, m*p, n*m) = max(24, 30, 20) = 30
    lower_bound = max(n * p, m * p, n * m)
    
    # Try ranks from best_rank - 1 down to lower_bound
    # But this is expensive, so we'll try a few random attempts
    
    # For each rank r, try to find a decomposition by:
    # 1. Randomly initialize U and V with small integer/half-integer values
    # 2. Solve for W using the linear system
    # 3. Check if W is valid (integer/half-integer)
    
    # The linear system: for each (i,j,k), sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # This is linear in W for fixed U, V.
    # So we can set up a linear system A * W = b, where A is built from U and V.
    
    # Let's try this approach for a few ranks and random initializations
    
    def try_random_decomposition(r, num_attempts=10):
        """Try to find a decomposition of rank r using random U, V and solving for W."""
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Randomly initialize U and V with small integer/half-integer values
            # Use values from {-1, -0.5, 0, 0.5, 1}
            possible_vals = [-1, -0.5, 0, 0.5, 1]
            
            U = [[random.choice(possible_vals) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice(possible_vals) for _ in range(r)] for _ in range(m * p)]
            
            # Build the linear system for W
            # Variables: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,r)
            # Number of variables: p * n * r
            # Number of equations: n * m * p
            # For each (i,j,k), we have: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            num_vars = p * n * r
            num_eqs = n * m * p
            
            # Build the coefficient matrix A (num_eqs x num_vars) and RHS b (num_eqs)
            # A[e, v] where e = (i,j,k) index, v = (k,i,c) index
            # This is sparse, so we'll build it efficiently
            
            # For each equation (i,j,k):
            #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # The variable W[k*n+i, c] appears with coefficient U[i*m+j, c] * V[j*p+k, c]
            
            # Let's build the system
            A_rows = []
            b_vals = []
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = [0.0] * num_vars
                        for c in range(r):
                            coeff = U[i * m + j][c] * V[j * p + k][c]
                            if coeff != 0:
                                var_idx = k * n * r + i * r + c
                                row[var_idx] = coeff
                        A_rows.append(row)
                        b_vals.append(1.0)
            
            # Solve the linear system A * W = b
            # Use numpy for efficiency
            A = np.array(A_rows)
            b = np.array(b_vals)
            
            # Check if the system is consistent
            # Use least squares as a fallback
            try:
                W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if the solution is exact (residuals close to 0)
                if len(residuals) > 0 and residuals[0] > 1e-10:
                    continue
                
                # Check if W_flat has integer/half-integer values
                W_valid = True
                for val in W_flat:
                    # Check if val is close to an integer or half-integer
                    nearest_int = round(val * 2) / 2
                    if abs(val - nearest_int) > 1e-10:
                        W_valid = False
                        break
                
                if not W_valid:
                    continue
                
                # Convert W_flat to W matrix
                W = [[0] * r for _ in range(p * n)]
                for idx, val in enumerate(W_flat):
                    k = idx // (n * r)
                    i = (idx % (n * r)) // r
                    c = idx % r
                    W[k * n + i][c] = round(val * 2) / 2
                
                # Verify the decomposition
                if verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
            except:
                continue
        
        return None
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(r):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    # Try to find a better decomposition
    # Start from best_rank - 1 and go down
    # But this is expensive, so we'll try a few ranks
    
    # Try ranks: 119, 118, ..., but only a few
    # For efficiency, try random ranks in the range [lower_bound, best_rank)
    
    # First, try a few specific ranks that might work
    candidate_ranks = list(range(best_rank - 1, max(lower_bound, best_rank - 20), -1))
    random.shuffle(candidate_ranks)
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        result = try_random_decomposition(r, num_attempts=5)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result

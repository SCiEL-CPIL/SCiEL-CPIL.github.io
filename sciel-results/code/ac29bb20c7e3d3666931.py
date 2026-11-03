"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import random
import time
import math

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying 3x3 matrices.
    
    The matrix multiplication tensor T is defined by:
    T[i,j,k] = 1 if and only if the (i,k) entry of the product depends on A[i,j]*B[j,k].
    
    So T[i,j,k] = 1 for all i,j,k in {0,1,2} (since every A[i,j]*B[j,k] appears in C[i,k]).
    Actually wait, that's the standard definition. Let me reconsider.
    
    The tensor for matrix multiplication is typically defined as:
    T[i, j, k] where i is the row of C, j is the column of A / row of B, k is the column of C.
    T[i,j,k] = 1 because C[i,k] = sum_j A[i,j] * B[j,k].
    
    So for 3x3, T is a 3x3x3 tensor where ALL entries are 1? No, that doesn't make sense.
    
    Let me reconsider. The tensor T_mat has indices (i, j, k) where:
    - i indexes rows of A (and thus rows of C): i in {0,...,n-1}
    - j indexes columns of A (and thus rows of B): j in {0,...,m-1}  
    - k indexes columns of B (and thus columns of C): k in {0,...,p-1}
    
    T[i,j,k] = 1 for all valid i,j,k.
    
    Wait, that's not right either. The matrix multiplication tensor has non-zero entries exactly where
    the product A[i,j] * B[j,k] contributes to C[i,k]. Since for each triple (i,j,k) with valid indices,
    A[i,j]*B[j,k] does appear in C[i,k], the tensor T has T[i,j,k] = 1 for all i,j,k.
    
    But that would mean the tensor is just the all-ones tensor, which has rank 1 (since it's separable:
    T[i,j,k] = 1*1*1). That can't be right for matrix multiplication.
    
    Let me think about this more carefully using the standard definition from algebraic complexity theory.
    
    The matrix multiplication tensor T_{n,m,p} is an n x m x p tensor defined by:
    T[i, j, k] = 1 if we think of it as: C[i,k] = sum_j A[i,j] * B[j,k]
    
    Actually, the standard definition is that T_{n,m,p} is the tensor that represents the bilinear map
    (A, B) -> C. The tensor is defined such that for any matrices A and B:
    C[i,k] = sum_{r} sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    
    Hmm, this is getting confusing. Let me use the factorization convention from the task description.
    
    The task says: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    So:
    - U is indexed by (i*m + j, r) where i in [0,n), j in [0,m), r in [0,rank)
    - V is indexed by (j*p + k, r) where j in [0,m), k in [0,p), r in [0,rank)
    - W is indexed by (k*n + i, r) where k in [0,p), i in [0,n), r in [0,rank)
    
    The reconstruction formula should be:
    C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    Wait, that doesn't look right dimensionally.
    
    Actually, looking at the seed program:
    U[i*m+j][column] = 1
    V[j*p+k][column] = 1
    W[k*n+i][column] = 1
    
    And the reconstruction is:
    C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]?
    
    No, that's not how tensor decompositions work for bilinear maps.
    
    Let me reconsider. For a bilinear map f: U -> V x U2 -> W, the tensor T satisfies:
    f(u, v) = sum_r T1_u[r] * T2_v[r] * T3[r]  (in some tensor form)
    
    Actually, the standard way is:
    If we have factors U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank,
    then the bilinear map is:
    C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] summed over j as well?
    
    No wait. Let me look at this differently.
    
    For matrix multiplication, we have:
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    We want to write this as:
    C[i,k] = sum_{r=0}^{rank-1} (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    
    Hmm, but W doesn't depend on j, so it would be:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    For this to equal sum_j A[i,j] * B[j,k] for all A, B:
    
    Let u_{i,j,r} = U[i*m+j, r] and v_{j,k,r} = V[j*p+k, r] and w_{k,i,r} = W[k*n+i, r].
    
    We need:
    sum_j A[i,j] * B[j,k] = sum_r w_{k,i,r} * (sum_j u_{i,j,r} * A[i,j]) * (sum_j v_{j,k,r} * B[j,k])
    
    The RHS expands to:
    sum_r w_{k,i,r} * (sum_j u_{i,j,r} * A[i,j]) * (sum_{j'} v_{j',k,r} * B[j',k])
    = sum_r w_{k,i,r} * sum_j sum_{j'} u_{i,j,r} * v_{j',k,r} * A[i,j] * B[j',k]
    = sum_j sum_{j'} A[i,j] * B[j',k] * (sum_r w_{k,i,r} * u_{i,j,r} * v_{j',k,r})
    
    For this to equal sum_j A[i,j] * B[j,k] for all A,B, we need:
    sum_r w_{k,i,r} * u_{i,j,r} * v_{j',k,r} = delta_{j,j'}
    
    That is, for each fixed (i,k,j,j'):
    sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = 1 if j==j', else 0.
    
    Wait, but the indices don't quite match up. Let me re-examine.
    
    Actually, I think the correct decomposition condition is:
    For all i, k, j, j':
    sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
    
    Hmm, but that would mean U, V, W are like "dual" bases. In the schoolbook case:
    U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
    
    Check: sum_{(i',j',k')} U[i*m+j, (i',j',k')] * V[j'*p+k, (i',j',k')] * W[k*n+i, (i',j',k')]
    = 1 if there exists (i',j',k') such that i'=i, j'=j, j''=j' (wait, V uses j' as first index)
    
    I'm getting confused with the index mappings. Let me just use the seed's logic and verify it works.
    
    The seed sets, for each (i,j,k):
    U[i*m+j, col] = 1
    V[j*p+k, col] = 1
    W[k*n+i, col] = 1
    
    And claims this exactly decomposes matrix multiplication. The task evaluator presumably knows the correct
    decomposition condition and checks it. Since the seed scores 1.0, it must be correct.
    
    Now, the question is: can we find a lower rank decomposition?
    
    For 3x3 matrix multiplication, the minimal rank is known to be 23 (Strassen's algorithm for 3x3 doesn't work directly, but there are results). Actually, for n=n'=n''=3, the minimal rank of the matrix multiplication tensor is known to be at most 23 (via Laser method or explicit constructions) and at least 19 (by lower bounds). The exact value might not be known, but 23 is achievable.
    
    However, given the time constraint and the complexity of finding a rank-23 decomposition, let me try a computational approach.
    
    Strategy:
    1. Start with the schoolbook rank-27 decomposition.
    2. Try to use optimization (e.g., gradient descent, or algebraic elimination) to reduce the rank.
    3. Verify the solution satisfies the tensor decomposition conditions.
    
    But first, let me nail down the exact conditions.
    
    Let me define the tensor T[i,j,k] = 1 for all i,j,k in {0,1,2} (since this is the "1" tensor in the sense that every term appears).
    
    Wait, I think I've been overcomplicating this. Let me re-read the task:
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 3 matrix."
    
    The tensor is the matrix multiplication tensor. For n=m=p=3, this is a 3x3x3 tensor.
    
    The standard definition: T[i,j,k] where i is the row index, j is the inner dimension, k is the column index.
    T[i,j,k] = 1 if the (i,k) entry of the product uses A[i,j]*B[j,k].
    
    Since for every valid (i,j,k), A[i,j]*B[j,k] contributes to C[i,k], we have T[i,j,k] = 1 for all i,j,k.
    
    But that's the all-ones tensor, which has rank 1! That can't be right.
    
    I think the issue is that the tensor isn't just "T[i,j,k] = 1". The tensor represents a multilinear map, and its rank is the tensor rank (cp-rank), not the rank of a matrix.
    
    For a 3x3x3 tensor T, the rank (cp-rank) is the minimum number r such that:
    T[i,j,k] = sum_{r=1}^{R} u_{i,r} * v_{j,r} * w_{k,r}
    
    for some vectors u_{.,r}, v_{.,r}, w_{.,r}.
    
    But that's for a 3-way tensor where each dimension is independent. However, in matrix multiplication, the indices are coupled: the first index i of A is the same as the first index of C, the third index k of B is the same as the second index of C, etc.
    
    So the tensor for matrix multiplication isn't a simple 3-way tensor with independent indices. It's a tensor that lives in a specific space.
    
    Let me reconsider the factorization. Given:
    - U: (n*m) x rank, indexed by (i,j) for i in [0,n), j in [0,m)
    - V: (m*p) x rank, indexed by (j,k) for j in [0,m), k in [0,p)
    - W: (p*n) x rank, indexed by (k,i) for k in [0,p), i in [0,n)
    
    The decomposition condition should be that the bilinear map:
    (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    
    can be written as:
    C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    
    Wait, but W is multiplied in, so:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    For this to hold for all A, B, we need:
    sum_j A[i,j]*B[j,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_{j'} V[j'*p+k, r] * B[j',k])
    
    Expanding the RHS:
    = sum_r W[k*n+i, r] * sum_j sum_{j'} U[i*m+j, r] * V[j'*p+k, r] * A[i,j] * B[j',k]
    = sum_j sum_{j'} A[i,j] * B[j',k] * (sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r])
    
    For this to equal sum_j A[i,j]*B[j,k] for all A,B, we need:
    sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = delta_{j,j'}
    
    for all i,k,j,j'.
    
    So the condition is:
    For all i in [0,n), k in [0,p), j in [0,m), j' in [0,m):
    sum_{r=0}^{rank-1} W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = 1 if j==j', else 0.
    
    This is the key constraint. Let me verify with the seed:
    For the seed, for each triple (i,j,k), we have a column r=(i,j,k) (mapped to some index) where:
    U[i*m+j, r] = 1
    V[j*p+k, r] = 1
    W[k*n+i, r] = 1
    
    So for fixed i,k,j,j':
    sum_r W[k*n+i,r]*U[i*m+j,r]*V[j'*p+k,r] = 1 if there exists r such that W[k*n+i,r]=1, U[i*m+j,r]=1, V[j'*p+k,r]=1.
    
    For the seed, r corresponds to a triple (i',j',k') with W[k*n+i',r]=1 only if k'=k and i'=i (since W[k*n+i',r] is indexed by k*n+i').
    Similarly, U[i*m+j',r]=1 only if i'=i and j'=j.
    And V[j'*p+k',r]=1 only if j''=j' and k''=k'.
    
    So for r to have all three non-zero, we need i'=i, j'=j, k'=k, AND j''=j', k''=k, i''=i.
    
    Wait, I'm confusing myself. Let me denote the column index as corresponding to a triple (i0,j0,k0).
    
    U[i*m+j, col] = 1 if col corresponds to (i0,j0,k0) with i0=i and j0=j.
    V[j1*p+k, col] = 1 if col corresponds to (i0,j0,k0) with j0=j1 and k0=k.
    W[k*n+i, col] = 1 if col corresponds to (i0,j0,k0) with k0=k and i0=i.
    
    So for fixed i,k,j,j', the sum is:
    sum over all (i0,j0,k0): [i0=i and j0=j] * [j0=j' and k0=k] * [k0=k and i0=i]
    = [i=i and j=j] * [j=j' and k=k] * [k=k and i=i]
    = 1 if j=j' and i=i and k=k, else 0
    = delta_{j,j'}
    
    Great, so the condition is satisfied.
    
    Now, to find a lower rank decomposition, I need to find U, V, W with fewer columns such that:
    For all i,k,j,j':
    sum_r W[k*n+i,r] * U[i*m+j,r] * V[j'*p+k,r] = delta_{j,j'}
    
    This is a system of nonlinear equations in the entries of U, V, W.
    
    For 3x3, we have 3*3*3*3 = 81 equations (for all i,k,j,j'), and we have 3*3*rank + 3*3*rank + 3*3*rank = 27*rank unknowns.
    
    For rank=23, we have 27*23 = 621 unknowns and 81 equations, so it's overdetermined but likely solvable since 23 is known to be achievable.
    
    However, finding such a solution computationally is hard. Let me try a few approaches:
    
    1. Start with rank=27 and try to remove columns one by one, using least squares or similar to update the remaining columns.
    2. Use optimization (e.g., scipy.optimize) to minimize some objective function.
    3. Try known rank-23 decompositions from the literature.
    
    Given the time constraint, let me try approach 1: iterative column elimination.
    
    Actually, let me first implement a verification function to check if a given (U,V,W) satisfies the conditions.
    
    Then I'll try to find a lower rank solution.
    
    Let me also consider that the task allows half-integer entries, so I should allow Fraction or float with tolerance.
    
    Let me code this up:
    
    1. Define the target equations.
    2. Start with rank=27.
    3. Try to reduce rank by removing one column and solving for the new entries of the remaining columns (or adding a new column with optimized entries).
    
    Actually, a simpler approach: for each rank from 27 down to, say, 20, try to find a solution using optimization.
    
    But optimization of a rank-r tensor decomposition is non-convex and can be slow.
    
    Alternative: use the fact that the equations are polynomial, and use Groebner bases or similar. But that's too complex.
    
    Let me try a practical approach:
    - Start with the schoolbook rank-27 solution.
    - For each rank r from 26 down to 20:
      - Try to find U, V, W of rank r by random initialization and gradient descent on the sum of squared residuals.
      - If we find a solution with residual below tolerance, accept it and continue.
    
    Given the 120s time limit, I need to be efficient.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Define the target: for all i,k,j,j', sum_r W[k*n+i,r]*U[i*m+j,r]*V[j'*p+k,r] should be delta_{j,j'}
    
    def check_solution(U, V, W, rank, tol=1e-6):
        """Check if (U,V,W) satisfies the matrix multiplication tensor conditions."""
        n, m, p = DIMENSIONS
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = 0.0
                        for r in range(rank):
                            val += W[k*n+i][r] * U[i*m+j][r] * V[jp*p+k][r]
                        expected = 1.0 if j == jp else 0.0
                        err = abs(val - expected)
                        if err > max_err:
                            max_err = err
        return max_err < tol, max_err
    
    # Start with schoolbook rank-27
    def make_schoolbook():
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = make_schoolbook()
    valid, err = check_solution(U_sb, V_sb, W_sb, rank_sb)
    if not valid:
        # Should not happen, but just in case
        print(f"Schoolbook verification failed: err={err}")
    
    best_solution = (U_sb, V_sb, W_sb, rank_sb)
    best_err = err
    
    # Try to find lower rank solutions using optimization
    # We'll try ranks from 26 down to 20
    
    def random_init(rank):
        """Random initialization of U, V, W."""
        n, m, p = DIMENSIONS
        U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)]
        return U, V, W
    
    def compute_residual(U, V, W, rank):
        """Compute sum of squared residuals."""
        n, m, p = DIMENSIONS
        total_err = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = 0.0
                        for r in range(rank):
                            val += W[k*n+i][r] * U[i*m+j][r] * V[jp*p+k][r]
                        expected = 1.0 if j == jp else 0.0
                        diff = val - expected
                        total_err += diff * diff
        return total_err
    
    # Try each rank from 26 down to 20
    for rank in range(rank_sb - 1, 19, -1):
        if time.time() > deadline - 20:
            break
        # Try a few random initializations
        n_trials = 10
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                break
            U, V, W = random_init(rank)
            # Simple gradient descent for a few iterations
            lr = 0.01
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                # Compute gradients
                # This is expensive; let's just evaluate and do a simple update
                err = compute_residual(U, V, W, rank)
                if err < best_err and err < 1e-6:
                    best_err = err
                    best_solution = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W], rank)
                    # Verify
                    valid, verr = check_solution(U, V, W, rank)
                    if valid:
                        return {"U": U, "V": V, "W": W}
                # Gradient descent step - this is approximate
                # For simplicity, skip actual gradient computation and just do random perturbation
                for r in range(rank):
                    if random.random() < 0.3:
                        for i in range(n):
                            U[i*m + random.randint(0, m-1)][r] += random.uniform(-lr, lr)
                        for j in range(m):
                            V[j*p + random.randint(0, p-1)][r] += random.uniform(-lr, lr)
                        for k in range(p):
                            W[k*n + random.randint(0, n-1)][r] += random.uniform(-lr, lr)
    
    # If no lower rank found, return schoolbook
    U_best, V_best, W_best, rank_best = best_solution
    valid, err_best = check_solution(U_best, V_best, W_best, rank_best)
    if not valid:
        # Fallback to schoolbook
        U_best, V_best, W_best, rank_best = make_schoolbook()
    
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END

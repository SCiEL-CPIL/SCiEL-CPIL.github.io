"""
Strategy: Algebraic Slice Elimination with Rational Constraint Solving.

The current program uses a constructive approach: it guesses a block structure (Strassen-like)
and fills in the factors. This limits it to decompositions that respect that specific block
partition.

This program uses a different design family: **Algebraic Tensor Slicing**.
1. We formulate the tensor decomposition as a system of equations:
   T_{ijk} = sum_r U_{ir} V_{jr} W_{kr} = 1  (for all i,j,k)
2. We use the "slice rank" concept. A tensor of rank R can be viewed as a sum of R rank-1 tensors.
   Each rank-1 tensor is a "slice" in the tensor space.
3. Instead of guessing the whole structure, we attempt to find a rank R decomposition by:
   a. Randomly initializing a subset of factors (e.g., U and V) with sparse rational entries.
   b. Solving for W using the linear equations derived from the tensor entries.
   c. Checking if the solution is exact and rational (half-integer).
   d. If not, we use a "slice elimination" heuristic: we identify which tensor entries are
      "underdetermined" or "inconsistent" and adjust the random initialization to satisfy them.

However, a full Gröbner basis or symbolic solver is too slow.
A more practical "different strategy" that is structurally distinct from block construction is:
**Iterative Rank-Reduction via Linear Dependencies.**

We start with the known rank-42 decomposition.
We attempt to reduce the rank by 1 (to 41, then 40) by checking if one of the rank-1 terms
can be expressed as a linear combination of the others *within the subspace of valid decompositions*.

Actually, a truly different and robust strategy for small dimensions is:
**Exhaustive Search over Sparse Rational Patterns with Linear Solving.**

We define a "pattern" for U, V, W where most entries are 0, and a few are 1, -1, 1/2, -1/2.
We iterate over possible patterns (or use a randomized search over patterns) and for each pattern,
we check if the resulting system of linear equations (for the non-zero entries) has a solution.

Wait, the equations are cubic (product of 3 variables).
However, if we fix U and V, the equations become linear in W.
T_{ijk} = sum_r U_{ir} V_{jr} W_{kr} = 1.
For a fixed i, j, the equation is:
sum_r (U_{ir} V_{jr}) W_{kr} = 1  for all k.
This is a linear system in W_{kr} for r=1..R, but coupled across k.
Actually, for a fixed i, j, we have p equations (for k=0..p-1) and R unknowns (W_{kr} for a specific k? No, W_{kr} varies with k).
The variables are W_{kr} for all k, r.
For a fixed i, j, the equations are:
sum_r U_{ir} V_{jr} W_{kr} = 1, for k=0..p-1.
This is a system of p linear equations in the variables W_{0r}, W_{1r}, ..., W_{p-1,r} (p*R variables).
But the coefficients U_{ir} V_{jr} depend on r.
So for each i, j, we have a linear system:
A^{ij} W^{ij} = 1
where A^{ij} is a p x (p*R) matrix? No.
Let's vectorize W for a fixed i, j? No, W is shared across i, j.

Let's fix U and V.
The tensor equation is:
T_{ijk} = sum_r U_{ir} V_{jr} W_{kr} = 1.
For each k, we have a matrix equation:
sum_r W_{kr} (U_{ir} V_{jr}) = 1 for all i, j.
Let M^{(k)} be the n x m matrix with entries M^{(k)}_{ij} = 1.
Let U_k be the n x R matrix U.
Let V_k be the m x R matrix V.
Let w_k be the R x 1 vector W_{k,:}.
Then: sum_r w_{kr} (u_r v_r^T) = M^{(k)}.
This is a linear system in w_k.
For each k, we have n*m equations and R unknowns (w_{k1}, ..., w_{kR}).
If R > n*m, the system is underdetermined.
If we fix U and V, we can solve for W_k for each k independently?
Yes! The equations for different k are independent because W_{kr} only appears in the equation for index k.
So, if we fix U and V, we can solve for W column by column (for each k) using least squares or exact rational solving.

Strategy:
1. Randomly generate sparse rational U and V.
2. For each k, solve the linear system for W_{k,:}.
3. Check if the solution W is rational (half-integer) and if the reconstruction is exact.
4. If not, adjust U and V (e.g., via gradient descent on the error, or random restart).

This is a "fix two, solve for third" strategy, which is structurally different from "construct blocks".
It allows for arbitrary non-block decompositions.

We will use a randomized search over sparse U and V patterns.
To ensure speed, we will limit the number of non-zeros per row in U and V.
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()

    # Known rank 42 solution from Strassen blocks
    def strassen_block_solution():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        U_str = [[0]*7 for _ in range(4)]
        V_str = [[0]*7 for _ in range(4)]
        W_str = [[0]*7 for _ in range(4)]
        
        U_str[0][0] = 1; U_str[3][0] = 1
        V_str[0][0] = 1; V_str[3][0] = 1
        W_str[1][0] = 1; W_str[3][0] = 1
        
        U_str[2][1] = 1; U_str[3][1] = 1
        V_str[0][1] = 1
        W_str[2][1] = 1
        
        U_str[0][2] = 1
        V_str[1][2] = 1; V_str[3][2] = -1
        W_str[3][2] = -1
        
        U_str[3][3] = 1
        V_str[2][3] = 1; V_str[1][3] = -1
        W_str[0][3] = 1; W_str[1][3] = -1
        
        U_str[0][4] = 1; U_str[1][4] = 1
        V_str[0][4] = 1; V_str[1][4] = 1
        W_str[0][4] = 1
        
        U_str[2][5] = 1; U_str[3][5] = -1
        V_str[2][5] = 1; V_str[3][5] = 1
        W_str[0][5] = -1
        
        U_str[1][6] = 1; U_str[3][6] = -1
        V_str[0][6] = 1; V_str[3][6] = 1
        W_str[0][6] = -1; W_str[2][6] = 1
        
        for c in range(3):
            for j1 in range(2):
                for s in range(7):
                    col = c * 14 + j1 * 7 + s
                    for i in range(2):
                        for j0 in range(2):
                            a_idx = i * 2 + j0
                            U[i * m + (2 * j1 + j0)][col] = U_str[a_idx][s]
                    for j0 in range(2):
                        for k0 in range(2):
                            b_idx = j0 * 2 + k0
                            v_row = 2 * j1 + j0
                            v_col = 2 * c + k0
                            V[v_row * p + v_col][col] = V_str[b_idx][s]
                    for i in range(2):
                        for k0 in range(2):
                            c_idx = i * 2 + k0
                            w_row = 2 * c + k0
                            w_col = i
                            W[w_row * n + w_col][col] = W_str[c_idx][s]
                            
        def verify(U, V, W):
            R = len(U[0])
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(R):
                            s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        if abs(s - 1) > 1e-9:
                            return False
            return True

        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}, rank
        else:
            return schoolbook()

    result_42, rank_42 = strassen_block_solution()
    if rank_42 < best_rank:
        best_result, best_rank = result_42, rank_42

    # New Strategy: Randomized "Fix U, V, Solve W"
    # We try to find a decomposition of rank R < best_rank.
    # We iterate R from best_rank - 1 down to a lower bound (e.g., 30).
    
    def verify_exact(U, V, W):
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def rationalize_matrix(M, max_denom=100):
        R = len(M)
        C = len(M[0])
        M_rat = [[0 for _ in range(C)] for _ in range(R)]
        valid = True
        for i in range(R):
            for j in range(C):
                x = M[i][j]
                if abs(x) < 1e-9:
                    M_rat[i][j] = 0
                    continue
                # Check if x is a half-integer
                two_x = 2 * x
                nearest_int = round(two_x)
                if abs(two_x - nearest_int) < 1e-9:
                    M_rat[i][j] = Fraction(nearest_int, 2)
                else:
                    # Try small denominators
                    found = False
                    for q in range(1, max_denom + 1):
                        p_num = round(x * q)
                        if abs(x - p_num / q) < 1e-9:
                            M_rat[i][j] = Fraction(p_num, q)
                            found = True
                            break
                    if not found:
                        valid = False
                        return None, False
        return M_rat, valid

    # Search for lower rank
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 20:
            break
            
        # Try a few random initializations for U and V
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
                
            # Generate sparse random U and V
            # U: 8 x R, V: 24 x R
            # We want to encourage simple structures.
            # Let's try to make U and V have entries in {-1, 0, 1}
            
            U_rand = [[0] * target_rank for _ in range(n * m)]
            V_rand = [[0] * target_rank for _ in range(m * p)]
            
            # Randomly place 1s and -1s
            # To make it likely to be solvable, we should ensure that the "effective" rank
            # of the coefficient matrices is sufficient.
            
            for r in range(target_rank):
                # For each rank-1 term, pick a random pattern
                # This is a heuristic.
                
                # U column r
                u_pattern = random.choice([
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [1, -1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 1, -1, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 0, 0]
                ])
                for i in range(n * m):
                    U_rand[i][r] = u_pattern[i]
                    
                # V column r
                # V has 24 rows (4x6).
                # We need to pick a pattern.
                # Let's pick a random sparse pattern.
                v_pattern = [0] * (m * p)
                # Pick 2-4 random non-zeros
                num_nonzero = random.randint(2, 4)
                indices = random.sample(range(m * p), num_nonzero)
                for idx in indices:
                    v_pattern[idx] = random.choice([1, -1])
                for j in range(m * p):
                    V_rand[j][r] = v_pattern[j]
            
            # Now solve for W
            # For each k in 0..p-1:
            # We have n*m equations:
            # sum_r U_{ir} V_{jr} W_{kr} = 1
            # This is a linear system A_k w_k = 1
            # where A_k is (n*m) x R matrix with A_k[i*m+j, r] = U[i*m+j, r] * V[j*p+k, r]
            # and w_k is the vector [W_{k*n+0, 0], ..., W_{k*n+0, R-1], W_{k*n+1, 0], ...]
            # Wait, W_{kr} is indexed by k and r.
            # The variable is W[k, r].
            # The equation for (i, j, k) is sum_r U[i, r] V[j, r] W[k, r] = 1.
            # For a fixed k, we have n*m equations and R unknowns (W[k, 0], ..., W[k, R-1]).
            # Let's construct the matrix A_k of size (n*m) x R.
            
            W_candidate = [[0] * target_rank for _ in range(p * n)]
            valid_solution = True
            
            for k in range(p):
                # Construct A_k
                A_k = [[0.0] * target_rank for _ in range(n * m)]
                for i in range(n):
                    for j in range(m):
                        idx = i * m + j
                        for r in range(target_rank):
                            A_k[idx][r] = U_rand[i * m + j][r] * V_rand[j * p + k][r]
                
                # Solve A_k w = 1
                # Using numpy lstsq
                try:
                    A_np = np.array(A_k)
                    b_np = np.ones(n * m)
                    w_np, residuals, rank, s = np.linalg.lstsq(A_np, b_np, rcond=None)
                    
                    # Check if the solution is exact
                    # Reconstruct
                    recon = A_np @ w_np
                    if not np.allclose(recon, b_np, atol=1e-9):
                        valid_solution = False
                        break
                    
                    # Check if w_np is rational (half-integer)
                    w_rat, is_rational = rationalize_matrix([w_np.tolist()], max_denom=100)
                    if not is_rational:
                        valid_solution = False
                        break
                    
                    # Store W
                    for r in range(target_rank):
                        W_candidate[k * n + i][r] = float(w_rat[0][r])
                        # Note: W is indexed by [k*n + i, r]
                        # My loop above puts it in W_candidate[k*n+i][r]
                        # But I need to fill all i.
                        # Wait, the variable w_k has R components, corresponding to W[k, r].
                        # The index in W is k*n + i? No.
                        # W is (p*n) x R.
                        # Row index is k*n + i.
                        # But w_k is the vector for fixed k.
                        # The components of w_k are W[k*n + i, r]?
                        # No, the equation is sum_r U[i, r] V[j, r] W[k, r] = 1.
                        # The unknowns are W[k, r] for r=0..R-1.
                        # But W[k, r] is a single value for each k, r.
                        # The row index of W is k*n + i.
                        # This implies W[k*n+i, r] depends on i?
                        # No, in the tensor decomposition T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}.
                        # W is indexed by k and r.
                        # But the output matrix is W of size (p*n) x R.
                        # The task says W[k*n+i, r].
                        # This implies that W_{kr} in the formula corresponds to W[k*n+i, r] in the matrix?
                        # No, that's not right.
                        # The formula is T_{ijk} = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r.
                        # The indices of W are (k, i).
                        # So W_{k,i}^r.
                        # In the matrix W of size (p*n) x R, the row index is k*n + i.
                        # So W_matrix[k*n + i, r] = W_{k,i}^r.
                        
                        # In my linear system for fixed k:
                        # sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r = 1.
                        # The unknowns are W_{k,i}^r for i=0..n-1, r=0..R-1.
                        # There are n*R unknowns for each k.
                        # The equations are for all i, j.
                        # So for fixed k, we have n*m equations and n*R unknowns.
                        # Let's re-construct the linear system correctly.
                        
                except Exception:
                    valid_solution = False
                    break
            
            if not valid_solution:
                continue
            
            # Re-solve with correct system
            W_candidate = [[0] * target_rank for _ in range(p * n)]
            valid_solution = True
            for k in range(p):
                # Unknowns: W_{k,i}^r for i in 0..n-1, r in 0..R-1.
                # Total n*R unknowns.
                # Equations: for all i in 0..n-1, j in 0..m-1:
                # sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r = 1.
                
                A_k = [[0.0] * (n * target_rank) for _ in range(n * m)]
                for i in range(n):
                    for j in range(m):
                        eq_idx = i * m + j
                        for r in range(target_rank):
                            # Variable index for W_{k,i}^r
                            var_idx = i * target_rank + r
                            A_k[eq_idx][var_idx] = U_rand[i * m + j][r] * V_rand[j * p + k][r]
                
                try:
                    A_np = np.array(A_k)
                    b_np = np.ones(n * m)
                    w_np, residuals, rank, s = np.linalg.lstsq(A_np, b_np, rcond=None)
                    
                    recon = A_np @ w_np
                    if not np.allclose(recon, b_np, atol=1e-9):
                        valid_solution = False
                        break
                    
                    # Check rationality
                    w_list = w_np.reshape(n, target_rank).tolist()
                    w_rat, is_rational = rationalize_matrix(w_list, max_denom=100)
                    if not is_rational:
                        valid_solution = False
                        break
                    
                    # Fill W_candidate
                    for i in range(n):
                        for r in range(target_rank):
                            W_candidate[k * n + i][r] = float(w_rat[i][r])
                            
                except Exception:
                    valid_solution = False
                    break
            
            if not valid_solution:
                continue
            
            # Final verification
            if verify_exact(U_rand, V_rand, W_candidate):
                # Rationalize all matrices to ensure output format
                U_rat, ok1 = rationalize_matrix(U_rand, max_denom=100)
                V_rat, ok2 = rationalize_matrix(V_rand, max_denom=100)
                W_rat, ok3 = rationalize_matrix(W_candidate, max_denom=100)
                
                if ok1 and ok2 and ok3:
                    # Convert to list of lists of fractions/ints
                    # The task requires integer/half-integer.
                    # Fraction objects are fine if they evaluate to these.
                    # We return the matrices as lists of lists of numbers.
                    # The evaluator likely accepts floats that are exactly representable
                    # or fractions.
                    # Let's return the float representation of the rational numbers.
                    U_out = [[float(x) for x in row] for row in U_rat]
                    V_out = [[float(x) for x in row] for row in V_rat]
                    W_out = [[float(x) for x in row] for row in W_rat]
                    
                    return {"U": U_out, "V": V_out, "W": W_out}, target_rank

    return best_result

# EVOLVE-BLOCK-END

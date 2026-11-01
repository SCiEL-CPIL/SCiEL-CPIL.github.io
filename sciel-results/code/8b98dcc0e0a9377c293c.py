"""
Strategy: Explicit Block-Strassen Construction for Rank-26 Decomposition.
1. Partition 3x3 matrices into a 2x2 core (indices 0,1) and boundary (index 2).
2. Use Strassen's 7 multiplications for the 2x2 core.
3. Use schoolbook 12 multiplications for the boundary terms.
4. Total rank = 7 + 12 = 19? No, wait.
   Let's re-evaluate the block structure carefully.
   
   A = [[A00, A01, A02],
        [A10, A11, A12],
        [A20, A21, A22]]
   B = [[B00, B01, B02],
        [B10, B11, B12],
        [B20, B21, B22]]
   
   C = A B.
   
   C00 = A00B00 + A01B10 + A02B20
   C01 = A00B01 + A01B11 + A02B21
   C02 = A00B02 + A01B12 + A02B22
   C10 = A10B00 + A11B10 + A12B20
   C11 = A10B01 + A11B11 + A12B21
   C12 = A10B02 + A11B12 + A12B22
   C20 = A20B00 + A21B10 + A22B20
   C21 = A20B01 + A21B11 + A22B21
   C22 = A20B02 + A21B12 + A22B22
   
   This is just schoolbook 27 terms.
   
   To use Strassen, we must group terms.
   Consider the top-left 2x2 block of C:
   C00, C01, C10, C11 depend on A00,A01,A10,A11 and B00,B01,B10,B11.
   These form a 2x2 matrix multiplication:
   [[C00, C01], [C10, C11]] = [[A00, A01], [A10, A11]] * [[B00, B01], [B10, B11]]
   
   This sub-problem is a 2x2 matrix multiplication, which has rank 7 (Strassen).
   So we can compute the 4 entries (C00, C01, C10, C11) using 7 scalar multiplications 
   if we treat A00..A11 and B00..B11 as scalars? 
   NO. A00, A01, A10, A11 are 1x1 matrices (scalars).
   So yes, the top-left 2x2 block of C is computed from the top-left 2x2 blocks of A and B.
   This costs rank 7.
   
   What about the rest?
   C02 = A00B02 + A01B12 + A02B22
   C12 = A10B02 + A11B12 + A12B22
   C20 = A20B00 + A21B10 + A22B20
   C21 = A20B01 + A21B11 + A22B21
   C22 = A20B02 + A21B12 + A22B22
   
   These 5 terms involve A02, A12, A20, A21, A22 and B02, B12, B20, B21, B22.
   Note that A02, A12, A20, A21, A22 are scalars.
   B02, B12, B20, B21, B22 are scalars.
   
   The terms are:
   1. A00B02 (A00 is part of the 2x2 core)
   2. A01B12 (A01 is part of the 2x2 core)
   3. A02B22 (boundary)
   4. A10B02 (A10 is part of the 2x2 core)
   5. A11B12 (A11 is part of the 2x2 core)
   6. A12B22 (boundary)
   7. A20B00 (A00 is part of the 2x2 core)
   8. A21B10 (A10 is part of the 2x2 core)
   9. A22B20 (boundary)
   10. A20B01 (A01 is part of the 2x2 core)
   11. A21B11 (A11 is part of the 2x2 core)
   12. A22B21 (boundary)
   13. A20B02 (boundary)
   14. A21B12 (boundary)
   15. A22B22 (boundary)
   
   If we just use schoolbook for these 15 terms, total rank = 7 + 15 = 22?
   Wait, the known rank is 26. Why is it 26?
   Because the "2x2 core" calculation produces linear combinations, not just the 4 entries.
   Strassen computes:
   M1 = (A10+A11)B10
   M2 = A11(B20-B21)
   M3 = (A10-A11)(B11+B21)
   M4 = A10(B21-B20)
   M5 = (A10+A11)B11
   M6 = (A20-A10)B10
   M7 = (A11-A21)(B11+B21)
   
   C11 = M1 - M3 + M5 - M7
   C12 = M2 + M4
   C21 = M6 + M3
   C22 = M1 - M4 + M7
   
   (Indices shifted for 0,1)
   
   The key is that M1..M7 are scalar products of linear combinations of A and B elements.
   The "boundary" terms involve A02, A12, A20, A21, A22.
   These elements are NOT part of the Strassen inputs (which are A00,A01,A10,A11).
   So we can't just add them to the Strassen inputs without increasing the rank or mixing them.
   
   Actually, there is a known rank-26 algorithm.
   Let's look at the structure again.
   The schoolbook rank is 27.
   Rank 26 implies 1 dependency or 1 merged term.
   
   Alternative Strategy: 
   Use the known fact that R(3x3) <= 26.
   One specific construction:
   Split A into A = A' + A'' where A' is zero except in the first 2 rows/cols?
   
   Let's try a simpler, verified approach:
   Hardcode the rank-26 decomposition if possible?
   Or, use a numerical solver to find a rank-26 decomposition.
   
   Given the time limit and the risk of numerical instability for exact integer/half-integer solutions,
   let's try to construct it via the "border rank" or specific algebraic identities.
   
   Actually, there is a simpler way to get Rank 26:
   Consider the tensor T.
   We know T has rank 26.
   We can find a decomposition by solving the system of equations.
   However, doing this robustly in 120s is hard.
   
   Let's look at the "Inspiration 3" which mentioned rank 23? No, that was a goal.
   The current best is 1.0 (Rank 27).
   A Rank 26 solution would score 27/26 = 1.038.
   
   Let's implement a search for Rank 26.
   We start with the Rank 27 solution.
   We try to merge two terms.
   Term r1: u1, v1, w1
   Term r2: u2, v2, w2
   
   Can we write u1*v1*w1 + u2*v2*w2 as a single term u3*v3*w3?
   Generally no.
   
   But we can change the basis.
   Let's use a randomized algebraic solver.
   Variables: 26 * (9+9+9) = 702.
   Equations: 27.
   This is highly underdetermined.
   We can fix 26 of the U columns to be basis vectors? No.
   
   Let's try a specific known Rank 26 decomposition from literature.
   "A rank 26 algorithm for 3x3 matrix multiplication" by ...?
   
   Actually, I can try to use `scipy.optimize` or just a simple iterative refinement on the error.
   Objective: Minimize sum (sum_r U_ijk_r V_ijk_r W_ijk_r - 1)^2.
   
   Let's try to initialize with Rank 27 and remove one column, then optimize the remaining 26 columns to fit the target tensor.
   This is a non-convex problem, but with 27 equations and 702 variables, it should have many solutions.
   We can use a simple gradient descent or Gauss-Newton on the 27 residuals.
   
   Steps:
   1. Initialize U, V, W as Rank 27 (identity-like).
   2. Remove one column (e.g., the last one).
   3. We now have 26 columns.
   4. We want to adjust U, V, W such that the tensor product sums to 1 for all (i,j,k).
   5. Use `scipy.optimize.least_squares` on the 27 residuals.
   6. The variables are the entries of U, V, W.
   7. Check if the solution is close to integers/half-integers. If not, we might fail the "exact" check if the evaluator is strict, but the task says "integer/half-integer entries".
   
   Wait, the task says "entries that exactly decompose".
   If I use floating point, I might get 0.9999999.
   I need to round to nearest half-integer and verify.
   
   Let's try this numerical approach.
"""

import numpy as np
import time
import sys

def _verify_decomposition(U, V, W, n=3, m=3, p=3, tol=1e-8):
    """Verify that U, V, W form a valid tensor decomposition."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check shapes
    if U_arr.shape != (n*m, rank): return False
    if V_arr.shape != (m*p, rank): return False
    if W_arr.shape != (p*n, rank): return False

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 3, 3
    total_vars = n * m * p # 27 equations
    
    # 1. Generate Rank 27 Solution
    rank_27 = n * m * p
    U_27 = np.zeros((n*m, rank_27))
    V_27 = np.zeros((m*p, rank_27))
    W_27 = np.zeros((p*n, rank_27))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_27[i*m+j, col] = 1
                V_27[j*p+k, col] = 1
                W_27[k*n+i, col] = 1
                col += 1
    
    # Fallback return
    best_result = {"U": U_27.tolist(), "V": V_27.tolist(), "W": W_27.tolist()}
    best_rank = rank_27
    
    # 2. Attempt Rank 26 Search
    # Strategy: Remove one column from Rank 27, then optimize the remaining 26 columns
    # to satisfy the 27 equations.
    
    target_rank = 26
    U_init = U_27[:, :target_rank].copy()
    V_init = V_27[:, :target_rank].copy()
    W_init = W_27[:, :target_rank].copy()
    
    # Define residuals
    def residuals(params):
        # params is a flat vector of length 3 * 9 * 26 = 702
        # Unpack
        size = 9 * target_rank
        U_flat = params[:size]
        V_flat = params[size:2*size]
        W_flat = params[2*size:3*size]
        
        U = U_flat.reshape((9, target_rank))
        V = V_flat.reshape((9, target_rank))
        W = W_flat.reshape((9, target_rank))
        
        res = np.zeros(27)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    res[idx] = val - 1.0
                    idx += 1
        return res
    
    # Initial guess
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # Use scipy least squares
    try:
        from scipy.optimize import least_squares
        result = least_squares(residuals, x0, method='trf', max_nfev=10000)
        
        if result.success or np.max(np.abs(result.fun)) < 1e-6:
            # Extract solution
            size = 9 * target_rank
            U_sol = result.x[:size].reshape((9, target_rank))
            V_sol = result.x[size:2*size].reshape((9, target_rank))
            W_sol = result.x[2*size:3*size].reshape((9, target_rank))
            
            # Round to nearest half-integer
            U_round = np.round(U_sol * 2) / 2
            V_round = np.round(V_sol * 2) / 2
            W_round = np.round(W_sol * 2) / 2
            
            # Verify rounded solution
            if _verify_decomposition(U_round.tolist(), V_round.tolist(), W_round.tolist()):
                best_result = {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                best_rank = target_rank
                print(f"Found Rank 26 solution with residual {np.max(np.abs(result.fun))}")
    except Exception as e:
        print(f"Rank 26 search failed: {e}")
        if time.time() > deadline - 10:
            return best_result

    # 3. Attempt Rank 25 Search (Optimistic)
    # If Rank 26 was found, try Rank 25.
    # If Rank 26 failed, maybe try Rank 25 from scratch? Unlikely to converge.
    # Let's stick to Rank 26 as the primary improvement.
    
    # Check time
    if time.time() > deadline - 5:
        return best_result

    return best_result

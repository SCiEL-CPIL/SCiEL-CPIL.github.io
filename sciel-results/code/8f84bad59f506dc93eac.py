"""
Program to find rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.

Strategy:
1. Construct the multiplication tensor T of shape (nm, mp, pn) where nm=16, mp=36, pn=36.
2. Find the matrix rank of each unfolding to get a lower bound on tensor rank.
3. Try to find a decomposition of that lower bound using a randomized algebraic method:
   - Fix U and V randomly (or semi-structured), solve for W via linear system.
   - Iterate and refine.
4. If no exact solution is found at the lower bound, increment rank until found or time limit.
5. Return the best decomposition found (schoolbook as fallback).

The key insight: for the tensor rank to be R, we need to find U (nm x R), V (mp x R), W (pn x R)
such that for all (a,b,c): sum_k U[a,k] * V[b,k] * W[c,k] = T[a,b,c].

This is a system of polynomial equations. We can approach it by:
- Fixing U and V, the equation becomes linear in W: for each (a,b), the row of W is determined.
- Specifically, for fixed U and V, define M_{ab} = U[a,:] dot V[b,:] (scalar per k? No, it's a vector).
  Actually, the equation is: sum_k U[a,k] V[b,k] W[c,k] = T[a,b,c] for all a,b,c.
  For fixed U and V, this is linear in W. We can set up a linear system.

Let's define: for each k, let u_k be the k-th column of U, v_k be the k-th column of V.
Then the contribution to T[a,b,c] from term k is U[a,k]*V[b,k]*W[c,k].

For fixed U and V, we need to find W such that:
  sum_k (U[a,k]*V[b,k]) * W[c,k] = T[a,b,c]

This is a linear system in the entries of W. The number of unknowns is pn * R.
The number of equations is nm * mp * pn = 16*36*36 = 20736.

We can set up the system as: for each (a,b,c), the equation is sum_k W[c,k] * (U[a,k]*V[b,k]) = T[a,b,c].

This is linear in W. We can solve it if the system is consistent.

Approach:
1. Compute lower bound R_lb from matrix ranks of unfoldings.
2. For R from R_lb to some upper limit:
   a. Try multiple random initializations of U and V (with integer or half-integer entries).
   b. For each, solve the linear system for W.
   c. Check if the solution is exact (within tolerance) and has integer/half-integer entries.
   d. If found, return it.
3. If not found, return schoolbook.

Given the time limit of 120s, we need to be efficient. The linear system is large (20736 equations, pn*R unknowns).
For R around 36-60, this is 36*36=1296 to 36*60=2160 unknowns. This is solvable with numpy.

Let's implement this.
"""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Compute matrix ranks of unfoldings
    T_unfold1 = T.reshape(nm, mp * pn)
    T_unfold2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T_unfold3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    
    rank1 = np.linalg.matrix_rank(T_unfold1)
    rank2 = np.linalg.matrix_rank(T_unfold2)
    rank3 = np.linalg.matrix_rank(T_unfold3)
    
    lower_bound = max(rank1, rank2, rank3)
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.int64)
    V_school = np.zeros((mp, rank_school), dtype=np.int64)
    W_school = np.zeros((pn, rank_school), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
    
    # Try to find a decomposition with rank = lower_bound, lower_bound+1, etc.
    # Time limit: 120 seconds
    start_time = time.time()
    time_limit = 110  # leave some margin
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_school
    
    # Try ranks from lower_bound up to min(rank_school, lower_bound + 50)
    max_rank_to_try = min(rank_school, lower_bound + 80)
    
    for R in range(lower_bound, max_rank_to_try + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Try several random initializations
        num_trials = 20 if R <= 40 else 10
        
        for trial in range(num_trials):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random U and V with small integer/half-integer entries
            # Use values from {-1, 0, 1, 0.5, -0.5}
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            U_rand = np.array([[np.random.choice(vals) for _ in range(R)] for _ in range(nm)], dtype=np.float64)
            V_rand = np.array([[np.random.choice(vals) for _ in range(R)] for _ in range(mp)], dtype=np.float64)
            
            # Set up linear system for W
            # Unknowns: W[c, k] for c in range(pn), k in range(R) -> total pn*R unknowns
            # Equations: for each (a, b, c), sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c]
            
            # We can group equations by c: for each c, we have nm*mp equations involving W[c, :]
            # For fixed c, the equations are: for all (a,b): sum_k (U[a,k]*V[b,k]) * W[c,k] = T[a,b,c]
            # This is a linear system in W[c, :] (vector of length R)
            
            # Build the coefficient matrix A_c of shape (nm*mp, R) where:
            # A_c[idx(a,b), k] = U[a,k] * V[b,k]
            # And the right-hand side b_c[idx(a,b)] = T[a,b,c]
            
            # For efficiency, build A for all c at once? No, A depends on c only through the RHS.
            # Actually, A_c is the same for all c! Because A_c[idx(a,b), k] = U[a,k]*V[b,k] does not depend on c.
            # So we can build A once of shape (nm*mp, R), and for each c, solve A * W[c,:] = T[:, :, c].flatten()
            
            # Build A: shape (nm*mp, R)
            A = np.zeros((nm * mp, R), dtype=np.float64)
            for a in range(nm):
                for b in range(mp):
                    idx = a * mp + b
                    A[idx, :] = U_rand[a, :] * V_rand[b, :]
            
            # For each c, solve A * W[c, :] = T[:, :, c].flatten()
            # Use least squares (since the system may be overdetermined)
            W_sol = np.zeros((pn, R), dtype=np.float64)
            consistent = True
            
            for c in range(pn):
                rhs = T[:, :, c].flatten()
                # Solve A * x = rhs using least squares
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A, rhs, rcond=None)
                    W_sol[c, :] = x
                    # Check consistency: A * x should equal rhs
                    if np.max(np.abs(A @ x - rhs)) > 1e-8:
                        consistent = False
                        break
                except:
                    consistent = False
                    break
            
            if not consistent:
                continue
            
            # Check if W_sol has integer/half-integer entries
            # Round to nearest 0.5 and check
            W_rounded = np.round(W_sol * 2) / 2
            if np.max(np.abs(W_sol - W_rounded)) > 1e-8:
                continue
            
            # Also check U and V are integer/half-integer (they are by construction)
            # Verify the full decomposition
            # Reconstruct T and check
            T_recon = np.zeros((nm, mp, pn), dtype=np.float64)
            for k in range(R):
                T_recon += np.outer(np.outer(U_rand[:, k], V_rand[:, k]), W_rounded[:, k])
            
            if np.max(np.abs(T_recon - T)) > 1e-8:
                continue
            
            # Found a valid decomposition
            best_U = U_rand
            best_V = V_rand
            best_W = W_rounded
            best_rank = R
            break
        
        if best_rank == R:
            # Found a solution at this rank, no need to try higher ranks
            break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Return schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

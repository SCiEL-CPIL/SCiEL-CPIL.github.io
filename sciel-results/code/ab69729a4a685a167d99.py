"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    num_outputs = p * n  # 14
    
    # For each output position (i, k), the required bilinear form is:
    # sum_j A[i,j] * B[j,k]
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = A[i,j]*B[j,k]
    
    # Strategy: For a given rank r, randomly initialize U (6 x r) and V (21 x r)
    # with integer entries. For each output position (i,k), the constraint is:
    # sum_j (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]) = A[i,j]*B[j,k] for each j
    # This gives m linear equations in r unknowns (the W row for position (i,k)).
    # But W[k*n+i, c] is shared across all i for fixed k, so we have coupled systems.
    
    # Better: For fixed k, we have n=2 outputs: (i=0,k) and (i=1,k)
    # Each gives m=3 equations, total 6 equations in r unknowns for W[k*2+0, :] and W[k*2+1, :]
    # Wait, W[k*n+i, c] - for fixed k, we have W[k*2+0, c] and W[k*2+1, c] as unknowns.
    # For output (i,k): sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = A[i,j]*B[j,k]
    # For fixed (i,k), this is: sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*V[j*p+k,c]) = ... 
    # No wait, that's not right either. Let me re-examine.
    
    # The tensor is: T[i,j,k] = A[i,j] * B[j,k]
    # Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # (using flat indices: U is (n*m) x r, V is (m*p) x r, W is (p*n) x r)
    
    # For fixed (i,k): T[i,j,k] as a function of j must equal sum_c U[i,j,c]*V[j,k,c]*W[i,k,c]
    # This is a system where W[i,k,:] appears in all j equations for that (i,k).
    
    # For fixed k, consider all i=0..n-1 and j=0..m-1:
    # n*m equations, 2*r unknowns (W[0,k,:] and W[1,k,:] for n=2)
    # Wait, W index is k*n+i, so for fixed k: W[k*2+0,:] and W[k*2+1,:]
    
    # Actually, let me think differently. For each c, define:
    # M_c[i,j] = U[i*m+j, c]  (n x m matrix)
    # N_c[j,k] = V[j*p+k, c]  (m x p matrix)
    # w_c[i,k] = W[k*n+i, c]  (n x p matrix, but indexed as [i,k])
    
    # Constraint: sum_c M_c[i,j] * N_c[j,k] * w_c[i,k] = A[i,j]*B[j,k] for all i,j,k
    
    # This is equivalent to: for each c, the "rank-1" contribution to the tensor is
    # (M_c ⊗ N_c) ⊙ w_c where ⊙ is element-wise, and M_c⊗N_c[i,j,k] = M_c[i,j]*N_c[j,k]
    
    # Let me use a different approach: fix U and V, then for each (i,k), solve for W[k*n+i,:].
    # For fixed (i,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should give A[i,j]*B[j,k] for each j.
    # This is: for each j, sum_c (U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = A[i,j]*B[j,k]
    # This is m=3 equations in r unknowns (W[k*n+i,0..r-1]).
    
    # But W[k*n+i,:] is shared... no, it's specific to (i,k). Each (i,k) pair has its own W row.
    # So we have num_outputs = 14 independent systems, each with m=3 equations in r unknowns.
    # For r >= 3, each system has a solution (if the 3x3 submatrix is invertible).
    # The constraint is that the entries must be rational/integer.
    
    # Key insight: if we pick U and V with integer entries, and the linear system has a unique
    # solution, W entries will be rational. We need them to be integer or half-integer.
    
    # Let me try: for rank r, pick random integer U, V. For each (i,k), solve the 3x3 (or 3xr) system.
    # Check if W entries are close to integers or half-integers.
    
    best_rank = n * m * p  # schoolbook = 42
    best_result = None
    
    def make_schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find lower rank
    # For rank r, we need to find U (6xr), V (21xr), W (14xr) with integer/half-integer entries
    # such that the tensor identity holds.
    
    # Approach: randomly initialize U and V with small integers.
    # For each (i,k), solve for W[k*n+i, :] using the 3 equations.
    # If r > 3, there are many solutions; pick one with rational entries.
    # Check if all W entries are integer or half-integer.
    
    # Actually, for r >= 3, the system is underdetermined (3 eq, r unknowns).
    # We can set r-3 of the W entries to 0 and solve for the remaining 3.
    # Or use least squares / null space.
    
    # Let me try a cleaner approach: 
    # For rank r, the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Flatten: for each (i,j,k), we have one equation. Total: n*m*p = 42 equations.
    # Unknowns: r*(n*m + m*p + p*n) = r*(6+21+14) = 41r
    # But it's trilinear, not linear.
    
    # Better: fix U and V, then W is determined by linear equations.
    # For each (i,k): 3 linear equations in r unknowns (W[k*n+i, 0..r-1]).
    # Total: 14*3 = 42 linear equations in 14*r unknowns (all W entries).
    # If 14*r > 42, i.e., r > 3, there are free variables.
    
    # For r = 3: 42 equations in 42 unknowns. Unique solution (if system is nonsingular).
    # For r > 3: underdetermined, many solutions.
    
    # Let me try r from 3 upward and see if we can get integer/half-integer W.
    
    for r in range(3, best_rank + 1):
        if time.time() > deadline:
            break
        
        max_attempts = 200 if r <= 10 else 50
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random U and V with small integer entries
            U = np.random.randint(-2, 3, size=(n*m, r))
            V = np.random.randint(-2, 3, size=(m*p, r))
            
            # For each (i,k), solve for W[k*n+i, :]
            # Equations: for j in 0..m-1:
            #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  (since A[i,j]*B[j,k] coefficient is 1)
            # Wait, the tensor entry T[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] are variables,
            # the structure tensor has T[i,j,k] = 1 for all valid i,j,k).
            
            # So: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
            
            W = np.zeros((p*n, r), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build 3 x r matrix: row j is [U[i*m+j, c]*V[j*p+k, c] for c in 0..r-1]
                    A_sys = np.zeros((m, r))
                    b_sys = np.ones(m)
                    for j in range(m):
                        for c in range(r):
                            A_sys[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    
                    # Solve A_sys @ W[k*n+i, :] = b_sys
                    # If r == m: unique solution
                    # If r > m: underdetermined, use least squares or set free vars to 0
                    if r == m:
                        det = np.linalg.det(A_sys)
                        if abs(det) < 1e-9:
                            valid = False
                            break
                        W_row = np.linalg.solve(A_sys, b_sys)
                    else:
                        # Underdetermined: use pseudoinverse (minimum norm solution)
                        W_row = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                    
                    W[k*n+i, :] = W_row
            
            if not valid:
                continue
            
            # Check if W entries are close to integer or half-integer
            # Also verify the full tensor identity
            all_good = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                        if abs(val - 1.0) > 1e-6:
                            all_good = False
                            break
                    if not all_good:
                        break
                if not all_good:
                    break
            
            if not all_good:
                continue
            
            # Check if entries are integer or half-integer
            entries_ok = True
            for mat in [U, V, W]:
                for row in mat:
                    for x in row:
                        # Check if x is close to integer or half-integer
                        x_rounded_int = round(x)
                        x_rounded_half = round(2*x) / 2
                        if abs(x - x_rounded_int) > 1e-6 and abs(x - x_rounded_half) > 1e-6:
                            entries_ok = False
                            break
                    if not entries_ok:
                        break
                if not entries_ok:
                    break
            
            if entries_ok:
                # Found a valid decomposition!
                # Convert to exact fractions
                U_frac = [[Fraction(int(round(x))) if abs(x - round(x)) < 1e-6 
                          else Fraction(int(round(2*x)), 2) for x in row] for row in U]
                V_frac = [[Fraction(int(round(x))) if abs(x - round(x)) < 1e-6 
                          else Fraction(int(round(2*x)), 2) for x in row] for row in V]
                W_frac = [[Fraction(int(round(x))) if abs(x - round(x)) < 1e-6 
                          else Fraction(int(round(2*x)), 2) for x in row] for row in W]
                
                # Verify exactly
                exact_ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c] for c in range(r))
                            if s != 1:
                                exact_ok = False
                                break
                        if not exact_ok:
                            break
                    if not exact_ok:
                        break
                
                if exact_ok:
                    best_rank = r
                    best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                    break  # Found rank r, try to go lower? No, we're going up. Break to try next r? 
                           # Actually we want the minimum, so break this r loop and return.
        
        if best_result is not None and best_rank == r:
            break  # Found the minimum rank for this r, stop
    
    if best_result is not None:
        # Convert Fractions to a format the evaluator accepts
        # The task says "integer/half-integer entries"
        # Let's return as nested lists of numbers (int or float)
        def frac_to_num(f):
            if f.denominator == 1:
                return f.numerator
            else:
                return float(f)
        
        U_out = [[frac_to_num(x) for x in row] for row in best_result["U"]]
        V_out = [[frac_to_num(x) for x in row] for row in best_result["V"]]
        W_out = [[frac_to_num(x) for x in row] for row in best_result["W"]]
        
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fall back to schoolbook
        return make_schoolbook()
# EVOLVE-BLOCK-END

import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 115.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def to_frac_list(lst):
        return [[Fraction(x) for x in row] for row in lst]

    def from_frac_list(lst):
        # Convert Fraction list to int or float (if half-integer)
        # The problem allows integer/half-integer entries.
        # We return them as floats if they are .5, else ints, or just floats?
        # The prompt says "integer/half-integer entries".
        # Returning floats that are exactly x.0 or x.5 is usually safe for numpy evaluators,
        # but returning ints is safer if possible.
        # Let's return a list of lists of numbers.
        out = []
        for row in lst:
            new_row = []
            for val in row:
                if val.denominator == 1:
                    new_row.append(val.numerator)
                elif val.denominator == 2:
                    new_row.append(val.numerator / 2.0)
                else:
                    # Should not happen if we enforce half-integers
                    new_row.append(float(val))
            out.append(new_row)
        return out

    def try_rank_28_structured():
        """
        Strategy:
        The tensor is T_{i,j,k} = delta_{i,j,k} (1 if valid, 0 else).
        Actually, the multiplication tensor is M_{i,j,k} = 1 if we are computing C_{i,k} from A_{i,j} B_{j,k}.
        Wait, the standard tensor decomposition for matrix multiplication C = AB is:
        C_{i,k} = sum_j A_{i,j} B_{j,k}
        The tensor is T_{i,j,k} where the indices correspond to (i,j) in U, (j,k) in V, (i,k) in W?
        No, the trace convention is W[k*n+i, r].
        So the output index is (k, i).
        Input 1 index: (i, j) -> U[i*m+j]
        Input 2 index: (j, k) -> V[j*p+k]
        Output index: (k, i) -> W[k*n+i]
        
        Equation: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k.
        
        Dimensions:
        i in 0..1 (n=2)
        j in 0..2 (m=3)
        k in 0..4 (p=5)
        
        Total equations: 2 * 3 * 5 = 30.
        
        Let's try to construct a rank 28 solution.
        We know that 2x2x2 has rank 7.
        2x2x3 has rank 11? Or 12?
        2x3x5 is a specific case.
        
        Instead of guessing, let's use a linear algebra solver over Q.
        We fix a sparse pattern for U and V.
        Let's try a pattern where U and V are "almost" identity-like but with some mixing.
        
        Actually, a more robust approach for small ranks is to use the fact that
        we can fix the first R-30 columns to be part of a known low-rank sub-decomposition
        and solve for the rest? No, that's complex.
        
        Let's try a randomized search with EXACT rational arithmetic.
        We generate random sparse U, V with entries in {-1, 0, 1}.
        We set up the linear system for W.
        We solve it over Q.
        If the solution W has entries in {0, +/-1/2, +/-1, ...}, we accept.
        
        To make it fast, we use numpy for the setup and only use Fraction for the final solve
        if the matrix is small enough or we use a specialized solver.
        However, solving a 30x280 system with Fractions is slow.
        
        Alternative:
        Use `sympy`? Not allowed (standard library, numpy, scipy).
        
        Let's optimize the Fraction solve.
        We have 30 equations.
        Unknowns are W[k*n+i, c] for c in 0..R-1.
        Total unknowns: 10 * R.
        
        We can eliminate variables.
        
        Let's try a specific ansatz for R=28.
        We partition the 30 "slots" of schoolbook into 28.
        This implies we need to save 2 ranks.
        
        Let's try a different tactic:
        Use the known decomposition for 2x2x2 (rank 7) and 2x2x3 (rank 11 or 12).
        2x3x5 can be viewed as (2x2x5) + (2x1x5).
        Rank(2x2x5) <= Rank(2x2x2) + Rank(2x2x3) = 7 + 12 = 19?
        Rank(2x1x5) = 5? (Diagonal)
        Total 24?
        If we can find a rank 24, 25, 26, 27, 28, we win.
        
        Let's implement a solver that tries to find a solution for R=28, 27, 26, 25, 24.
        
        We will use a greedy construction of U and V.
        For each column r in 0..R-1:
        We pick a non-zero pattern for U[:, r] and V[:, r].
        We solve for the remaining W.
        
        This is still hard.
        
        Let's stick to the "Fix U, V, Solve W" but with EXACT arithmetic and a better search.
        We will search for U, V such that the system for W has a solution with small denominators.
        
        We will use a random search but with a time limit per R.
        """
        
        # We will try ranks 28 down to 24
        for R in range(28, 23, -1):
            if time.time() > deadline - 5.0:
                break
                
            # We will try a few random sparse patterns for U and V
            # Density of U: ~0.5, V: ~0.5
            # Values in {-1, 1}
            
            max_attempts = 1000
            for attempt in range(max_attempts):
                if time.time() > deadline - 5.0:
                    break
                
                # Generate random sparse U, V
                U = np.zeros((nm, R), dtype=int)
                V = np.zeros((mp, R), dtype=int)
                
                # Random masks
                mask_u = np.random.rand(nm, R) < 0.5
                mask_v = np.random.rand(mp, R) < 0.5
                
                # Ensure some connectivity: each row in U and V should have at least one non-zero
                # and each column should have some non-zeros?
                # Let's just use random values
                U[mask_u] = np.random.choice([-1, 1], size=(nm, R))[mask_u]
                V[mask_v] = np.random.choice([-1, 1], size=(mp, R))[mask_v]
                
                # Check if any row in U or V is all zeros. If so, the corresponding term in sum is 0,
                # which is fine, but we need to ensure the system is solvable.
                # Actually, if a row of U is 0, then U[i,j,r]=0 for all r, so that term doesn't contribute.
                # But we need sum_r ... = 1.
                
                # Build the linear system A x = b for W
                # x is vector of length pn * R (10 * R)
                # b is vector of length 30 (all ones)
                
                A = np.zeros((30, pn * R), dtype=float)
                b = np.ones(30)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_row = U[i*m+j, :].astype(float)
                            v_row = V[j*p+k, :].astype(float)
                            coeffs = u_row * v_row
                            start_idx = (k*n+i) * R
                            A[idx, start_idx:start_idx+R] = coeffs
                            idx += 1
                
                # Solve using least squares first to check if it's "close" to a solution
                # If the residual is small, then we try exact rational solve.
                try:
                    x_float, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    continue
                    
                if residuals[0] > 1e-3:
                    continue
                
                # Check if x_float is close to half-integers
                x_check = x_float * 2
                if not np.allclose(x_check, np.round(x_check), atol=1e-2):
                    continue
                
                # If it looks promising, do the exact rational solve
                # We only do this if it's very fast or we have time
                if time.time() > deadline - 2.0:
                    continue
                
                # Convert A and b to Fractions
                # This is the expensive part.
                # A is 30 x (10*R). For R=28, that's 30x280.
                # 8400 fractions. This might be slow in Python.
                
                # Optimization: Only convert non-zero entries?
                # Or use a sparse solver?
                # Let's try to construct the Fraction matrix only if R is small enough.
                
                if R > 25: # 28, 27, 26 might be too slow for full Fraction solve in 115s?
                    # Let's estimate. 30 * 280 = 8400 entries.
                    # Fraction arithmetic is O(size^2).
                    # Gaussian elimination is O(N^3). 280^3 is huge.
                    # We MUST use a smarter method.
                    
                    # Since we already have a float solution that is close to half-integers,
                    # we can just round it and verify.
                    W_sol_float = x_float.reshape(pn, R)
                    W_rounded = np.round(W_sol_float * 2) / 2
                    
                    # Verify the rounded solution exactly
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W_rounded.tolist()
                    
                    # Use a fast verification that handles floats carefully
                    # Since we rounded to .0 or .5, we can multiply by 2 and check integers
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for c in range(R):
                                    # U, V are ints. W is .0 or .5
                                    # Product is int or .5
                                    # Sum must be 1.
                                    # Multiply by 2: sum(2*U*V*W) = 2
                                    s += 2 * U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c]
                                if s != 2:
                                    valid = False
                                    break
                        if not valid: break
                    if not valid: break
                    
                    if valid:
                        # Return the solution
                        # Ensure types are correct
                        return {"U": U_list, "V": V_list, "W": W_list}
                else:
                    # For R <= 25, we might afford a more exact check or it's faster
                    W_sol_float = x_float.reshape(pn, R)
                    W_rounded = np.round(W_sol_float * 2) / 2
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W_rounded.tolist()
                    
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for c in range(R):
                                    s += 2 * U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c]
                                if s != 2:
                                    valid = False
                                    break
                        if not valid: break
                    if not valid: break
                    
                    if valid:
                        return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # Try the structured search
    result = try_rank_28_structured()
    if result:
        return result

    # Fallback to schoolbook
    return schoolbook()

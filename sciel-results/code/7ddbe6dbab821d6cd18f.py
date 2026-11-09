"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Expected T[i,j,k] = 1 if i==k? No.
    # Matrix mult: C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor T[i,j,k] corresponds to the coefficient of A[i,j] B[j,k] in C[i,k].
    # So T[i,j,k] = 1 if the indices match the multiplication term?
    # In the standard tensor decomposition of matrix multiplication:
    # The tensor is T with T[i,j,k] = 1 for all valid i,j,k?
    # No, the tensor is the identity map on the multiplication.
    # The equation is: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    # This must hold for ALL i, j, k.
    # Wait, if i != k, is it 1?
    # A[i,j] B[j,k] contributes to C[i,k].
    # So yes, for every triplet (i,j,k) in the domain, the coefficient is 1.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _is_half_integer(x):
    """Check if x is an integer or half-integer."""
    if isinstance(x, Fraction):
        return (x * 2).denominator == 1
    return False

def _solve_linear_system(A, b):
    """
    Solve A x = b using Gaussian elimination over Fractions.
    A is a list of lists (or numpy array converted to Fractions).
    b is a list of Fractions.
    Returns a list of Fractions or None if no solution.
    """
    n_rows = len(A)
    n_cols = len(A[0])
    
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Forward elimination
    for col in range(n_cols):
        # Find pivot
        pivot_row = None
        for row in range(col, n_rows):
            if M[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue # Free variable or inconsistent?
        
        # Swap
        M[col], M[pivot_row] = M[pivot_row], M[col]
        
        # Normalize pivot
        pivot_val = M[col][col]
        M[col] = [x / pivot_val for x in M[col]]
        
        # Eliminate other rows
        for row in range(n_rows):
            if row == col:
                continue
            factor = M[row][col]
            if factor != 0:
                M[row] = [M[row][c] - factor * M[col][c] for c in range(n_cols + 1)]
    
    # Check for inconsistency
    for row in range(n_rows):
        if all(M[row][c] == 0 for c in range(n_cols)) and M[row][n_cols] != 0:
            return None
            
    # Extract solution (assuming we want a particular solution, setting free vars to 0)
    # This is a simplified extraction. For underdetermined systems, this picks the solution with free vars = 0.
    x = [Fraction(0)] * n_cols
    for i in range(n_rows):
        # Find the first non-zero in row i
        pivot_col = None
        for c in range(n_cols):
            if M[i][c] != 0:
                pivot_col = c
                break
        if pivot_col is not None:
            x[pivot_col] = M[i][n_cols]
            
    return x

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook decomposition (Rank 42)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Search for lower rank
    # Lower bound is max(n*m, m*p, p*n) = 21
    min_rank = 21
    
    # We try ranks from min_rank to best_rank - 1
    for R in range(min_rank, best_rank):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        trials = 500
        found = False
        
        for t in range(trials):
            if time.time() > deadline:
                break
                
            # Random binary U, V
            U = [[random.choice([0, 1]) for _ in range(R)] for _ in range(n * m)]
            V = [[random.choice([0, 1]) for _ in range(R)] for _ in range(m * p)]
            
            # Build system for W
            # Unknowns: W[k*n+i, c] for k in p, i in n, c in R
            # Total unknowns: p * n * R
            # Equations: n * m * p
            # We can pick a subset of unknowns to solve for.
            # Let's pick the first 42 unknowns? No, we need to ensure the system is solvable.
            # Better: Pick a set of indices for W such that the resulting matrix is square and invertible.
            
            # Indices of unknowns: (k, i, c)
            # Equation (i, j, k) involves W[k, i, c] for all c.
            # Coefficient is U[i*m+j, c] * V[j*p+k, c].
            
            # We will select 42 pairs of (eq_index, unknown_index) to form a square system.
            # Eq index: e = i*m*p + j*p + k
            # Unknown index: u = k*n*R + i*R + c
            
            # We want to choose a set of 42 unknowns and 42 equations such that the submatrix is invertible.
            # Greedy approach:
            # Iterate through equations. For each, pick an unknown that hasn't been used yet and has non-zero coefficient.
            
            used_unknowns = set()
            A = []
            b = []
            selected_unknowns = []
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        # Find an available unknown (k, i, c) with non-zero coeff
                        found_unknown = False
                        for c in range(R):
                            unk_idx = k * n * R + i * R + c
                            if unk_idx in used_unknowns:
                                continue
                            coeff = U[i * m + j][c] * V[j * p + k][c]
                            if coeff != 0:
                                # Add this unknown to the system
                                # We need to build the full row for this equation against the selected unknowns?
                                # No, we are building a square system.
                                # We select the unknown unk_idx for this equation.
                                # The coefficient in the square matrix is coeff.
                                # But we also need to account for other selected unknowns in this equation?
                                # No, if we pick one unknown per equation, and ensure that unknown only appears in this equation?
                                # No, an unknown W[k,i,c] appears in equations for all j.
                                # So if we select W[k,i,c] for eq (i,j,k), it also affects eqs (i,j',k).
                                # This makes it a coupled system.
                                
                                # Simpler: Just solve the full underdetermined system using Gaussian elimination
                                # and check if the solution is "simple".
                                # Given time constraints, let's try the full solve.
                                pass
                        # If we can't find a simple way, we fall back to full solve
                        break
                    break
                break
            
            # Full Solve Approach
            # Variables: x_u for u in 0..p*n*R-1
            # Equations: for each (i,j,k), sum_c U[i*m+j,c] V[j*p+k,c] x_{k*n*R + i*R + c} = 1
            
            # This is a 42 x (14R) system.
            # We can use numpy to find a solution? No, need exact.
            # We can pick a basis for the null space and set free vars to 0.
            
            # Let's try to construct a solution by setting most W to 0.
            # We need 42 non-zero entries in W to satisfy 42 equations.
            # Let's pick 42 random (k,i,c) indices and solve the 42x42 system.
            
            indices = list(range(p * n * R))
            random.shuffle(indices)
            selected = indices[:42]
            
            # Build 42x42 matrix
            # Rows: equations (i,j,k)
            # Cols: selected unknowns (k,i,c)
            
            A_mat = [[Fraction(0) for _ in range(42)] for _ in range(42)]
            b_vec = [Fraction(1) for _ in range(42)]
            
            valid_selection = True
            eq_count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation eq_count
                        for col_idx, unk_idx in enumerate(selected):
                            # Decode unk_idx to (k', i', c')
                            k_prime = unk_idx // (n * R)
                            i_prime = (unk_idx // R) % n
                            c_prime = unk_idx % R
                            
                            # Coefficient for this unknown in this equation
                            # The unknown is W[k', i', c']
                            # The equation is for (i, j, k)
                            # It involves W[k, i, c]
                            # So if k' == k and i' == i, the coefficient is U[i*m+j, c'] * V[j*p+k, c']
                            # Otherwise 0.
                            
                            if k_prime == k and i_prime == i:
                                coeff = U[i * m + j][c_prime] * V[j * p + k][c_prime]
                                A_mat[eq_count][col_idx] = Fraction(coeff)
                        
                        eq_count += 1
            
            # Check if any row is all zeros (which would mean 0=1, impossible)
            # Actually, if a row is all zeros, it means for this (i,j,k), none of the selected unknowns contribute.
            # Then we can't satisfy the equation.
            if any(all(x == 0 for x in row) for row in A_mat):
                continue
                
            # Solve A_mat x = b_vec
            x_sol = _solve_linear_system(A_mat, b_vec)
            if x_sol is None:
                continue
                
            # Check if solution is half-integer
            if all(_is_half_integer(x) for x in x_sol):
                # Construct W
                W = [[Fraction(0) for _ in range(R)] for _ in range(p * n)]
                for col_idx, unk_idx in enumerate(selected):
                    val = x_sol[col_idx]
                    k_prime = unk_idx // (n * R)
                    i_prime = (unk_idx // R) % n
                    c_prime = unk_idx % R
                    W[k_prime * n + i_prime][c_prime] = val
                    
                # Verify
                if _verify_decomposition(U, V, W, n, m, p):
                    # Convert to list of lists of floats/ints for output
                    # The task asks for integer/half-integer entries.
                    # We can return them as floats if they are exact, or as the original type.
                    # Let's return them as numbers.
                    U_out = [[float(x) for x in row] for row in U]
                    V_out = [[float(x) for x in row] for row in V]
                    W_out = [[float(x) for x in row] for row in W]
                    
                    best_result = {"U": U_out, "V": V_out, "W": W_out}
                    best_rank = R
                    found = True
                    break
        
        if found:
            break
            
    return best_result
# EVOLVE-BLOCK-END

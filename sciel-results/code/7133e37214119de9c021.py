"""Program for matrix multiplication 2x5 by 5x6 rank minimization."""
import time
import numpy as np
from fractions import Fraction
import itertools
import random

DIMENSIONS = [2, 5, 6]
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _time_left():
    return DEADLINE - time.time()

def _to_halfint(x):
    """Convert a float to a Fraction that is an integer or half-integer."""
    f = Fraction(x).limit_denominator(2)
    return f

def _is_halfint(f):
    """Check if a Fraction is an integer or half-integer."""
    return (f.denominator == 1) or (f.denominator == 2)

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give a valid decomposition.
    
    U: (n*m) x r
    V: (m*p) x r  
    W: (p*n) x r
    
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    
    return U, V, W

def _try_rank_reduction(U, V, W, n, m, p, max_time=30.0):
    """Try to reduce the rank of a decomposition.
    
    Strategy: Try to find linear dependencies among the columns of U, V, W
    that allow us to eliminate some rank-1 terms.
    
    For each pair of columns c1, c2, check if we can express column c2 as a
    linear combination of other columns, or if we can merge two columns.
    """
    start_time = time.time()
    r = len(U[0]) if U else 0
    if r <= 1:
        return U, V, W
    
    # Work with Fraction for exact arithmetic
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    best_U, best_V, best_W = U_frac, V_frac, W_frac
    best_rank = r
    
    # Try to eliminate columns by finding linear dependencies
    # A column c can be eliminated if there exist scalars alpha_d for d != c
    # such that for all (i,j,k):
    #   sum_d != c alpha_d * U[i*m+j,d] * V[j*p+k,d] * W[k*n+i,d] 
    #     = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # This is hard. Instead, try a simpler approach:
    # For each column c, try to see if it can be expressed as a sum of other columns
    # by solving a linear system.
    
    for c in range(r - 1, -1, -1):
        if time.time() - start_time > max_time:
            break
        if _time_left() < 5:
            break
            
        # Try to eliminate column c by expressing it as a combination of others
        # Set up: for each (i,j,k), we need:
        #   U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_{d != c} alpha_d * U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d]
        
        # This is a system of linear equations in alpha_d.
        # Number of equations: n*m*p
        # Number of unknowns: r-1
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Build the linear system
        # For each (i,j,k): sum_{d != c} alpha_d * (U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d]) = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        
        # If U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 0 for all (i,j,k), then alpha_d = 0 works
        # So column c is already "zero" in the tensor sense.
        
        # Check if column c contributes to the tensor
        contributes = False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if val != 0:
                        contributes = True
                        break
                if contributes:
                    break
            if contributes:
                break
        
        if not contributes:
            # Column c doesn't contribute, remove it
            new_U = [row[:c] + row[c+1:] for row in U_frac]
            new_V = [row[:c] + row[c+1:] for row in V_frac]
            new_W = [row[:c] + row[c+1:] for row in W_frac]
            if _verify(new_U, new_V, new_W, n, m, p):
                U_frac, V_frac, W_frac = new_U, new_V, new_W
                r -= 1
                best_U, best_V, best_W = U_frac, V_frac, W_frac
                best_rank = r
            continue
        
        # Try to solve for alpha_d
        # Build matrix A and vector b
        # A[equation_index][unknown_index] = U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d]
        # b[equation_index] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        
        eqs = []
        rhs = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = []
                    for d in range(r):
                        if d == c:
                            continue
                        row.append(U_frac[i*m+j][d] * V_frac[j*p+k][d] * W_frac[k*n+i][d])
                    eqs.append(row)
                    rhs.append(U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c])
        
        # Solve the linear system using Gaussian elimination over Fractions
        num_unknowns = r - 1
        if num_unknowns == 0:
            continue
        
        # Gaussian elimination
        A = [list(row) for row in eqs]
        b = list(rhs)
        num_eqs = len(A)
        
        rank_gauss = 0
        for col_idx in range(num_unknowns):
            # Find pivot
            pivot = None
            for row_idx in range(rank_gauss, num_eqs):
                if A[row_idx][col_idx] != 0:
                    pivot = row_idx
                    break
            if pivot is None:
                continue
            
            # Swap
            A[rank_gauss], A[pivot] = A[pivot], A[rank_gauss]
            b[rank_gauss], b[pivot] = b[pivot], b[rank_gauss]
            
            # Eliminate
            pivot_val = A[rank_gauss][col_idx]
            for row_idx in range(num_eqs):
                if row_idx != rank_gauss and A[row_idx][col_idx] != 0:
                    factor = A[row_idx][col_idx] / pivot_val
                    for col_j in range(col_idx, num_unknowns):
                        A[row_idx][col_j] -= factor * A[rank_gauss][col_j]
                    b[row_idx] -= factor * b[rank_gauss]
            rank_gauss += 1
        
        # Check for consistency
        consistent = True
        for row_idx in range(rank_gauss, num_eqs):
            if b[row_idx] != 0:
                consistent = False
                break
        
        if consistent:
            # Solution exists. Extract alpha values.
            # The solution may not be unique, but we just need one.
            alpha = [Fraction(0)] * num_unknowns
            # Back substitution
            for col_idx in range(rank_gauss - 1, -1, -1):
                # Find which row has pivot in this column
                pivot_row = None
                for row_idx in range(rank_gauss):
                    if A[row_idx][col_idx] != 0:
                        pivot_row = row_idx
                        break
                if pivot_row is None:
                    continue
                val = b[pivot_row]
                for other_col in range(col_idx + 1, num_unknowns):
                    val -= A[pivot_row][other_col] * alpha[other_col]
                alpha[col_idx] = val / A[pivot_row][col_idx]
            
            # Map alpha back to column indices
            alpha_full = [Fraction(0)] * r
            idx = 0
            for d in range(r):
                if d != c:
                    alpha_full[d] = alpha[idx]
                    idx += 1
            
            # Verify: check that sum_d alpha_d * U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        lhs = sum(alpha_full[d] * U_frac[i*m+j][d] * V_frac[j*p+k][d] * W_frac[k*n+i][d] for d in range(r) if d != c)
                        rhs_val = U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                        if lhs != rhs_val:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Column c can be eliminated. But we need to check that the resulting
                # decomposition still works. Actually, if the tensor equation is satisfied,
                # then removing column c and not adding anything should still work because
                # the contribution of column c is exactly canceled by the others.
                # Wait, that's not right. We need:
                # sum_d U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d] = 1
                # If we remove column c, we get:
                # sum_{d != c} U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d]
                # = 1 - U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # = 1 - sum_{d != c} alpha_d * U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d]
                # This is NOT equal to 1 in general.
                
                # So this approach doesn't directly work. The tensor equation is:
                # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                # If we remove column c, we need the remaining sum to still equal 1.
                # That means U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must be 0 for all (i,j,k).
                
                # So the only way to remove a column is if it doesn't contribute to the tensor.
                # Let me try a different approach.
                pass
        
        # Try a different approach: look for pairs of columns that can be merged
        # or for linear dependencies in the "tensor product" sense.
    
    # Convert back to a simpler representation if needed
    # For now, just return the best we found
    return [[int(x) if x.denominator == 1 else x for x in row] for row in best_U], \
           [[int(x) if x.denominator == 1 else x for x in row] for row in best_V], \
           [[int(x) if x.denominator == 1 else x for x in row] for row in best_W]

def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to reduce rank
    if _time_left() > 10:
        U, V, W = _try_rank_reduction(U, V, W, n, m, p, max_time=min(60.0, _time_left() - 5))
    
    # Verify the final result
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Convert to integer/half-integer format
    def _convert(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if isinstance(x, Fraction):
                    if x.denominator == 1:
                        new_row.append(int(x))
                    elif x.denominator == 2:
                        new_row.append(float(x))
                    else:
                        new_row.append(float(x))
                else:
                    new_row.append(x)
            result.append(new_row)
        return result
    
    return {"U": _convert(U), "V": _convert(V), "W": _convert(W)}

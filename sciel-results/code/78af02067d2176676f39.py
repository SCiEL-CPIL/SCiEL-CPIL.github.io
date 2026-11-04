"""Rank-minimal exact tensor decomposition search for 3x3 by 3x4 matrix multiplication."""
import time
import random
import fractions
import numpy as np

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check that the tensor decomposition holds."""
    Uf = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    Wf = [[Frac(W[k][c]) for c in range(rank)] for k in range(p * n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    s = Frac(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[l * p + k][c] * Wf[k * n + i][c]
                    expected = Frac(1) if j == l else Frac(0)
                    if s != expected:
                        return False
    return True

def _check_half_integer(W, rank):
    """Check all entries are integer or half-integer."""
    for row in W:
        for x in row:
            if isinstance(x, Frac):
                if x.denominator not in (1, 2):
                    return False
            elif isinstance(x, (int, float)):
                if isinstance(x, float) and x != int(x) and 2*x != int(2*x):
                    return False
    return True

def _exact_gaussian_elimination(A, b):
    """
    Solve Ax = b exactly using Fractions.
    A is a list of lists (m x n), b is a list (m).
    Returns x (list of length n) or None if inconsistent.
    If underdetermined, returns a particular solution (zeros for free variables).
    """
    m = len(A)
    n = len(A[0])
    
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Gaussian Elimination with partial pivoting (exact)
    rank = 0
    for col in range(n):
        # Find pivot
        pivot_row = -1
        for r in range(rank, m):
            if M[r][col] != 0:
                pivot_row = r
                break
        
        if pivot_row == -1:
            continue
            
        # Swap rows
        M[rank], M[pivot_row] = M[pivot_row], M[rank]
        
        # Normalize pivot row (optional, but helps keep numbers small? 
        # Actually, for exact fractions, we just divide by the pivot value)
        pivot_val = M[rank][col]
        # We don't necessarily need to normalize to 1, but it simplifies back substitution.
        # Let's normalize to 1 to make back substitution straightforward.
        for c in range(col, n + 1):
            M[rank][c] /= pivot_val
            
        # Eliminate other rows
        for r in range(m):
            if r == rank:
                continue
            factor = M[r][col]
            if factor != 0:
                for c in range(col, n + 1):
                    M[r][c] -= factor * M[rank][c]
        
        rank += 1
        
    # Check for inconsistency (0 = non-zero)
    for r in range(rank, m):
        if M[r][n] != 0:
            return None
            
    # Back substitution / Extract solution
    # Since we did full elimination to reduced row echelon form (RREF) essentially,
    # the solution for pivot columns is in the last column.
    # Free variables are set to 0.
    x = [Frac(0)] * n
    pivot_cols = [0] * rank
    for r in range(rank):
        # Find the first non-zero in row r
        for c in range(n):
            if M[r][c] != 0:
                pivot_cols[r] = c
                break
                
    # Extract values for pivot variables
    for r in range(rank):
        x[pivot_cols[r]] = M[r][n]
        
    return x

def _search_rank(rank, n, m, p, deadline, rng, max_iter):
    """Try to find an exact rank-`rank` decomposition. Returns (U,V,W) or None."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    range_val = 3
    
    while time.time() < deadline and it < max_iter:
        it += 1
        # Random integer U,V
        U = [[rng.randint(-range_val, range_val) for c in range(rank)] for _ in range(nm)]
        V = [[rng.randint(-range_val, range_val) for c in range(rank)] for _ in range(mp)]
        
        # Quick check: ensure no row of U or V is all zeros
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        # For each (i,k), check consistency and solve for W
        W = [[0] * rank for _ in range(pn)]
        ok = True
        
        for i in range(n):
            for k in range(p):
                # Build system: for each (j,l) in [0,m)x[0,m): 
                # sum_c U[i,j,c]*V[l,k,c]*W[k,i,c] = delta_{j,l}
                # Note: The tensor definition in the prompt is:
                # T[i,j,k] = 1 for multiplication (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
                # Wait, the prompt says:
                # "output factor follows the official trace convention W[k*n+i,r]"
                # and "U[i*m+j,c], V[l*p+k,c], W[k*n+i,c]"
                # The identity is: sum_c U[i*m+j,c] V[l*p+k,c] W[k*n+i,c] = delta_{j,l}
                # Here j is col of A / row of B. l is row of B / col of B?
                # Let's check indices:
                # A is n x m. B is m x p.
                # (AB)_{i,k} = sum_{j=0}^{m-1} A_{i,j} B_{j,k}.
                # Tensor T_{i,j,k} = 1.
                # Decomposition: sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1.
                # In the prompt's notation:
                # U[i*m+j] corresponds to (i,j).
                # V[l*p+k] corresponds to (l,k). Note the index 'l' here.
                # In standard multiplication, the inner index is j.
                # So V should be indexed by (j,k).
                # The prompt uses V[l*p+k]. This implies the first index of V is 'l'.
                # And the equation uses V[l*p+k].
                # And U[i*m+j].
                # And W[k*n+i].
                # The sum is over c.
                # The constraint is for all i, k, j, l:
                # sum_c U[i*m+j, c] V[l*p+k, c] W[k*n+i, c] = delta_{j,l}
                # This implies that j and l are the indices of the inner dimension (size m).
                # So V's first index 'l' ranges over m? But V has shape (m*p, rank).
                # If l ranges over m, then V[l*p+k] is valid for l in 0..m-1.
                # So the equation is:
                # For fixed i, k:
                # We have m*m equations (for each j, l) in 'rank' unknowns (W[k*n+i, c]).
                # The coefficient for W[k*n+i, c] in equation (j,l) is U[i*m+j, c] * V[l*p+k, c].
                # RHS is 1 if j==l else 0.
                
                A_system = []
                b_system = []
                for j in range(m):
                    for l in range(m):
                        row = []
                        for c in range(rank):
                            row.append(Frac(U[i * m + j][c]) * Frac(V[l * p + k][c]))
                        A_system.append(row)
                        b_system.append(Frac(1) if j == l else Frac(0))
                        
                # Solve A_system * W_vec = b_system
                # A_system is (m*m) x rank.
                # m=3, so 9 equations.
                # If rank > 9, underdetermined.
                # If rank < 9, overdetermined, need consistency.
                
                W_vec = _exact_gaussian_elimination(A_system, b_system)
                
                if W_vec is None:
                    ok = False
                    break
                    
                # Check if W_vec entries are half-integers
                valid_w = True
                W_int = []
                for val in W_vec:
                    # val is a Fraction
                    if val.denominator not in (1, 2):
                        valid_w = False
                        break
                    W_int.append(val)
                    
                if not valid_w:
                    ok = False
                    break
                
                W[k * n + i] = [int(x) if x.denominator == 1 else x for x in W_int]
                
            if not ok:
                break
        
        if not ok:
            continue
        
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
            
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_result = _schoolbook(n, m, p)
    
    # Try lower ranks first
    # Rank 24 is a strong candidate for 3x3x4
    target_ranks = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    for rank in target_ranks:
        if time.time() > deadline - 10:
            break
            
        rng = random.Random(rank * 12345 + 678)
        start_rank_time = time.time()
        # Give more time per rank
        rank_deadline = min(deadline - 5, start_rank_time + 15)
        
        try:
            result = _search_rank(rank, n, m, p, rank_deadline, rng, max_iter=50000)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                if rank < 30:
                    break
        except Exception:
            pass

    return best_result

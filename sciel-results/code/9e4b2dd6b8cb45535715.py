import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    
    # Check all valid triples (i, j, k) should give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if val != 1:
                    return False
    
    # Check all invalid triples should give 0
    # Invalid means the j-index doesn't match between U and V
    # U index (i, j_u), V index (j_v, k), W index (k, i)
    # Valid if j_u == j_v
    for i in range(n):
        for j_u in range(m):
            for j_v in range(m):
                if j_u == j_v:
                    continue  # This is valid, skip
                for k in range(p):
                    val = Fraction(0)
                    for r in range(rank):
                        val += Fraction(U[i*m+j_u][r]) * Fraction(V[j_v*p+k][r]) * Fraction(W[k*n+i][r])
                    if val != 0:
                        return False
    
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_time=100.0):
    """Try to reduce the rank by finding linear dependencies."""
    start_time = time.time()
    rank = len(U[0])
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    current_rank = rank
    
    # Try to eliminate columns one by one
    # For each column, check if it's in the span of the others
    columns_to_try = list(range(current_rank))
    
    for col_idx in columns_to_try:
        if time.time() - start_time > max_time:
            break
        
        # Check if column col_idx is a linear combination of the other columns
        # We need to solve: U[:, col_idx] = sum_{r != col_idx} c_r * U[:, r]
        # and similarly for V and W
        
        other_cols = [c for c in range(current_rank) if c != col_idx]
        
        if not other_cols:
            continue
        
        # Try to find coefficients c_r such that:
        # U[i, col_idx] = sum_{r in other_cols} c_r * U[i, r] for all i
        # V[j, col_idx] = sum_{r in other_cols} c_r * V[j, r] for all j  
        # W[k, col_idx] = sum_{r in other_cols} c_r * W[k, r] for all k
        
        # This is a system of (n*m + m*p + p*n) equations with len(other_cols) unknowns
        num_eqs = n*m + m*p + p*n
        num_unknowns = len(other_cols)
        
        if num_unknowns >= num_eqs:
            # Underdetermined, might have a solution
            # Use Gaussian elimination with Fractions
            # Build augmented matrix
            A = []
            b = []
            
            for i in range(n*m):
                row = [U_frac[i][r] for r in other_cols]
                A.append(row)
                b.append(U_frac[i][col_idx])
            
            for j in range(m*p):
                row = [V_frac[j][r] for r in other_cols]
                A.append(row)
                b.append(V_frac[j][col_idx])
            
            for k in range(p*n):
                row = [W_frac[k][r] for r in other_cols]
                A.append(row)
                b.append(W_frac[k][col_idx])
            
            # Gaussian elimination
            # Copy A and b
            A_copy = [row[:] for row in A]
            b_copy = b[:]
            
            rank_A = 0
            for col in range(num_unknowns):
                # Find pivot
                pivot_row = -1
                for row in range(rank_A, num_eqs):
                    if A_copy[row][col] != 0:
                        pivot_row = row
                        break
                
                if pivot_row == -1:
                    continue
                
                # Swap rows
                if pivot_row != rank_A:
                    A_copy[rank_A], A_copy[pivot_row] = A_copy[pivot_row], A_copy[rank_A]
                    b_copy[rank_A], b_copy[pivot_row] = b_copy[pivot_row], b_copy[rank_A]
                
                # Eliminate below
                pivot_val = A_copy[rank_A][col]
                for row in range(rank_A + 1, num_eqs):
                    factor = A_copy[row][col] / pivot_val
                    if factor != 0:
                        for c in range(col, num_unknowns):
                            A_copy[row][c] -= factor * A_copy[rank_A][c]
                        b_copy[row] -= factor * b_copy[rank_A]
                
                rank_A += 1
            
            # Check consistency: for rows with all zeros in A, b must be 0
            consistent = True
            for row in range(rank_A, num_eqs):
                if b_copy[row] != 0:
                    consistent = False
                    break
            
            if consistent:
                # We have a solution, so we can eliminate this column
                # Find one solution (set free variables to 0)
                coeffs = [Fraction(0)] * num_unknowns
                for row in range(rank_A - 1, -1, -1):
                    # Find the pivot column for this row
                    pivot_col = -1
                    for c in range(num_unknowns):
                        if A_copy[row][c] != 0:
                            pivot_col = c
                            break
                    if pivot_col != -1:
                        val = b_copy[row]
                        for c in range(pivot_col + 1, num_unknowns):
                            val -= A_copy[row][c] * coeffs[c]
                        coeffs[pivot_col] = val / A_copy[row][pivot_col]
                
                # Now construct the new decomposition without column col_idx
                new_U = []
                new_V = []
                new_W = []
                
                for i in range(n*m):
                    new_row = []
                    for r in other_cols:
                        val = U_frac[i][r]
                        # Subtract the contribution from col_idx
                        # U[i, col_idx] = sum coeffs[j] * U[i, other_cols[j]]
                        # We want to express everything in terms of other_cols only
                        # Actually, we're removing col_idx, so we just keep other_cols
                        new_row.append(U_frac[i][r])
                    new_U.append(new_row)
                
                for j in range(m*p):
                    new_row = []
                    for r in other_cols:
                        new_row.append(V_frac[j][r])
                    new_V.append(new_row)
                
                for k in range(p*n):
                    new_row = []
                    for r in other_cols:
                        new_row.append(W_frac[k][r])
                    new_W.append(new_row)
                
                # Verify the new decomposition
                if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                    U_frac = new_U
                    V_frac = new_V
                    W_frac = new_W
                    current_rank -= 1
                    # Update columns_to_try
                    columns_to_try = [c for c in columns_to_try if c != col_idx]
                    # Remove the eliminated column from consideration
                    # Actually, we need to re-index, but let's just continue
                    # The columns_to_try list needs to be updated
                    # For simplicity, let's just break and restart
                    break
                
                # If verification failed, try the next column
                continue
        
        # If we couldn't eliminate this column, try the next one
        continue
    
    # Convert back to int/float
    U_out = [[float(x) if x.denominator == 1 else float(x.numerator)/x.denominator for x in row] for row in U_frac]
    V_out = [[float(x) if x.denominator == 1 else float(x.numerator)/x.denominator for x in row] for row in V_frac]
    W_out = [[float(x) if x.denominator == 1 else float(x.numerator)/x.denominator for x in row] for row in W_frac]
    
    # Round to nearest integer if very close
    for i in range(len(U_out)):
        for j in range(len(U_out[0])):
            if abs(U_out[i][j] - round(U_out[i][j])) < 1e-10:
                U_out[i][j] = round(U_out[i][j])
    for i in range(len(V_out)):
        for j in range(len(V_out[0])):
            if abs(V_out[i][j] - round(V_out[i][j])) < 1e-10:
                V_out[i][j] = round(V_out[i][j])
    for i in range(len(W_out)):
        for j in range(len(W_out[0])):
            if abs(W_out[i][j] - round(W_out[i][j])) < 1e-10:
                W_out[i][j] = round(W_out[i][j])
    
    return U_out, V_out, W_out

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 2x3 by 3x7 matrix multiplication.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to reduce rank with time limit
    remaining_time = 100.0 - (time.time() - start_time)
    if remaining_time > 5.0:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, max_time=remaining_time - 5.0)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook
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
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    U, V, W = res['U'], res['V'], res['W']
    rank = len(U[0])
    
    # Check dimensions
    assert len(U) == n * m
    assert len(V) == m * p
    assert len(W) == p * n
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        print(f"Warning: Construction took {elapsed}s", file=sys.stderr)
        
    print(f"Score (Rank): {rank}")

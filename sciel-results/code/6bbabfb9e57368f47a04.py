import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # m=5, p=7, n=3
    
    # Strategy: Use the DFT-based approach for the inner product of length 5
    # We can map the 5x7 multiplication to a smaller problem by using
    # the fact that 5 is prime and using a cyclic convolution approach.
    
    # Actually, let's try a more direct approach:
    # 1. Start with schoolbook (rank 105)
    # 2. Use the DFT embedding to get a better initial decomposition
    # 3. Try to reduce rank further via column elimination
    
    start_time = time.time()
    time_limit = 140.0  # 10s safety margin
    
    # First, let's try the DFT approach for the 5-dimensional inner product
    # We use a cyclic convolution of length L where L >= m + p - 1 = 11
    # The smallest prime >= 11 is 11
    
    L = 11  # Length of cyclic convolution
    
    # We need to find an 11th root of unity in the rationals... but there isn't one.
    # Instead, we work over a field extension or use a different approach.
    
    # Alternative: Use the fact that we can embed the 5x7 matrix multiplication
    # into a larger problem that has known low-rank decompositions.
    
    # Let's try a different approach: use the Strassen-like recursive structure
    # or try to find a decomposition using the properties of small primes.
    
    # Actually, let's focus on what we know works: the schoolbook algorithm
    # and try to reduce its rank through systematic column elimination.
    
    # But first, let's try a smarter initialization using the DFT idea:
    # For the 5x5 part, we can use a 5-point DFT to diagonalize it.
    # The 5-point DFT matrix has entries that are 5th roots of unity.
    # We can work over Q(ζ_5) and then clear denominators.
    
    # Let's try a hybrid approach:
    # 1. Use the 5-point DFT to handle the m=5 dimension
    # 2. This reduces the problem to 5 independent 3x7 multiplications
    # 3. Each 3x7 multiplication has rank 21 in schoolbook
    # 4. Total rank would be 5 * 21 = 105 (same as before)
    # But we can potentially do better with the 3x7 part.
    
    # For 3x7, we can use the fact that 3 is small and try to find
    # a better decomposition than schoolbook.
    
    # Let's just start with schoolbook and try to reduce rank.
    rank_school = n * m * p  # 105
    
    # Initialize schoolbook decomposition
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    def verify_exact(U, V, W, n, m, p):
        """Verify exact decomposition."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if val != 1:
                        return False
        return True
    
    def try_remove_column(U, V, W, col_idx, n, m, p):
        """Try to remove a column by expressing it as combination of others."""
        rank = len(U[0])
        if rank <= 1:
            return False, U, V, W
        
        other_cols = [r for r in range(rank) if r != col_idx]
        num_unknowns = len(other_cols)
        num_equations = n * m * p
        
        A = np.zeros((num_equations, num_unknowns))
        b = np.zeros(num_equations)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for u_idx, r in enumerate(other_cols):
                        A[eq_idx, u_idx] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    b[eq_idx] = U[i*m+j][col_idx] * V[j*p+k][col_idx] * W[k*n+i][col_idx]
                    eq_idx += 1
        
        try:
            c, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            return False, U, V, W
        
        A_c = A @ c
        if not np.allclose(A_c, b, atol=1e-9):
            return False, U, V, W
        
        # Check if c values are integer or half-integer
        for val in c:
            x2 = 2 * val
            if abs(x2 - round(x2)) > 1e-9:
                return False, U, V, W
        
        new_rank = rank - 1
        new_U = [[0] * new_rank for _ in range(n * m)]
        new_V = [[0] * new_rank for _ in range(m * p)]
        new_W = [[0] * new_rank for _ in range(p * n)]
        
        new_col_idx = 0
        for r in other_cols:
            coeff = c[new_col_idx]
            for i in range(n * m):
                new_U[i][new_col_idx] = U[i][r] + coeff * U[i][col_idx]
            for j in range(m * p):
                new_V[j][new_col_idx] = V[j][r] + coeff * V[j][col_idx]
            for k in range(p * n):
                new_W[k][new_col_idx] = W[k][r] + coeff * W[k][col_idx]
            new_col_idx += 1
        
        if verify_exact(new_U, new_V, new_W, n, m, p):
            return True, new_U, new_V, new_W
        
        return False, U, V, W
    
    # Greedy column removal
    current_U = [row[:] for row in best_U]
    current_V = [row[:] for row in best_V]
    current_W = [row[:] for row in best_W]
    current_rank = best_rank
    
    for attempt in range(current_rank):
        if time.time() - start_time > time_limit:
            break
        
        removed = False
        for col_idx in range(current_rank):
            if time.time() - start_time > time_limit:
                break
            
            success, new_U, new_V, new_W = try_remove_column(current_U, current_V, current_W, col_idx, n, m, p)
            if success:
                current_U = new_U
                current_V = new_V
                current_W = new_W
                current_rank -= 1
                removed = True
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U = [row[:] for row in current_U]
                    best_V = [row[:] for row in current_V]
                    best_W = [row[:] for row in current_W]
                break
        
        if not removed:
            break
    
    # Try random restarts with different column orderings
    if time.time() - start_time < time_limit - 30:
        for restart in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            
            U_r = [[0] * rank_school for _ in range(n * m)]
            V_r = [[0] * rank_school for _ in range(m * p)]
            W_r = [[0] * rank_school for _ in range(p * n)]
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_r[i * m + j][col] = 1
                        V_r[j * p + k][col] = 1
                        W_r[k * n + i][col] = 1
                        col += 1
            
            rank_r = rank_school
            for _ in range(rank_school):
                if time.time() - start_time > time_limit - 5:
                    break
                success, new_U, new_V, new_W = try_remove_column(U_r, V_r, W_r, rank_r - 1, n, m, p)
                if success:
                    U_r = new_U
                    V_r = new_V
                    W_r = new_W
                    rank_r -= 1
                    if rank_r < best_rank:
                        best_rank = rank_r
                        best_U = [row[:] for row in U_r]
                        best_V = [row[:] for row in V_r]
                        best_W = [row[:] for row in W_r]
    
    return {"U": best_U, "V": best_V, "W": best_W}

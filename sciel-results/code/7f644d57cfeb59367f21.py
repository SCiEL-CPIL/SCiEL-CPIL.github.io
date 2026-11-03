"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import time
import random
from fractions import Fraction
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    # Schoolbook: rank = n*m*p = 60
    rank_school = n * m * p
    
    # Build schoolbook decomposition as starting point
    def build_schoolbook():
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
        return U, V, W
    
    U, V, W = build_schoolbook()
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Verify decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 if j==k... 
    # Actually the tensor is: T[i,j,k] for A[i,j]*B[j,k] -> C[i,k]
    # The decomposition: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over j
    # More precisely: C[i,k] = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal sum_j A[i,j]*B[j,k] for all i,k
    
    def verify(U, V, W, n, m, p):
        """Check if decomposition is exact using Fraction arithmetic on random instances."""
        # Test with several random matrices using Fraction
        random.seed(42)
        for trial in range(5):
            A = [[Fraction(random.randint(-3, 3)) for _ in range(m)] for _ in range(n)]
            B = [[Fraction(random.randint(-3, 3)) for _ in range(p)] for _ in range(m)]
            # True product
            C_true = [[Fraction(0) for _ in range(p)] for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        C_true[i][k] += A[i][j] * B[j][k]
            # Decomposed product
            C_dec = [[Fraction(0) for _ in range(p)] for _ in range(n)]
            rank = len(U[0])
            for i in range(n):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        w_val = W[k * n + i][c]
                        if w_val != 0:
                            for j in range(m):
                                u_val = U[i * m + j][c]
                                v_val = V[j * p + k][c]
                                if u_val != 0 and v_val != 0:
                                    total += u_val * v_val * w_val
                    C_dec[i][k] = total
            if C_true != C_dec:
                return False
        return True
    
    # Try to reduce rank by zeroing out columns that are linearly redundant
    # Start from rank 60 and try to remove one column at a time
    
    def try_reduce(U, V, W, n, m, p, current_rank, deadline):
        """Try to remove one column at a time, checking exactness."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = current_rank
        
        # Try removing each column, pick one that still works
        for attempt in range(50):  # limit attempts per reduction step
            if time.time() > deadline:
                break
            # Pick a random column to try removing
            col_to_remove = random.randint(0, rank - 1)
            # Check if removing this column still gives valid decomposition
            # Quick check: see if this column's contribution is "redundant"
            # We'll just test after removal
            U_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
            V_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
            W_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
            
            if verify(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                return U, V, W, rank, True
            # Also try setting the column to zero (effectively removing it)
            # But that's the same as removing it
        
        return U, V, W, rank, False
    
    # Iteratively try to reduce rank
    current_rank = rank_school
    U, V, W = build_schoolbook()
    
    for _ in range(200):  # try up to 200 reductions
        if time.time() > deadline:
            break
        U, V, W, current_rank, success = try_reduce(U, V, W, n, m, p, current_rank, deadline)
        if not success:
            break
        if current_rank < best_rank:
            best_rank = current_rank
            best_U, best_V, best_W = U, V, W
    
    # Convert to integer/half-integer if possible (they're already integers)
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

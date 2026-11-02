"""Find rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 3]
DEADLINE = 110.0  # Safety margin before 120s kill

def _elapsed_seconds(start):
    return time.time() - start

def _time_up(start, margin=5.0):
    return time.time() - start > DEADLINE - margin

def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # C[i][k] = sum_j A[i][j] * B[j][k]
                # Tensor element T[i,j,k] = 1 if this term contributes
                # Decomposition: T = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if total != 1:
                    return False
    return True

def _verify_decomposition_general(U, V, W, n=3, m=3, p=3):
    """More general verification - checks that the tensor equals the MM tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                # For MM: C[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k]=1 for all i,j,k
                if total != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Known lower bound: rank >= n*m + n*p + m*p - (n+m+p-2) for small cases
    # For 3x3x3: lower bound is 19 (Alon-Boppana, Strassen proved 20 is achievable? No, Strassen was 7 for 2x2x2)
    # Actually for 3x3x3, the known best is 23 (Strassen 1969), and lower bound is 19
    # Let me try to find a decomposition with rank as low as possible
    
    # First, let's try to use a known construction or search
    # The schoolbook rank is 27
    # Strassen's 3x3x3 result: rank 23
    
    # Let me try a random search with structured initializations
    # We need U (9, rank), V (9, rank), W (9, rank) with entries in {0, 1/2, 1} or integers/half-integers
    
    best_rank = 27
    best_solution = None
    
    # Start with schoolbook
    U_school = [[0]*27 for _ in range(9)]
    V_school = [[0]*27 for _ in range(9)]
    W_school = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    if _verify_decomposition(U_school, V_school, W_school, n, m, p):
        best_rank = 27
        best_solution = (U_school, V_school, W_school)
    
    # Try random search with small ranks
    # Candidate values for entries
    candidate_vals = [0, 1, -1, 0.5, -0.5]
    
    for target_rank in range(19, 27):
        if _time_up(start_time, 10):
            break
        
        # Multiple random attempts per rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if _time_up(start_time, 5):
                break
            
            random.seed(attempt * 1000 + target_rank)
            
            U = [[random.choice(candidate_vals) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice(candidate_vals) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice(candidate_vals) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_solution = (U, V, W)
                break
        
        if best_rank <= target_rank:
            break
    
    # Try more structured approaches
    if best_rank > 20 and not _time_up(start_time, 10):
        # Try to find solutions with more symmetry
        for target_rank in range(19, best_rank):
            if _time_up(start_time, 5):
                break
            
            # Try sparse structured decompositions
            # Each column of U,V,W should be somewhat sparse
            for attempt in range(30):
                if _time_up(start_time, 5):
                    break
                
                random.seed(10000 + attempt * 7 + target_rank)
                
                U = [[0.0]*target_rank for _ in range(n*m)]
                V = [[0.0]*target_rank for _ in range(m*p)]
                W = [[0.0]*target_rank for _ in range(p*n)]
                
                # Make each row have a few non-zero entries
                for row in range(n*m):
                    num_nz = random.randint(1, 3)
                    cols = random.sample(range(target_rank), min(num_nz, target_rank))
                    for c in cols:
                        U[row][c] = random.choice([1, -1, 0.5, -0.5])
                
                for row in range(m*p):
                    num_nz = random.randint(1, 3)
                    cols = random.sample(range(target_rank), min(num_nz, target_rank))
                    for c in cols:
                        V[row][c] = random.choice([1, -1, 0.5, -0.5])
                
                for row in range(p*n):
                    num_nz = random.randint(1, 3)
                    cols = random.sample(range(target_rank), min(num_nz, target_rank))
                    for c in cols:
                        W[row][c] = random.choice([1, -1, 0.5, -0.5])
                
                if _verify_decomposition(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_solution = (U, V, W)
                    break
            
            if best_rank <= target_rank:
                break
    
    if best_solution is not None:
        U, V, W = best_solution
    else:
        U, V, W = U_school, V_school, W_school
    
    return {"U": U, "V": V, "W": W}

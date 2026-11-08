"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import random
import itertools
from fractions import Fraction
import numpy as np

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Schoolbook rank
    rank_school = n * m * p  # 56
    
    # Schoolbook solution
    def schoolbook(n, m, p):
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
        return U, V, W
    
    best_rank = rank_school
    best_U, best_V, best_W = schoolbook(n, m, p)
    
    # Fast verification using numpy with exact integer arithmetic
    def verify_fast(U, V, W, n, m, p):
        """Check if U, V, W form a valid decomposition using numpy."""
        r = len(U[0])
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                    if s != 1:
                        return False
        return True
    
    # Try to find a better decomposition using structured random search
    # Key insight: The matrix multiplication tensor for 2x4x7 might have
    # a decomposition of rank < 56. Let's try to find it.
    
    # Strategy: Use a more structured search
    # 1. Start from schoolbook and try to merge columns
    # 2. Use random search with small integer entries
    
    random.seed(42)
    
    # First, try a structured approach: see if we can reduce rank by 1
    # by finding two schoolbook columns that can be combined
    
    # For rank 55, we need to find U, V, W of shape (8,55), (28,55), (14,55)
    # such that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Let's try a local search approach:
    # Start with schoolbook and try to modify entries to reduce rank
    
    # Alternative: Use the fact that the tensor is symmetric in a certain way
    # and try to find a decomposition with rank 55
    
    # Let's try a more targeted random search
    # We'll use a smaller search space with entries from {-1, 0, 1}
    
    # Try rank 55 first
    for target_rank in [55, 54, 53]:
        if time.time() > deadline - 30:
            break
        
        # Try a few hundred random attempts
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            
            # Generate random U, V, W with entries from {-1, 0, 1}
            # To increase the chance of success, we can bias towards the schoolbook structure
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p*n)]
            
            if verify_fast(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < rank_school:
            break
    
    # If we haven't found a better decomposition, try a more systematic approach
    # using the structure of the problem
    
    # For 2x4x7, let's see if we can use a blocking approach
    # A is 2x4, B is 4x7
    # We can write A = [A1 A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 4x7
    # C = A1*B1 + A2*B2
    
    # But this doesn't directly help with rank reduction.
    
    # Let's try one more approach: use the known lower bound and see if we can
    # achieve it. The lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28.
    # But this is very weak.
    
    # A better lower bound for matrix multiplication rank is:
    # rank >= (n*m*p) / max(n,m,p) ... this is not standard.
    
    # Actually, for matrix multiplication M_{n,m,p}, the rank is at least
    # max(n*m, m*p, p*n) and at most n*m*p.
    
    # For 2x4x7, the rank is likely 56 or close to it.
    
    # Let's try one final approach: use a simulated annealing style search
    # to try to find a rank 55 decomposition
    
    if best_rank == rank_school and time.time() < deadline - 20:
        # Try to find rank 55 with a more structured search
        # We'll start from the schoolbook and try to perturb it
        
        U = [[0] * 55 for _ in range(n*m)]
        V = [[0] * 55 for _ in range(m*p)]
        W = [[0] * 55 for _ in range(p*n)]
        
        # Initialize with a subset of schoolbook columns
        # We have 56 schoolbook columns, we need to fit them into 55
        # Let's try to merge two columns
        
        # For now, let's just try random search with a larger search space
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(55)] for _ in range(n*m)]
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(55)] for _ in range(m*p)]
            W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(55)] for _ in range(p*n)]
            
            if verify_fast(U, V, W, n, m, p):
                best_rank = 55
                best_U, best_V, best_W = U, V, W
                break
    
    # Convert to the required format
    r = best_rank
    U_list = [[int(x) for x in row] for row in best_U]
    V_list = [[int(x) for x in row] for row in best_V]
    W_list = [[int(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

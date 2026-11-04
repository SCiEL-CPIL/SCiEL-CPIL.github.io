"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import math
from fractions import Fraction
from itertools import product
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    def build_schoolbook():
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
    
    def verify_exact(U, V, W):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # The tensor T[i, j, k] = 1 for all i, j, k (it's the identity mapping for matrix mult)
        # Actually, for matrix multiplication C = AB:
        # C[i, k] = sum_j A[i, j] * B[j, k]
        # The tensor is T[i, j, k] = 1 (delta function on j)
        # So T[i,j,k] = 1 for all valid i, j, k
        # Check: sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] == 1 for all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(len(U[i*m+j])):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if total != 1:
                        return False
        return True
    
    def verify_exact_fast(U, V, W):
        """Faster verification using numpy."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        U_np = np.array(U, dtype=float)  # shape (n*m, rank)
        V_np = np.array(V, dtype=float)  # shape (m*p, rank)
        W_np = np.array(W, dtype=float)  # shape (p*n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_np[i*m + j]
                    v_val = V_np[j*p + k]
                    w_val = W_np[k*n + i]
                    # sum_c u_val[c] * v_val[c] * w_val[c]
                    total = np.sum(u_val * v_val * w_val)
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Start with schoolbook
    best_result = build_schoolbook()
    
    # Try to find a lower rank decomposition
    # The matrix multiplication tensor for (n,m,p) has rank n*m*p for schoolbook
    # But there might be a lower rank decomposition
    
    # For small dimensions, we can try to find a decomposition
    # by solving a system of equations
    
    def try_rank(rank):
        """Try to find a decomposition of the given rank."""
        if rank >= n * m * p:
            return None
        
        n, m, p = DIMENSIONS
        # We need to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
        # such that sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] == 1 for all i, j, k
        
        # This is a non-convex problem, so we'll use random search
        # with local improvement
        
        # Initialize with random values
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Try different random seeds
        for seed in range(100):
            random.seed(seed)
            
            # Initialize with random values
            for i in range(n * m):
                for c in range(rank):
                    U[i][c] = random.uniform(-1, 1)
            for j in range(m * p):
                for c in range(rank):
                    V[j][c] = random.uniform(-1, 1)
            for k in range(p * n):
                for c in range(rank):
                    W[k][c] = random.uniform(-1, 1)
            
            # Try to improve using local search
            for iteration in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Compute the error
                U_np = np.array(U, dtype=float)
                V_np = np.array(V, dtype=float)
                W_np = np.array(W, dtype=float)
                
                # Compute the tensor
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U_np[i*m + j]
                            v_val = V_np[j*p + k]
                            w_val = W_np[k*n + i]
                            total = np.sum(u_val * v_val * w_val)
                            error += (total - 1) ** 2
                
                if error < 1e-6:
                    # Check if it's exactly 1
                    if verify_exact(U, V, W):
                        return {"U": U, "V": V, "W": W}
                
                # Try to improve one component at a time
                # Pick a random component
                comp = random.choice(['U', 'V', 'W'])
                if comp == 'U':
                    idx = random.randrange(n * m)
                elif comp == 'V':
                    idx = random.randrange(m * p)
                else:
                    idx = random.randrange(p * n)
                
                c = random.randrange(rank)
                
                # Try changing this component
                old_val = U[idx][c] if comp == 'U' else (V[idx][c] if comp == 'V' else W[idx][c])
                new_val = old_val + random.gauss(0, 0.1)
                
                if comp == 'U':
                    U[idx][c] = new_val
                elif comp == 'V':
                    V[idx][c] = new_val
                else:
                    W[idx][c] = new_val
                
                # Compute new error
                U_np = np.array(U, dtype=float)
                V_np = np.array(V, dtype=float)
                W_np = np.array(W, dtype=float)
                
                new_error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U_np[i*m + j]
                            v_val = V_np[j*p + k]
                            w_val = W_np[k*n + i]
                            total = np.sum(u_val * v_val * w_val)
                            new_error += (total - 1) ** 2
                
                if new_error < error:
                    pass  # Keep the change
                else:
                    # Revert
                    if comp == 'U':
                        U[idx][c] = old_val
                    elif comp == 'V':
                        V[idx][c] = old_val
                    else:
                        W[idx][c] = old_val
        
        return None
    
    # Try to find a lower rank decomposition
    # The schoolbook rank is n*m*p = 60
    # Try ranks from lower to higher
    for rank in range(1, n * m * p + 1):
        if time.time() > deadline - 30:
            break
        
        result = try_rank(rank)
        if result is not None:
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END

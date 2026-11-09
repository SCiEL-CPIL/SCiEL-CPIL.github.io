"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import sys
import signal
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 110.0

    # Construct the schoolbook decomposition
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Convert to numpy for efficient validation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    best_rank = schoolbook_rank
    best_U = U_np
    best_V = V_np
    best_W = W_np

    # Precompute the target tensor non-zero indices
    non_zero_indices = set()
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zero_indices.add((a, b, c))
                target_entries.append((a, b, c))

    # Function to check if a decomposition is valid (efficient version)
    def is_valid(U, V, W, tol=1e-9):
        # Check all non-zero entries
        for a, b, c in target_entries:
            val = np.dot(U[a], np.multiply(V[b], W[c]))
            if abs(val - 1.0) > tol:
                return False
        # Check a sample of zero entries
        np.random.seed(123)
        for _ in range(200):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            if (a, b, c) not in non_zero_indices:
                val = np.dot(U[a], np.multiply(V[b], W[c]))
                if abs(val) > tol:
                    return False
        return True

    # Strategy: Use structured random search with Strassen-inspired initialization
    # For 3x3x8, we can try to find a decomposition with rank < 72
    # Key insight: use the algebraic structure to guide the search
    
    # Try various ranks from low to high
    for target_rank in range(24, min(73, schoolbook_rank)):
        if time.time() > DEADLINE - 15:
            break
        
        # Multiple attempts at each rank
        for attempt in range(10):
            if time.time() > DEADLINE - 5:
                break
            
            np.random.seed(42 + target_rank * 1000 + attempt * 7)
            
            # Initialize with structured random values
            # Use values from {-1, 0, 1} or {-0.5, 0, 0.5, 1} for better chance of exact solutions
            scale = np.random.choice([0.5, 1.0, 2.0])
            U_r = np.random.randint(-2, 3, size=(n * m, target_rank)) * scale
            V_r = np.random.randint(-2, 3, size=(m * p, target_rank)) * scale
            W_r = np.random.randint(-2, 3, size=(p * n, target_rank)) * scale
            
            # Normalize to have reasonable magnitude
            U_r = U_r / np.sqrt(target_rank)
            V_r = V_r / np.sqrt(target_rank)
            W_r = W_r / np.sqrt(target_rank)
            
            if is_valid(U_r, V_r, W_r):
                best_rank = target_rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break
        
        if best_rank == target_rank:
            break

    # Convert to list of lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    new_row.append(int(round(x)))
                else:
                    if abs(2*x - round(2*x)) < 1e-9:
                        new_row.append(round(2*x)/2.0)
                    else:
                        new_row.append(round(x, 6))
            result.append(new_row)
        return result

    U_result = to_list(best_U)
    V_result = to_list(best_V)
    W_result = to_list(best_W)

    return {"U": U_result, "V": V_result, "W": W_result}

# EVOLVE-BLOCK-END

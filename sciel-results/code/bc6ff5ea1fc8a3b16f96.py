"""Rewritten program: rank-minimal exact tensor decomposition for 2x3 x 3x8 matmul using block decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    # Convert to arrays for speed if not already
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over c
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _search_decomp(n, m, p, target_rank, time_limit, seed=42):
    """
    Search for a decomposition of rank target_rank for n x m by m x p.
    Returns (U, V, W) as lists of lists of ints/floats, or None.
    """
    start_time = time.time()
    random.seed(seed)
    
    # Shapes
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # We search for integer solutions in a small range
    # If not found, we could try half-integers, but integers are more likely for small ranks
    max_val = 3
    
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        # Random initialization
        U = np.zeros((dim_u, target_rank), dtype=int)
        V = np.zeros((dim_v, target_rank), dtype=int)
        W = np.zeros((dim_w, target_rank), dtype=int)
        
        for i in range(dim_u):
            for c in range(target_rank):
                U[i, c] = random.randint(-max_val, max_val)
        for i in range(dim_v):
            for c in range(target_rank):
                V[i, c] = random.randint(-max_val, max_val)
        for i in range(dim_w):
            for c in range(target_rank):
                W[i, c] = random.randint(-max_val, max_val)
        
        # Verify
        if _verify(U, V, W, n, m, p):
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return U_list, V_list, W_list
            
        # Simple local search: perturb one element
        if attempts > 100:
             # Pick a random column and element to change
             c = random.randint(0, target_rank - 1)
             mat = random.choice([U, V, W])
             idx = random.randint(0, mat.shape[0] - 1)
             old_val = mat[idx, c]
             new_val = random.randint(-max_val, max_val)
             mat[idx, c] = new_val
             
             if _verify(U, V, W, n, m, p):
                 U_list = U.tolist()
                 V_list = V.tolist()
                 W_list = W.tolist()
                 return U_list, V_list, W_list
             else:
                 mat[idx, c] = old_val

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s margin
    
    # Strategy: Decompose B (3x8) into B1 (3x3), B2 (3x3), B3 (3x2)
    # Find decomp for (2, 3, 3) and (2, 3, 2)
    
    # Try to find rank 11 for (2, 3, 3)
    # Try to find rank 10 for (2, 3, 2)
    
    # Time allocation
    # 60s for (2, 3, 3)
    # 40s for (2, 3, 2)
    # If not found, try higher ranks or fall back
    
    d1 = None
    d2 = None
    
    # Search for (2, 3, 3)
    # Try ranks 11, 12, 13
    for r1 in [11, 12, 13]:
        if time.time() > deadline - 50:
            break
        d1 = _search_decomp(2, 3, 3, r1, time_limit=15, seed=r1*100)
        if d1:
            print(f"Found (2,3,3) decomp with rank {r1}")
            break
            
    # Search for (2, 3, 2)
    # Try ranks 10, 11, 12
    for r2 in [10, 11, 12]:
        if time.time() > deadline - 10:
            break
        d2 = _search_decomp(2, 3, 2, r2, time_limit=15, seed=r2*100)
        if d2:
            print(f"Found (2,3,2) decomp with rank {r2}")
            break

    # Construct final result
    if d1 and d2:
        U1, V1, W1 = d1
        U2, V2, W2 = d2
        
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = 2 * r1 + r2
        
        # Initialize full matrices
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Fill Block 1 (cols 0-2 of B)
        for i in range(n * m):
            for c in range(r1):
                U[i][c] = U1[i][c]
        for j in range(m):
            for k in range(3):
                v_idx = j * p + k
                v1_idx = j * 3 + k
                for c in range(r1):
                    V[v_idx][c] = V1[v1_idx][c]
        for k in range(3):
            for i in range(n):
                w_idx = k * n + i
                w1_idx = k * n + i
                for c in range(r1):
                    W[w_idx][c] = W1[w1_idx][c]
                    
        # Fill Block 2 (cols 3-5 of B)
        offset1 = r1
        for i in range(n * m):
            for c in range(r1):
                U[i][offset1 + c] = U1[i][c]
        for j in range(m):
            for k in range(3):
                v_idx = j * p + (3 + k)
                v1_idx = j * 3 + k
                for c in range(r1):
                    V[v_idx][offset1 + c] = V1[v1_idx][c]
        for k in range(3):
            for i in range(n):
                w_idx = (3 + k) * n + i
                w1_idx = k * n + i
                for c in range(r1):
                    W[w_idx][offset1 + c] = W1[w1_idx][c]
                    
        # Fill Block 3 (cols 6-7 of B)
        offset2 = 2 * r1
        for i in range(n * m):
            for c in range(r2):
                U[i][offset2 + c] = U2[i][c]
        for j in range(m):
            for k in range(2):
                v_idx = j * p + (6 + k)
                v2_idx = j * 3 + k
                for c in range(r2):
                    V[v_idx][offset2 + c] = V2[v2_idx][c]
        for k in range(2):
            for i in range(n):
                w_idx = (6 + k) * n + i
                w2_idx = k * n + i
                for c in range(r2):
                    W[w_idx][offset2 + c] = W2[w2_idx][c]
                    
        # Final verification
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            print("Verification failed for combined decomp")

    # Fallback: Schoolbook
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

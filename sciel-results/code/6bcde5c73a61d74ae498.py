"""
Program for finding a rank-minimal exact tensor decomposition for multiplying
a 3 by 4 matrix by a 4 by 4 matrix.
"""
import time
import numpy as np
import random

DIMENSIONS = [3, 4, 4]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_integer_or_half_integer(val, tol=1e-10):
    """Check if val is integer or half-integer."""
    if abs(val - round(val)) < tol:
        return True
    if abs(val - (round(val*2.0)/2.0)) < tol:
        return True
    return False

def _find_rank23_3x3x3(deadline):
    """
    Search for a rank-23 decomposition of the 3x3x3 all-ones tensor.
    Returns (U, V, W) as lists, or None if not found.
    U: (9, 23), V: (9, 23), W: (9, 23)
    Constraint: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all i,j,k in {0,1,2}
    Entries must be integer or half-integer.
    """
    n = m = p = 3
    target_rank = 23
    
    # Try random search with local refinement
    best_error = float('inf')
    best_solution = None
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Random initialization with half-integer values
        scale = random.choice([0.5, 1.0, 2.0])
        U = np.zeros((9, target_rank))
        V = np.zeros((9, target_rank))
        W = np.zeros((9, target_rank))
        
        for r in range(target_rank):
            for idx in range(9):
                # Use small half-integer values
                val = random.choice([-2, -1, -0.5, 0, 0.5, 1, 2]) * scale
                U[idx, r] = val
                V[idx, r] = val
                W[idx, r] = val
        
        # Compute error
        error = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(target_rank):
                        s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                    error += (s - 1.0) ** 2
        
        if error < 1e-10:
            # Check if entries are integer/half-integer
            valid = True
            for arr in [U, V, W]:
                for val in arr.flatten():
                    if not _is_integer_or_half_integer(val):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
        
        # Local search: try small perturbations
        for _ in range(100):
            if time.time() > deadline:
                break
            # Pick a random element and perturb it
            arr_idx = random.choice([0, 1, 2])
            idx = random.randint(0, 8)
            r = random.randint(0, target_rank - 1)
            old_val = [U, V, W][arr_idx][idx, r]
            # Try nearby half-integer values
            candidates = [old_val - 0.5, old_val + 0.5, old_val - 1.0, old_val + 1.0]
            candidates = [c for c in candidates if _is_integer_or_half_integer(c)]
            if not candidates:
                continue
            new_val = random.choice(candidates)
            
            # Compute new error
            new_error = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        s = 0.0
                        for rr in range(target_rank):
                            u_val = U[i*3+j, rr]
                            v_val = V[j*3+k, rr]
                            w_val = W[k*3+i, rr]
                            if arr_idx == 0 and idx == i*3+j and rr == r:
                                u_val = new_val
                            elif arr_idx == 1 and idx == j*3+k and rr == r:
                                v_val = new_val
                            elif arr_idx == 2 and idx == k*3+i and rr == r:
                                w_val = new_val
                            s += u_val * v_val * w_val
                        new_error += (s - 1.0) ** 2
                        if new_error > error + 0.1:
                            break
                    if new_error > error + 0.1:
                        break
                if new_error > error + 0.1:
                    break
            
            if new_error < error:
                [U, V, W][arr_idx][idx, r] = new_val
                error = new_error
                if error < 1e-10:
                    valid = True
                    for arr in [U, V, W]:
                        for val in arr.flatten():
                            if not _is_integer_or_half_integer(val):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120s limit
    
    # Fallback: Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Strategy: Combine rank-23 3x3x3 decomposition with rank-1 parts
    # Split: C = A_3*B_3 + A_3*B_31 + A_1*B_1
    # where A_3 is 3x3, A_1 is 3x1, B_3 is 3x3, B_31 is 3x1, B_1 is 1x4
    # Part 1: 3x3x3 (rank 23) - indices i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
    # Part 2: 3x3x1 (rank 1)  - indices i in {0,1,2}, j in {0,1,2}, k=3
    # Part 3: 3x1x4 (rank 1)  - indices i in {0,1,2}, j=3, k in {0,1,2,3}
    # Total rank: 25
    
    # First, try to find the rank-23 3x3x3 decomposition
    result_23 = None
    if time.time() < deadline - 5:
        result_23 = _find_rank23_3x3x3(deadline - 5)
    
    if result_23 is not None:
        U23, V23, W23 = result_23
        # Build the full 3x4x4 decomposition with rank 25
        total_rank = 25
        U_full = [[0] * total_rank for _ in range(n * m)]  # (12, 25)
        V_full = [[0] * total_rank for _ in range(m * p)]  # (16, 25)
        W_full = [[0] * total_rank for _ in range(p * n)]  # (12, 25)
        
        # Copy the 3x3x3 part (rank 23) into columns 0-22
        # For i in {0,1,2}, j in {0,1,2}: U[i*4+j, r] = U23[i*3+j, r] for r in 0..22
        for i in range(3):
            for j in range(3):
                for r in range(23):
                    U_full[i*4+j][r] = U23[i*3+j][r]
        
        # For j in {0,1,2}, k in {0,1,2}: V[j*4+k, r] = V23[j*3+k, r] for r in 0..22
        for j in range(3):
            for k in range(3):
                for r in range(23):
                    V_full[j*4+k][r] = V23[j*3+k][r]
        
        # For k in {0,1,2}, i in {0,1,2}: W[k*3+i, r] = W23[k*3+i, r] for r in 0..22
        # Note: W index is k*n+i = k*3+i
        for k in range(3):
            for i in range(3):
                for r in range(23):
                    W_full[k*3+i][r] = W23[k*3+i][r]
        
        # Part 2: 3x3x1 rank-1 term at column 23
        # For i in {0,1,2}, j in {0,1,2}, k=3: U[i*4+j, 23]=1, V[j*4+3, 23]=1, W[3*3+i, 23]=1
        for i in range(3):
            for j in range(3):
                U_full[i*4+j][23] = 1
                V_full[j*4+3][23] = 1
                W_full[3*3+i][23] = 1
        
        # Part 3: 3x1x4 rank-1 term at column 24
        # For i in {0,1,2}, j=3, k in {0,1,2,3}: U[i*4+3, 24]=1, V[3*4+k, 24]=1, W[k*3+i, 24]=1
        for i in range(3):
            for k in range(4):
                U_full[i*4+3][24] = 1
                V_full[3*4+k][24] = 1
                W_full[k*3+i][24] = 1
        
        # Verify the decomposition
        if _verify_decomposition(U_full, V_full, W_full, n, m, p):
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))

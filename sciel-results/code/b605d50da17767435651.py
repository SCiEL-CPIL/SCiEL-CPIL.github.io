import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Schoolbook fallback: rank = n*m*p = 56
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

    # Try to find a lower rank decomposition using structured search
    # Theoretical lower bound is 28
    
    # Strategy: Use a structured approach based on the algebraic structure
    # For 2x4 x 4x7, we can exploit the small size of n=2
    
    # Approach 1: Try rank 28 with a systematic construction
    # The key insight: for each (j,k), we need to handle the 2-dimensional i-index
    # This suggests using 2x2 matrix multiplication structure
    
    # Let's try a construction based on splitting into blocks
    # A (2x4) * B (4x7) = C (2x7)
    # Split B into two 2x7 blocks: B = [B_top; B_bot]
    # C = A_top_block * B_top + A_bot_block * B_bot
    # where A_top_block and A_bot_block are 2x2
    
    # Actually, let's use a more direct approach:
    # For rank r, we need to assign values such that the tensor equation holds
    
    # Try multiple ranks from 28 upward
    for target_rank in [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]:
        if time.time() > deadline - 10:
            break
        
        result = find_rank_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            return result
    
    return {"U": U_school, "V": V_school, "W": W_school}

def find_rank_decomposition(n, m, p, rank, deadline):
    """
    Find a rank-rank decomposition for n x m by m x p matrix multiplication.
    Uses a structured approach with local optimization.
    """
    
    # Precompute all (i,j,k) tuples
    tuples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_tuples = len(tuples)  # 56
    
    # For efficient error computation, precompute which (i,j,k) each entry affects
    # U[i*m+j, r] affects all (i, j, k) for all k
    # V[j*p+k, r] affects all (i, j, k) for all i
    # W[k*n+i, r] affects all (i, j, k) for all j
    
    def compute_partial_sum(U, V, W, i, j, k):
        """Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
        total = 0
        for r in range(rank):
            total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        return total
    
    def compute_error(U, V, W):
        """Compute total squared error"""
        error = 0
        for (i, j, k) in tuples:
            val = compute_partial_sum(U, V, W, i, j, k)
            error += (val - 1) ** 2
        return error
    
    # Initialize with a structured pattern
    # Use a pattern based on the algebraic structure
    # For n=2, we can use the fact that each (j,k) pair has 2 i-values
    
    # Try multiple random initializations
    best_result = None
    best_error = float('inf')
    
    num_attempts = 5
    for attempt in range(num_attempts):
        if time.time() > deadline - 15:
            break
        
        # Initialize with small random values
        U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)]
        
        # Local search with gradient-like updates
        current_error = compute_error(U, V, W)
        if current_error < best_error:
            best_error = current_error
            if current_error < 1e-10:
                # Verify exactness
                if verify_exact(U, V, W, n, m, p, rank):
                    return {"U": U, "V": V, "W": W}
                best_result = {"U": U, "V": V, "W": W}
        
        # Perform local optimization
        max_iterations = 200
        for it in range(max_iterations):
            if time.time() > deadline - 15:
                break
            
            # Pick a random element to update
            mat = random.choice(["U", "V", "W"])
            
            if mat == "U":
                idx = random.randint(0, n*m-1)
                r = random.randint(0, rank-1)
                i, j = divmod(idx, m)
                
                # Try small perturbations
                old_val = U[idx][r]
                for delta in [-0.5, -0.1, -0.01, 0.01, 0.1, 0.5]:
                    U[idx][r] = old_val + delta
                    new_error = compute_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        if current_error < best_error:
                            best_error = current_error
                            if current_error < 1e-10:
                                if verify_exact(U, V, W, n, m, p, rank):
                                    return {"U": U, "V": V, "W": W}
                                best_result = {"U": U, "V": V, "W": W}
                        break
                    else:
                        U[idx][r] = old_val
            elif mat == "V":
                idx = random.randint(0, m*p-1)
                r = random.randint(0, rank-1)
                j, k = divmod(idx, p)
                
                old_val = V[idx][r]
                for delta in [-0.5, -0.1, -0.01, 0.01, 0.1, 0.5]:
                    V[idx][r] = old_val + delta
                    new_error = compute_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        if current_error < best_error:
                            best_error = current_error
                            if current_error < 1e-10:
                                if verify_exact(U, V, W, n, m, p, rank):
                                    return {"U": U, "V": V, "W": W}
                                best_result = {"U": U, "V": V, "W": W}
                        break
                    else:
                        V[idx][r] = old_val
            else:
                idx = random.randint(0, p*n-1)
                r = random.randint(0, rank-1)
                k, i = divmod(idx, n)
                
                old_val = W[idx][r]
                for delta in [-0.5, -0.1, -0.01, 0.01, 0.1, 0.5]:
                    W[idx][r] = old_val + delta
                    new_error = compute_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        if current_error < best_error:
                            best_error = current_error
                            if current_error < 1e-10:
                                if verify_exact(U, V, W, n, m, p, rank):
                                    return {"U": U, "V": V, "W": W}
                                best_result = {"U": U, "V": V, "W": W}
                        break
                    else:
                        W[idx][r] = old_val
        
        # After local search, try to snap to integer/half-integer values
        if current_error < 1.0:
            U_snapped = [[round(U[i][r] * 2) / 2 for r in range(rank)] for i in range(n*m)]
            V_snapped = [[round(V[i][r] * 2) / 2 for r in range(rank)] for i in range(m*p)]
            W_snapped = [[round(W[i][r] * 2) / 2 for r in range(rank)] for i in range(p*n)]
            
            if verify_exact(U_snapped, V_snapped, W_snapped, n, m, p, rank):
                return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
    
    return None

def verify_exact(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact (all values are exactly 1)"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if total != 1:
                    return False
    return True

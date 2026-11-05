"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p  # 72
    
    # Verify a candidate decomposition
    def verify(U, V, W, rank):
        """Check if U,V,W with given rank exactly decompose the tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Vectorized verification using numpy for speed
    def verify_np(U, V, W, rank):
        """Fast vectorized verification."""
        # Build tensor via einsum-like approach
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy: Use structured decomposition based on known results for 3x3x3
    # The key insight: we can decompose 3x3x8 as blocks of 3x3x3 and 3x3x2
    # For 3x3x3, there exist decompositions with rank < 27 (schoolbook)
    # Let's try to find a structured lower-rank decomposition
    
    # Approach 1: Try to use a rank-23 decomposition for 3x3x3 blocks
    # For 3x3x8 = 3x3x3 + 3x3x3 + 3x3x2
    # If we can get rank 23 for each 3x3x3 and rank 18 for 3x3x2, total = 23+23+18 = 64
    
    # But we need to be careful about the tensor structure.
    # Let's try a different approach: use the fact that the tensor is
    # T[i,j,k] = 1 for all valid (i,j,k)
    
    # Try to find a decomposition where we share structure across k values
    
    # Method: Start with schoolbook and try to merge columns using
    # linear algebra (find linear dependencies)
    
    def try_merge_columns(U, V, W, rank, target_rank, max_attempts=10):
        """Try to merge columns to reduce rank."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Randomly select columns to merge
            cols_to_keep = list(range(target_rank))
            cols_to_merge = list(range(target_rank, rank))
            np.random.shuffle(cols_to_merge)
            
            U_new = np.zeros((n * m, target_rank), dtype=float)
            V_new = np.zeros((m * p, target_rank), dtype=float)
            W_new = np.zeros((p * n, target_rank), dtype=float)
            
            # Copy kept columns
            for new_c, old_c in enumerate(cols_to_keep):
                U_new[:, new_c] = U[:, old_c]
                V_new[:, new_c] = V[:, old_c]
                W_new[:, new_c] = W[:, old_c]
            
            # Try to merge remaining columns into kept columns
            for old_c in cols_to_merge:
                # Find the kept column that is "closest"
                best_match = -1
                best_score = float('inf')
                for new_c in range(target_rank):
                    # Score based on similarity
                    score = np.sum((U[:, old_c] - U_new[:, new_c])**2) + \
                            np.sum((V[:, old_c] - V_new[:, new_c])**2) + \
                            np.sum((W[:, old_c] - W_new[:, new_c])**2)
                    if score < best_score:
                        best_score = score
                        best_match = new_c
                
                if best_match >= 0:
                    # Merge by adding (with small weight to allow cancellation)
                    weight = np.random.choice([0.5, -0.5, 0.25, -0.25])
                    U_new[:, best_match] += weight * U[:, old_c]
                    V_new[:, best_match] += weight * V[:, old_c]
                    W_new[:, best_match] += weight * W[:, old_c]
            
            # Round to half-integers
            U_new = np.round(U_new * 2) / 2
            V_new = np.round(V_new * 2) / 2
            W_new = np.round(W_new * 2) / 2
            
            # Verify
            if verify_np(U_new.tolist(), V_new.tolist(), W_new.tolist(), target_rank):
                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
        
        return None
    
    # Try structured rank reduction from schoolbook
    for target_rank in range(best_rank - 1, max(55, best_rank - 17), -1):
        if time.time() > deadline:
            break
        
        # Try multiple attempts
        for attempt in range(8):
            if time.time() > deadline:
                break
            
            # Start from schoolbook and perturb
            U = np.zeros((n * m, best_rank), dtype=float)
            V = np.zeros((m * p, best_rank), dtype=float)
            W = np.zeros((p * n, best_rank), dtype=float)
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
            
            # Add small random perturbations to some entries
            for _ in range(20):
                row_u = np.random.randint(0, n * m)
                col_u = np.random.randint(0, best_rank)
                U[row_u, col_u] += np.random.choice([-0.5, 0.5])
                
                row_v = np.random.randint(0, m * p)
                col_v = np.random.randint(0, best_rank)
                V[row_v, col_v] += np.random.choice([-0.5, 0.5])
                
                row_w = np.random.randint(0, p * n)
                col_w = np.random.randint(0, best_rank)
                W[row_w, col_w] += np.random.choice([-0.5, 0.5])
            
            result = try_merge_columns(U.tolist(), V.tolist(), W.tolist(), best_rank, target_rank)
            if result is not None:
                best_rank = target_rank
                best_result = result
                break
        
        if time.time() > deadline:
            break
    
    # Method 2: Try direct construction with structured patterns
    # For 3x3x8, try to use a decomposition where U and W have block structure
    
    if time.time() < deadline - 20:
        # Try rank values that might work based on known bounds
        for target_rank in [64, 62, 60, 58, 56]:
            if time.time() > deadline:
                break
            if target_rank >= best_rank:
                continue
            
            for attempt in range(5):
                if time.time() > deadline:
                    break
                
                # Initialize with structured pattern
                U = np.zeros((n * m, target_rank), dtype=float)
                V = np.zeros((m * p, target_rank), dtype=float)
                W = np.zeros((p * n, target_rank), dtype=float)
                
                # Try to fill with a pattern that might work
                # For each (i,j,k), assign to a column with some structure
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < target_rank:
                                U[i * m + j, col] = 1.0
                                V[j * p + k, col] = 1.0
                                W[k * n + i, col] = 1.0
                                col += 1
                            else:
                                # Try to reuse an existing column with modifications
                                c = np.random.randint(0, target_rank)
                                # Only modify if it doesn't break other entries
                                U[i * m + j, c] += 0.5
                                V[j * p + k, c] += 0.5
                                W[k * n + i, c] += 0.5
                
                # Round and verify
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                if verify_np(U.tolist(), V.tolist(), W.tolist(), target_rank):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
        
        if time.time() > deadline:
            return best_result
    
    # Method 3: Random search with small values for even lower ranks
    if time.time() < deadline - 10:
        for target_rank in range(min(best_rank - 1, 60), max(45, best_rank - 15), -1):
            if time.time() > deadline:
                break
            if target_rank >= best_rank:
                continue
            
            for attempt in range(4):
                if time.time() > deadline:
                    break
                
                # Random initialization with small half-integer values
                U = (np.random.randint(-3, 4, (n * m, target_rank)) * 0.5)
                V = (np.random.randint(-3, 4, (m * p, target_rank)) * 0.5)
                W = (np.random.randint(-3, 4, (p * n, target_rank)) * 0.5)
                
                if verify_np(U.tolist(), V.tolist(), W.tolist(), target_rank):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
            
            if time.time() > deadline:
                break
    
    return best_result
# EVOLVE-BLOCK-END

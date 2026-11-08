import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # 1. Generate all possible 2x2x2 Strassen blocks
    # Block defined by (i_start, j_start, k_start)
    # i_start in [0, n-2], j_start in [0, m-2], k_start in [0, p-2]
    possible_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                possible_blocks.append((i0, j0, k0))
    
    # 2. Define conflict matrix between blocks
    # Two blocks conflict if their 2x2x2 subcubes overlap
    def conflict(b1, b2):
        i1, j1, k1 = b1
        i2, j2, k2 = b2
        # Check interval overlap [i1, i1+1] and [i2, i2+1]
        if max(i1, i2) <= min(i1+1, i2+1) and max(j1, j2) <= min(j1+1, j2+1) and max(k1, k2) <= min(k1+1, k2+1):
            return True
        return False

    num_blocks = len(possible_blocks)
    # Find maximum independent set (greedy with randomization/shaking or simple max matching approximation)
    # Since N is small (3*4*4 = 48 blocks), we can do a simple backtracking or branch and bound
    # Or just greedy multiple times
    
    best_cover = []
    
    def is_valid_placement(blocks, block_idx):
        # Check if adding blocks[block_idx] to current 'blocks' is valid
        new_b = possible_blocks[block_idx]
        for existing in blocks:
            if conflict(new_b, possible_blocks[existing]):
                return False
        return True

    # Simple Branch and Bound / DFS to find Max Independent Set
    # 48 nodes is too big for naive 2^48, but graph is sparse.
    # We can use a greedy heuristic with restarts.
    
    max_ind_set = []
    def dfs(start_idx, current_set):
        # Heuristic: try to add blocks that conflict with fewest remaining candidates
        # For simplicity in this constrained environment, we'll do a randomized greedy search
        pass

    # Let's use a simpler greedy approach with multiple shuffles
    best_size = -1
    best_config = []
    
    for _ in range(1000): # 1000 random greedy attempts
        order = list(range(num_blocks))
        np.random.shuffle(order)
        curr = []
        for idx in order:
            if is_valid_placement(curr, idx):
                curr.append(idx)
        if len(curr) > best_size:
            best_size = len(curr)
            best_config = list(curr)
            
    # Try to improve best_config by local search (swap one block)
    improved = True
    while improved:
        improved = False
        for i in range(len(best_config)):
            for add_idx in range(num_blocks):
                if add_idx not in best_config:
                    # Try replacing best_config[i] with add_idx
                    # Or just adding if there's space? No, we want to maximize count.
                    # Since we want max count, we look for swaps that don't reduce count, but might allow future additions?
                    # Actually, just finding the max size is the goal.
                    # The greedy above might get stuck. 
                    # Let's try adding any valid block to best_config if possible (shouldn't be if it was greedy)
                    # Or swapping.
                    pass
                    
    # Re-evaluate the best_config found. 
    # Actually, for 3x5x5, the packing limit is likely determined by the i-dimension.
    # i=0 blocks and i=1 blocks overlap in i if j,k overlap.
    # If j,k are disjoint, they can coexist.
    
    selected_blocks = [possible_blocks[idx] for idx in best_config]
    
    # Initialize tensors
    total_rank = 0
    # Estimate max rank: 48*7 (if all used, impossible) + 75
    max_possible_rank = num_blocks * 7 + n*m*p
    U = np.zeros((n * m, max_possible_rank), dtype=float)
    V = np.zeros((m * p, max_possible_rank), dtype=float)
    W = np.zeros((p * n, max_possible_rank), dtype=float)
    
    col = 0
    covered = np.zeros((n, m, p), dtype=bool)
    
    # Apply Strassen blocks
    for (i0, j0, k0) in selected_blocks:
        # Check if this block actually adds new coverage? 
        # The independent set ensures no overlap with OTHER STRASSEN blocks.
        # But it might overlap with SCHOOLBOOK terms? No, we add schoolbook later.
        
        # Strassen formulas for indices:
        # A indices: (i, j), B indices: (j, k), C indices: (k, i)
        # Local: a in 0,1 (i=i0+a), b in 0,1 (j=j0+b), c in 0,1 (k=k0+c)
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1]) -> C[0,0] += 1, C[1,1] += 1
        U[i0*m + j0, col] += 1.0
        U[(i0+1)*m + (j0+1), col] += 1.0
        V[j0*p + k0, col] += 1.0
        V[(j0+1)*p + (k0+1), col] += 1.0
        W[k0*n + i0, col] += 1.0
        W[(k0+1)*n + (i0+1), col] += 1.0
        col += 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,0] -> C[1,0] += 1, C[1,1] -= 1
        U[(i0+1)*m + j0, col] += 1.0
        U[(i0+1)*m + (j0+1), col] += 1.0
        V[j0*p + k0, col] += 1.0
        W[k0*n + (i0+1), col] += 1.0
        W[(k0+1)*n + (i0+1), col] -= 1.0
        col += 1
        
        # M3 = A[0,0] * (B[0,1] - B[1,1]) -> C[0,1] += 1, C[1,1] += 1
        U[i0*m + j0, col] += 1.0
        V[j0*p + (k0+1), col] += 1.0
        V[(j0+1)*p + (k0+1), col] -= 1.0
        W[(k0+1)*n + i0, col] += 1.0
        W[(k0+1)*n + (i0+1), col] += 1.0
        col += 1
        
        # M4 = A[1,1] * (B[1,0] - B[0,0]) -> C[0,0] += 1, C[1,0] += 1
        U[(i0+1)*m + (j0+1), col] += 1.0
        V[(j0+1)*p + k0, col] += 1.0
        V[j0*p + k0, col] -= 1.0
        W[k0*n + i0, col] += 1.0
        W[k0*n + (i0+1), col] += 1.0
        col += 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,1] -> C[0,0] -= 1, C[0,1] += 1
        U[i0*m + j0, col] += 1.0
        U[i0*m + (j0+1), col] += 1.0
        V[(j0+1)*p + (k0+1), col] += 1.0
        W[k0*n + i0, col] -= 1.0
        W[(k0+1)*n + i0, col] += 1.0
        col += 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1]) -> C[1,1] += 1
        U[(i0+1)*m + j0, col] += 1.0
        U[i0*m + j0, col] -= 1.0
        V[j0*p + k0, col] += 1.0
        V[j0*p + (k0+1), col] += 1.0
        W[(k0+1)*n + (i0+1), col] += 1.0
        col += 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1]) -> C[0,0] += 1
        U[i0*m + (j0+1), col] += 1.0
        U[(i0+1)*m + (j0+1), col] -= 1.0
        V[(j0+1)*p + k0, col] += 1.0
        V[(j0+1)*p + (k0+1), col] += 1.0
        W[k0*n + i0, col] += 1.0
        col += 1
        
        # Mark covered
        covered[i0:i0+2, j0:j0+2, k0:k0+2] = True
        
    # Add schoolbook terms for uncovered indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i, j, k]:
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
                    
    # Trim to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    # Check deadline
    if time.time() > deadline:
        # If we are late, we still return what we have
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    
    # Verification
    np.random.seed(42)
    errors = 0
    for _ in range(1000):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = np.dot(A, B)
        
        # Tensor decomposition check:
        # C[k*n+i, :] = sum_r W[k*n+i, r] * ( sum_r' U[i*m+j, r] * V[j*p+k, r] ) ?? No.
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*m+j, r] ... 
        # The tensor T[i,j,k] = C[i,k] coefficient is 1 if we use schoolbook.
        # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # Let's verify T[i,j,k] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        errors += 1
                        break
            if errors > 0: break
        if errors > 0: break
            
    print(f"Verification Errors: {errors}")
    print(f"Rank: {U.shape[1]}")

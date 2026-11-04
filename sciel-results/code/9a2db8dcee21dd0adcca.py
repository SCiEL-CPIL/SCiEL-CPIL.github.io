import time
import numpy as np
import random
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    total_terms = n * m * p
    
    # Precompute Strassen coefficients for a 2x2x2 block
    # Local indices: i,j,k in {0,1}
    # U index: i*m + j (local: i*2 + j)
    # V index: j*p + k (local: j*2 + k)
    # W index: k*n + i (local: k*2 + i)
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # M2: (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[1, 1] = 1; W_s[3, 1] = -1
    
    # M3: A00(B01-B11)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[2, 2] = 1; W_s[3, 2] = 1
    
    # M4: A11(B10-B00)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[1, 3] = 1
    
    # M5: (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1

    # Generate all possible 2x2x2 blocks
    blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                terms = frozenset((i0 + di, j0 + dj, k0 + dk) 
                                  for di in range(2) 
                                  for dj in range(2) 
                                  for dk in range(2))
                blocks.append((i0, j0, k0, terms))
    
    num_blocks_total = len(blocks)
    
    # Precompute conflict matrix
    # conflict[i][j] is True if block i and block j overlap
    conflict = [[False] * num_blocks_total for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if blocks[i][3] & blocks[j][3]:
                conflict[i][j] = True
                conflict[j][i] = True

    def is_valid_set(block_indices):
        for i in range(len(block_indices)):
            for j in range(i + 1, len(block_indices)):
                if conflict[block_indices[i]][block_indices[j]]:
                    return False
        return True

    def greedy_init():
        # Sort blocks by degree (number of conflicts) to pick "least conflicting" first?
        # Or just random order.
        indices = list(range(num_blocks_total))
        random.shuffle(indices)
        selected = []
        for idx in indices:
            if all(not conflict[idx][s] for s in selected):
                selected.append(idx)
        return selected

    def local_search(initial_set, max_iters=1000):
        current_set = set(initial_set)
        current_score = len(current_set)
        best_set = set(current_set)
        best_score = current_score
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Try to add a block
            added = False
            candidates = [i for i in range(num_blocks_total) if i not in current_set]
            random.shuffle(candidates)
            for idx in candidates:
                if all(not conflict[idx][s] for s in current_set):
                    current_set.add(idx)
                    added = True
                    break
            
            if added:
                current_score += 1
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
                continue
            
            # Try to swap: remove 1, add 1 or more
            # Heuristic: Try removing a block that allows adding multiple others
            # For simplicity, try swapping 1 for 1 first, then 1 for 2
            # Let's try a simple random swap: remove one, try to add one different one
            if current_set:
                remove_candidates = list(current_set)
                random.shuffle(remove_candidates)
                for rem_idx in remove_candidates:
                    # Temporarily remove
                    temp_set = current_set - {rem_idx}
                    # Try to add a block that was previously conflicting with rem_idx
                    # or just any block not in temp_set
                    add_candidates = [i for i in range(num_blocks_total) if i not in temp_set]
                    random.shuffle(add_candidates)
                    for add_idx in add_candidates:
                        if all(not conflict[add_idx][s] for s in temp_set):
                            # This swap is valid and maintains size (or increases if we could add more)
                            # Actually, we removed 1 and added 1, so size is same.
                            # We need to find if removing 1 allows adding >1.
                            pass
                    
                    # Let's try a more aggressive move: Remove 1, then greedily add as many as possible
                    temp_set = current_set - {rem_idx}
                    new_adds = []
                    remaining_candidates = [i for i in range(num_blocks_total) if i not in temp_set]
                    random.shuffle(remaining_candidates)
                    for idx in remaining_candidates:
                        if all(not conflict[idx][s] for s in temp_set):
                            temp_set.add(idx)
                            new_adds.append(idx)
                            
                    if len(new_adds) > 1: # Net gain
                        current_set = temp_set
                        current_score += len(new_adds) - 1
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(current_set)
                        break
        
        return best_set

    best_blocks_indices = []
    best_count = 0
    
    # Run multiple restarts
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        init_set = greedy_init()
        improved_set = local_search(init_set, max_iters=200)
        
        if len(improved_set) > best_count:
            best_count = len(improved_set)
            best_blocks_indices = list(improved_set)
            
            # Early exit if we hit the theoretical max
            # Max blocks is floor(105/8) = 13
            if best_count >= 13:
                break

    # Construct U, V, W
    num_blocks = len(best_blocks_indices)
    rank = total_terms - num_blocks
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    covered = set()
    
    # Add Strassen blocks
    for idx in best_blocks_indices:
        i0, j0, k0, terms = blocks[idx]
        
        for t in range(7):
            # U contributions
            for li in range(2):
                for lj in range(2):
                    lu = li * 2 + lj
                    val = U_s[lu, t]
                    if val != 0:
                        gu = (i0 + li) * m + (j0 + lj)
                        U[gu, col] += val
                        
            # V contributions
            for lj in range(2):
                for lk in range(2):
                    lv = lj * 2 + lk
                    val = V_s[lv, t]
                    if val != 0:
                        gv = (j0 + lj) * p + (k0 + lk)
                        V[gv, col] += val
                        
            # W contributions
            for lk in range(2):
                for li in range(2):
                    lw = lk * 2 + li
                    val = W_s[lw, t]
                    if val != 0:
                        gw = (k0 + lk) * n + (i0 + li)
                        W[gw, col] += val
                        
            col += 1
        covered |= terms
        
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    
    # Trim if necessary
    if col < rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        rank = col
        
    # Verify
    def verify(U, V, W):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            T += np.einsum('ij,jk,ki->ijk', U_r[:, :, c], V_r[:, :, c], W_r[:, :, c])
            
        return np.allclose(T, 1.0, atol=1e-9)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        U_sb = np.zeros((n * m, total_terms), dtype=float)
        V_sb = np.zeros((m * p, total_terms), dtype=float)
        W_sb = np.zeros((p * n, total_terms), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END

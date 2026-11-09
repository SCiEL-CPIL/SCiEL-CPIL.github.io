import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Generate all space words
    # q=3, n=6 -> 729 words
    space_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    Q = space_words.shape[0]
    
    # Precompute distance from each space word to all other space words?
    # dist_space[i, j] = hamming distance between space_words[i] and space_words[j]
    # This is (Q, Q) = (729, 729). 729*729*6 ~ 3M ops. Fast.
    # Then dist_matrix for a code is just dist_space[code_indices, :]
    dist_space = (space_words[:, None, :] != space_words[None, :, :]).sum(axis=2)
    
    # 2. Initialize with Tetracode direct sum (81 words)
    # Tetracode: 9 words of length 4
    tetracode = np.array([[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)], dtype=np.int8)
    # Direct sum with 2 extra coordinates (since n=6, 6-4=2)
    tails = np.array(list(itertools.product(range(q), repeat=n-4)), dtype=np.int8)
    
    # Construct initial code: 9 * 9 = 81 words
    # We need to combine each tetracode head with each tail
    heads_expanded = np.repeat(tetracode, tails.shape[0], axis=0)
    tails_expanded = np.tile(tails, (tetracode.shape[0], 1))
    code = np.hstack([heads_expanded, tails_expanded])
    
    # Map code words to space indices for fast distance lookup
    # Create a dictionary mapping tuple(word) -> index in space_words
    space_index_map = {}
    for i, w in enumerate(space_words):
        space_index_map[tuple(w)] = i
    
    code_indices = np.array([space_index_map[tuple(cw)] for cw in code], dtype=np.int32)
    M = code.shape[0]
    
    # dist_matrix: (M, Q)
    # dist_matrix[i, j] = distance from code[i] to space_words[j]
    # = dist_space[code_indices[i], j]
    dist_matrix = dist_space[code_indices, :]
    
    def is_valid_removal(dist_mat, k):
        # Check if removing codeword k leaves all words covered
        covered_by_k = dist_mat[k] <= radius
        
        if not np.any(covered_by_k):
            return True
        
        # Efficient check:
        # We need to know for words covered by k, is there another codeword covering them?
        # min_excl_k = min_{j != k} dist_mat[j, :]
        
        # Optimization: 
        # min_all = np.min(dist_mat, axis=0)
        # argmin_all = np.argmin(dist_mat, axis=0)
        # If argmin_all[w] == k, we need the second minimum.
        # Otherwise, min_all[w] is the min excluding k (since k is not the unique min, or not the min at all).
        # Actually, if argmin_all[w] != k, then min_{j!=k} dist_mat[j, w] <= min_all[w] <= radius?
        # Wait. If argmin_all[w] != k, it means some j != k has distance min_all[w].
        # Since the code is valid, min_all[w] <= radius.
        # So if argmin_all[w] != k, then min_{j!=k} dist_mat[j, w] = min_all[w] <= radius. Safe.
        # Only if argmin_all[w] == k, we need to check if there is another codeword with distance <= radius.
        # Specifically, we need min_{j != k} dist_mat[j, w] <= radius.
        # This is equivalent to: is there any j != k with dist_mat[j, w] <= radius?
        
        # Let's compute min_excl_k efficiently.
        # dist_mat_temp = dist_mat.copy()
        # dist_mat_temp[k, :] = np.inf
        # min_excl_k = np.min(dist_mat_temp, axis=0)
        
        # This copy is O(M*Q). M~80, Q=729. 58k ops. Fast enough.
        dist_mat_temp = dist_mat.copy()
        dist_mat_temp[k, :] = np.inf
        min_excl_k = np.min(dist_mat_temp, axis=0)
        
        critical_words = dist_mat[k] <= radius
        if np.any(critical_words):
            if np.any(min_excl_k[critical_words] > radius):
                return False
        return True

    def is_valid_swap(dist_mat, k, new_code_idx):
        # Check if replacing codeword k with space_words[new_code_idx] results in a valid cover
        
        # Identify critical words for k:
        # Words covered by k (dist_mat[k, w] <= r) but NOT covered by any j != k.
        
        dist_mat_temp = dist_mat.copy()
        dist_mat_temp[k, :] = np.inf
        min_excl_k = np.min(dist_mat_temp, axis=0)
        
        critical_mask = (dist_mat[k] <= radius) & (min_excl_k > radius)
        
        if not np.any(critical_mask):
            return True
        
        # Check if new codeword covers all critical words
        # dist_new = dist_space[new_code_idx, :]
        # return np.all(dist_new[critical_mask] <= radius)
        
        dist_new = dist_space[new_code_idx, :]
        return np.all(dist_new[critical_mask] <= radius)

    # Local Search
    best_code = code.copy()
    best_M = M
    stuck_count = 0
    
    while time.time() < deadline:
        removed_any = False
        M = code.shape[0]
        
        # Try to remove words
        for k in range(M - 1, -1, -1):
            if time.time() > deadline:
                break
            
            if is_valid_removal(dist_matrix, k):
                # Remove k
                code = np.delete(code, k, axis=0)
                code_indices = np.delete(code_indices, k, axis=0)
                dist_matrix = np.delete(dist_matrix, k, axis=0)
                M -= 1
                removed_any = True
                break
        
        if removed_any:
            if M < best_M:
                best_M = M
                best_code = code.copy()
            stuck_count = 0
            continue
        
        # If no removal, try swaps
        swap_found = False
        # Try moving each codeword to a random new position
        # We can be smarter: try to move to a word that is "far" from current or random.
        # Random is fine.
        for k in range(M):
            if time.time() > deadline:
                break
            
            # Generate a few random candidates for new position
            # Use a set to avoid trying the same index multiple times
            tried_indices = set()
            for _ in range(20):
                if time.time() > deadline:
                    break
                new_idx = random.randint(0, Q - 1)
                if new_idx in tried_indices:
                    continue
                tried_indices.add(new_idx)
                
                if is_valid_swap(dist_matrix, k, new_idx):
                    # Perform swap
                    code[k] = space_words[new_idx]
                    code_indices[k] = new_idx
                    dist_matrix[k, :] = dist_space[new_idx, :]
                    swap_found = True
                    break
            
            if swap_found:
                break
        
        if swap_found:
            stuck_count = 0
            # After a swap, we might be able to remove more, so restart loop
            continue
        
        # Stuck in local minimum
        stuck_count += 1
        
        if stuck_count > 10:
            # Try a random restart? Or kick?
            # Let's try to break the local minimum by adding a random word and then removing.
            # This increases M temporarily.
            # Pick a random word not in code.
            # Then try to remove any word.
            
            # 1. Add a random word
            new_idx = random.randint(0, Q - 1)
            # Check if it's already in code?
            # If it is, it's a no-op swap essentially.
            # Let's just add it.
            code = np.vstack([code, space_words[new_idx]])
            code_indices = np.append(code_indices, new_idx)
            dist_matrix = np.vstack([dist_matrix, dist_space[new_idx, :]])
            M += 1
            
            # 2. Try to remove a word
            removed = False
            for k in range(M - 1, -1, -1):
                if time.time() > deadline:
                    break
                if is_valid_removal(dist_matrix, k):
                    code = np.delete(code, k, axis=0)
                    code_indices = np.delete(code_indices, k, axis=0)
                    dist_matrix = np.delete(dist_matrix, k, axis=0)
                    M -= 1
                    removed = True
                    break
            
            if removed:
                if M < best_M:
                    best_M = M
                    best_code = code.copy()
                stuck_count = 0
                continue
            else:
                # Could not remove, revert?
                # Or just accept the larger code and try to optimize?
                # Better to revert to best known or current state?
                # Let's just break if we can't improve.
                break

    return [list(cw) for cw in best_code]

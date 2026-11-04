import time
import numpy as np

_start_time = time.time()
_DEADLINE = 120.0  # Safety margin before 150s kill

def _tetracode_solution(q, n, radius):
    """Direct sum of ternary tetracode (9 words len 4, perfect radius 1) with full tails."""
    if n < 4:
        return [list(range(q))[:n]]
    # Ternary tetracode: 9 words of length 4
    # Perfect covering code of length 4, radius 1, size 9
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
    
    words = []
    for h in tetracode:
        # Iterate over all combinations for the remaining n-4 positions
        for tail in np.ndindex(*([3] * (n - 4))):
            words.append(h + list(tail))
    return words

def _build_coverage_matrix(q, n, radius, words):
    """
    Builds a boolean matrix cov of shape (q**n, len(words))
    where cov[i, j] is True if word j covers point i (Hamming distance <= radius).
    """
    space_size = q ** n
    N = len(words)
    cov = np.zeros((space_size, N), dtype=bool)
    
    # Generate all points in the space as an array of shape (space_size, n)
    # Using base-q expansion
    digits = np.zeros((space_size, n), dtype=np.int32)
    for i in range(space_size):
        x = i
        for pos in range(n - 1, -1, -1):
            digits[i, pos] = x % q
            x //= q
            
    # Convert words to indices or keep as is?
    # Let's keep words as list of lists for now, or convert to array
    words_arr = np.array(words, dtype=np.int32)
    
    # For each point i, check distance to each word j
    # Vectorized approach:
    # For each position p, check if digits[i, p] == words[j, p]
    # This is expensive if done naively in a loop over N.
    # Better: 
    # We know radius is 1.
    # Point i is covered by word j if:
    # 1. i == j
    # 2. There exists exactly one position p where i[p] != j[p], and all others are equal.
    
    # Let's use the property of radius 1.
    # Initialize cov with False
    # 1. Exact matches
    # Convert words to indices for easy lookup?
    # idx = sum(w[k] * q^(n-1-k))
    word_indices = np.zeros(N, dtype=np.int64)
    for j in range(N):
        idx = 0
        for k in range(n):
            idx = idx * q + words[j][k]
        word_indices[j] = idx
        
    # Map point index to its coordinate vector is already in 'digits'
    # But we need to check if point i is within dist 1 of word j.
    
    # Efficient Radius 1 Covering Check:
    # A point i is covered by j if:
    # i == j OR
    # (i differs from j in exactly 1 position)
    
    # Let's compute the "sphere" of each word and mark those points.
    # Since N is ~500-700 and space is 6561, this is small enough.
    
    # Precompute weights for positions
    weights = np.array([q ** (n - 1 - pos) for pos in range(n)], dtype=np.int64)
    
    # For each word j, generate its neighbors and set cov[neighbor, j] = True
    # Neighbors of word j:
    # j itself
    # j with one digit changed to d (d != j[p])
    
    # This loop is O(N * q * n). 700 * 3 * 8 = 16800 operations. Very fast.
    for j in range(N):
        w = words[j]
        base_idx = word_indices[j]
        cov[base_idx, j] = True
        for p in range(n):
            original_val = w[p]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                # Calculate index of the modified word
                # delta = (new_val - original_val) * weights[p]
                delta = (new_val - original_val) * weights[p]
                new_idx = base_idx + delta
                cov[new_idx, j] = True
                
    return cov

def _local_search(q, n, radius, words, time_limit=120.0):
    """
    Local search to minimize the number of codewords.
    Uses vectorized removal and swap heuristics.
    """
    space_size = q ** n
    N = len(words)
    
    if N == 0:
        return []
        
    cov = _build_coverage_matrix(q, n, radius, words)
    # cov shape: (space_size, N)
    
    # cover_count[i] = number of codewords covering point i
    cover_count = cov.sum(axis=1).astype(np.int32)
    
    # active[j] = True if codeword j is in the current set
    active = np.ones(N, dtype=bool)
    current_size = N
    
    improved = True
    while improved:
        if time.time() - _start_time > time_limit - 5:
            break
        improved = False
        
        # --- Phase 1: Removal ---
        # A codeword j can be removed if it does not uniquely cover any point.
        # i.e., for all i where cov[i, j] is True, cover_count[i] > 1.
        # Equivalently, there is NO i such that cov[i, j] is True and cover_count[i] == 1.
        
        # unique_mask[i] is True if cover_count[i] == 1
        unique_mask = (cover_count == 1)
        
        # For each active codeword j, check if it covers any unique point.
        # covers_unique[j] = np.sum(cov[:, j] & unique_mask) > 0
        # If covers_unique[j] is False, j can be removed.
        
        # Vectorized check for all active j
        # cov is (space_size, N). unique_mask is (space_size,)
        # broadcast unique_mask to (space_size, 1)
        # unique_points_covered_by_j = cov & unique_mask[:, np.newaxis]
        # sum over axis 0
        
        # To save memory/time, we can do this in chunks or just compute for all.
        # 6561 * 700 bools is ~4.5 MB. Fine.
        unique_coverage_counts = (cov[:, active] & unique_mask[:, np.newaxis]).sum(axis=0)
        
        # Find indices in the 'active' subarray that have 0 unique coverage
        removable_indices_in_active = np.where(unique_coverage_counts == 0)[0]
        
        if len(removable_indices_in_active) > 0:
            # Map back to original indices
            active_indices = np.where(active)[0]
            to_remove = active_indices[removable_indices_in_active]
            
            # Remove them
            active[to_remove] = False
            current_size -= len(to_remove)
            
            # Update cover_count
            # cover_count -= cov[:, to_remove].sum(axis=1)
            # But we must be careful: cov[:, to_remove] might be large.
            # We can just recompute cover_count? No, update is faster.
            # cov is (space_size, N). to_remove is list of indices.
            # We need to subtract the coverage of these removed words.
            # Since we are removing multiple, we can sum them.
            removed_cov = cov[:, to_remove].sum(axis=1).astype(np.int32)
            cover_count -= removed_cov
            
            improved = True
            # Continue loop to check for more removals immediately
            continue
            
        # --- Phase 2: Swap ---
        # If no removals, try swaps.
        # For each active codeword j, we want to replace it with an inactive codeword k
        # such that the new set is valid.
        # Condition: The points uniquely covered by j (U_j) must be covered by k.
        # i.e., for all i in U_j, cov[i, k] must be True.
        
        # Additionally, we prefer k that "relieves" other codewords.
        # Benefit: Number of points i (not in U_j) where cov[i, k] is True and cover_count[i] == 1.
        # These points will become covered by 2 codewords (j's old coverer is removed, k is added? No, j is removed, k is added).
        # Wait, if we swap j -> k:
        # Points in U_j: covered by j (count 1). j removed -> count 0. k added -> count 1 (if k covers them).
        # So U_j remains covered (count 1).
        # Points in U_k (points uniquely covered by k in the CURRENT set? No, k is inactive).
        # Let's define V_k = { i | cov[i, k] is True and cover_count[i] == 1 }.
        # If we add k, these points i in V_k will have cover_count increase from 1 to 2 (assuming j doesn't cover them, or if j covers them, it was already 2+).
        # Actually, if j covers i, and i is in V_k, then cover_count[i] was 1 (only k? No, k is inactive).
        # cover_count is for the ACTIVE set.
        # If i is covered by k (inactive) and cover_count[i] == 1 (by active set), then adding k makes cover_count[i] == 2.
        # This reduces the number of "unique" points, making it easier to remove other codewords.
        
        # So we want to maximize |V_k \ U_j|? Or just |V_k|?
        # Constraint: k must cover U_j.
        
        active_indices = np.where(active)[0]
        inactive_indices = np.where(~active)[0]
        
        if len(inactive_indices) == 0:
            break # No swaps possible
        
        # Precompute U_j for all active j
        # U_j mask: cov[:, j] & unique_mask
        # unique_mask is current cover_count == 1
        
        best_swap_found = False
        for j_idx in active_indices:
            if time.time() - _start_time > time_limit - 5:
                break
                
            # Get unique points covered by j
            U_j_mask = cov[:, j_idx] & unique_mask
            U_j_indices = np.where(U_j_mask)[0]
            
            if len(U_j_indices) == 0:
                # j is removable, but phase 1 should have caught this.
                # Maybe cover_count changed? No, we only change on removal.
                # If U_j is empty, j is removable.
                # Force removal?
                active[j_idx] = False
                cover_count -= cov[:, j_idx].astype(np.int32)
                current_size -= 1
                improved = True
                best_swap_found = True
                break
            
            # Find k in inactive_indices that covers all points in U_j_indices
            # Check cov[U_j_indices, k] is all True
            
            # cov[U_j_indices, :] shape (len(U_j), N)
            # We only care about inactive_indices
            # cand_cov = cov[U_j_indices, inactive_indices] shape (len(U_j), len(inactive))
            
            # To save memory, if len(U_j) is large, this is heavy.
            # But len(U_j) is usually small.
            
            if len(U_j_indices) > 0:
                # Check which inactive k covers all U_j
                # sum of False in cand_cov should be 0
                # i.e., all True
                cand_cov = cov[U_j_indices, inactive_indices]
                # all_true_mask[k_local] = np.all(cand_cov[:, k_local])
                all_true_mask = np.all(cand_cov, axis=0)
                valid_k_locals = np.where(all_true_mask)[0]
                
                if len(valid_k_locals) == 0:
                    continue
                
                valid_k_globals = inactive_indices[valid_k_locals]
                
                # Calculate benefit for each valid k
                # Benefit = number of points i where cov[i, k] is True and cover_count[i] == 1
                # AND i is NOT in U_j (since those stay 1, they don't contribute to "robustness gain" in the same way, 
                # actually they stay 1, so they don't help remove other codewords).
                
                # V_k_mask = cov[:, k] & unique_mask
                # We want sum(V_k_mask) - sum(V_k_mask & U_j_mask)
                # i.e., unique points covered by k that are NOT uniquely covered by j.
                
                # Vectorize over valid_k_globals
                # This might be slow if many valid k.
                # But usually few k cover a specific U_j.
                
                best_k = -1
                best_benefit = -1
                
                for k_global in valid_k_globals:
                    V_k_mask = cov[:, k_global] & unique_mask
                    # Subtract those already in U_j
                    # U_j_mask is already defined
                    net_gain = np.sum(V_k_mask) - np.sum(V_k_mask & U_j_mask)
                    if net_gain > best_benefit:
                        best_benefit = net_gain
                        best_k = k_global
                        
                if best_k != -1 and best_benefit > 0:
                    # Perform swap
                    active[j_idx] = False
                    active[best_k] = True
                    
                    # Update cover_count
                    # Remove j
                    cover_count -= cov[:, j_idx].astype(np.int32)
                    # Add k
                    cover_count += cov[:, best_k].astype(np.int32)
                    
                    # Update unique_mask for next iteration?
                    # unique_mask is derived from cover_count.
                    # We just set improved = True and break to restart loop.
                    improved = True
                    best_swap_found = True
                    break # Restart loop to recalculate unique_mask
                
        if not improved:
            break
            
    # Construct result
    result_indices = np.where(active)[0]
    return [words[i] for i in result_indices]

def construct(q, n, radius):
    global _start_time
    _start_time = time.time()
    time_limit = _DEADLINE - 5  # 5s safety margin

    if q != 3 or n != 8 or radius != 1:
        return _tetracode_solution(q, n, radius)

    # Start with tetracode-based solution (729 words)
    seed_words = _tetracode_solution(q, n, radius)
    
    # Local search to prune
    pruned = _local_search(q, n, radius, seed_words, time_limit)
    
    return pruned

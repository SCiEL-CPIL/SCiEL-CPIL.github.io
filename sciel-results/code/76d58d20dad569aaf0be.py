import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Generate all space words
    # q=3, n=6 -> 729 words
    space_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    Q = space_words.shape[0]
    
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
    
    M = code.shape[0]
    
    # Helper to compute distance matrix
    def compute_dist_matrix(code_mat):
        # code_mat: (M, n)
        # space_words: (Q, n)
        # Dist: (M, Q)
        # d(c, s) = sum(c != s)
        # Using broadcasting
        # c[:, None, :] vs s[None, :, :]
        # This creates (M, 1, n) and (1, Q, n)
        # diff = c[:, None, :] != s[None, :, :]
        # dist = diff.sum(axis=2)
        return (code_mat[:, None, :] != space_words[None, :, :]).sum(axis=2)

    dist_matrix = compute_dist_matrix(code)
    
    def is_valid_removal(dist_mat, k):
        # Check if removing codeword k leaves all words covered
        # For all words w where dist_mat[k, w] <= radius,
        # there must exist j != k such that dist_mat[j, w] <= radius
        
        # Get words covered by k
        covered_by_k = dist_mat[k] <= radius
        
        if not np.any(covered_by_k):
            return True # It covers nothing, so removal is safe (though unlikely in a minimal cover)
        
        # For those words, check min distance among other codewords
        # We need min_{j != k} dist_mat[j, w] <= radius
        
        # Create a copy or mask
        # To avoid copying the whole matrix, we can compute min and check if the min was k
        
        # Efficient check:
        # Let min_dist = min_j dist_mat[j, w]
        # Let second_min_dist = second_min_j dist_mat[j, w]
        # If min_dist <= radius, the word is covered.
        # If we remove k, the new min for word w is:
        #   if dist_mat[k, w] == min_dist and count of min == 1: new_min = second_min
        #   else: new_min = min_dist
        
        # We only care about words where dist_mat[k, w] <= radius.
        # If for any such word, the new_min > radius, removal is invalid.
        
        # Calculate min and second min for all words
        # This is O(M*Q) per check, which is 80*729 ~ 58k ops. Very fast.
        
        # Sort or use argpartition?
        # Simple approach:
        # Mask out k, find min.
        # But we can do it vectorized.
        
        # Let's just compute the min distance for all words excluding k.
        # dist_mat_masked = dist_mat.copy()
        # dist_mat_masked[k, :] = np.inf
        # min_excl_k = np.min(dist_mat_masked, axis=0)
        
        # Optimization:
        # We only need to check words w where dist_mat[k, w] <= radius.
        # For these words, if min_{j!=k} dist_mat[j, w] > radius, it's invalid.
        
        # Let's compute min_excl_k for all words.
        # To avoid full copy, we can use:
        # min1, idx1 = np.min(dist_mat, axis=0)
        # if idx1 == k for some w, we need min2.
        
        # Let's use the mask approach, it's simple and M is small.
        dist_mat_temp = dist_mat.copy()
        dist_mat_temp[k, :] = np.inf
        min_excl_k = np.min(dist_mat_temp, axis=0)
        
        # Check condition for words covered by k
        # If dist_mat[k, w] <= radius, then min_excl_k[w] must be <= radius
        # If dist_mat[k, w] > radius, w was not covered by k, so its coverage status doesn't change 
        # (it was covered by others or not at all, but since current code is valid, it was covered by others).
        # Wait, if current code is valid, all w are covered.
        # If w is NOT covered by k (dist > r), it MUST be covered by some j != k. So min_excl_k[w] <= r is guaranteed.
        # So we only need to check w where dist_mat[k, w] <= r.
        
        critical_words = dist_mat[k] <= radius
        if np.any(critical_words):
            # Check if min_excl_k is <= radius for these words
            if np.any(min_excl_k[critical_words] > radius):
                return False
        return True

    def is_valid_swap(dist_mat, k, new_codeword):
        # Check if replacing codeword k with new_codeword results in a valid cover
        # This is equivalent to:
        # 1. Removing k leaves all words covered by OTHERS, EXCEPT possibly those only covered by k.
        # 2. The new_codeword covers those remaining words.
        
        # Actually, simpler:
        # Construct new dist matrix? No, too slow.
        # Logic:
        # A word w is covered in the new set if:
        # (dist_mat[j, w] <= r for some j != k) OR (dist(new_codeword, w) <= r)
        
        # We know current set is valid.
        # Words that are "safe" (covered by j != k) don't need checking.
        # Words that are "critical" (covered by k, and NOT covered by any j != k) MUST be covered by new_codeword.
        
        # Identify critical words for k:
        # dist_mat[k, w] <= r AND min_{j!=k} dist_mat[j, w] > r
        
        # Calculate min_excl_k
        dist_mat_temp = dist_mat.copy()
        dist_mat_temp[k, :] = np.inf
        min_excl_k = np.min(dist_mat_temp, axis=0)
        
        critical_mask = (dist_mat[k] <= radius) & (min_excl_k > radius)
        
        if not np.any(critical_mask):
            return True # No critical words, swap is always valid (as long as new word is in space, which it is)
        
        # Check if new_codeword covers all critical words
        # dist_new = (new_codeword != space_words).sum(axis=1)
        # return np.all(dist_new[critical_mask] <= radius)
        
        dist_new = (new_codeword[None, :] != space_words).sum(axis=1)
        return np.all(dist_new[critical_mask] <= radius)

    # Local Search
    best_code = code.copy()
    best_M = M
    
    while time.time() < deadline:
        removed_any = False
        M = code.shape[0]
        
        # Try to remove words
        # Iterate in reverse to handle index shifts, or restart loop
        # Restart loop is safer for logic clarity
        for k in range(M - 1, -1, -1):
            if time.time() > deadline:
                break
            
            if is_valid_removal(dist_matrix, k):
                # Remove k
                code = np.delete(code, k, axis=0)
                dist_matrix = np.delete(dist_matrix, k, axis=0)
                M -= 1
                removed_any = True
                # Restart removal loop
                break
        
        if removed_any:
            if M < best_M:
                best_M = M
                best_code = code.copy()
            continue # Restart outer loop
        
        # If no removal, try swaps
        swap_found = False
        # Try moving each codeword to a random new position
        for k in range(M):
            if time.time() > deadline:
                break
            
            # Generate a few random candidates for new position
            # Bias towards words that are currently "under-covered" or random
            # For simplicity, random
            for _ in range(10):
                if time.time() > deadline:
                    break
                new_cw = np.random.randint(0, q, size=n)
                
                if is_valid_swap(dist_matrix, k, new_cw):
                    # Perform swap
                    old_cw = code[k].copy()
                    code[k] = new_cw
                    dist_matrix[k, :] = (new_cw[None, :] != space_words).sum(axis=1)
                    swap_found = True
                    break
            
            if swap_found:
                # Restart search after a swap
                break
        
        if not swap_found and not removed_any:
            # Stuck in local minimum
            # We can try a "kick": randomly change a few codewords to escape, 
            # but we must maintain validity? No, SA allows invalid states.
            # However, implementing full SA with validity checks is complex.
            # Given 81 -> 73 is close, random swaps might suffice.
            # If stuck, we might just return best_code found so far.
            # Or, we can try to add a random word and then remove, but that increases size.
            # Let's just break if we've been stuck for a while?
            # For now, if no swap found in a full pass, we might be at a local min.
            # We can try to restart with a slightly perturbed initial code?
            # Or just accept the best found.
            
            # To improve chances, we can try "2-swap" or just break.
            # Given the constraint, breaking and returning best is safe.
            # But let's try one more thing: if stuck, pick a random codeword and move it to a random word, 
            # even if it breaks coverage temporarily? No, our check requires valid swap.
            
            # If we are stuck at 74 or 75, we might need to accept a worse state temporarily.
            # But for simplicity and safety, we return the best valid code found.
            break

    return [list(cw) for cw in best_code]

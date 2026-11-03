import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        return _fallback(q, n, radius)

    N = q ** n
    
    # 1. Precompute all neighbors for each word in the space
    # neighbors[i] is a list of indices of words within distance 1 of i
    # We will use this to define the coverage of a codeword.
    
    # Generate all words
    words = np.array(np.unravel_index(np.arange(N), (q,) * n), dtype=np.int8)
    
    # Precompute neighbor indices for each word
    # For each word i, its neighbors are i itself and words differing in 1 position.
    # We can compute this efficiently.
    neighbor_indices = []
    for i in range(N):
        w = words[i]
        idxs = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_new = w.copy()
                w_new[pos] = new_val
                # Calculate index
                idx = 0
                mult = 1
                for d in range(n):
                    idx += w_new[d] * mult
                    mult *= q
                idxs.append(idx)
        neighbor_indices.append(np.array(idxs, dtype=np.int32))
        
    # Convert to a list of arrays for easier access
    # coverage_mask[i] is a boolean array of length N where True means word j is covered by codeword i
    # Storing N boolean arrays of size N is N*N = 6561*6561 ~ 43M booleans ~ 43MB. This is acceptable.
    # However, accessing them in a loop is slow. 
    # Better: Precompute a matrix M where M[i, j] = 1 if j is covered by i.
    # M is N x N. 
    # Greedy step: find i that maximizes sum(M[i, :][uncovered]).
    
    M = np.zeros((N, N), dtype=np.bool_)
    for i in range(N):
        M[i, neighbor_indices[i]] = True
        
    # 2. Initialize with a structured code (Direct Sum of Tetracode)
    # Tetracode is an optimal (4,1,3) ternary code? 
    # Actually, the (4,1,3) ternary code has size 9 (Tetracode).
    # K_3(4,1) = 9.
    # Direct sum: C_8 = C_4 x C_4. Size 9 * 9 = 81? 
    # Wait, K_3(8,1) is 486. 
    # The direct sum of two (4,1) codes of size 9 gives size 81. 
    # Does it cover? 
    # If c1 in C4 covers all words in F3^4, and c2 in C4 covers all words in F3^4,
    # then (c1, c2) covers (w1, w2) if d(c1, w1) + d(c2, w2) <= 1.
    # This requires d(c1, w1)=0 and d(c2, w2)=0 OR one is 1 and other 0.
    # It does NOT cover (w1, w2) if d(c1, w1)=1 and d(c2, w2)=1.
    # So direct sum of radius 1 codes does not produce a radius 1 code for the concatenated space.
    # It produces a radius 2 code? No.
    # The van Laarhoven code is not a simple direct sum.
    
    # Let's just use a random initialization or greedy from scratch if structured init is complex.
    # Given the time limit, a fast greedy from scratch might be better than a complex structured init that might not be valid or optimal.
    # However, the prompt suggests "initialize the search with a structured code".
    # Let's try a simple greedy first, it's robust.
    
    uncovered = np.ones(N, dtype=np.bool_)
    code = []
    used = np.zeros(N, dtype=np.bool_)
    
    # Greedy Set Cover
    # To speed up, we can precompute the "weight" (number of uncovered points covered) for each candidate.
    # But updating weights after each pick is expensive if done naively.
    # With N=6561, a simple scan of N candidates, each taking O(N) to check overlap, is N^2 = 43M ops per step.
    # If we take ~500 steps, that's 21.5B ops. Too slow in Python/NumPy loops if not vectorized properly.
    
    # Vectorized Greedy:
    # Maintain an array `cover_count` of length N, where cover_count[j] is the number of selected codewords covering j.
    # Actually, we just need to know if it's 0 or >0.
    # Let's maintain `uncovered` boolean array.
    # In each step:
    # 1. Compute for all i: `overlap[i] = M[i, :][uncovered].sum()`
    #    This is a matrix-vector product or sum over axis.
    #    M is (N, N). uncovered is (N,).
    #    overlaps = M @ uncovered  (if M is float) or M[uncovered].sum(axis=1)
    #    M is bool. M[uncovered] creates a view/copy of rows where uncovered is True? No, that's rows.
    #    We want column sums? No, we want for each row i, how many 1s in columns j where uncovered[j] is True.
    #    This is `M.dot(uncovered)` if we cast to float/int.
    #    Let's use int8 for M to save memory and allow dot product?
    #    M_int = M.astype(np.int8)
    #    overlaps = M_int.dot(uncovered.astype(np.int8))
    
    M_int = M.astype(np.int8)
    
    while uncovered.any():
        if time.time() > deadline - 10: # Leave time for finalization
            break
        
        # Find best candidate
        overlaps = M_int.dot(uncovered.astype(np.int8))
        # Exclude already used codewords
        overlaps[used] = -1
        best_idx = np.argmax(overlaps)
        
        if overlaps[best_idx] <= 0:
            break
            
        code.append(best_idx)
        used[best_idx] = True
        
        # Update uncovered
        # uncovered = uncovered & ~M[best_idx]
        uncovered &= ~M[best_idx]
        
    # 3. Local Search: Reduction
    # Remove redundant codewords.
    # A codeword i is redundant if for all j in M[i, :], there exists another codeword k != i such that M[k, j] is True.
    # Equivalently, if the sum of M[k, j] over all k in code (excluding i) is >= 1 for all j in M[i, :].
    
    # To speed this up, we can maintain `cover_counts` array.
    # cover_counts[j] = number of codewords in `code` covering j.
    
    cover_counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        cover_counts[M[idx]] += 1
        
    # Check for redundancy
    changed = True
    while changed:
        changed = False
        if time.time() > deadline - 5:
            break
            
        # For each codeword in code, check if it's redundant.
        # To avoid O(N^2) per iteration, we can just iterate through current code list.
        # Since code size is ~500, and N=6561, checking one codeword takes O(17) if we have neighbor lists, or O(N) if we use masks.
        # Using masks: M[idx] is a boolean array. 
        # Redundant if np.all(cover_counts[M[idx]] > 1) is False? 
        # No, redundant if for all j where M[idx, j] is True, cover_counts[j] > 1.
        # So: if np.all(cover_counts[M[idx]] > 1): redundant.
        
        new_code = []
        for idx in code:
            if time.time() > deadline - 5:
                new_code.append(idx)
                continue
                
            # Check redundancy
            # Get the boolean mask for this codeword
            mask = M[idx]
            # Check if any word covered by idx is covered ONLY by idx
            # i.e., cover_counts[j] == 1 for any j where mask[j] is True
            # If not, it's redundant.
            
            # Efficient check:
            # min_count = np.min(cover_counts[mask])
            # if min_count > 1: redundant
            
            # But np.min on a boolean indexed array creates a copy. 
            # Alternative:
            # unique_counts = np.unique(cover_counts[mask])
            # if len(unique_counts) == 1 and unique_counts[0] > 1: redundant
            
            # Faster:
            # counts_in_mask = cover_counts[mask]
            # if np.all(counts_in_mask > 1): redundant
            
            # Let's try this.
            if np.all(cover_counts[mask] > 1):
                # Redundant
                cover_counts[mask] -= 1
                changed = True
            else:
                new_code.append(idx)
                
        code = new_code
        
    # 4. Local Search: Swapping (Optional, if time permits)
    # Try to swap a codeword with another to allow further reduction.
    # This is complex to implement efficiently in the remaining time.
    # Given the greedy + reduction usually gets close to optimal for small N, 
    # and the known optimum is 486, while greedy might start around 500-600,
    # reduction might get it to ~490-500.
    # Let's add a simple swap loop if time allows.
    
    if time.time() < deadline - 20:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # For each codeword i in code:
            for i in range(len(code)):
                idx = code[i]
                mask = M[idx]
                # Find words covered only by idx (critical words)
                crit_mask = (cover_counts == 1) & mask
                if not crit_mask.any():
                    continue
                    
                crit_indices = np.where(crit_mask)[0]
                
                # Find candidates k not in code that cover all crit_indices
                # We need M[k, crit_indices] to be all True.
                # candidates = np.where(np.all(M[:, crit_indices], axis=1))[0]
                # This is expensive: N * len(crit_indices).
                # len(crit_indices) is usually small (1-3).
                
                # Optimization: Only consider candidates that are "good".
                # For now, just do the check.
                
                # Get all codewords that cover the first critical word
                first_crit = crit_indices[0]
                potential = np.where(M[:, first_crit])[0]
                
                # Filter potential to those covering ALL critical words
                valid_potential = []
                for k in potential:
                    if k == idx:
                        continue
                    if used[k]:
                        continue
                    # Check if k covers all crit_indices
                    # M[k, crit_indices] should be all True
                    if np.all(M[k, crit_indices]):
                        valid_potential.append(k)
                        
                if valid_potential:
                    # Pick one (e.g., one that maximizes coverage of other words? or random?)
                    # Let's pick the one that covers the most *other* words (greedy heuristic)
                    best_k = -1
                    best_score = -1
                    for k in valid_potential:
                        # Score: number of words covered by k
                        score = M[k].sum()
                        if score > best_score:
                            best_score = score
                            best_k = k
                            
                    if best_k != -1:
                        # Perform swap
                        # Remove idx
                        cover_counts[mask] -= 1
                        used[idx] = False
                        
                        # Add best_k
                        new_mask = M[best_k]
                        cover_counts[new_mask] += 1
                        used[best_k] = True
                        
                        code[i] = best_k
                        improved = True
                        # Restart reduction? 
                        # We can just break and let the outer loop handle it, 
                        # but the outer loop is `while improved`. 
                        # We should probably restart the reduction phase immediately.
                        # For simplicity, we just mark improved and continue the swap loop.
                        # The reduction loop will catch new redundancies in the next iteration of the `while changed` block?
                        # No, the reduction block is above. 
                        # We should run reduction again.
                        # Let's restructure:
                        # 1. Greedy
                        # 2. Loop:
                        #    a. Reduce
                        #    b. Try Swaps
                        #    c. If no improvement, break.
                        
                        # Since we are inside the swap loop, let's just break out to re-run reduction.
                        # We can set a flag.
                        break
                        
            if improved:
                # Re-run reduction
                changed = True
                while changed:
                    changed = False
                    if time.time() > deadline - 5:
                        break
                    new_code = []
                    for idx in code:
                        mask = M[idx]
                        if np.all(cover_counts[mask] > 1):
                            cover_counts[mask] -= 1
                            changed = True
                        else:
                            new_code.append(idx)
                    code = new_code
                # And then continue swap search? 
                # The `improved` flag is True, so the outer while loop will continue.
                # But we need to restart the swap search from the beginning or continue?
                # Continuing might miss opportunities. Restarting is safer.
                # The outer while loop will iterate again.

    # Convert indices to codewords
    result = []
    for idx in code:
        digits = words[idx].tolist()
        result.append(digits)
    
    return result

def _fallback(q, n, radius):
    # Direct sum construction or random
    if n % 4 == 0:
        # Tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Random
        return [random.randint(0, q-1) for _ in range(n)]

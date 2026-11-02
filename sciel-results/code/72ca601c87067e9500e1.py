"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # For the specific case q=3, n=6, radius=1, we aim for a covering code of size 73.
    # If time runs out, we fall back to the direct sum construction (size 81).
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: direct sum of tetracode
        # This is a valid covering code for q=3, n=4, r=1 (size 9)
        # For general n, we can extend it.
        # For n=6, direct sum of (3,4,1) code gives 9 * 3^(6-4) = 9*9=81.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    start_time = time.time()
    deadline = 115.0  # Leave 5s safety margin for 120s limit (or 150s limit, but 120s is safer)

    # Precompute neighbors for all 729 words
    # neighbors[i] is a list of indices j such that dist(i, j) <= 1
    # We represent words as integers 0..728 (base 3)
    
    def num_to_vec(num, length=6):
        vec = [0] * length
        for i in range(length):
            vec[i] = num % 3
            num //= 3
        return vec

    def vec_to_num(vec):
        num = 0
        for i in range(len(vec)):
            num += vec[i] * (3 ** i)
        return num

    # Precompute coverage sets as lists of indices
    # cover_set[i] is a list of words covered by codeword i
    cover_set = []
    for i in range(729):
        v = num_to_vec(i)
        s = [i]
        for pos in range(6):
            orig = v[pos]
            for new_val in range(3):
                if new_val != orig:
                    v[pos] = new_val
                    s.append(vec_to_num(v))
            v[pos] = orig
        cover_set.append(s)
        
    # Convert cover_set to numpy arrays for faster operations if needed, 
    # but lists are fine for 729 elements.
    
    # Greedy Set Cover
    # Start with empty set
    selected = set()
    uncovered = np.ones(729, dtype=bool)
    
    # To speed up greedy, we can compute gain = number of uncovered words covered by c
    # gain(c) = sum(uncovered[j] for j in cover_set[c])
    
    while np.any(uncovered) and time.time() < deadline - 5:
        best_c = -1
        max_gain = 0
        # Iterate over all candidates
        for c in range(729):
            if c in selected:
                continue
            # Calculate gain
            # Optimization: precompute gain? No, uncovered changes.
            # Use numpy for fast sum
            gain = np.sum(uncovered[cover_set[c]])
            if gain > max_gain:
                max_gain = gain
                best_c = c
        if max_gain == 0:
            break
        selected.add(best_c)
        uncovered[cover_set[best_c]] = False
        
    if not np.any(uncovered):
        best_code = list(selected)
    else:
        # If greedy failed to cover (shouldn't happen), fallback
        best_code = None

    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Local Search to minimize size
    # State: selected set, cover_count array
    # cover_count[w] = number of codewords in selected that cover w
    
    def update_cover_count(selected, cover_count):
        cover_count[:] = 0
        for c in selected:
            for w in cover_set[c]:
                cover_count[w] += 1
        return cover_count

    cover_count = np.zeros(729, dtype=np.int32)
    update_cover_count(best_code, cover_count)
    
    best_size = len(best_code)
    best_code_set = set(best_code)
    
    # Target size is 73.
    # We try to reduce size.
    
    def is_valid(cover_count):
        return np.min(cover_count) >= 1
    
    # Local Search Loop
    # Moves:
    # 1. Delete c if valid.
    # 2. Swap c_out with c_in if it improves "potential" or allows deletion.
    #    Potential: number of words with count 1 (exposed). Minimize this.
    #    Or just accept swap if it leads to a valid solution with same size? No, that doesn't reduce size.
    #    We need to reduce size.
    #    Strategy: 
    #    - Try to delete.
    #    - If no deletion, try to swap to create a "redundant" codeword (count >= 2 for all words in its cover set, allowing deletion of another).
    #    - Actually, a standard approach is:
    #      1. Try all 1-1 swaps (remove c1, add c2). If the new set is valid, accept? No, size is same.
    #      2. After a swap, try to delete.
    #      3. If stuck, do a 2-2 swap or random perturbation.
    
    # Let's use a simpler heuristic:
    # While size > 73:
    #   Try to find a codeword c that can be removed.
    #   If found, remove it.
    #   If not, try to find a swap (c_out, c_in) such that after swap, a deletion is possible.
    #   If not, perturb.
    
    current_set = set(best_code)
    current_size = len(current_set)
    
    # Precompute cover_set as list of numpy arrays for faster indexing?
    # cover_set_np = [np.array(cs) for cs in cover_set]
    
    for iteration in range(10000):
        if time.time() > deadline:
            break
        if current_size <= 73:
            break
            
        # Update cover_count for current_set
        cover_count[:] = 0
        for c in current_set:
            for w in cover_set[c]:
                cover_count[w] += 1
        
        # 1. Try deletion
        deleted = False
        candidates = list(current_set)
        random.shuffle(candidates)
        for c in candidates:
            # Check if removing c keeps all counts >= 1
            # We can do this by checking min(cover_count[w] - 1) for w in cover_set[c]
            # But we need to be careful not to modify cover_count permanently until we decide.
            # Let's just compute min count for words covered by c.
            # If min(cover_count[w]) >= 2 for all w in cover_set[c], then we can remove c.
            # Actually, if cover_count[w] >= 2 for all w in cover_set[c], removing c makes them >= 1.
            # What about words NOT covered by c? Their count doesn't change.
            # So we just need min(cover_count[w] for w in cover_set[c]) >= 2.
            
            # Optimization: if any word in cover_set[c] has count 1, we can't remove c.
            # min_val = np.min(cover_count[cover_set[c]])
            # if min_val >= 2:
            #    remove c
            #    break
            
            # Let's use a simple loop for speed check first
            can_remove = True
            for w in cover_set[c]:
                if cover_count[w] == 1:
                    can_remove = False
                    break
            if can_remove:
                current_set.remove(c)
                current_size -= 1
                deleted = True
                if current_size < best_size:
                    best_size = current_size
                    best_code_set = set(current_set)
                break
        
        if deleted:
            continue
            
        # 2. Try 1-1 swap
        # Find c_out in current_set, c_in not in current_set
        # such that after swap, we can delete something?
        # Or simply, swap to increase redundancy.
        # Let's try to swap to minimize the number of "critical" words (count 1).
        # This is complex. Let's try random swaps and see if they enable deletion.
        
        # Simpler: Try all pairs (c_out, c_in) where c_out is "most redundant" and c_in is "most useful"?
        # Let's just try random swaps.
        
        # Identify words with count 1.
        # If we swap c_out for c_in, and c_in covers more critical words than c_out loses, maybe we can delete later.
        
        # Heuristic: Pick c_out with max redundancy (min count in its cover set is high).
        # Pick c_in with max coverage of critical words.
        
        # Let's just do a few random swaps and check for deletion.
        swapped = False
        if current_size > 73:
            # Try a few random swaps
            for _ in range(50):
                c_out = random.choice(list(current_set))
                c_in = random.randint(0, 728)
                if c_in in current_set:
                    continue
                
                # Perform swap
                current_set.remove(c_out)
                current_set.add(c_in)
                
                # Update cover_count
                cover_count[:] = 0
                for c in current_set:
                    for w in cover_set[c]:
                        cover_count[w] += 1
                        
                # Try to delete
                del_candidates = list(current_set)
                random.shuffle(del_candidates)
                del_found = False
                for c in del_candidates:
                    can_remove = True
                    for w in cover_set[c]:
                        if cover_count[w] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        current_set.remove(c)
                        current_size -= 1
                        del_found = True
                        if current_size < best_size:
                            best_size = current_size
                            best_code_set = set(current_set)
                        break
                
                if not del_found:
                    # Revert swap if it didn't lead to deletion?
                    # Or keep it if it improves some metric?
                    # For now, revert if no deletion, to stay in valid space?
                    # Actually, the swap itself is valid (covers all).
                    # But if we revert, we might loop.
                    # Let's keep the swap if it reduces the number of words with count 1.
                    # Count words with count 1 before and after?
                    # We didn't store "before".
                    # Let's just revert to avoid random walk away from solution.
                    current_set.remove(c_in)
                    current_set.add(c_out)
                    # Re-update cover_count for next iteration?
                    # We'll re-update at start of loop.
                else:
                    swapped = True
                    break
                
        if not swapped and not deleted:
            # Perturbation: remove 2 random, add 2 random?
            # Or just restart greedy?
            # Let's try a 2-2 swap.
            # Or just accept we are stuck at this local min and try a different random restart of the whole process?
            # Given time, let's just break and try another greedy restart if we have time.
            # But we only have one greedy so far.
            
            # Let's try to force a change by removing a random codeword and adding a random one, 
            # even if it temporarily increases "critical" words, hoping to escape.
            # Actually, let's just break the local search and do a new greedy restart.
            # But greedy is slow.
            
            # Alternative: Simulated Annealing.
            # Let's just break and return best found.
            # Or try 2-2 swap.
            
            # 2-2 Swap:
            # Remove c1, c2. Add c3, c4.
            # Check validity.
            # This is expensive.
            
            # Let's just break and return.
            break

    # Verify best_code_set
    if best_code_set:
        cover_count_final = np.zeros(729, dtype=np.int32)
        for c in best_code_set:
            for w in cover_set[c]:
                cover_count_final[w] += 1
        if np.min(cover_count_final) >= 1:
            return [num_to_vec(c) for c in best_code_set]
            
    # Fallback
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

# EVOLVE-BLOCK-END

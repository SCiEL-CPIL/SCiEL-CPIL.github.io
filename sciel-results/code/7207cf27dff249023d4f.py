import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    # Generate all words in {0, 1, ..., q-1}^n
    # We represent words as tuples of integers.
    # Total words = q^n
    total_words = q ** n
    
    # Create list of all words
    # For q=3, n=6, total_words = 729. Small enough to keep in memory.
    words = []
    for i in range(total_words):
        w = []
        rem = i
        for _ in range(n):
            w.append(rem % q)
            rem //= q
        words.append(tuple(w))
        
    # Precompute Hamming distance spheres
    # sphere[i] is a list of indices of words within 'radius' of words[i]
    # Since radius is small (1) and n is small, we can compute neighbors explicitly.
    # However, checking distance for every pair is O(N^2 * n) = 729^2 * 6 ~ 3M ops. Acceptable.
    # But we can be smarter: generate neighbors by flipping bits.
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Generate neighbors for each word
    neighbors = [set() for _ in range(total_words)]
    for i, w in enumerate(words):
        neighbors[i].add(i)
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                # Create new word
                w_list = list(w)
                w_list[pos] = val
                new_w = tuple(w_list)
                j = word_to_idx[new_w]
                neighbors[i].add(j)
                # Note: neighbors are symmetric, but we build full sets for easy lookup
                # We only need radius 1 here based on task description (radius passed is 1)
                # If radius > 1, this simple flip logic fails. 
                # The task says "radius" is passed. 
                # For general radius, we need to generate all combinations of up to 'radius' positions changed.
                # Given the specific task is r=1, but the function signature is general.
                # Let's assume radius is small. If radius is large, q^n is small or n is small.
                # For safety, if radius is not 1, we might need a generic distance check.
                # But the prompt specifies the task is K_3(6,1). 
                # I will implement the general case using itertools for neighbor generation if radius > 1.
    
    # If radius != 1, the neighbor generation above is incomplete.
    # Let's rewrite neighbor generation to be robust for small radius.
    if radius != 1:
        neighbors = [set() for _ in range(total_words)]
        for i, w in enumerate(words):
            neighbors[i].add(i)
            # Choose 'radius' positions to change
            # For small n and radius, this is fine.
            # If n=6, radius=1, 6*2 = 12 neighbors.
            # If n=6, radius=2, 15*2*2 + 6*2 = 60+12 = 72 neighbors.
            # Still small.
            for pos_combo in __import__('itertools').combinations(range(n), radius):
                for val_combo in __import__('itertools').product(range(q), repeat=radius):
                    w_list = list(w)
                    for p, v in zip(pos_combo, val_combo):
                        if w_list[p] == v:
                            # If value doesn't change, it's not a distinct neighbor in terms of distance increase?
                            # Actually, distance is count of positions where values differ.
                            # If we change pos p to same value, distance doesn't increase.
                            # So we should only generate words at distance exactly k?
                            # No, we want distance <= radius.
                            # The set of words at distance <= R is the union of words at distance 0, 1, ..., R.
                            # My loop above for radius=1 works.
                            # For general radius, it's easier to just check distance for all pairs if N is small.
                            # 729 * 729 * 6 = 3.1M operations. Python loop is slow (~1s).
                            # NumPy vectorized distance is faster.
                            pass
                            
        # Let's use a NumPy approach for distance calculation for generality and speed.
        words_np = np.array(words, dtype=np.int8)
        # Compute pairwise Hamming distance
        # diff = words_np[:, np.newaxis, :] != words_np[np.newaxis, :, :]
        # dist = diff.sum(axis=2)
        # This creates a 729x729x6 array. 729*729*6 = 3.1M bytes (int8). Fine.
        dist_matrix = (words_np[:, np.newaxis, :] != words_np[np.newaxis, :, :]).sum(axis=2)
        
        neighbors = [set() for _ in range(total_words)]
        for i in range(total_words):
            # Find indices where distance <= radius
            # np.where(dist_matrix[i] <= radius)[0]
            idxs = np.where(dist_matrix[i] <= radius)[0]
            neighbors[i] = set(int(x) for x in idxs)
            
    # 2. Greedy Set Cover Initialization
    uncovered = set(range(total_words))
    code_set = set()
    
    # Pre-compute coverage sets for all words? 
    # coverage[i] = neighbors[i]
    # We already have neighbors.
    
    while uncovered:
        if time.time() > deadline - 5.0:
            break
        best_c = -1
        max_covered = 0
        # To speed up, we can sample or iterate. 729 is small, iterate all.
        for c in range(total_words):
            if c in code_set:
                continue
            # Number of newly covered words
            # len(neighbors[c] & uncovered)
            # This is a set intersection.
            # Optimization: if max possible is small, break?
            # Just iterate.
            covered_count = len(neighbors[c] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_c = c
        
        if best_c == -1:
            break
            
        code_set.add(best_c)
        uncovered -= neighbors[best_c]
        
    # 3. Local Search Optimization
    best_code = list(code_set)
    best_size = len(best_code)
    
    # Helper to check if code_set is a valid covering code
    def is_valid(code_list):
        covered = set()
        for c in code_list:
            covered.update(neighbors[c])
        return len(covered) == total_words
        
    # Helper to remove redundant codewords
    def remove_redundant(code_list):
        code_set_local = set(code_list)
        changed = True
        while changed:
            changed = False
            for c in list(code_set_local):
                # Temporarily remove c
                code_set_local.remove(c)
                if is_valid(list(code_set_local)):
                    changed = True
                    # Keep it removed
                else:
                    # Put it back
                    code_set_local.add(c)
        return list(code_set_local)
        
    # Initial cleanup
    best_code = remove_redundant(best_code)
    best_size = len(best_code)
    
    # Simulated Annealing / Randomized Local Search
    # We try to swap a codeword or just remove one if possible.
    # Strategy: 
    # 1. Pick a random codeword c from best_code.
    # 2. Remove it.
    # 3. If still valid, accept (size decreased).
    # 4. If not valid, find a codeword c_new that covers the most uncovered words.
    # 5. Add c_new.
    # 6. Try to remove any redundant codeword.
    # 7. If size < best_size, update best.
    
    iterations = 0
    while time.time() < deadline - 5.0:
        iterations += 1
        
        # Get a random codeword to remove
        if not best_code:
            break
            
        c_remove = random.choice(best_code)
        
        # Try removing it
        trial_code = [c for c in best_code if c != c_remove]
        
        if is_valid(trial_code):
            # Success!
            best_code = trial_code
            best_size = len(best_code)
            continue
            
        # Failed. Find best replacement
        # Calculate uncovered words
        covered = set()
        for c in trial_code:
            covered.update(neighbors[c])
        uncovered = set(range(total_words)) - covered
        
        if not uncovered:
            # Should not happen if is_valid is correct
            best_code = trial_code
            best_size = len(best_code)
            continue
            
        best_new_c = -1
        max_gain = 0
        for c_new in range(total_words):
            if c_new in trial_code:
                continue
            gain = len(neighbors[c_new] & uncovered)
            if gain > max_gain:
                max_gain = gain
                best_new_c = c_new
                
        if best_new_c == -1:
            break
            
        trial_code.append(best_new_c)
        
        # Try to clean up redundancies
        # This might be expensive if done every step.
        # Only do it if we have time or periodically.
        if iterations % 10 == 0 or len(trial_code) < best_size:
            trial_code = remove_redundant(trial_code)
            
        if len(trial_code) < best_size:
            best_code = trial_code
            best_size = len(best_code)
            
    # Return the best code found
    # Convert indices to lists of integers
    return [list(words[c]) for c in best_code]

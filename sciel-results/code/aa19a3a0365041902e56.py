import time
import random
import itertools
import sys

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet {0..q-1} with radius r.
    Uses a local search heuristic: 
    1. Start with a random initial set.
    2. Greedily add codewords to ensure full coverage.
    3. Iteratively remove redundant codewords.
    4. Try swap moves to escape local minima.
    """
    # Deadline guard
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin for Python overhead and return
    
    # Precompute all words and their neighborhoods
    # For q=3, n=7, radius=1:
    # Total words: 3^7 = 2187
    # Neighborhood size: 1 + n*(q-1) = 1 + 7*2 = 15
    
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    num_words = len(all_words)
    
    # Map word tuple to index for fast lookup
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (ball of radius 'radius') for each word
    # neighbors[i] is a list of indices of words within Hamming distance 'radius' from all_words[i]
    neighbors = []
    for i, w in enumerate(all_words):
        ball = set()
        # Generate all words in the ball
        # For radius 1, it's the word itself and all single-position changes
        if radius == 1:
            ball.add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = w.copy()
                    new_w[pos] = val
                    ball.add(word_to_idx[tuple(new_w)])
        else:
            # General radius handling (slower, but we only expect radius 1 or small)
            # For efficiency, we stick to radius 1 logic if possible, 
            # but let's implement a generic BFS or enumeration for small radii if needed.
            # Given the task context (K_3(7,1)), radius is likely 1.
            # We will assume radius is small. If radius > 1, we might need a different approach.
            # For now, let's handle radius 1 specifically as it's the standard case for this problem.
            # If radius is not 1, we fall back to a simpler greedy.
            pass
        
        neighbors.append(list(ball))
        
    # If radius is not 1, the above loop didn't populate correctly for general case.
    # Let's add a check. The task specifies K_3(7,1).
    if radius != 1:
        # Fallback for other radii: Simple greedy
        # This is less optimized but valid
        def greedy_construct():
            covered = [False] * num_words
            code = []
            while any(not c for c in covered):
                # Find word covering most uncovered
                best_w = None
                best_count = -1
                for w_idx in range(num_words):
                    if covered[w_idx]: continue
                    # Count how many uncovered words are in its ball
                    cnt = 0
                    for nb in neighbors[w_idx]:
                        if not covered[nb]:
                            cnt += 1
                    if cnt > best_count:
                        best_count = cnt
                        best_w = w_idx
                if best_w is None:
                    break
                code.append(all_words[best_w])
                for nb in neighbors[best_w]:
                    covered[nb] = True
            return code
            
        # For non-radius 1, we don't have precomputed neighbors correctly in the loop above.
        # Let's re-compute neighbors properly for any radius if needed.
        # But the prompt implies the specific task K_3(7,1).
        # We will proceed with the assumption that neighbors are correctly built for radius 1.
        # If radius != 1, we should have built neighbors differently.
        # Let's rebuild neighbors correctly for any radius to be safe.
        neighbors = []
        for i, w in enumerate(all_words):
            ball = set()
            # Generate all words within Hamming distance 'radius'
            # This is expensive for large n/r, but n=7 is small.
            # Use itertools to generate variations?
            # Or BFS?
            # For n=7, radius=1, it's fast. For radius=2, it's 1 + 14 + 21 = 36 words. Still small.
            # We can use a recursive generation or just check distance for all words (2187*2187 is ~5M ops, acceptable once).
            # Let's just check distance for all words if radius > 1 to be safe and simple.
            if radius == 1:
                 # Use the fast path from before
                 ball.add(i)
                 for pos in range(n):
                     for val in range(q):
                         if val == w[pos]: continue
                         new_w = w.copy()
                         new_w[pos] = val
                         ball.add(word_to_idx[tuple(new_w)])
            else:
                 for j, v in enumerate(all_words):
                     dist = sum(1 for a, b in zip(w, v) if a != b)
                     if dist <= radius:
                         ball.add(j)
            neighbors.append(list(ball))

        return greedy_construct()

    # Main Algorithm for Radius 1 (and potentially others if neighbors are correct)
    best_code = None
    best_size = float('inf')
    
    # Initial random seed
    random.seed(42)
    
    max_restarts = 50 # We will adjust based on time
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        # 1. Initialization
        # Start with a random subset of size ~ 1.2 * lower_bound
        # Lower bound for K_3(7,1) is 156.
        # Let's start with 200 random words.
        init_size = 200
        if init_size > num_words:
            init_size = num_words
            
        initial_idx = random.sample(range(num_words), init_size)
        current_code_idx = list(initial_idx)
        
        # 2. Ensure Coverage (Greedy Addition)
        # Calculate coverage
        coverage = [0] * num_words
        for idx in current_code_idx:
            for nb in neighbors[idx]:
                coverage[nb] += 1
        
        # If not fully covered, add words
        uncovered = [i for i in range(num_words) if coverage[i] == 0]
        while uncovered:
            if time.time() - start_time > timeout:
                break
            # Find the codeword that covers the most uncovered words
            best_add_idx = -1
            best_cover_count = 0
            # Optimization: Only check words that are not already in the code?
            # Or just any word.
            # To speed up, we can prioritize words that are close to uncovered regions.
            # Simple: iterate all words, count uncovered neighbors.
            # 2187 words * 15 neighbors = ~30k ops. Very fast.
            for w_idx in range(num_words):
                if w_idx in current_code_idx: # Skip if already present? 
                    # Actually, if it's in the code, its neighbors are already counted.
                    # We want to add a NEW word.
                    # Checking `in` on a list is O(N). N is ~200. 2187*200 = 400k ops. OK.
                    continue
                cnt = 0
                for nb in neighbors[w_idx]:
                    if coverage[nb] == 0:
                        cnt += 1
                if cnt > best_cover_count:
                    best_cover_count = cnt
                    best_add_idx = w_idx
            
            if best_add_idx == -1:
                break # Should not happen if space is connected
                
            current_code_idx.append(best_add_idx)
            for nb in neighbors[best_add_idx]:
                coverage[nb] += 1
                
            # Update uncovered list efficiently? 
            # Just recalculate or filter.
            uncovered = [i for i in uncovered if coverage[i] == 0]
            
        if time.time() - start_time > timeout:
            break
            
        # 3. Optimization (Pruning and Swaps)
        # We have a valid code now. Let's minimize it.
        code_set = set(current_code_idx)
        
        # Recalculate coverage for the current set
        coverage = [0] * num_words
        for idx in code_set:
            for nb in neighbors[idx]:
                coverage[nb] += 1
        
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > timeout:
                break
                
            # 3a. Pruning: Remove any codeword that doesn't reduce coverage below 1
            removable = []
            for idx in list(code_set):
                # Check if any neighbor has coverage 1
                is_removable = True
                for nb in neighbors[idx]:
                    if coverage[nb] == 1:
                        is_removable = False
                        break
                if is_removable:
                    removable.append(idx)
                    
            if removable:
                # Remove a random one
                idx_to_remove = random.choice(removable)
                code_set.remove(idx_to_remove)
                for nb in neighbors[idx_to_remove]:
                    coverage[nb] -= 1
                improved = True
                continue # Restart loop as coverage changed
                
            # 3b. Swaps: Try to replace one codeword with another to allow further pruning or just reduce size?
            # Swaps don't reduce size directly. They help escape local minima where no single removal is possible.
            # Strategy: Pick a codeword c_out. Remove it. Find c_in that covers the "lost" coverage.
            # If c_in covers all words that dropped to 0 (and maybe others), accept.
            # This maintains size but changes the set. Then we try to prune again.
            
            # To save time, we only do a few swap attempts per restart.
            if random.random() < 0.5: # 50% chance to attempt swap phase
                # Pick a random codeword to try removing
                candidates = list(code_set)
                random.shuffle(candidates)
                swap_found = False
                for c_out in candidates[:10]: # Try first 10 random candidates
                    if time.time() - start_time > timeout:
                        break
                    # Simulate removal
                    # Find words that become uncovered (coverage drops to 0)
                    exposed = []
                    for nb in neighbors[c_out]:
                        if coverage[nb] == 1:
                            exposed.append(nb)
                            
                    if not exposed:
                        # This is a prune case, handled above. Why is it here?
                        # Maybe coverage changed? No, we check removable first.
                        # So exposed should not be empty if it wasn't removable.
                        # Wait, if it was removable, we would have removed it in 3a.
                        # So here, exposed is non-empty.
                        pass
                        
                    # Find a codeword c_in that covers all 'exposed' words
                    # And ideally covers more to help future pruning.
                    best_c_in = -1
                    best_score = -1
                    for c_in in range(num_words):
                        if c_in in code_set:
                            continue
                        # Check if c_in covers all exposed
                        covers_all = True
                        extra_cover = 0
                        for nb in exposed:
                            if nb not in neighbors[c_in]:
                                covers_all = False
                                break
                        if covers_all:
                            # Score: how many words with coverage 1 does it cover? (to make them 2, safer for pruning)
                            # Or just how many total words it covers?
                            # Let's prioritize covering words with low coverage.
                            score = 0
                            for nb in neighbors[c_in]:
                                if coverage[nb] == 1:
                                    score += 10
                                elif coverage[nb] == 0:
                                    score += 100 # Should not happen if it covers exposed, but maybe other 0s?
                                else:
                                    score += 1
                            if score > best_score:
                                best_score = score
                                best_c_in = c_in
                                
                    if best_c_in != -1:
                        # Perform swap
                        code_set.remove(c_out)
                        for nb in neighbors[c_out]:
                            coverage[nb] -= 1
                            
                        code_set.add(best_c_in)
                        for nb in neighbors[best_c_in]:
                            coverage[nb] += 1
                            
                        improved = True # Trigger another round of pruning
                        swap_found = True
                        break # Restart swap search with new set
                        
                if swap_found:
                    continue # Loop back to pruning
            
        # End of optimization for this restart
        
        current_size = len(code_set)
        if current_size < best_size:
            best_size = current_size
            best_code = [all_words[i] for i in code_set]
            # If we hit the theoretical lower bound or best known, we can stop early?
            # Best known is 186. Lower bound 156.
            if best_size <= 186:
                # We might still find something smaller, but 186 is a strong target.
                # Let's run a few more iterations to see if we can beat 186.
                pass

    if best_code is None:
        # Fallback: return the initial tetracode construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return best_code

# EVOLVE-BLOCK-END

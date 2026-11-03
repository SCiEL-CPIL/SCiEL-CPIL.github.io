import itertools
import random
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        # Simple greedy
        total_words = q ** n
        words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        # Precompute neighbors
        neighbors = [[] for _ in range(total_words)]
        for i, w in enumerate(words):
            nbrs = [i]
            for pos in range(n):
                for new_digit in range(q):
                    if new_digit != w[pos]:
                        new_w = list(w)
                        new_w[pos] = new_digit
                        new_w = tuple(new_w)
                        nbrs.append(word_to_idx[new_w])
            neighbors[i] = nbrs
        
        covered = [False] * total_words
        code = []
        while any(not c for c in covered):
            best_count = -1
            best_candidates = []
            for w in range(total_words):
                count = sum(1 for nbr in neighbors[w] if not covered[nbr])
                if count > best_count:
                    best_count = count
                    best_candidates = [w]
                elif count == best_count and count > 0:
                    best_candidates.append(w)
            selected = random.choice(best_candidates)
            code.append(selected)
            for nbr in neighbors[selected]:
                covered[nbr] = True
        return [list(words[w]) for w in code]
    
    # Optimized path for q=3, n=6, radius=1
    total_words = 729
    
    # Precompute neighbors for each word (integer representation)
    neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        nbrs = [w]
        for pos in range(n):
            div = 3 ** (n - 1 - pos)
            curr_digit = (w // div) % q
            for new_digit in range(q):
                if new_digit != curr_digit:
                    new_w = w + (new_digit - curr_digit) * div
                    nbrs.append(new_w)
        neighbors[w] = nbrs
    
    # Convert to numpy arrays for speed
    neighbors_np = [np.array(nbrs) for nbrs in neighbors]

    def greedy_init(rng):
        covered = np.zeros(total_words, dtype=bool)
        code = []
        while not covered.all():
            # Find best candidate
            best_count = -1
            best_candidates = []
            # Vectorized calculation of uncovered neighbors
            # For each word, count how many of its neighbors are not covered
            # This is slow if done naively. 
            # Optimization: 
            # We can maintain an array of "uncovered count" for each potential center?
            # No, simple loop is probably fine for 729 words.
            
            # To speed up, we can precompute the neighbor lists as arrays.
            # Let's just iterate.
            for w in range(total_words):
                nbrs = neighbors_np[w]
                count = np.sum(~covered[nbrs])
                if count > best_count:
                    best_count = count
                    best_candidates = [w]
                elif count == best_count and count > 0:
                    best_candidates.append(w)
            
            if best_count == 0:
                break # Should not happen if not covered.all()
                
            selected = rng.choice(best_candidates)
            code.append(selected)
            covered[neighbors_np[selected]] = True
            
        return code

    def local_search(code, rng):
        # Convert to list of ints
        code = list(code)
        # Initialize cover counts
        cover_count = np.zeros(total_words, dtype=int)
        for c in code:
            cover_count[neighbors_np[c]] += 1
        
        improved = True
        max_swaps = 500 # Limit swaps to avoid infinite loops
        
        while improved:
            improved = False
            # Deletion Phase
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Remove c
                cover_count[neighbors_np[c]] -= 1
                if np.min(cover_count) > 0:
                    code.pop(i)
                    improved = True
                    break # Restart deletion
                else:
                    # Restore c
                    cover_count[neighbors_np[c]] += 1
            
            if improved:
                continue

            # Swap Phase
            # Find a codeword with unique points
            # We want to swap a codeword to make the code "more redundant"
            
            # Identify candidates for swap: codewords that have unique points
            # If a codeword has no unique points, it should have been deleted.
            
            # To be efficient, we can just pick a random codeword and try to swap it.
            # But we must ensure the swap is valid (covers unique points).
            
            # Let's try to swap each codeword once per pass?
            # Or just pick a random one.
            
            # Strategy: Pick a random index i.
            # Find U_i (unique points of code[i]).
            # Find a word w that covers U_i.
            # If found, swap.
            
            # To avoid getting stuck, we try a few random swaps if the first one fails or leads to a loop.
            
            for _ in range(max_swaps):
                if not code:
                    break
                
                i = rng.randint(0, len(code) - 1)
                c = code[i]
                
                # Find unique points of c
                # Points p where cover_count[p] == 1 and p in neighbors[c]
                nbrs = neighbors_np[c]
                # mask for nbrs where cover_count is 1
                unique_mask = (cover_count[nbrs] == 1)
                unique_points = nbrs[unique_mask]
                
                if len(unique_points) == 0:
                    # Should have been deleted, but maybe due to order?
                    # Try to delete again?
                    # If cover_count[p] == 1 for all p in nbrs[c], and we remove c, they become 0.
                    # Wait, if len(unique_points) == 0, it means NO point is uniquely covered by c.
                    # So removing c would leave all its neighbors with count >= 1.
                    # So c CAN be deleted.
                    # Why didn't the deletion phase catch it?
                    # Maybe it was checked earlier and failed? No, deletion checks all.
                    # If it's here, it means the deletion phase passed it.
                    # This implies there IS a unique point.
                    # Let's double check.
                    # If unique_points is empty, it means for all p in nbrs[c], cover_count[p] != 1.
                    # Since c covers p, cover_count[p] >= 1.
                    # So cover_count[p] >= 2 for all p in nbrs[c].
                    # Then removing c would result in cover_count[p] >= 1 for all p in nbrs[c].
                    # And for p not in nbrs[c], cover_count[p] is unchanged and >= 1.
                    # So min(cover_count) would be >= 1.
                    # So it SHOULD be deletable.
                    # If it wasn't deleted, maybe my deletion logic is flawed?
                    # Ah, the deletion loop breaks after the first success.
                    # If this c is later in the list, it might have been skipped?
                    # No, the loop goes from end to start.
                    # If we are in Swap phase, it means NO deletion was possible in the last pass.
                    # So for ALL i, removing code[i] results in min(cover_count) == 0.
                    # This implies for every i, there exists at least one p in nbrs[code[i]] with cover_count[p] == 1.
                    # So unique_points should NOT be empty.
                    # If it is, there is a bug.
                    # Let's force a deletion check.
                    cover_count[neighbors_np[c]] -= 1
                    if np.min(cover_count) > 0:
                        code.pop(i)
                        improved = True
                        break # Restart whole while loop
                    else:
                        cover_count[neighbors_np[c]] += 1
                        continue # Try next swap candidate
                
                # Find a word w that covers all unique_points
                # w must be in neighbors of all p in unique_points?
                # No, w must cover p, so p must be in neighbors[w].
                # So w must be in the intersection of neighbors[p] for all p in unique_points.
                
                # Efficient way:
                # Start with all 729 words.
                # Filter by: w covers unique_points[0]
                # Filter by: w covers unique_points[1]
                # ...
                
                # Precompute reverse neighbors?
                # rev_neighbors[p] = list of words that cover p.
                # This is just neighbors[p] in our symmetric definition?
                # Yes, distance is symmetric.
                
                candidates = set(range(total_words))
                for p in unique_points:
                    candidates = candidates.intersection(set(neighbors_np[p].tolist()))
                    
                if not candidates:
                    continue # No valid swap for this c
                
                # Pick a random candidate
                # Prefer candidates that are not c?
                # If c is in candidates, swapping c with c is useless.
                candidates_list = list(candidates)
                if c in candidates_list:
                    candidates_list.remove(c)
                
                if not candidates_list:
                    continue
                    
                w = rng.choice(candidates_list)
                
                # Perform Swap
                # Remove c, Add w
                cover_count[neighbors_np[c]] -= 1
                cover_count[neighbors_np[w]] += 1
                code[i] = w
                improved = True # Structure changed
                break # Restart deletion phase
            
            if not improved and len(code) > 0:
                # No swap found that changes structure?
                # This can happen if all valid swaps are self-swaps or we ran out of swaps.
                # Break to avoid infinite loop.
                break

        return code

    best_code = None
    best_len = float('inf')
    
    # Run multiple restarts
    for restart in range(20):
        rng = random.Random(1000 + restart)
        code = greedy_init(rng)
        code = local_search(code, rng)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
            
        # Early stop if we hit the known lower bound? 71 is lower bound, 73 is best known.
        # If we find 73, we can stop? No, we want to minimize.
        # If we find 71, it's a breakthrough.
        
    # Convert integer codes back to lists of integers
    final_code = []
    for w in best_code:
        word = [0] * n
        temp = w
        for i in range(n-1, -1, -1):
            word[i] = temp % q
            temp //= q
        final_code.append(word)
        
    return final_code

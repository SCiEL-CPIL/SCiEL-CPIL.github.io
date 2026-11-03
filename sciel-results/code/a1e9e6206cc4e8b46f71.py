import time
import random
import numpy as np
import sys

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: 150s limit, stop at 145s
    DEADLINE = start_time + 145.0
    
    # Parameters for the specific task: q=3, n=8, radius=1
    N = q ** n
    
    # Precomputation: Map word indices to their neighbors (distance <= radius)
    # For q=3, n=8, radius=1, N=6561.
    # Neighbors count: 1 + n*(q-1) = 17
    
    pos_step = [q ** (n - 1 - i) for i in range(n)]
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = (idx // pos_step[i]) % q
        return word

    def word_to_idx(word):
        idx = 0
        for i, val in enumerate(word):
            idx += val * pos_step[i]
        return idx

    # Precompute neighbors for all N words
    # neighbors[i] is a list of indices
    neighbors = [[] for _ in range(N)]
    neighbor_sets = [set() for _ in range(N)]
    
    for i in range(N):
        w = idx_to_word(i)
        neighbors[i].append(i)
        neighbor_sets[i].add(i)
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    neighbor_idx = i + (new_val - orig_val) * pos_step[pos]
                    neighbors[i].append(neighbor_idx)
                    neighbor_sets[i].add(neighbor_idx)
    
    # Data structures for the search
    cover_count = np.zeros(N, dtype=np.int32)
    in_code = np.zeros(N, dtype=bool)
    code_list = []
    
    # 1. Greedy Initialization
    uncovered_mask = np.ones(N, dtype=bool)
    remaining = N
    
    while remaining > 0:
        if time.time() > DEADLINE - 10:
            break
            
        best_idx = -1
        best_score = -1
        
        # Iterate through all possible codewords
        for c in range(N):
            if in_code[c]:
                continue
            
            # Score: number of neighbors of c that are currently uncovered
            score = 0
            for nb in neighbors[c]:
                if uncovered_mask[nb]:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_idx = c
                
        if best_idx == -1 or best_score == 0:
            break
            
        # Add best_idx to code
        code_list.append(best_idx)
        in_code[best_idx] = True
        
        # Update coverage
        for nb in neighbors[best_idx]:
            if not in_code[nb]: # Optimization: only update if not already covered? No, cover_count is global.
                pass
            cover_count[nb] += 1
            if cover_count[nb] == 1:
                uncovered_mask[nb] = False
                remaining -= 1

    # 2. Local Search Refinement
    
    def is_valid():
        return np.all(cover_count >= 1)
        
    def try_remove_redundant():
        """Remove any codeword that doesn't uniquely cover any word."""
        removed = False
        i = 0
        while i < len(code_list):
            if time.time() > DEADLINE - 5:
                break
                
            c = code_list[i]
            
            # Check if c is redundant
            # c is redundant if for all nb in neighbors[c], cover_count[nb] > 1
            redundant = True
            for nb in neighbors[c]:
                if cover_count[nb] == 1:
                    redundant = False
                    break
                    
            if redundant:
                # Remove c
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                in_code[c] = False
                code_list.pop(i)
                removed = True
                # Do not increment i
            else:
                i += 1
                
        return removed

    def update_cover_add(c):
        for nb in neighbors[c]:
            cover_count[nb] += 1
        in_code[c] = True
        code_list.append(c)

    def update_cover_remove(c):
        for nb in neighbors[c]:
            cover_count[nb] -= 1
        in_code[c] = False

    # Pass 1: Remove all redundant words
    while try_remove_redundant():
        pass
        
    # Pass 2: Swap Optimization using Inverse-Index Enumeration
    # We iterate until no improvement is found in a full pass
    
    improved = True
    passes = 0
    while improved and time.time() < DEADLINE - 5:
        improved = False
        passes += 1
        # Randomize order to avoid local minima traps due to ordering
        indices = list(range(len(code_list)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > DEADLINE - 5:
                break
            if i >= len(code_list):
                continue
                
            c = code_list[i]
            
            # 1. Identify critical words (uniquely covered by c)
            critical = []
            for nb in neighbors[c]:
                if cover_count[nb] == 1:
                    critical.append(nb)
                    
            if not critical:
                # c is redundant
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                in_code[c] = False
                code_list.pop(i)
                improved = True
                break # Restart loop as indices changed
            
            # 2. Find candidate replacements
            # Candidates must cover all critical words.
            # candidate c' must be in neighbors[h] for all h in critical.
            
            # Start with the neighbors of the first critical word
            # Optimization: Pick the critical word with the fewest neighbors? 
            # All have 17 neighbors.
            # So just take the set of neighbors of critical[0].
            candidates = set(neighbors[critical[0]])
            
            # Intersect with neighbors of other critical words
            for h in critical[1:]:
                candidates.intersection_update(neighbors[h])
                
            # Remove current codewords from candidates
            # We can keep c itself? No, we are replacing c.
            # If we keep c, it's not a swap, it's a no-op.
            # But we removed c from 'in_code' logically? No, we haven't modified state yet.
            # We are looking for c' != c such that c' covers all criticals.
            # Also c' should not be in the current code (unless it's c, but we are removing c).
            # If c' is in the code, adding it is invalid (duplicates not allowed, though 
            # the problem says "codewords must be distinct").
            
            valid_candidates = []
            for c_prime in candidates:
                if c_prime == c:
                    continue
                if in_code[c_prime]:
                    continue
                valid_candidates.append(c_prime)
                
            if valid_candidates:
                # Pick a candidate.
                # Heuristic: Pick the one that maximizes coverage of non-critical words?
                # Or just pick the first one?
                # To be more effective, we can score them by how many "extra" words they cover
                # that are currently covered by multiple codewords (reducing redundancy elsewhere).
                
                best_c_prime = valid_candidates[0]
                best_score = -1
                
                for c_prime in valid_candidates:
                    # Score: Sum of (cover_count[nb] - 1) for nb in neighbors[c_prime]
                    # This measures how much "redundant" coverage we are adding.
                    # We want to minimize this? Or maximize?
                    # If we add c_prime, cover_count increases for its neighbors.
                    # If those neighbors already had count > 1, we are adding waste.
                    # If they had count == 1, we are making them count == 2 (still valid, but less robust).
                    # Actually, we just want to maintain validity.
                    # A better heuristic: Pick the c_prime that results in the most 
                    # immediate redundancy removals? Hard to calculate.
                    # Let's just pick the first one for speed, or randomize.
                    # Randomizing helps escape local minima.
                    best_c_prime = random.choice(valid_candidates)
                    break # Just pick one randomly to keep it fast and diverse
                    
                # Perform Swap
                # Remove c
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                in_code[c] = False
                code_list[i] = best_c_prime # Replace in list
                
                # Add best_c_prime
                for nb in neighbors[best_c_prime]:
                    cover_count[nb] += 1
                in_code[best_c_prime] = True
                
                # Check for new redundancies
                if try_remove_redundant():
                    improved = True
                    break # Restart loop
                else:
                    # No immediate redundancy removed.
                    # Should we keep the swap?
                    # If we revert, we might cycle.
                    # If we keep, we might enable future swaps.
                    # Given the "inverse index" finds *all* valid swaps, 
                    # accepting one that doesn't immediately reduce size might be risky.
                    # However, if we revert, we might get stuck.
                    # Let's accept the swap if it doesn't increase size (it doesn't) 
                    # and mark improved=False for this specific step, 
                    # but continue searching for other swaps.
                    # Actually, if no redundancy is removed, the size is the same.
                    # We will just keep the swap.
                    # But to avoid infinite loops of swaps with no gain, 
                    # we should probably only accept if it leads to a gain?
                    # Let's try: Accept swap. If no gain, revert?
                    # If we revert, we are back to square one.
                    # Let's accept the swap. The randomization of indices and 
                    # random choice of candidate should help.
                    pass

    # Final cleanup
    try_remove_redundant()
    
    # Convert indices back to words
    result = [idx_to_word(i) for i in code_list]
    
    return result

# EVOLVE-BLOCK-END

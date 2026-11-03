"""
Improved Ternary Covering Code Constructor for K_3(8,1)
Combines Greedy Set Cover initialization with Local Search (Redundancy Removal + Swaps).
"""
import time
import random
import numpy as np
import math
import sys

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: 150s limit, stop at 145s
    DEADLINE = start_time + 145.0
    
    # Parameters for the specific task: q=3, n=8, radius=1
    # But keep it general enough to not crash if params differ slightly, 
    # though the task contract implies fixed q,n,r for the score.
    
    N = q ** n
    
    # Precomputation: Map word indices to their neighbors (distance <= radius)
    # For q=3, n=8, radius=1, N=6561.
    # Neighbors count: 1 + n*(q-1) = 1 + 8*2 = 17? 
    # Wait. Distance 1: change 1 position to 1 of (q-1) other values.
    # So 1 (self) + n*(q-1) neighbors.
    # For q=3, n=8: 1 + 8*2 = 17 neighbors.
    
    # Helper to convert index to word and vice versa
    # Using base-q representation
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
    for i in range(N):
        w = idx_to_word(i)
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    # Calculate index of neighbor
                    # delta = (new_val - orig_val) * pos_step[pos]
                    # neighbor_idx = i + delta
                    # This works because we are just changing one digit in base-q
                    # and no carry-over happens since we are changing a single digit 
                    # to another digit in 0..q-1.
                    # Wait, is this always true?
                    # Yes, if we treat the word as a number in base q, changing digit d 
                    # to d' changes the value by (d'-d)*q^k. No carry/borrow.
                    neighbor_idx = i + (new_val - orig_val) * pos_step[pos]
                    neighbors[i].append(neighbor_idx)
        # If radius > 1, this logic needs extension, but task is radius 1.
        # We assume radius 1 for the specific task.
        # If radius != 1, we fallback to a slower generic method or just handle r=1.
        # The task says "checked over the whole space", implying general q,n,r.
        # But the specific instance is q=3, n=8, r=1.
        # To be safe, if radius != 1, we use a slower generation.
    
    if radius != 1:
        # Generic neighbor generation for small N
        # This is only for robustness, not the main path
        neighbors = []
        for i in range(N):
            w = idx_to_word(i)
            nb = set([i])
            if radius >= 1:
                # Generate all words within distance radius
                # For small N, brute force is fine
                for dist in range(1, radius + 1):
                    for positions in __import__('itertools').combinations(range(n), dist):
                        for vals in __import__('itertools').product(range(q), repeat=dist):
                            if all(vals[j] == w[positions[j]] for j in range(dist)):
                                continue
                            new_w = w.copy()
                            for j, pos in enumerate(positions):
                                new_w[pos] = vals[j]
                            nb.add(word_to_idx(new_w))
            neighbors.append(list(nb))

    # Data structures for the search
    # cover_count: how many codewords cover each word
    cover_count = np.zeros(N, dtype=np.int32)
    # in_code: boolean mask
    in_code = np.zeros(N, dtype=bool)
    # code_list: list of indices currently in the code
    code_list = []
    
    # 1. Greedy Initialization
    # We want to pick words that cover the most *uncovered* words.
    # uncovered_mask: boolean array
    uncovered_mask = np.ones(N, dtype=bool)
    
    # To speed up greedy, we can maintain a list of candidate scores.
    # But N=6561 is small enough to just scan all candidates each time?
    # Greedy steps: roughly K steps. K ~ 500.
    # 500 * 6561 * (cost to score)
    # Cost to score: iterate over neighbors (17). 500 * 6561 * 17 ~ 55M ops.
    # In Python, 55M simple ops might take ~5-10 seconds.
    # We can optimize by using NumPy or just accepting the time if it's < 10s.
    # Let's try to optimize the scoring.
    
    # We can precompute neighbor arrays as NumPy arrays for faster access?
    # Or just use lists.
    
    # Greedy Loop
    while np.any(uncovered_mask):
        if time.time() > DEADLINE - 10:
            break
            
        best_idx = -1
        best_score = -1
        
        # Iterate through all possible codewords
        # We can skip words already in code? Yes.
        # We can also prune: if a word is already fully covered, its score is 0.
        
        for c in range(N):
            if in_code[c]:
                continue
            
            # Score: number of neighbors of c that are currently uncovered
            # neighbors[c] is a list of indices
            # We need to count how many are in uncovered_mask
            # Using a loop:
            score = 0
            for nb in neighbors[c]:
                if uncovered_mask[nb]:
                    score += 1
                    if score > best_score:
                        # Early termination if we can't beat best? 
                        # Max possible score is len(neighbors[c])
                        # If best_score is already max, we can break?
                        # No, we just want the best.
                        pass
            
            if score > best_score:
                best_score = score
                best_idx = c
                
            # Optimization: If best_score reaches the size of the ball (17), 
            # we can't do better.
            if best_score == len(neighbors[c]): 
                 # Actually, different c might have different neighbor counts? No, all same for r=1.
                 # So if best_score == 17, we can break the loop?
                 # Yes, 17 is the max.
                 # But we found best_idx. We can break.
                 # However, there might be multiple with 17. Any is fine.
                 # Let's break to save time.
                 # But wait, we are inside the `for c` loop.
                 # We should break the `for c` loop.
                 pass
        
        if best_score == len(neighbors[best_idx]) and best_idx != -1:
             # We found a perfect cover for this step. 
             # We can't break the `for c` loop from here easily without a flag.
             # Let's just let it finish or add a flag.
             # For simplicity, we'll let it run, 6561 iterations is fast.
             pass
        
        if best_idx == -1:
            # Should not happen if N > 0
            break
            
        # Add best_idx to code
        code_list.append(best_idx)
        in_code[best_idx] = True
        
        # Update coverage
        for nb in neighbors[best_idx]:
            cover_count[nb] += 1
            if cover_count[nb] == 1:
                uncovered_mask[nb] = False

    # 2. Local Search Refinement
    
    def is_valid():
        return np.all(cover_count >= 1)
        
    def try_remove_redundant():
        """Remove any codeword that doesn't uniquely cover any word."""
        removed = False
        # Iterate through a copy of the list to allow popping
        # Or iterate indices
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
                    # This word is uniquely covered by c
                    # So c is NOT redundant
                    redundant = False
                    break
                    
            if redundant:
                # Remove c
                update_cover_remove(c)
                code_list.pop(i)
                in_code[c] = False
                removed = True
                # Do not increment i, check the next element at the same index
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
        # Note: code_list removal is handled by the caller (pop)
        
    # Pass 1: Remove all redundant words
    while try_remove_redundant():
        pass
        
    # Pass 2: Swap Optimization
    # Try to swap c with c' if it allows removal of another word or maintains validity with same size?
    # Actually, the goal is to minimize size.
    # A swap keeps size constant.
    # But a swap might make another word redundant.
    # So:
    # 1. Try swap c -> c'.
    # 2. If valid, check if any word (including c' or others) is now redundant.
    # 3. If size decreased, accept.
    
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        
        # Iterate through current code
        # We need to be careful with indices if we modify the list
        # We will collect potential moves and apply one at a time
        
        # To speed up, we can randomize the order of c
        indices = list(range(len(code_list)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > DEADLINE - 5:
                break
                
            if i >= len(code_list):
                continue
                
            c = code_list[i]
            
            # Tentatively remove c
            for nb in neighbors[c]:
                cover_count[nb] -= 1
            in_code[c] = False
            # Remove from list? 
            # To keep list consistent, we can just mark it and handle logic, 
            # but `code_list` is used for iteration. 
            # Let's just use `in_code` and `cover_count` for logic, 
            # and `code_list` for the set of active words.
            # We will manipulate `code_list` carefully.
            
            # Find holes
            holes = []
            for nb in neighbors[c]:
                if cover_count[nb] == 0:
                    holes.append(nb)
                    
            if not holes:
                # c was redundant!
                # This should have been caught by try_remove_redundant, 
                # but maybe it became redundant due to previous swaps.
                # Just remove it.
                code_list.pop(i)
                # We already decremented cover_count.
                # We don't need to update cover_count again.
                improved = True
                # Break to restart loop as indices changed
                break
            else:
                # We need to find a c' that covers all holes.
                # c' must not be in code (except c, which is removed).
                # We want to maximize coverage of holes.
                # Ideally, c' covers ALL holes.
                
                best_c_prime = -1
                best_holes_covered = 0
                total_holes = len(holes)
                
                # Iterate through all possible c'
                # Optimization: Only consider c' that is NOT in code.
                # We can iterate all N and check in_code.
                
                for c_prime in range(N):
                    if in_code[c_prime]:
                        continue
                    if c_prime == c:
                        continue # c is effectively removed (in_code False)
                        
                    # Check how many holes c_prime covers
                    holes_covered = 0
                    for h in holes:
                        # Check if h is in neighbors[c_prime]
                        # This is O(len(neighbors)) = 17
                        # We can precompute a boolean matrix? 6561x6561 is too big.
                        # Just loop.
                        # Optimization: 
                        # Instead of checking each h in holes against neighbors[c_prime],
                        # we can check if h is in neighbors[c_prime].
                        # Since holes are small (usually 1-3), this is fast.
                        
                        # To check "is h in neighbors[c_prime]", we can use a set?
                        # Or just linear search in neighbors[c_prime] (size 17).
                        found = False
                        for nb in neighbors[c_prime]:
                            if nb == h:
                                found = True
                                break
                        if found:
                            holes_covered += 1
                            
                    if holes_covered > best_holes_covered:
                        best_holes_covered = holes_covered
                        best_c_prime = c_prime
                        
                    if holes_covered == total_holes:
                        # We found a perfect cover for the holes.
                        # We can break early? 
                        # Yes, this is the best possible.
                        break
                        
                if best_holes_covered == total_holes and best_c_prime != -1:
                    # We can swap c with c_prime
                    # Apply swap
                    # c is already removed from cover_count and in_code
                    # Add c_prime
                    for nb in neighbors[best_c_prime]:
                        cover_count[nb] += 1
                    in_code[best_c_prime] = True
                    code_list[i] = best_c_prime # Replace in list
                    
                    # Now check if any word is redundant
                    # We can call try_remove_redundant
                    if try_remove_redundant():
                        improved = True
                        # Break to restart
                        break
                    else:
                        # No size reduction. 
                        # Should we keep the swap?
                        # It might enable future swaps.
                        # But if we don't reduce size, we might cycle.
                        # Let's keep it if it doesn't increase size (it doesn't).
                        # But to avoid cycling, we can revert if no immediate gain?
                        # Or keep it.
                        # Let's keep it, but mark improved=False for this specific swap 
                        # unless we find a reduction.
                        # Actually, if we keep it, we might get stuck in a loop of swaps.
                        # Better to only accept if it leads to a removal.
                        # So, revert the swap.
                        for nb in neighbors[best_c_prime]:
                            cover_count[nb] -= 1
                        in_code[best_c_prime] = False
                        # Restore c
                        for nb in neighbors[c]:
                            cover_count[nb] += 1
                        in_code[c] = True
                        code_list[i] = c
                        
                else:
                    # Cannot fix holes with a single swap.
                    # Revert removal of c
                    for nb in neighbors[c]:
                        cover_count[nb] += 1
                    in_code[c] = True
                    # code_list[i] is still c

    # Final cleanup
    try_remove_redundant()
    
    # Convert indices back to words
    result = [idx_to_word(i) for i in code_list]
    
    return result

# EVOLVE-BLOCK-END

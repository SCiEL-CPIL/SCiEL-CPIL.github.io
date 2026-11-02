"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    # For the specific case q=3, n=6, radius=1, we know the optimal size is 73.
    # We implement a fast search to find a covering code of size 73.
    # If time runs out, we fall back to the direct sum construction (size 81).
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: direct sum of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Precompute neighbors for all 729 words
    # neighbors[i] is a list of indices j such that dist(i, j) <= 1
    # Actually, for set cover, we want for each codeword c, the set of words it covers.
    # Let's use numpy for speed.
    
    start_time = time.time()
    deadline = 110.0  # Leave 10s safety margin
    
    # Generate all 729 words as base-3 numbers 0..728
    # Convert number to base-3 vector of length 6
    def num_to_vec(num, length=6):
        vec = [0] * length
        for i in range(length):
            vec[i] = num % 3
            num //= 3
        return vec
    
    # Precompute coverage sets
    # cover_set[i] is a bitmask or set of words covered by word i
    # Since 729 words, we can use a numpy array of shape (729, 729) boolean or just lists of sets.
    # Using numpy bitsets might be faster for intersection/union, but let's try simple sets first.
    # Or even better, since we want to maximize coverage, we can represent coverage as a set of integers.
    
    # Let's precompute the list of words covered by each possible codeword.
    # covered_by[i] = set of j such that dist(i, j) <= 1
    covered_by = [None] * 729
    for i in range(729):
        v = num_to_vec(i)
        covered = set()
        covered.add(i)
        for pos in range(6):
            orig = v[pos]
            for new_val in range(3):
                if new_val != orig:
                    v[pos] = new_val
                    # convert back to num
                    num = 0
                    for k in range(6):
                        num = num * 3 + v[5-k] # Wait, num_to_vec uses little-endian?
                        # Let's stick to one convention.
                        # My num_to_vec: v[0] is least significant digit.
                        # So num = sum(v[i] * 3**i)
                    # Let's rewrite num_to_vec and vec_to_num properly.
        
        # Let's restart the precomputation logic cleanly.
        pass

    def vec_to_num(vec):
        num = 0
        for i in range(len(vec)):
            num += vec[i] * (3 ** i)
        return num

    def num_to_vec(num):
        vec = [0] * 6
        for i in range(6):
            vec[i] = num % 3
            num //= 3
        return vec

    covered_by = [set() for _ in range(729)]
    for i in range(729):
        v = num_to_vec(i)
        covered_by[i].add(i)
        for pos in range(6):
            orig = v[pos]
            for new_val in range(3):
                if new_val != orig:
                    v[pos] = new_val
                    j = vec_to_num(v)
                    covered_by[i].add(j)
            v[pos] = orig

    # Greedy Set Cover
    # We want to select a subset of codewords S such that union(covered_by[s] for s in S) == {0..728}
    # Minimize |S|.
    
    # We can try to force size 73.
    # Start with a random subset or greedy, then local search.
    
    # A known good starting point: The direct sum gives 81.
    # Let's try to reduce it.
    
    # Heuristic: 
    # 1. Start with a random set of size 73? No, unlikely to cover.
    # 2. Start with greedy set cover, then try to remove/swap to reduce size.
    
    # Greedy Set Cover:
    # At each step, pick the codeword that covers the most *uncovered* words.
    
    best_code = None
    best_size = 81 # Initial fallback
    
    # Try a few random restarts with different greedy strategies or initial seeds?
    # Given time constraints, let's do one robust greedy + local search.
    
    # Greedy:
    uncovered = set(range(729))
    selected = []
    while uncovered:
        if time.time() > deadline:
            break
        best_c = -1
        max_gain = 0
        # To speed up, we can iterate over all 729 candidates.
        for c in range(729):
            if c in selected: # Optimization: skip already selected? No, we add to list.
                continue
            gain = len(covered_by[c] & uncovered)
            if gain > max_gain:
                max_gain = gain
                best_c = c
        if max_gain == 0:
            # Should not happen if space is connected
            break
        selected.append(best_c)
        uncovered -= covered_by[best_c]
        
    if not uncovered:
        best_code = selected
        best_size = len(selected)
    
    # Now try to reduce the size from best_size (likely ~80-90) to 73.
    # Local Search:
    # 1. Try removing a codeword if coverage is still complete.
    # 2. Try swapping a codeword with another if it reduces size or maintains size but allows further reduction.
    
    if best_code is not None:
        code_set = set(best_code)
        # Verify coverage
        all_covered = set()
        for c in code_set:
            all_covered |= covered_by[c]
        
        # Iterative improvement
        # We want to reach size 73.
        # If current size > 73, we need to remove elements or swap.
        # Simple approach: Randomly remove an element, check coverage. If valid, keep it.
        # If not, try to find a replacement that covers the lost words while allowing removal of another?
        # That's complex. 
        # Alternative: Start with the greedy result. 
        # While size > 73:
        #   Try to remove a random codeword. If coverage is lost, try to add a different one that covers the lost words and allows removal of the original?
        #   Actually, just "remove if possible" is very effective for set cover heuristics.
        
        # Let's try "remove if possible" repeatedly.
        changed = True
        while changed and len(code_set) > 73 and time.time() < deadline:
            changed = False
            # Shuffle to break symmetry
            candidates = list(code_set)
            random.shuffle(candidates)
            for c in candidates:
                if len(code_set) <= 73:
                    break
                # Try removing c
                code_set.remove(c)
                # Check coverage
                # We can maintain a count of how many times each word is covered to make this O(1) per removal check?
                # Let's just recompute coverage for now, 729 * 73 operations is small.
                all_covered = set()
                for cc in code_set:
                    all_covered |= covered_by[cc]
                if len(all_covered) == 729:
                    changed = True
                    break # Restart loop with new set
                else:
                    code_set.add(c) # Restore
            
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            
        # If we are still > 73, we might need swaps.
        # But "remove if possible" often gets close to optimal for these small instances.
        # If we hit 73, we are done.
        
        if len(code_set) > 73:
             # Try swapping
             # Pick a codeword c to remove.
             # Find a codeword c' not in set such that (set - {c} + {c'}) covers all.
             # And ideally, this helps remove more later.
             # This is expensive. 
             # Given the time, let's see if "remove if possible" gets us there.
             # For K_3(6,1), greedy usually starts around 80-85. Removing redundant ones might get us to 73.
             pass

    # Final check and return
    if best_code is not None and len(best_code) <= 81:
        # Verify one last time
        all_covered = set()
        for c in best_code:
            all_covered |= covered_by[c]
        if len(all_covered) == 729:
            return [num_to_vec(c) for c in best_code]
    
    # Fallback
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

# EVOLVE-BLOCK-END

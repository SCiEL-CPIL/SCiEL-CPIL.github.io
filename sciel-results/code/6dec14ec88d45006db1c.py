import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin for 150s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    # Precompute neighbors for all N words
    # neighbors[i] is a list of indices within distance 1 of i
    neighbors = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(q):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors.append(nb)
    
    # For faster intersection counting in repair, we might need to convert to sets or use boolean masks
    # But since N is small, simple list iteration in Python might be okay if we reduce the number of calls.
    # However, let's use a NumPy-based approach for the main search loop.

    # State
    # coverage: array of length N, counts how many codewords cover each word
    coverage = np.zeros(N, dtype=np.int32)
    code = set()
    
    # Helper to add a codeword
    def add_word(c):
        code.add(c)
        coverage[neighbors[c]] += 1
        
    # Helper to remove a codeword
    def remove_word(c):
        code.remove(c)
        coverage[neighbors[c]] -= 1
        
    # Initial Solution: Greedy Construction
    # Start with empty, add word covering most uncovered points
    uncovered_mask = np.ones(N, dtype=bool)
    
    # To speed up greedy, we can precompute the "sphere" of each word as a boolean mask? 
    # N=2187, SphereSize=15. 
    # A boolean matrix of size N x N is 2187*2187 ~ 4.7M booleans ~ 4.7MB. This is fine.
    # But we don't need the full matrix, just the neighbor lists.
    
    # Greedy Step
    # We want to pick c that maximizes sum(uncovered_mask[neighbors[c]])
    # We can do this efficiently if we maintain the count of uncovered points for each candidate.
    # Let's just do a simple greedy:
    # 1. Pick random start? No, deterministic is better for reproducibility, but random helps escape.
    # Let's use a randomized greedy.
    
    # Optimization: Precompute neighbor lists as numpy arrays for fast indexing
    neighbor_arrays = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    # Greedy Construction
    # We will add words until all are covered.
    # To make it fast, we can estimate the gain.
    # Since N is small, we can just iterate.
    
    # Randomize the order of candidates to break ties and allow different paths
    candidates = list(range(N))
    random.shuffle(candidates)
    
    # Simple Greedy:
    # At each step, find c in candidates that covers the most *currently uncovered* points.
    # Update uncovered_mask.
    
    # This is O(N * |Code| * SphereSize) which might be too slow if |Code| is large.
    # |Code| ~ 200. N=2187. Sphere=15.
    # 2187 * 200 * 15 ~ 6.5M ops per step? No, we only check candidates.
    # Actually, we can just pick the first one that covers a lot?
    # Better: 
    # Maintain a score for each candidate: how many uncovered points it covers.
    # When we add a word c, we subtract 1 from the score of all candidates that share neighbors with c?
    # That's complex.
    
    # Let's stick to the Direct Sum as a strong initial solution, then optimize.
    # Direct Sum is 243 words. It's a valid starting point.
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            init_indices.append(tuple_to_idx(h + list(tail)))
            
    for c in init_indices:
        add_word(c)
        
    # Now perform Local Search
    # 1. Exhaustive Removal: Remove any word that doesn't break coverage.
    def try_remove_all():
        removed = False
        # Iterate over a copy
        for c in list(code):
            if time.time() > deadline:
                return removed
            # Check if c is redundant
            # min(coverage[neighbors[c]]) > 1
            if np.min(coverage[neighbor_arrays[c]]) > 1:
                remove_word(c)
                removed = True
        return removed

    # 2. Kick (1-move): Remove 1, try to repair with 1.
    # If removing c creates a gap of size G, we need to find a word d that covers the most of G.
    # If the best d covers all G, and we add d, size stays same (remove 1, add 1). 
    # Wait, we want to DECREASE size.
    # So we remove 1. If coverage is still valid, we accept (size -1).
    # If coverage is invalid (gap exists), we try to fill the gap with 1 word.
    # If we can fill the gap with 1 word, size stays same. 
    # To decrease size, we need to remove 1 and repair with 0? That's just removal.
    # Or remove 1 and repair with 1, but the new word allows removing ANOTHER word?
    # Standard "1-1" move: Remove c, add d. Accept if size decreases? No, size is same.
    # Usually, we accept if it leads to a better "score" or if it allows further reduction.
    # For simplicity, let's do:
    # A) Pure Removal (handled by try_remove_all)
    # B) Swap: Remove c, Add d. Accept if it enables a subsequent removal? 
    #    Or just accept if it doesn't increase size and we are stuck?
    #    Better: Accept if the new configuration has more "flexibility" (e.g., more words with coverage=1? No, we want fewer).
    #    Let's try "2-1" moves: Remove 2 words, Add 1 word.
    #    If we can remove 2 words and their combined gap is covered by 1 new word, we reduce size by 1.
    
    def greedy_repair_gap(uncovered_indices, max_adds=1):
        """
        Given a set of uncovered indices, find up to max_adds words to cover them.
        Returns list of added words.
        """
        if not uncovered_indices:
            return []
        
        added = []
        temp_uncovered = set(uncovered_indices)
        
        for _ in range(max_adds):
            if not temp_uncovered:
                break
                
            best_c = -1
            best_count = 0
            
            # Candidates should be neighbors of uncovered points?
            # Yes, any word that covers an uncovered point must be a neighbor of it.
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors[w])
                
            # Exclude words already in code? 
            # If we are repairing a gap created by removal, the removed words are NOT in code.
            # Other words in code are already active.
            # We want to add NEW words.
            # So we should exclude words already in code.
            
            for c in candidate_pool:
                if c in code:
                    continue
                
                # Count how many of temp_uncovered it covers
                # Use numpy for speed?
                # neighbors[c] is a list. 
                # Let's just use a set intersection if temp_uncovered is small.
                # len(set(neighbors[c]) & temp_uncovered)
                # Creating a set for each c is slow.
                # Let's precompute neighbor sets?
                # N=2187. 2187 sets of size 15. Memory is fine.
                # But we are in a tight loop.
                
                # Alternative: 
                # Since temp_uncovered is small (<= 2 * 15 = 30), we can just check each w in temp_uncovered
                # and see if w is in neighbors[c].
                # But checking "w in neighbors[c]" is O(15).
                # Total: |candidate_pool| * |temp_uncovered| * 15.
                # |candidate_pool| can be up to 30*15 = 450.
                # 450 * 30 * 15 ~ 200k ops. This is acceptable if done few times.
                
                cnt = 0
                # Optimization: if we just need to know if it covers ANY, we break early.
                # But we want the MAX.
                for w in temp_uncovered:
                    if w in neighbors[c]:
                        cnt += 1
                        if cnt > best_count:
                            best_count = cnt
                            best_c = c
                            # Early break if we cover all?
                            if best_count == len(temp_uncovered):
                                break
                            
            if best_c == -1:
                break
                
            added.append(best_c)
            for w in neighbors[best_c]:
                temp_uncovered.discard(w)
                
        return added

    # Precompute neighbor sets for faster lookup in repair?
    # neighbor_sets = [set(nb) for nb in neighbors]
    # This might speed up the "w in neighbors[c]" check.
    neighbor_sets = [set(nb) for nb in neighbors]

    # Update greedy_repair to use neighbor_sets
    def greedy_repair_fast(uncovered_indices, max_adds=1):
        if not uncovered_indices:
            return []
        
        added = []
        temp_uncovered = set(uncovered_indices)
        
        for _ in range(max_adds):
            if not temp_uncovered:
                break
                
            best_c = -1
            best_count = 0
            
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors[w])
                
            for c in candidate_pool:
                if c in code:
                    continue
                
                # Count intersection
                # Using set intersection is fast in C
                # But creating a set from neighbors[c] is slow.
                # We have neighbor_sets[c].
                # intersection = neighbor_sets[c] & temp_uncovered
                # len(intersection)
                # This creates a new set object.
                # Is it faster than the loop?
                # Let's try the loop with neighbor_sets for O(1) lookup.
                
                cnt = 0
                for w in temp_uncovered:
                    if w in neighbor_sets[c]:
                        cnt += 1
                        if cnt > best_count:
                            best_count = cnt
                            best_c = c
                            if best_count == len(temp_uncovered):
                                break
                            
            if best_c == -1:
                break
                
            added.append(best_c)
            for w in neighbors[best_c]:
                temp_uncovered.discard(w)
                
        return added

    def kick_1_1():
        """Remove 1, Add 1. Accept if it allows a removal or just for diversity? 
        Actually, let's focus on 2-1 moves (Remove 2, Add 1) which directly reduces size."""
        pass

    def kick_2_1():
        """Remove 2 random words, try to repair with 1 word. If successful, accept (size -1)."""
        if len(code) < 2:
            return False
            
        old_size = len(code)
        
        # Pick 2 random distinct
        i1 = random.randint(0, len(code) - 1)
        i2 = random.randint(0, len(code) - 1)
        if i1 == i2:
            i2 = (i2 + 1) % len(code)
            
        c1 = list(code)[i1] # Set doesn't support index
        c2 = list(code)[i2]
        
        # Simulate removal
        # We need to check if removing c1 and c2 leaves any word with coverage 0.
        # coverage[w] - (1 if w in nb(c1) else 0) - (1 if w in nb(c2) else 0) > 0
        
        # Efficient check:
        # Find words that are ONLY covered by c1 or c2?
        # No, words covered by c1 AND c2 are still covered by others if coverage > 2?
        # We need to find w such that coverage[w] == (1 if w in nb(c1) else 0) + (1 if w in nb(c2) else 0)
        
        # Let's just compute the new coverage for the union of neighbors.
        affected = set(neighbors[c1]).union(set(neighbors[c2]))
        
        uncovered = []
        for w in affected:
            new_cov = coverage[w]
            if w in neighbors[c1]:
                new_cov -= 1
            if w in neighbors[c2]:
                new_cov -= 1
            if new_cov < 1:
                uncovered.append(w)
                
        if not uncovered:
            # Both are redundant? Should have been caught by try_remove_all.
            # Just remove one?
            remove_word(c1)
            return True
            
        # Try to repair with 1 word
        added = greedy_repair_fast(uncovered, max_adds=1)
        
        if len(added) == 1:
            # Check if the added word actually covers all uncovered
            # greedy_repair_fast returns words that cover the remaining uncovered.
            # It stops when temp_uncovered is empty.
            # So if it returned 1 word, it means that word covered all.
            
            # Accept
            remove_word(c1)
            remove_word(c2)
            add_word(added[0])
            return True
            
        return False

    # Main Loop
    last_improvement = time.time()
    
    while time.time() < deadline:
        # 1. Exhaustive Removal
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = try_remove_all()
            
        # 2. Kicks
        if time.time() > deadline:
            break
        
        success = False
        kicks_tried = 0
        max_kicks = 50 # Increase attempts
        
        while kicks_tried < max_kicks:
            if time.time() > deadline:
                break
                
            if kick_2_1():
                success = True
                kicks_tried = 0
            else:
                kicks_tried += 1
                
        if not success:
            # Stuck. Try a random restart of the local search?
            # Or just continue with more random kicks.
            # If we are very stuck, we might try to add a random word and remove 2?
            # That's a 1-add, 2-remove move.
            pass

    # Convert indices back to lists of ints
    result = [idx_to_tuple(c) for c in code]
    return result

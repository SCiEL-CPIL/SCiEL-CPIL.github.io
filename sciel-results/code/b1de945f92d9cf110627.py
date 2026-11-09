import time
import random
from itertools import combinations

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0 # Safety margin
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET
    
    best_words = []
    
    # Precompute popcount for all 18-bit values using a lookup table
    # This is much faster than calling .bit_count() repeatedly
    popcount = [0] * (1 << N)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Helper to check if candidate c is compatible with list 'words'
    def is_compatible(c, words):
        for w in words:
            if popcount[c ^ w] < MIN_DIST:
                return False
        return True
    
    # Bitset-based Bron-Kerbosch variant for maximum independent set
    # We fix the first codeword to 0 (symmetry) and search for additional words
    # that are at distance >= MIN_DIST from 0 and from each other
    
    def bron_kerbosch_backtrack(candidates, current_set, depth, max_depth):
        """
        Backtracking search for maximum independent set.
        candidates: list of candidate vertices (words) to consider
        current_set: current independent set
        depth: current recursion depth
        max_depth: maximum depth to search (for pruning)
        Returns: best independent set found
        """
        best = current_set[:]
        
        if not candidates or depth > max_depth:
            return best
        
        # Try adding each candidate
        for i, c in enumerate(candidates):
            # Check if c is compatible with all in current_set
            if not is_compatible(c, current_set):
                continue
            
            new_set = current_set + [c]
            if len(new_set) > len(best):
                best = new_set[:]
            
            # Remove c and all vertices incompatible with c from candidates
            remaining = []
            for j in range(i + 1, len(candidates)):
                cj = candidates[j]
                # cj must be compatible with c
                if popcount[c ^ cj] >= MIN_DIST:
                    remaining.append(cj)
            
            # Recurse
            result = bron_kerbosch_backtrack(remaining, new_set, depth + 1, max_depth)
            if len(result) > len(best):
                best = result
        
        return best
    
    # Strategy 1: Start with 0 and use deterministic backtracking
    # Generate all words at distance >= MIN_DIST from 0
    candidates_from_zero = [c for c in range(1, 1 << N) if popcount[c] >= MIN_DIST]
    
    # Sort candidates by some heuristic (e.g., by weight, then by value)
    # Words with higher weight might be more "spread out"
    candidates_from_zero.sort(key=lambda x: (-popcount[x], x))
    
    # Limit candidates to avoid exponential blowup
    # Take top K candidates by weight
    K = 200
    candidates_from_zero = candidates_from_zero[:K]
    
    # Run backtracking with a reasonable depth limit
    # The first word is fixed to 0, so we search for up to max_depth additional words
    max_depth = 20  # We'll find at least 20 words including 0
    
    if time.time() < deadline - 5.0:
        result = bron_kerbosch_backtrack(candidates_from_zero, [0], 0, max_depth)
        if len(result) > len(best_words):
            best_words = result
    
    # Strategy 2: Multiple random greedy passes with bitset optimization
    # This can sometimes find better solutions than pure backtracking
    
    max_restarts = 50
    
    for run_id in range(max_restarts):
        # Break if we are close to deadline
        if time.time() > deadline - 1.0:
            break
        
        words = []
        
        # Start with 0 for symmetry
        words.append(0)
        
        # Generate a random permutation of candidates
        all_candidates = [c for c in range(1, 1 << N) if popcount[c] >= MIN_DIST]
        random.shuffle(all_candidates)
        
        stall_counter = 0
        max_stall = 300
        last_added = 1
        
        # Process candidates in random order
        for c in all_candidates:
            if time.time() > deadline - 0.5:
                break
            
            if is_compatible(c, words):
                words.append(c)
                stall_counter = 0
                last_added = len(words)
                
                # Update global best
                if len(words) > len(best_words):
                    best_words = words[:]
            else:
                stall_counter += 1
            
            # Check for local maximum (stall)
            if stall_counter > max_stall:
                break
    
    # Strategy 3: Local improvement - try to add more words to the best solution
    # by relaxing constraints or using a different order
    
    if best_words and time.time() < deadline - 2.0:
        # Create a copy to work on
        improved = best_words[:]
        
        # Try to add words that are compatible with the current set
        # but were missed due to random ordering
        current_set = set(improved)
        
        # Generate candidates systematically
        for c in range(1, 1 << N):
            if time.time() > deadline - 0.5:
                break
            
            if c in current_set:
                continue
            
            if popcount[c] < MIN_DIST:
                continue
            
            if is_compatible(c, improved):
                improved.append(c)
                current_set.add(c)
        
        if len(improved) > len(best_words):
            best_words = improved
    
    # Ensure we return a non-empty list
    if not best_words:
        # The all-zero vector is always a valid codeword in a code of size 1
        # But usually we can do better. 
        # If the search failed completely (unlikely), return [0]
        return [0]
    
    return best_words

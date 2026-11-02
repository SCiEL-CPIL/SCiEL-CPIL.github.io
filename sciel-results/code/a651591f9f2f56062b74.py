"""Rewritten program for ternary covering code K_3(8,1)."""
import time
import random
import numpy as np

def _all_neighbors_fast(cw, q, n, radius=1):
    """Generate all words within Hamming distance 1 of cw. Returns list of tuples."""
    # cw is a tuple of length n
    neighbors = [cw]
    for i in range(n):
        for v in range(q):
            if v != cw[i]:
                new_cw = list(cw)
                new_cw[i] = v
                neighbors.append(tuple(new_cw))
    return neighbors

def _greedy_cover(q, n, radius=1, max_time=10.0):
    """
    Fast greedy heuristic to get a feasible covering.
    Returns a list of codewords (as lists).
    """
    start = time.time()
    # Total space size
    total = q ** n
    
    # Represent covered set as a boolean array of size total
    covered = np.zeros(total, dtype=bool)
    
    # We need to map a word (tuple of ints) to an index
    # index = sum(w[i] * q^(n-1-i))
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    # Precompute neighbors for all possible words? No, too much.
    # Instead, we maintain a set of "uncovered" words.
    # But checking uncovered is O(uncovered).
    
    # Alternative: Greedy by picking codeword that covers most uncovered words.
    # This is expensive if done naively.
    
    # Simpler greedy: Randomized greedy.
    # Start with empty code.
    # While not all covered:
    #   Pick a random uncovered word u.
    #   Pick a random word v within distance radius of u (or just u itself, or random neighbor).
    #   Actually, to be fast, we can just pick random words until covered.
    #   But that might be huge.
    
    # Let's use a structured greedy:
    # The tetracode construction gives 9 * 3^4 = 729 words.
    # We can start from there and try to remove.
    
    # Tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = []
    for h in tetracode:
        for tail in np.ndindex(3, 3, 3, 3):
            code.append(h + list(tail))
    
    # Convert to tuples for hashing
    code_tuples = [tuple(c) for c in code]
    
    # Build covered set
    covered = set()
    for c in code_tuples:
        for nb in _all_neighbors_fast(c, q, n, radius):
            covered.add(nb)
    
    # Now try to remove codewords
    # Iterate multiple times
    for _ in range(20):
        if time.time() - start > max_time:
            break
        improved = False
        for i in range(len(code_tuples)):
            if time.time() - start > max_time:
                break
            # Check if code_tuples[i] is redundant
            # Temporarily remove it
            c = code_tuples[i]
            # Find words only covered by c
            # This is expensive. Instead, check if all neighbors of c are covered by others.
            # For each neighbor nb of c, check if there exists another c' in code such that nb is in neighbors(c').
            
            # Optimization: Only check if c is the unique cover for some nb.
            # We can maintain a count of covers for each word.
            # But that's O(total * |code|) memory/time to update.
            
            # Let's just do a simple check:
            # For each nb in neighbors(c):
            #   is there another c' != c such that nb in neighbors(c')?
            # If yes for all nb, then c is redundant.
            
            redundant = True
            nbs = _all_neighbors_fast(c, q, n, radius)
            for nb in nbs:
                found = False
                for j, c2 in enumerate(code_tuples):
                    if i == j:
                        continue
                    # Check if nb is in neighbors(c2)
                    # Hamming distance
                    dist = sum(1 for a, b in zip(nb, c2) if a != b)
                    if dist <= radius:
                        found = True
                        break
                if not found:
                    redundant = False
                    break
            
            if redundant:
                code_tuples.pop(i)
                improved = True
                # Restart loop? Or continue?
                # Continue, but indices shift.
                break # Restart outer loop to be safe with indices
        
        if not improved:
            break
    
    return [list(c) for c in code_tuples]

def _local_search(code, q, n, radius=1, max_time=100.0):
    """
    Local search to minimize code size.
    Uses swap and removal moves.
    """
    start = time.time()
    code = [tuple(c) for c in code]
    
    # Precompute neighbors for current code words
    # To speed up redundancy checks, we can maintain a cover count array.
    # total = q**n = 6561
    total = q ** n
    cover_count = np.zeros(total, dtype=np.int32)
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)
    
    # Initialize cover_count
    for c in code:
        for nb in _all_neighbors_fast(c, q, n, radius):
            cover_count[word_to_idx(nb)] += 1
    
    def is_valid_cover():
        return np.all(cover_count > 0)
    
    best_code = code[:]
    best_len = len(code)
    
    # Local search loop
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start < max_time:
        improved = False
        
        # 1. Try removals
        for i in range(len(code)):
            if time.time() - start > max_time:
                return best_code
            c = code[i]
            # Check if removing c keeps cover valid
            # i.e., all neighbors of c have cover_count > 1
            nbs = _all_neighbors_fast(c, q, n, radius)
            can_remove = True
            for nb in nbs:
                if cover_count[word_to_idx(nb)] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c
                for nb in nbs:
                    cover_count[word_to_idx(nb)] -= 1
                code.pop(i)
                improved = True
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = code[:]
                no_improve_count = 0
                break # Restart scan
        
        if improved:
            continue
        
        # 2. Try swaps: replace c_i with c_new
        # Heuristic: find a codeword c_i that is "least useful" and replace with a random one
        # Or: for each c_i, try replacing with random c_new, check if valid.
        
        # To be efficient, we can pick a random i, and a random c_new.
        # Check if valid.
        
        # Better: pick i where c_i has many neighbors with cover_count == 1 (critical) vs few.
        # Actually, we want to remove c_i only if it's redundant, which we already tried.
        # Swap is useful if c_i is not redundant, but c_new covers words that c_i doesn't,
        # and c_i covers words that are covered by others.
        
        # Let's try: Pick random i.
        # Pick random c_new (not in code).
        # Check if (code - {c_i}) U {c_new} is valid.
        # This requires checking:
        # 1. All neighbors of c_i that are NOT covered by others (cover_count==1) must be covered by c_new.
        # 2. All neighbors of c_new must be covered by (code - {c_i}) or c_new itself.
        
        for _ in range(100): # Try 100 swaps per iteration
            if time.time() - start > max_time:
                return best_code
            
            i = random.randint(0, len(code)-1)
            c_i = code[i]
            
            # Generate random c_new
            c_new = tuple(random.randint(0, q-1) for _ in range(n))
            if c_new in code:
                continue
            
            # Check validity of swap
            # Step 1: Check words only covered by c_i
            # These are neighbors of c_i with cover_count == 1.
            # They must be neighbors of c_new.
            
            nbs_i = _all_neighbors_fast(c_i, q, n, radius)
            nbs_new = _all_neighbors_fast(c_new, q, n, radius)
            
            # Create set of nbs_new for fast lookup
            set_new = set(nbs_new)
            
            valid_swap = True
            for nb in nbs_i:
                idx = word_to_idx(nb)
                if cover_count[idx] == 1:
                    # Only c_i covers this
                    if nb not in set_new:
                        valid_swap = False
                        break
            
            if not valid_swap:
                continue
            
            # Step 2: Check words covered by c_new but not by others
            # Actually, we just need to ensure no word becomes uncovered.
            # Words that might become uncovered are those only covered by c_i (checked above).
            # Words covered by c_new are fine because c_new is added.
            # So if Step 1 passes, the swap is valid?
            # Wait, what if c_new covers a word w, and c_i also covers w, and cover_count[w] == 1?
            # Then w is only covered by c_i. If we remove c_i, w is uncovered UNLESS c_new covers w.
            # So Step 1 checks exactly this: for all w in N(c_i) with count 1, w must be in N(c_new).
            # What about w in N(c_new) with count 0? Impossible, count is current count.
            # If w in N(c_new) and count[w] == 0, then w is currently uncovered, which means current code is invalid.
            # But we assume current code is valid.
            
            # So if Step 1 passes, the swap is valid.
            
            # Perform swap
            for nb in nbs_i:
                cover_count[word_to_idx(nb)] -= 1
            for nb in nbs_new:
                cover_count[word_to_idx(nb)] += 1
            code[i] = c_new
            
            improved = True
            no_improve_count = 0
            break # Restart scan
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_code

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # 1. Greedy initialization
    code = _greedy_cover(q, n, radius, max_time=20.0)
    
    # 2. Local search
    code = _local_search(code, q, n, radius, max_time=deadline - time.time())
    
    return code

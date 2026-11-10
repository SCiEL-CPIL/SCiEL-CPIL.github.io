"""Football pool problem for 8 matches: ternary covering code K_3(8,1)"""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    total_words = q ** n

    def int_to_word(x):
        w = [0] * n
        for i in range(n):
            w[i] = x % q
            x //= q
        return w

    def word_to_int(w):
        x = 0
        for i in range(n):
            x = x * q + w[i]
        return x

    # Precompute neighbors for all words
    # For radius 1: each word has 1 + n*(q-1) neighbors
    neighbors = {}
    for x in range(total_words):
        w = int_to_word(x)
        nb = [x]
        for i in range(n):
            for v in range(q):
                if v != w[i]:
                    nw = w[:]
                    nw[i] = v
                    nb.append(word_to_int(nw))
        neighbors[x] = nb

    # Greedy construction: start with empty code, add codewords that cover most uncovered words
    covered = np.zeros(total_words, dtype=bool)
    code = []
    
    # Precompute coverage contribution of each potential codeword
    # cov[x] = set of words covered by adding codeword x
    # We'll track uncovered count per potential codeword dynamically
    
    # For efficiency, precompute for each word x, the list of words it covers
    # and use a counter for how many times each word is covered by current code
    
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    # We need to efficiently find which codeword to add next
    # Strategy: maintain for each candidate codeword c, the number of newly covered words
    # new_cover[c] = number of words in neighbors[c] that are not yet covered
    
    # But updating new_cover when we add a codeword is expensive
    # Alternative: greedy with periodic recomputation
    
    # Let's use a different approach: 
    # 1. Start with a good initial solution (direct product)
    # 2. Remove redundant codewords
    # 3. Try local search to improve
    
    # Initial solution: tetracode x 3^4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert to indices
    code_indices = [word_to_int(c) for c in initial_code]
    code_set = set(code_indices)
    
    # Compute coverage counts
    for c in code_indices:
        for nb in neighbors[c]:
            cover_count[nb] += 1
    
    # Verify all covered
    if not np.all(cover_count >= 1):
        # Fallback to greedy
        pass
    
    # Phase 1: Remove redundant codewords
    # A codeword c is redundant if all words it covers are covered by other codewords
    # i.e., for all nb in neighbors[c], cover_count[nb] > 1
    
    removed = set()
    progress = True
    while progress and time.time() < deadline - 30:
        progress = False
        for c in code_indices:
            if c in removed:
                continue
            # Check if c is redundant
            redundant = True
            for nb in neighbors[c]:
                if cover_count[nb] <= 1:
                    redundant = False
                    break
            if redundant:
                removed.add(c)
                progress = True
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
    
    # Build current code
    current_code = [c for c in code_indices if c not in removed]
    
    # Phase 2: Local search - try to replace pairs of codewords with one
    # or move a codeword to a better position
    best_code = current_code[:]
    best_size = len(best_code)
    
    # Try to reduce by moving codewords
    # For each codeword c, try moving it to a position c' that covers more uncovered words
    # when c is removed
    
    # First, compute which words are uniquely covered by each codeword
    # unique_cover[c] = set of words only covered by c (among current code)
    
    # This is expensive, so let's use a simpler approach:
    # Try removing each codeword and see if coverage is maintained
    # If not, try to find a replacement
    
    # Actually, let's try a different local search:
    # For each codeword c, try all possible positions c' and see if replacing c with c'
    # maintains coverage and allows further reduction
    
    # Simpler: try to remove codewords in different orders
    # Use random ordering to escape local minima
    
    for iteration in range(50):
        if time.time() > deadline - 20:
            break
        
        # Try to remove codewords in random order
        code_copy = best_code[:]
        removed_local = set()
        cover_local = np.zeros(total_words, dtype=np.int32)
        
        for c in code_copy:
            for nb in neighbors[c]:
                cover_local[nb] += 1
        
        random.shuffle(code_copy)
        
        for c in code_copy:
            if c in removed_local:
                continue
            # Check if c is redundant
            redundant = True
            for nb in neighbors[c]:
                if cover_local[nb] <= 1:
                    redundant = False
                    break
            if redundant:
                removed_local.add(c)
                for nb in neighbors[c]:
                    cover_local[nb] -= 1
        
        new_code = [c for c in best_code if c not in removed_local]
        if len(new_code) < best_size:
            best_code = new_code
            best_size = len(best_code)
    
    # Phase 3: Try to add codewords that allow removing multiple others
    # For now, return the best code found
    
    # Convert back to list of lists
    final_code = [int_to_word(c) for c in best_code]
    
    # Verify coverage
    verify_count = np.zeros(total_words, dtype=np.int32)
    for c in best_code:
        for nb in neighbors[c]:
            verify_count[nb] += 1
    
    if not np.all(verify_count >= 1):
        # Fallback to initial code
        return initial_code
    
    return final_code

# EVOLVE-BLOCK-END

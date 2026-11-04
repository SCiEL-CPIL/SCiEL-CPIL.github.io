import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s limit
    
    # This task is specifically for q=3, n=6, radius=1
    # We use the tetracode direct-sum construction as a starting point
    # Tetracode: perfect 1-error-correcting code of length 4 over GF(3) with 9 words
    
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Construct initial code: direct sum with all tails
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h) + tail)
            
    m = len(code)  # 81 for q=3, n=6
    total_words = q ** n  # 729
    
    # Precompute coverage sets for each codeword
    # A codeword covers itself and all words at distance 1
    # For q=3, n=6: each codeword covers 1 + 6*2 = 13 words
    
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    covers = []
    for c in code:
        c_set = set()
        c_set.add(word_to_index(c))
        c_list = list(c)
        for pos in range(n):
            orig_val = c_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                c_list[pos] = new_val
                c_set.add(word_to_index(tuple(c_list)))
            c_list[pos] = orig_val
        covers.append(c_set)
    
    # Build a lookup from word index to list of codeword indices that cover it
    # This helps in local search: when we remove a codeword, we quickly find what becomes uncovered
    # and can check if another codeword covers those words
    
    # Precompute: for each word, which codewords cover it
    word_to_codewords = [set() for _ in range(total_words)]
    for ci in range(m):
        for wi in covers[ci]:
            word_to_codewords[wi].add(ci)
    
    # Initial coverage count
    coverage_count = [0] * total_words
    for ci in range(m):
        for wi in covers[ci]:
            coverage_count[wi] += 1
    
    best_code = code[:]
    best_count = m
    
    # Phase 1: Greedy removal with multiple random restarts
    max_restarts = 3000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Reset state
        curr_coverage = coverage_count[:]
        active = [True] * m
        
        # Random order for removal attempts
        order = list(range(m))
        random.shuffle(order)
        
        # Multiple passes of greedy removal
        for sweep in range(3):
            for ci in order:
                if not active[ci]:
                    continue
                
                # Check if ci is redundant: every word it covers has coverage > 1
                is_redundant = True
                for wi in covers[ci]:
                    if curr_coverage[wi] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    active[ci] = False
                    for wi in covers[ci]:
                        curr_coverage[wi] -= 1
        
        current_code = [code[i] for i in range(m) if active[i]]
        if len(current_code) < best_count:
            best_count = len(current_code)
            best_code = current_code[:]
    
    # Phase 2: Local search with add-remove moves
    # Key insight: sometimes we need to remove one codeword and add another to escape local minima
    # We use symmetry breaking by always trying to remove the "largest" codeword first
    # and adding the "smallest" possible replacement
    
    if time.time() < deadline - 10.0:
        best_set = set(best_code)
        m_best = len(best_code)
        
        # Recompute coverage for best_code
        best_covers = []
        for c in best_code:
            c_set = set()
            c_set.add(word_to_index(c))
            c_list = list(c)
            for pos in range(n):
                orig_val = c_list[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    c_list[pos] = new_val
                    c_set.add(word_to_index(tuple(c_list)))
                c_list[pos] = orig_val
            best_covers.append(c_set)
        
        best_coverage = [0] * total_words
        for ci in range(m_best):
            for wi in best_covers[ci]:
                best_coverage[wi] += 1
        
        # Local search: try to remove codeword ri and find best replacement
        # A replacement is valid if it covers all words that become uncovered
        # We score replacements by how many additional words they cover (to enable future removals)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            m_best = len(best_code)
            best_set = set(best_code)
            
            # Recompute coverage
            best_covers = []
            for c in best_code:
                c_set = set()
                c_set.add(word_to_index(c))
                c_list = list(c)
                for pos in range(n):
                    orig_val = c_list[pos]
                    for new_val in range(q):
                        if new_val == orig_val:
                            continue
                        c_list[pos] = new_val
                        c_set.add(word_to_index(tuple(c_list)))
                    c_list[pos] = orig_val
                best_covers.append(c_set)
            
            best_coverage = [0] * total_words
            for ci in range(m_best):
                for wi in best_covers[ci]:
                    best_coverage[wi] += 1
            
            # Try each codeword for removal
            for ri in range(m_best):
                if time.time() > deadline - 5.0:
                    break
                
                # Compute coverage after removing ri
                temp_coverage = best_coverage[:]
                for wi in best_covers[ri]:
                    temp_coverage[wi] -= 1
                
                # Find words that become uncovered
                uncovered = [wi for wi in range(total_words) if temp_coverage[wi] == 0]
                
                if len(uncovered) == 0:
                    # Can remove ri entirely
                    new_code = [best_code[i] for i in range(m_best) if i != ri]
                    best_code = new_code
                    best_count = len(best_code)
                    improved = True
                    break  # Restart the while loop with updated state
                
                # If there are uncovered words, find the best replacement
                if len(uncovered) <= 10:
                    uncovered_set = set(uncovered)
                    best_replacement = -1
                    best_score = -1
                    
                    for ci in range(m):
                        if code[ci] in best_set:
                            continue
                        c_set = covers[ci]
                        # Check if ci covers all uncovered words
                        if uncovered_set.issubset(c_set):
                            # Score: how many words in c_set have coverage 1 (would become 2)
                            # This indicates the replacement is "tight" and may enable future removals
                            score = 0
                            for wi in c_set:
                                if temp_coverage[wi] == 1:
                                    score += 1
                            if score > best_score:
                                best_score = score
                                best_replacement = ci
                    
                    if best_replacement >= 0:
                        # Perform the swap
                        new_code = [best_code[i] for i in range(m_best) if i != ri]
                        new_code.append(code[best_replacement])
                        best_code = new_code
                        improved = True
                        break  # Restart with updated state
    
    # Phase 3: Final aggressive removal pass
    if time.time() < deadline - 3.0:
        best_set = set(best_code)
        m_best = len(best_code)
        
        # Recompute coverage
        best_covers = []
        for c in best_code:
            c_set = set()
            c_set.add(word_to_index(c))
            c_list = list(c)
            for pos in range(n):
                orig_val = c_list[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    c_list[pos] = new_val
                    c_set.add(word_to_index(tuple(c_list)))
                c_list[pos] = orig_val
            best_covers.append(c_set)
        
        best_coverage = [0] * total_words
        for ci in range(m_best):
            for wi in best_covers[ci]:
                best_coverage[wi] += 1
        
        # Try all possible single removals
        for ri in range(m_best):
            if time.time() > deadline - 2.0:
                break
            
            temp_coverage = best_coverage[:]
            for wi in best_covers[ri]:
                temp_coverage[wi] -= 1
            
            uncovered = [wi for wi in range(total_words) if temp_coverage[wi] == 0]
            
            if len(uncovered) == 0:
                new_code = [best_code[i] for i in range(m_best) if i != ri]
                best_code = new_code
                best_count = len(best_code)
                # Restart coverage computation
                m_best = len(best_code)
                best_covers = best_covers[:ri] + best_covers[ri+1:]
                best_coverage = [0] * total_words
                for ci in range(m_best):
                    for wi in best_covers[ci]:
                        best_coverage[wi] += 1
    
    return [list(c) for c in best_code]

import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # 8s safety margin for 120s limit
    
    total_words = q ** n
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbors using bitmask representation for speed
    # For q=3, n=8, total_words=6561
    # We'll use a list of sets of neighbors
    
    # For radius=1, neighbors of word i are i itself plus all words differing in exactly one position
    # For general radius, we need to handle it more carefully
    
    if radius == 1:
        # Fast path for radius 1
        neighbors = [set() for _ in range(total_words)]
        for i in range(total_words):
            d = list(digits_of[i])
            neighbors[i].add(i)
            for pos in range(n):
                orig = d[pos]
                for new_val in range(q):
                    if new_val != orig:
                        d[pos] = new_val
                        # Convert back to index
                        idx = 0
                        for j in range(n):
                            idx = idx * q + d[j]
                        neighbors[i].add(idx)
                d[pos] = orig
    else:
        # General case
        neighbors = [set() for _ in range(total_words)]
        for i in range(total_words):
            d = list(digits_of[i])
            neighbors[i].add(i)
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    for variation in itertools.product(range(q), repeat=r):
                        is_diff = False
                        new_d = d[:]
                        for k, pos in enumerate(pos_comb):
                            if variation[k] != d[pos]:
                                is_diff = True
                            new_d[pos] = variation[k]
                        if not is_diff:
                            continue
                        idx = 0
                        for j in range(n):
                            idx = idx * q + new_d[j]
                        neighbors[i].add(idx)
    
    # Greedy Set Cover with local search
    best_code = None
    min_size = float('inf')
    
    # Convert neighbors to lists for faster iteration
    neighbor_lists = [list(neigh) for neigh in neighbors]
    
    def greedy_cover():
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        
        while uncovered.any():
            # Find codeword that covers most uncovered words
            # Use numpy for speed
            best_idx = -1
            max_cover = 0
            candidates = []
            
            for i in range(total_words):
                # Count uncovered neighbors
                nb = neighbor_lists[i]
                cnt = sum(1 for x in nb if uncovered[x])
                if cnt > max_cover:
                    max_cover = cnt
                    best_idx = i
                    candidates = [i]
                elif cnt == max_cover:
                    candidates.append(i)
                    
            if max_cover == 0:
                break
                
            best_idx = random.choice(candidates)
            code.append(best_idx)
            for x in neighbor_lists[best_idx]:
                uncovered[x] = False
                
        return code
    
    def local_search(code):
        code_set = set(code)
        active_code = list(code)
        
        # Build coverage map: for each word, which codewords cover it
        coverage = [0] * total_words
        for c in active_code:
            for x in neighbor_lists[c]:
                coverage[x] += 1
        
        # Try to remove redundant codewords
        changed = True
        while changed:
            changed = False
            for idx, c in enumerate(active_code):
                # Check if removing c leaves any word uncovered
                removable = True
                for x in neighbor_lists[c]:
                    if coverage[x] == 1:
                        removable = False
                        break
                if removable:
                    # Remove c
                    for x in neighbor_lists[c]:
                        coverage[x] -= 1
                    active_code.pop(idx)
                    code_set.discard(c)
                    changed = True
                    break  # Restart the loop since indices changed
        
        # Try to add codewords that allow removal of multiple existing ones
        # This is more complex, so let's do a simpler version:
        # Try swapping: remove c, add c' if it allows net reduction
        
        for _ in range(50):  # Limited number of swap attempts
            if time.time() > deadline - 5:
                break
            if not active_code:
                break
            
            # Pick a random codeword to potentially remove
            c_idx = random.randint(0, len(active_code) - 1)
            c = active_code[c_idx]
            
            # Find words that would become uncovered if we remove c
            at_risk = [x for x in neighbor_lists[c] if coverage[x] == 1]
            
            if not at_risk:
                # Can remove c directly
                for x in neighbor_lists[c]:
                    coverage[x] -= 1
                active_code.pop(c_idx)
                continue
            
            # Find codewords that cover at least one at-risk word
            candidate_replacements = set()
            for x in at_risk:
                for c2 in active_code:
                    if c2 != c and x in neighbors[c2]:
                        candidate_replacements.add(c2)
            
            if not candidate_replacements:
                continue
            
            # Try each candidate replacement
            for c2 in list(candidate_replacements):
                # Check if adding c2 and removing c is beneficial
                # c2 must cover all at-risk words
                covers_all = all(x in neighbors[c2] for x in at_risk)
                if not covers_all:
                    continue
                
                # Check if this allows removal of other codewords
                # For now, just do the swap if it doesn't increase size
                # (it keeps size same, but might enable future reductions)
                
                # Temporarily add c2
                new_coverage = coverage[:]
                for x in neighbor_lists[c2]:
                    new_coverage[x] += 1
                
                # Remove c
                for x in neighbor_lists[c]:
                    new_coverage[x] -= 1
                
                # Check if any word becomes uncovered
                valid = all(new_coverage[x] >= 1 for x in range(total_words))
                
                if valid:
                    # Perform the swap
                    coverage = new_coverage
                    active_code[c_idx] = c2
                    break
        
        # Final cleanup: remove any remaining redundant codewords
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for idx in range(len(active_code) - 1, -1, -1):
                c = active_code[idx]
                removable = True
                for x in neighbor_lists[c]:
                    if coverage[x] == 1:
                        removable = False
                        break
                if removable:
                    for x in neighbor_lists[c]:
                        coverage[x] -= 1
                    active_code.pop(idx)
                    changed = True
        
        return active_code
    
    # Run multiple greedy + local search iterations
    iterations = 0
    while time.time() < deadline - 10:
        iterations += 1
        
        # Greedy initialization
        code = greedy_cover()
        
        # Local search refinement
        refined = local_search(code)
        
        if len(refined) < min_size:
            min_size = len(refined)
            best_code = refined[:]
        
        # If we've found a good solution, spend more time on local search
        if min_size <= 500 and time.time() < deadline - 5:
            # Try more aggressive local search
            extra = local_search(refined)
            if len(extra) < min_size:
                min_size = len(extra)
                best_code = extra[:]
    
    if best_code is None:
        # Fallback to simple construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(digits_of[i]) for i in best_code]

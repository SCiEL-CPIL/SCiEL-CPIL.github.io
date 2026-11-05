import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    
    # Precompute neighbors (Hamming distance <= radius)
    neighbors = {}
    for w in all_words:
        nbs = set()
        nbs.add(w)
        for i in range(n):
            for val in range(q):
                if val == w[i]:
                    continue
                new_w = list(w)
                new_w[i] = val
                nbs.add(tuple(new_w))
        neighbors[w] = nbs
    
    # 1. Greedy Construction
    covered = set()
    code = []
    selected_set = set()
    
    while len(covered) < num_words and time.time() < deadline:
        best_w = None
        max_new = -1
        for w in all_words:
            if w in selected_set:
                continue
            # Calculate new coverage
            new_covers_count = 0
            for nb in neighbors[w]:
                if nb not in covered:
                    new_covers_count += 1
            
            if new_covers_count > max_new:
                max_new = new_covers_count
                best_w = w
        
        if best_w is None:
            break
            
        code.append(best_w)
        selected_set.add(best_w)
        covered.update(neighbors[best_w])
        
    # 2. Local Search
    # We will try to reduce the size of the code.
    # Strategy:
    # Loop:
    #   1. Try to remove any redundant codeword.
    #   2. If no removals, try to swap a codeword to enable removals.
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to remove redundant codewords
        for i in range(len(code)):
            c = code[i]
            # Check if c is essential
            # A codeword c is essential if there exists a word x in Ball(c) 
            # such that x is not in Ball(c') for any other c' in code.
            
            essential = False
            for x in neighbors[c]:
                # Check if x is covered by any other codeword
                covered_by_other = False
                for j in range(len(code)):
                    if i == j:
                        continue
                    if x in neighbors[code[j]]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    essential = True
                    break
            
            if not essential:
                # Remove c
                code.pop(i)
                improved = True
                break # Restart loop to check for more removals
        
        if improved:
            continue
        
        # 2. Try swaps if no removals
        # If we are stuck, try to swap a codeword c with a new word w
        # such that the new configuration allows for a reduction.
        
        swap_done = False
        for i in range(len(code)):
            c = code[i]
            
            # Calculate uncovered words if c is removed
            uncovered = set()
            for x in neighbors[c]:
                covered_by_other = False
                for j in range(len(code)):
                    if i == j:
                        continue
                    if x in neighbors[code[j]]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    uncovered.add(x)
            
            if not uncovered:
                # Should have been removed in step 1
                code.pop(i)
                improved = True
                break
            
            # Try to find a replacement w
            # w must cover all words in `uncovered`
            # w must not be in code
            
            found_swap = False
            for w in all_words:
                if w in code:
                    continue
                # Check if Ball(w) covers `uncovered`
                if uncovered.issubset(neighbors[w]):
                    # Perform swap
                    code[i] = w
                    found_swap = True
                    improved = True
                    break
            
            if found_swap:
                break
        
        if improved:
            continue
        
        # If no improvement, we are stuck in a local minimum.
        # Break to avoid infinite loop.
        break

    # Return as list of lists
    return [list(c) for c in code]

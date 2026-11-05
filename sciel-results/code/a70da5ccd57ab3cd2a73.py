import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        return []

    total_words = 3 ** n
    powers = [3**i for i in range(n)]
    
    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        digits = [0]*n
        tmp = w
        for i in range(n):
            digits[i] = tmp % 3
            tmp //= 3
        
        neighbors[w].append(w)
        for i in range(n):
            d = digits[i]
            p = powers[i]
            new_d = (d + 1) % 3
            neighbors[w].append(w + (new_d - d) * p)
            new_d = (d - 1) % 3
            neighbors[w].append(w + (new_d - d) * p)
            
    def greedy_prune(seed):
        random.seed(seed)
        covered = [False] * total_words
        code_indices = []
        uncovered_set = set(range(total_words))
        
        # Greedy Phase
        while uncovered_set:
            best_word = -1
            best_count = 0
            # Randomize tie-breaking by shuffling the range or just picking randomly among bests
            # To save time, we just scan and pick the first best, but add random noise to counts?
            # Better: Collect top candidates.
            # For speed, we just scan. 
            # To introduce randomness, we can iterate through words in a random order?
            # No, that's slow to generate random order every step.
            # We can just add a small random value to the count to break ties.
            
            # Optimization: We can stop early if we find a count equal to len(uncovered_set)?
            # Unlikely.
            
            for w in range(total_words):
                cnt = 0
                for nb in neighbors[w]:
                    if nb in uncovered_set:
                        cnt += 1
                # Add random noise to break ties and explore different paths
                cnt += random.random() * 0.1 
                
                if cnt > best_count:
                    best_count = cnt
                    best_word = w
                    
            code_indices.append(best_word)
            for nb in neighbors[best_word]:
                if nb in uncovered_set:
                    uncovered_set.remove(nb)
                    
        # Pruning Phase
        cover_count = [0] * total_words
        for c in code_indices:
            for nb in neighbors[c]:
                cover_count[nb] += 1
                
        changed = True
        while changed:
            changed = False
            # Iterate in reverse
            for i in range(len(code_indices) - 1, -1, -1):
                c = code_indices[i]
                safe = True
                for nb in neighbors[c]:
                    if cover_count[nb] == 1:
                        safe = False
                        break
                if safe:
                    code_indices.pop(i)
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    changed = True
                    break # Restart to be safe, though single pass works too
                    
        return code_indices

    best_code = []
    start_time = time.time()
    
    # Run multiple times with different seeds
    # Each run takes ~5-10s. We have 120s. So ~10-20 runs possible.
    for i in range(15):
        code = greedy_prune(i)
        if not best_code or len(code) < len(best_code):
            best_code = code
            
        # Check time
        if time.time() - start_time > 100:
            break
            
    # Convert to list of lists
    code_out = []
    for c in best_code:
        digits = [0]*n
        tmp = c
        for i in range(n):
            digits[i] = tmp % 3
            tmp //= 3
        code_out.append(digits)
        
    return code_out

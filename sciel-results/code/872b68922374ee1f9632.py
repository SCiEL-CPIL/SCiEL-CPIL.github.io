import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # Seed: Direct sum of ternary tetracode
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    # Note: The tetracode logic in the seed assumed q=3.
    # For general q, the tetracode is specific to q=3.
    # The task specifies q=3, n=7, radius=1.
    # We should handle the general case or specialize.
    # The prompt says: "the evaluator passes this task's alphabet size q, length n and covering radius."
    # But the specific task is K_3(7,1).
    # I will assume q=3, n=7, radius=1 for optimization, but write code that works generally if possible.
    # However, the tetracode is only for q=3.
    # If q != 3, we fall back to a simple greedy or random init.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Simple greedy set cover
        # This might be slow for large n, but for small n it's ok.
        # For this specific task, we focus on the 3,7,1 case.
        pass
        
    # Initialize with seed for q=3, n=7
    if q == 3 and n == 7:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # The seed code used n-4=3 tail coordinates
        # Wait, the seed code in the prompt is:
        # tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # This produces 9 * 3^3 = 243 codewords.
        
        codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Generic initialization: Random or Greedy
        # For safety, use a simple greedy for other params
        codewords = []
        # This is a placeholder. The main focus is the specific task.
        # We will assume the evaluator uses the specific task parameters.
        # If not, we need a valid fallback.
        # Let's implement a basic greedy for the fallback.
        all_words = list(itertools.product(range(q), repeat=n))
        covered = set()
        for w in all_words:
            if w not in covered:
                codewords.append(list(w))
                for i in range(n):
                    for val in range(q):
                        if w[i] != val:
                            nw = list(w)
                            nw[i] = val
                            covered.add(tuple(nw))
                covered.add(tuple(w))
                if len(codewords) > 1000: break # Safety

    # Convert to list of tuples for hashing
    codewords_set = set(tuple(c) for c in codewords)
    
    # Precompute balls for all possible words? 
    # Space size 3^7 = 2187.
    # Ball size for radius 1: 1 + 7*2 = 15.
    # Total operations ~ 2187 * 15. Very small.
    
    # Map word index to list of neighbors (including self)
    # Index = base-q number
    def to_idx(word):
        idx = 0
        for i, v in enumerate(word):
            idx = idx * q + v
        return idx
        
    def from_idx(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
        
    total_words = q ** n
    neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        word = from_idx(w)
        neighbors[w].append(w)
        for i in range(n):
            for val in range(q):
                if word[i] != val:
                    nw = list(word)
                    nw[i] = val
                    neighbors[w].append(to_idx(nw))
                    
    # Coverage array: how many codewords cover each word
    coverage = [0] * total_words
    
    def update_coverage(idx, delta):
        for nb in neighbors[idx]:
            coverage[nb] += delta
            
    # Initialize coverage
    for c in codewords_set:
        update_coverage(to_idx(c), 1)
        
    def is_covered():
        return all(c > 0 for c in coverage)
        
    def can_remove(c_idx):
        # Check if all neighbors of c_idx have coverage > 1
        for nb in neighbors[c_idx]:
            if coverage[nb] <= 1:
                return False
        return True
        
    # Prune phase
    def prune():
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of the set
            for c in list(codewords_set):
                c_idx = to_idx(c)
                if can_remove(c_idx):
                    update_coverage(c_idx, -1)
                    codewords_set.remove(c)
                    changed = True
                    # Restart loop as indices/coverage changed
                    break
                    
    # Initial prune
    prune()
    
    # Local Search
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    # Helper to find best replacement for a set of uncovered words
    def find_best_replacement(uncovered_set):
        best_score = -1
        best_word = None
        # Iterate over all possible words
        # Optimization: Only check words not in codewords_set
        for w in range(total_words):
            if tuple(from_idx(w)) in codewords_set:
                continue
            # Count how many uncovered words are in neighbors[w]
            score = 0
            for nb in neighbors[w]:
                if nb in uncovered_set:
                    score += 1
            if score > best_score:
                best_score = score
                best_word = w
        return best_word, best_score
        
    # Search Loop
    while time.time() - start_time < time_limit:
        if not codewords_set:
            break
            
        # Try to remove a random codeword
        c = random.choice(list(codewords_set))
        c_idx = to_idx(c)
        
        # Temporarily remove
        update_coverage(c_idx, -1)
        codewords_set.remove(c)
        
        # Check coverage
        uncovered = []
        for i in range(total_words):
            if coverage[i] < 0: # Should be 0 if valid, but we subtracted
                uncovered.append(i)
            elif coverage[i] == 0:
                # It was covered only by c
                uncovered.append(i)
                
        if not uncovered:
            # Successfully removed, size reduced
            prune()
            continue
        else:
            # Need to repair
            # Find best replacement
            unc_set = set(uncovered)
            best_w, best_score = find_best_replacement(unc_set)
            
            if best_w is not None:
                best_c = tuple(from_idx(best_w))
                # Add the replacement
                codewords_set.add(best_c)
                update_coverage(best_w, 1)
                
                # After adding, try to prune
                prune()
                
                # If we are stuck (size didn't decrease after prune), 
                # we have just performed a swap. 
                # We can continue searching.
                
            else:
                # No good replacement found? Restore c
                codewords_set.add(c)
                update_coverage(c_idx, 1)
                
    return [list(c) for c in codewords_set]

import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0
    
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    num_words = len(all_words)
    
    # Precompute coverage sets
    covered_by = [set() for _ in range(num_words)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    for i, w in enumerate(all_words):
        covered_by[i].add(i)
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                neighbor = w.copy()
                neighbor[pos] = new_val
                j = word_to_idx[tuple(neighbor)]
                covered_by[i].add(j)
                
    best_solution = None
    best_size = num_words
    
    def is_valid(selected_indices):
        # Check if all words are covered
        covered = [False] * num_words
        for i in selected_indices:
            for j in covered_by[i]:
                covered[j] = True
        return all(covered)
        
    # Run multiple random greedy restarts
    max_restarts = 5000
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        random.seed(r)
        uncovered = set(range(num_words))
        selected = []
        
        while uncovered:
            # Find best candidate
            max_count = 0
            candidates = []
            for i in range(num_words):
                # Count uncovered covered by i
                count = 0
                for j in covered_by[i]:
                    if j in uncovered:
                        count += 1
                if count > max_count:
                    max_count = count
                    candidates = [i]
                elif count == max_count and count > 0:
                    candidates.append(i)
                    
            if not candidates:
                break
                
            chosen = random.choice(candidates)
            selected.append(chosen)
            uncovered.difference_update(covered_by[chosen])
            
        current_size = len(selected)
        if current_size < best_size:
            best_size = current_size
            best_solution = [all_words[i] for i in selected]
            
        if best_size <= 73:
            break
            
    # Local Search on the best solution found
    if best_solution is not None:
        # Convert to set of indices for fast lookup
        current_set = set([all_words.index(w) for w in best_solution]) 
        # all_words.index is O(N). Better to use word_to_idx.
        current_set = set([word_to_idx[tuple(w)] for w in best_solution])
        
        # Local Search Loop
        for _ in range(1000): # Max iterations
            if time.time() > deadline:
                break
                
            improved = False
            
            # Try 1-Removal
            for w in list(current_set):
                new_set = current_set - {w}
                if is_valid(new_set):
                    current_set = new_set
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try 1-Exchange
            # Pick a random w in current_set
            w_in = random.choice(list(current_set))
            # Pick a random w_out not in current_set
            # To speed up, maintain a list of non-selected words?
            # Or just pick random index and check.
            w_out = random.randint(0, num_words - 1)
            if w_out in current_set:
                continue
                
            new_set = (current_set - {w_in}) | {w_out}
            if is_valid(new_set):
                current_set = new_set
                # Do not break, keep searching for removals
                
    # Finalize
    if best_solution is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert current_set back to list of words
    final_indices = list(current_set)
    # Check if this is better than best_solution
    if len(final_indices) < best_size:
        best_size = len(final_indices)
        best_solution = [all_words[i] for i in final_indices]
        
    return best_solution

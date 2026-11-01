"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    """
    Construct a covering code using Simulated Annealing.
    This approach models the problem as an optimization landscape where we minimize
    the number of codewords plus a penalty for uncovered words. By accepting
    worse states with a certain probability (based on temperature), we escape
    local minima that trap deterministic local searches.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback: Simple greedy for other parameters
        return _greedy_construct(q, n, radius)

    # Specific optimization for q=3, n=6, radius=1
    total_words = 3 ** n
    all_words = list(itertools.product(range(3), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    
    # Precompute neighbors for each word index (distance <= 1)
    # neighbors[i] is a list of indices
    neighbors = []
    for i, w in enumerate(all_words):
        nb = [i]
        for pos in range(n):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                nb.append(word_to_idx[tuple(new_w)])
        neighbors.append(nb)
        
    # Helper to calculate coverage count for a set of word indices
    def calc_coverage(code_indices):
        covered = np.zeros(total_words, dtype=bool)
        for idx in code_indices:
            covered[neighbors[idx]] = True
        return np.sum(covered)
    
    # Initialize with a valid covering code (Tetracode construction, size 81)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            initial_indices.append(word_to_idx[w])
            
    current_code = set(initial_indices)
    current_coverage = calc_coverage(current_code)
    current_cost = len(current_code) + 100 * (total_words - current_coverage)
    
    best_code = set(current_code)
    best_cost = current_cost
    
    # Simulated Annealing parameters
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    T = 10.0
    cooling_rate = 0.9995
    min_T = 0.1
    
    random.seed(42)
    
    while time.time() - start_time < max_time:
        if T < min_T:
            break
            
        # Generate a neighbor state
        move_type = random.random()
        new_code = set(current_code)
        
        if move_type < 0.4:
            # Remove a random word
            if current_code:
                to_remove = random.choice(list(current_code))
                new_code.discard(to_remove)
        elif move_type < 0.8:
            # Add a random word
            to_add = random.randint(0, total_words - 1)
            new_code.add(to_add)
        else:
            # Swap
            if current_code:
                to_remove = random.choice(list(current_code))
                to_add = random.randint(0, total_words - 1)
                new_code.discard(to_remove)
                new_code.add(to_add)
                
        if len(new_code) == len(current_code) and new_code == current_code:
            continue
            
        new_coverage = calc_coverage(new_code)
        new_cost = len(new_code) + 100 * (total_words - new_coverage)
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0:
            # Accept improvement
            current_code = new_code
            current_coverage = new_coverage
            current_cost = new_cost
            
            if new_coverage == total_words and len(new_code) < len(best_code):
                best_code = set(new_code)
                best_cost = new_cost
        else:
            # Accept worse state with probability exp(-delta/T)
            if random.random() < np.exp(-delta_cost / T):
                current_code = new_code
                current_coverage = new_coverage
                current_cost = new_cost
                
        T *= cooling_rate
        
    # Return the best valid code found
    # Ensure best_code is valid. If not, fall back to initial or a greedy repair
    if calc_coverage(best_code) < total_words:
         # This shouldn't happen if we track best correctly, but just in case
         # Try to repair best_code or use initial
         if calc_coverage(set(initial_indices)) == total_words:
             best_code = set(initial_indices)
         else:
             # Greedy repair
             best_code = set(initial_indices)
             for idx in range(total_words):
                 if not np.any([idx in neighbors[c] for c in best_code]):
                     # This is slow, just use initial for safety
                     pass
             
    return [list(idx_to_word[i]) for i in best_code]

def _greedy_construct(q, n, radius):
    """Fallback greedy construction."""
    all_words = list(itertools.product(range(q), repeat=n))
    neighbors = {}
    for w in all_words:
        nb = set()
        for v in all_words:
            dist = sum(1 for i in range(n) if w[i] != v[i])
            if dist <= radius:
                nb.add(v)
        neighbors[w] = nb
        
    uncovered = set(all_words)
    code = []
    while uncovered:
        best_word = None
        max_covered = -1
        for w in all_words:
            if w in code:
                continue
            covered_count = len(neighbors[w] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = w
        if best_word is None:
            break
        code.append(best_word)
        uncovered -= neighbors[best_word]
    return [list(c) for c in code]

# EVOLVE-BLOCK-END

"""Combined approach: direct sum initialization + simulated annealing local search for ternary covering code K_3(6,1)."""
import itertools
import time
import random
import numpy as np

def _direct_sum_construction(q, n, radius):
    """Fallback: direct sum of ternary tetracode with all words on remaining coords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

def _word_to_idx(word, q, n):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx

def _idx_to_word(idx, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    total_words = q ** n
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = all_words  # all_words[i] is the word with index i
    
    # Precompute neighbors (sphere of radius 1) for each word as a set of indices
    neighbors = []
    for w in all_words:
        w_list = list(w)
        nbrs = {word_to_idx[w]}
        for i in range(n):
            orig = w_list[i]
            for v in range(q):
                if v != orig:
                    nw = w_list[:]
                    nw[i] = v
                    nbrs.add(word_to_idx[tuple(nw)])
        neighbors.append(nbrs)
    
    # Convert neighbors to frozensets for faster hashing and membership
    neighbors_fs = [frozenset(nb) for nb in neighbors]
    
    # Start with direct sum construction
    initial_code = _direct_sum_construction(q, n, radius)
    code_set = set(tuple(cw) for cw in initial_code)
    code_indices = set(word_to_idx[cw] for cw in code_set)
    
    # Track coverage counts
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for idx in code_indices:
        for nbr in neighbors[idx]:
            coverage_count[nbr] += 1
    
    # Helper to check if removing idx is safe
    def is_redundant(idx):
        for nbr in neighbors[idx]:
            if coverage_count[nbr] == 1:
                return False
        return True

    # Helper to check if swap is valid (add covers all uniquely-covered words of remove)
    def try_swap(remove_idx, add_idx):
        for nbr in neighbors[remove_idx]:
            if coverage_count[nbr] == 1:
                if nbr not in neighbors_fs[add_idx]:
                    return False
        return True

    best_code_size = len(code_set)
    
    # Simulated Annealing parameters
    temperature = 2.0
    cooling_rate = 0.9995
    
    current_fragility = int(np.sum(coverage_count == 1))
    
    # Main search loop
    while time.time() < deadline - 5:
        # 1. Try to remove redundant codewords (multiple passes until stable)
        removed_any = False
        for _pass in range(3):  # Multiple removal passes
            removed_this_pass = False
            code_list = list(code_set)
            random.shuffle(code_list)
            for cw_tuple in code_list:
                if time.time() > deadline - 5:
                    break
                idx = word_to_idx[cw_tuple]
                if is_redundant(idx):
                    for nbr in neighbors[idx]:
                        coverage_count[nbr] -= 1
                    code_set.remove(cw_tuple)
                    code_indices.remove(idx)
                    removed_this_pass = True
                    removed_any = True
                    current_fragility = int(np.sum(coverage_count == 1))
                    if len(code_set) < best_code_size:
                        best_code_size = len(code_set)
            if not removed_this_pass:
                break
        
        if removed_any:
            continue  # Restart removal phase as state changed significantly
        
        # 2. If no removals, try swaps with simulated annealing
        cw_tuple = random.choice(list(code_set))
        remove_idx = word_to_idx[cw_tuple]
        
        # Pick a random candidate to add
        add_idx = random.randint(0, total_words - 1)
        
        if add_idx == remove_idx:
            continue
            
        if add_idx in code_indices:  # Check if already in code (O(1) set lookup)
            continue
            
        # Check if swap is valid
        if not try_swap(remove_idx, add_idx):
            continue
        
        # Perform swap and measure fragility change
        old_fragility = current_fragility
        
        # Update coverage
        for nbr in neighbors[remove_idx]:
            coverage_count[nbr] -= 1
        for nbr in neighbors[add_idx]:
            coverage_count[nbr] += 1
            
        new_fragility = int(np.sum(coverage_count == 1))
        delta = new_fragility - old_fragility
        
        # Acceptance criterion: simulated annealing
        if delta <= 0:
            accept = True
        else:
            # Accept with probability exp(-delta / temperature)
            accept = random.random() < np.exp(-delta / max(temperature, 0.01))
        
        if accept:
            code_set.remove(cw_tuple)
            code_set.add(idx_to_word[add_idx])
            code_indices.remove(remove_idx)
            code_indices.add(add_idx)
            current_fragility = new_fragility
            if len(code_set) < best_code_size:
                best_code_size = len(code_set)
        else:
            # Revert
            for nbr in neighbors[remove_idx]:
                coverage_count[nbr] += 1
            for nbr in neighbors[add_idx]:
                coverage_count[nbr] -= 1
        
        # Cool down
        temperature *= cooling_rate
        temperature = max(temperature, 0.1)

    # Final verification
    coverage_final = np.zeros(total_words, dtype=np.int32)
    for idx in code_indices:
        for nbr in neighbors[idx]:
            coverage_final[nbr] += 1
    
    if np.all(coverage_final > 0):
        return [list(cw) for cw in code_set]
    else:
        return _direct_sum_construction(q, n, radius)

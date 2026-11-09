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
    
    # Precompute neighbors (sphere of radius 1) for each word as a set of indices
    # Using numpy arrays for faster intersection/union if needed, but sets are fine for n=6
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
    
    # Start with direct sum construction
    initial_code = _direct_sum_construction(q, n, radius)
    code_set = set(tuple(cw) for cw in initial_code)
    
    # Track coverage counts
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for cw in code_set:
        idx = word_to_idx[cw]
        for nbr in neighbors[idx]:
            coverage_count[nbr] += 1
            
    # Helper to check if removing idx is safe
    def is_redundant(idx):
        for nbr in neighbors[idx]:
            if coverage_count[nbr] == 1:
                return False
        return True

    # Helper to perform swap
    def try_swap(remove_idx, add_idx):
        # Check if add covers all uniquely covered words of remove
        # And if the result is a valid cover
        # A word is covered if coverage_count > 0.
        # After removing remove_idx, words in neighbors[remove_idx] with count 1 become 0.
        # These MUST be in neighbors[add_idx].
        
        # Optimization: Pre-compute uniquely covered set for remove_idx? 
        # It changes as we swap.
        
        # Check validity
        for nbr in neighbors[remove_idx]:
            if coverage_count[nbr] == 1:
                if nbr not in neighbors[add_idx]:
                    return False
        return True

    best_code_size = len(code_set)
    
    # Simulated Annealing / Local Search
    # We use a temperature that decays, allowing "bad" moves (increasing fragility) to escape local minima
    # Objective: Minimize |C|. Secondary: Minimize number of singly-covered words (fragility)
    
    current_fragility = np.sum(coverage_count == 1)
    
    # Main search loop
    while time.time() < deadline - 5:
        # 1. Try to remove redundant codewords
        removed_any = False
        for cw_tuple in list(code_set):
            if time.time() > deadline - 5:
                break
            idx = word_to_idx[cw_tuple]
            if is_redundant(idx):
                # Remove it
                for nbr in neighbors[idx]:
                    coverage_count[nbr] -= 1
                code_set.remove(cw_tuple)
                removed_any = True
                current_fragility = np.sum(coverage_count == 1)
        
        if removed_any:
            continue # Restart removal phase as state changed significantly
        
        # 2. If no removals, try swaps
        # Pick a random codeword to potentially swap
        # Or pick the one with highest "potential" (covers many fragility points?)
        # Let's try random swaps with acceptance based on fragility reduction
        
        cw_tuple = random.choice(list(code_set))
        remove_idx = word_to_idx[cw_tuple]
        
        # Pick a random candidate to add
        # Bias towards words that cover many currently singly-covered words?
        # For now, simple random
        add_idx = random.randint(0, total_words - 1)
        
        if add_idx == remove_idx:
            continue
            
        if add_idx in [word_to_idx[cw] for cw in code_set]: # Check if already in code
            continue
            
        # Check if swap is valid
        if try_swap(remove_idx, add_idx):
            # Calculate new fragility
            # We need to simulate the swap to count fragility, or approximate
            # Words in neighbors[remove_idx] \ neighbors[add_idx]: count decreases by 1
            # Words in neighbors[add_idx] \ neighbors[remove_idx]: count increases by 1
            # Words in intersection: count unchanged
            
            old_fragility = current_fragility
            
            # Count changes
            # This is O(n) per swap if we iterate neighbors.
            # Let's just do it if we accept, to save time?
            # Or estimate:
            # If we reduce fragility, accept.
            
            # Perform swap to check fragility precisely
            # (This is expensive, but n=6 is small)
            
            # Simulate
            new_fragility = 0
            # We can't easily compute new_fragility without updating array.
            # Let's just accept if it seems "good" or random.
            
            # Actually, let's just accept if it doesn't increase fragility?
            # Or use a simple heuristic: Accept if it covers more "rare" words.
            
            # Let's just perform the swap and measure fragility.
            # Update coverage
            for nbr in neighbors[remove_idx]:
                coverage_count[nbr] -= 1
            for nbr in neighbors[add_idx]:
                coverage_count[nbr] += 1
                
            new_fragility = np.sum(coverage_count == 1)
            
            # Acceptance criterion:
            # If fragility decreased, accept.
            # If fragility increased, accept with probability exp(-delta/T)?
            # For simplicity, let's just accept if fragility <= old_fragility.
            # This might get stuck.
            
            if new_fragility <= old_fragility:
                code_set.remove(cw_tuple)
                code_set.add(tuple(all_words[add_idx]))
                current_fragility = new_fragility
            else:
                # Revert
                for nbr in neighbors[remove_idx]:
                    coverage_count[nbr] += 1
                for nbr in neighbors[add_idx]:
                    coverage_count[nbr] -= 1
                # Maybe accept with small probability to escape?
                if random.random() < 0.1:
                    code_set.remove(cw_tuple)
                    code_set.add(tuple(all_words[add_idx]))
                    current_fragility = new_fragility

    # Final verification
    coverage_final = np.zeros(total_words, dtype=np.int32)
    for cw in code_set:
        idx = word_to_idx[cw]
        for nbr in neighbors[idx]:
            coverage_final[nbr] += 1
    
    if np.all(coverage_final > 0):
        return [list(cw) for cw in code_set]
    else:
        return _direct_sum_construction(q, n, radius)

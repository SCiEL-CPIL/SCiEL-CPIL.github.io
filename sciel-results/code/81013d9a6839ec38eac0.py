"""Combined approach: direct sum initialization + simulated annealing + local search for ternary covering code K_3(6,1)."""
import itertools
import time
import random
import numpy as np

def _direct_sum_construction(q, n, radius):
    """Fallback: direct sum of ternary tetracode with all words on remaining coords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

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
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    total_words = q ** n
    sphere_size = 1 + n * (q - 1)
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
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
    
    # Start with direct sum construction
    initial_code = _direct_sum_construction(q, n, radius)
    code_set = set(tuple(cw) for cw in initial_code)
    
    # Track which words are covered and by how many codewords
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for cw in code_set:
        idx = word_to_idx[cw]
        for nbr in neighbors[idx]:
            coverage_count[nbr] += 1
    
    # Verify initial code covers everything
    if np.any(coverage_count == 0):
        return initial_code
    
    # Phase 1: Local search - try to remove redundant codewords
    def try_remove_redundant(code_set, coverage_count, deadline):
        """Remove any codeword that doesn't uniquely cover any word."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for cw_tuple in list(code_set):
                if time.time() > deadline - 5:
                    break
                idx = word_to_idx[cw_tuple]
                # Check if removing this codeword still covers all words
                uniquely_covered = False
                for nbr in neighbors[idx]:
                    if coverage_count[nbr] == 1:
                        uniquely_covered = True
                        break
                if not uniquely_covered:
                    # Safe to remove
                    for nbr in neighbors[idx]:
                        coverage_count[nbr] -= 1
                    code_set.remove(cw_tuple)
                    changed = True
    
    try_remove_redundant(code_set, coverage_count, deadline)
    
    # Phase 2: Simulated annealing with swap moves
    # State: code_set (set of tuples), coverage_count (numpy array)
    # Objective: minimize len(code_set) while maintaining full coverage
    
    best_code_set = set(code_set)
    best_size = len(code_set)
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    n_sa_steps = 5000
    cooling = (T_end / T_start) ** (1.0 / max(1, n_sa_steps - 1))
    
    current_size = len(code_set)
    
    for step in range(n_sa_steps):
        if time.time() > deadline - 10:
            break
        
        T = T_start * (cooling ** step)
        
        # Random move: either swap (replace one codeword with another) or try removal
        move_type = random.random()
        
        if move_type < 0.3 and current_size > 1:
            # Try to remove a random codeword
            candidates = list(code_set)
            if not candidates:
                break
            cw_tuple = random.choice(candidates)
            idx = word_to_idx[cw_tuple]
            
            # Check if removal is valid
            uniquely_covered = False
            for nbr in neighbors[idx]:
                if coverage_count[nbr] == 1:
                    uniquely_covered = True
                    break
            
            if uniquely_covered:
                continue  # Can't remove
            
            # Perform removal
            for nbr in neighbors[idx]:
                coverage_count[nbr] -= 1
            code_set.remove(cw_tuple)
            current_size -= 1
            
            # Try to remove more redundant codewords
            try_remove_redundant(code_set, coverage_count, deadline)
            current_size = len(code_set)
            
            # Update best
            if current_size < best_size:
                best_size = current_size
                best_code_set = set(code_set)
                
        elif move_type < 0.8:
            # Swap: remove one codeword, add another
            if not code_set:
                break
            
            # Pick a random codeword to potentially remove
            cw_tuple = random.choice(list(code_set))
            rem_idx = word_to_idx[cw_tuple]
            
            # Pick a random candidate to add
            add_idx = random.randint(0, total_words - 1)
            if add_idx == rem_idx:
                continue
            
            # Check if this swap maintains coverage
            # Words uniquely covered by rem_idx must be covered by add_idx
            can_swap = True
            for nbr in neighbors[rem_idx]:
                if coverage_count[nbr] == 1:
                    if nbr not in neighbors[add_idx]:
                        can_swap = False
                        break
            
            if not can_swap:
                continue
            
            # Perform swap
            for nbr in neighbors[rem_idx]:
                coverage_count[nbr] -= 1
            for nbr in neighbors[add_idx]:
                coverage_count[nbr] += 1
            code_set.remove(cw_tuple)
            code_set.add(all_words[add_idx])
            
            # Size unchanged, but structure changed
            # Try to remove redundant codewords
            try_remove_redundant(code_set, coverage_count, deadline)
            current_size = len(code_set)
            
            if current_size < best_size:
                best_size = current_size
                best_code_set = set(code_set)
                
        else:
            # Add a random codeword (may enable later removals)
            if current_size > total_words:
                break
            add_idx = random.randint(0, total_words - 1)
            if all_words[add_idx] in code_set:
                continue
            
            for nbr in neighbors[add_idx]:
                coverage_count[nbr] += 1
            code_set.add(all_words[add_idx])
            current_size += 1
            
            # Try to remove redundant codewords
            try_remove_redundant(code_set, coverage_count, deadline)
            current_size = len(code_set)
            
            if current_size < best_size:
                best_size = current_size
                best_code_set = set(code_set)
    
    # Phase 3: Final local search on the best solution found
    if best_size < len(code_set):
        # Restore best solution
        code_set = set(best_code_set)
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in code_set:
            idx = word_to_idx[cw]
            for nbr in neighbors[idx]:
                coverage_count[nbr] += 1
    
    # Final aggressive local search
    try_remove_redundant(code_set, coverage_count, deadline)
    
    # Phase 4: Swap-based improvement on the final solution
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        # Find words that are covered by exactly one codeword
        singly_covered = np.where(coverage_count == 1)[0]
        if len(singly_covered) == 0:
            break
        
        # For each codeword, see how many singly-covered words it covers
        best_removal = None
        best_removal_count = 0
        for cw_tuple in code_set:
            idx = word_to_idx[cw_tuple]
            count = 0
            for nbr in neighbors[idx]:
                if coverage_count[nbr] == 1:
                    count += 1
            if count > best_removal_count:
                best_removal_count = count
                best_removal = cw_tuple
        
        if best_removal is None or best_removal_count == 0:
            break
        
        # Try to find a replacement codeword that covers most of the same words
        rem_idx = word_to_idx[best_removal]
        rem_neighbors = neighbors[rem_idx]
        
        # Find a candidate replacement
        best_add = None
        best_overlap = 0
        for i in range(total_words):
            if i == rem_idx:
                continue
            # How many of rem_neighbors does candidate i cover?
            overlap = 0
            for nbr in rem_neighbors:
                if nbr in neighbors[i]:
                    overlap += 1
            if overlap > best_overlap:
                best_overlap = overlap
                best_add = i
        
        if best_add is not None and best_overlap >= best_removal_count:
            # Try the swap
            add_idx = best_add
            add_neighbors = neighbors[add_idx]
            
            # Check if after removing best_removal and adding best_add, all words are still covered
            can_swap = True
            for nbr in rem_neighbors:
                if coverage_count[nbr] == 1:
                    if nbr not in add_neighbors:
                        can_swap = False
                        break
            
            if can_swap:
                # Perform swap
                for nbr in rem_neighbors:
                    coverage_count[nbr] -= 1
                for nbr in add_neighbors:
                    coverage_count[nbr] += 1
                code_set.remove(best_removal)
                code_set.add(all_words[add_idx])
                
                # Now try to remove more codewords
                try_remove_redundant(code_set, coverage_count, deadline)
    
    # Final verification
    coverage_final = np.zeros(total_words, dtype=np.int32)
    for cw in code_set:
        idx = word_to_idx[cw]
        for nbr in neighbors[idx]:
            coverage_final[nbr] += 1
    
    if np.all(coverage_final > 0):
        return [list(cw) for cw in code_set]
    else:
        # Fallback to direct sum
        return _direct_sum_construction(q, n, radius)

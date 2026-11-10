"""Simulated annealing for ternary covering code K_3(7,1)."""
import numpy as np
import random
import time


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius.
    Uses simulated annealing to minimize the number of codewords while ensuring
    full coverage.
    """
    if q != 3 or n != 7 or radius != 1:
        # Fallback: use the direct sum construction for other parameters
        import itertools
        # For general case, use a simple greedy or the tetracode-based approach
        # But since the task is specifically for q=3, n=7, radius=1, we focus there.
        # For robustness, return a valid (if not optimal) code.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Brute force for small n
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words
    
    # Specific optimization for q=3, n=7, radius=1
    total_words = 3 ** n  # 2187
    
    # Precompute neighbors: for each word, list of words within Hamming distance 1
    # To save memory and time, we can compute coverage on the fly or use bitsets.
    # Let's use a bitset approach for efficiency.
    # Each word is an index 0..2186.
    # We need to know, for each selected codeword, which words it covers.
    # A codeword covers itself and its 6 neighbors (change one position to one of the other 2 values).
    
    # Precompute the coverage mask for each possible codeword.
    # mask[i] is a set (or bitset) of words covered by word i.
    # Using Python integers as bitsets: 2187 bits fits in ~300 bytes per int, but Python ints are arbitrary precision.
    # Alternatively, use numpy arrays of bools for each codeword? That's 2187 * 2187 bools ~ 4.7 MB, which is fine.
    # But updating coverage counts is easier with an array of counts.
    
    # Let's precompute the list of neighbors for each word.
    # word_to_idx: mapping from tuple to index
    # idx_to_word: mapping from index to tuple
    
    all_words = list(np.ndindex(3, 3, 3, 3, 3, 3, 3))
    # This is slow. Let's use itertools.
    import itertools
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage: for each word i, what words does it cover?
    # It covers i and all words at distance 1.
    # Let's store this as a list of lists of indices.
    coverage = [None] * total_words
    for i, w in enumerate(all_words):
        covered = [i]
        w_list = list(w)
        for pos in range(7):
            original = w_list[pos]
            for val in range(3):
                if val == original:
                    continue
                w_list[pos] = val
                neighbor_idx = word_to_idx[tuple(w_list)]
                covered.append(neighbor_idx)
            w_list[pos] = original
        coverage[i] = covered
    
    # Initial solution: random selection or greedy
    # Let's start with a random set of codewords, say 200.
    # Or better, start with a good heuristic: random 180 words.
    # We want to minimize the number of codewords.
    # Start with a feasible solution? Or infeasible and anneal?
    # Let's start with a large random set to ensure coverage, then remove.
    # Or start with a small set and add if needed.
    
    # Strategy: Start with a random set of size ~190.
    # Track coverage count for each word.
    # If a word is uncovered, we must add a codeword that covers it.
    # If a word is over-covered, we might remove a codeword.
    
    # Let's use a simulated annealing approach:
    # State: set of selected codewords (boolean array)
    # Energy: (number of uncovered words) * large_penalty + (number of selected codewords)
    # Moves:
    # 1. Add a random codeword.
    # 2. Remove a random codeword (if it doesn't make coverage infeasible? Or allow infeasibility).
    # 3. Swap: remove one, add one.
    
    # To handle infeasibility, we use a penalty for uncovered words.
    # Penalty should be large enough to prioritize feasibility, but allow temporary violations.
    
    # Let's initialize with a greedy set cover to get a good starting point.
    # Greedy: repeatedly pick the codeword that covers the most uncovered words.
    
    uncovered = np.ones(total_words, dtype=bool)
    covered_count = np.zeros(total_words, dtype=int)
    selected = np.zeros(total_words, dtype=bool)
    
    # Greedy initialization
    for _ in range(300):  # Cap the greedy steps to avoid too large a code
        if not uncovered.any():
            break
        # Find the codeword that covers the most uncovered words
        best_idx = -1
        best_count = 0
        # This is O(N^2) per step, which is too slow for 2187 words.
        # We need a faster way.
        # Let's just pick a random codeword that covers at least one uncovered word.
        # Or use a random subset.
        
        # Let's just start with a random set of 200 codewords and hope it's feasible.
        # If not, the annealing will fix it.
        
        pass
    
    # Better initialization: Random set of size 190.
    initial_size = 190
    selected_indices = np.random.choice(total_words, size=initial_size, replace=False)
    selected[selected_indices] = True
    
    # Compute coverage counts
    for idx in selected_indices:
        for word_idx in coverage[idx]:
            covered_count[word_idx] += 1
    
    uncovered_count = np.sum(covered_count == 0)
    
    def compute_energy(selected, covered_count):
        num_selected = np.sum(selected)
        uncovered = np.sum(covered_count == 0)
        return uncovered * 1000 + num_selected
    
    current_energy = compute_energy(selected, covered_count)
    
    # Simulated Annealing
    start_temp = 100.0
    end_temp = 0.1
    cooling_rate = 0.9995
    max_iterations = 50000
    time_limit = 100  # seconds
    
    start_time = time.time()
    temp = start_temp
    best_energy = current_energy
    best_selected = selected.copy()
    best_covered_count = covered_count.copy()
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Choose a move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Pick a random word to add
            idx = random.randint(0, total_words - 1)
            if selected[idx]:
                continue
            # Add it
            selected[idx] = True
            delta_energy = 1  # +1 codeword
            for word_idx in coverage[idx]:
                if covered_count[word_idx] == 0:
                    delta_energy -= 1000  # Reduced uncovered count by 1
                covered_count[word_idx] += 1
            
            new_energy = current_energy + delta_energy
            if new_energy < current_energy or random.random() < np.exp(-(new_energy - current_energy) / temp):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = selected.copy()
                    best_covered_count = covered_count.copy()
            else:
                # Revert
                selected[idx] = False
                for word_idx in coverage[idx]:
                    covered_count[word_idx] -= 1
        
        elif move_type == 'remove':
            # Pick a random selected word to remove
            selected_indices = np.where(selected)[0]
            if len(selected_indices) == 0:
                continue
            idx = random.choice(selected_indices)
            # Check if removal causes uncovered words
            delta_energy = -1  # -1 codeword
            for word_idx in coverage[idx]:
                if covered_count[word_idx] == 1:
                    delta_energy += 1000  # Increased uncovered count by 1
                covered_count[word_idx] -= 1
            
            new_energy = current_energy + delta_energy
            if new_energy < current_energy or random.random() < np.exp(-(new_energy - current_energy) / temp):
                selected[idx] = False
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = selected.copy()
                    best_covered_count = covered_count.copy()
            else:
                # Revert
                for word_idx in coverage[idx]:
                    covered_count[word_idx] += 1
        
        else:  # swap
            # Remove a random selected word, add a random unselected word
            selected_indices = np.where(selected)[0]
            if len(selected_indices) == 0:
                continue
            idx_remove = random.choice(selected_indices)
            idx_add = random.randint(0, total_words - 1)
            if selected[idx_add]:
                continue
            
            # Apply removal
            delta_energy = -1
            for word_idx in coverage[idx_remove]:
                if covered_count[word_idx] == 1:
                    delta_energy += 1000
                covered_count[word_idx] -= 1
            
            # Apply addition
            delta_energy += 1
            for word_idx in coverage[idx_add]:
                if covered_count[word_idx] == 0:
                    delta_energy -= 1000
                covered_count[word_idx] += 1
            
            new_energy = current_energy + delta_energy
            if new_energy < current_energy or random.random() < np.exp(-(new_energy - current_energy) / temp):
                selected[idx_remove] = False
                selected[idx_add] = True
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = selected.copy()
                    best_covered_count = covered_count.copy()
            else:
                # Revert
                for word_idx in coverage[idx_remove]:
                    covered_count[word_idx] += 1
                for word_idx in coverage[idx_add]:
                    covered_count[word_idx] -= 1
        
        temp *= cooling_rate
    
    # Return the best solution found
    # Ensure it's feasible. If not, we might need to add codewords.
    # But the penalty should have driven it to feasibility.
    
    final_selected = best_selected
    final_indices = np.where(final_selected)[0]
    
    # Verify coverage
    final_covered_count = np.zeros(total_words, dtype=int)
    for idx in final_indices:
        for word_idx in coverage[idx]:
            final_covered_count[word_idx] += 1
    
    if np.any(final_covered_count == 0):
        # Not feasible, add codewords to cover the rest
        uncovered_words = np.where(final_covered_count == 0)[0]
        for w in uncovered_words:
            # Find a codeword that covers w
            for c in range(total_words):
                if w in coverage[c]:
                    final_selected[c] = True
                    for word_idx in coverage[c]:
                        final_covered_count[word_idx] += 1
                    break
    
    final_indices = np.where(final_selected)[0]
    return [list(all_words[idx]) for idx in final_indices]

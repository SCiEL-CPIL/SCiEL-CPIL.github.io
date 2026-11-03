import itertools
import time
import random
import numpy as np
from collections import Counter

def construct(q, n, radius):
    """
    Construct a ternary covering code K_3(6,1).
    Returns a list of codewords (lists of integers).
    """
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin for 120s limit (evaluator kills at 150s, but we want to be safe)
    
    # --- 1. Setup and Fallback ---
    # The task specifies q=3, n=6, radius=1.
    # If parameters differ, fallback to a simple valid construction.
    if q != 3 or n != 6 or radius != 1:
        if n <= 4:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        else:
            # For robustness, return a list of a few words, though likely invalid for large n.
            # We assume q=3, n=6, radius=1 based on the task description.
            # A trivial valid code is all words.
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n  # 729
    
    # --- 2. Precompute Data Structures ---
    
    # Helper: Convert index to word tuple
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[n - 1 - i] = temp % 3
            temp //= 3
        return tuple(w)
    
    # Helper: Convert word tuple to index
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * 3 + val
        return idx

    # Precompute spheres (Hamming balls of radius 1) as sets of indices
    # spheres[i] contains all indices within distance 1 of index i
    spheres = [None] * total_words
    for idx in range(total_words):
        w = list(idx_to_word(idx))
        sphere_set = {idx}
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                sphere_set.add(word_to_idx(w))
                w[pos] = orig_val
        spheres[idx] = sphere_set

    # --- 3. Initial Code Construction ---
    # Direct sum of tetracode (9 words, len 4) and all words on remaining n-4 coords
    # Tetracode: { (a, b, a+b, a+2b) mod 3 }
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(h + list(tail))
    
    # Convert to indices for fast processing
    initial_indices = [word_to_idx(cw) for cw in initial_codewords]
    
    # --- 4. Optimization Loop ---
    
    best_code = initial_codewords[:]
    best_len = len(best_code)
    
    rng = random.Random(42)
    
    def compute_counts(indices):
        """Compute coverage counts for a list of codeword indices."""
        counts = [0] * total_words
        for idx in indices:
            for neighbor in spheres[idx]:
                counts[neighbor] += 1
        return counts
    
    def is_valid_code(indices):
        """Check if all words are covered by the given codeword indices."""
        if not indices:
            return total_words == 0
        counts = compute_counts(indices)
        return min(counts) >= 1
    
    def greedy_remove(indices, order=None):
        """Greedy removal: remove redundant codewords in given order."""
        if not indices:
            return []
        
        counts = compute_counts(indices)
        active = [True] * len(indices)
        
        if order is None:
            order = list(range(len(indices)))
            rng.shuffle(order)
        
        # Process in given order, removing non-essential codewords
        for i in order:
            if not active[i]:
                continue
            idx = indices[i]
            essential = False
            for neighbor in spheres[idx]:
                if counts[neighbor] == 1:
                    essential = True
                    break
            if not essential:
                active[i] = False
                for neighbor in spheres[idx]:
                    counts[neighbor] -= 1
        
        return [indices[i] for i in range(len(indices)) if active[i]]
    
    def simulated_anneal_removal(indices, max_iter=500):
        """Simulated annealing for codeword removal to escape local optima."""
        if len(indices) <= 1:
            return indices
        
        current = list(indices)
        counts = compute_counts(current)
        current_len = len(current)
        best = current[:]
        best_len = current_len
        
        # Precompute set of current codewords for fast lookup
        current_set = set(current)
        
        # Temperature schedule
        T_start = 2.0
        T_end = 0.1
        cooling = (T_end / T_start) ** (1.0 / max_iter)
        T = T_start
        
        for iteration in range(max_iter):
            T *= cooling
            
            # Pick a random codeword to try removing
            i = rng.randint(0, len(current) - 1)
            idx = current[i]
            
            # Check if removable (no word covered only by it)
            can_remove = True
            for neighbor in spheres[idx]:
                if counts[neighbor] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Always accept removal
                for neighbor in spheres[idx]:
                    counts[neighbor] -= 1
                current.pop(i)
                current_set.discard(idx)
                current_len -= 1
                if current_len < best_len:
                    best = current[:]
                    best_len = current_len
            else:
                # Try adding a random word from the initial set that's not in current
                if rng.random() < 0.3 and len(current) < len(initial_indices):
                    # Find a word from initial_indices not in current
                    candidates = [x for x in initial_indices if x not in current_set]
                    if candidates:
                        add_idx = rng.choice(candidates)
                        # Check if adding helps (reduces uncovered words)
                        # Simple heuristic: add if it covers at least one word with count 0
                        helpful = False
                        for neighbor in spheres[add_idx]:
                            if counts[neighbor] == 0:
                                helpful = True
                                break
                        if helpful or rng.random() < 0.1 * T:
                            current.append(add_idx)
                            current_set.add(add_idx)
                            current_len += 1
                            for neighbor in spheres[add_idx]:
                                counts[neighbor] += 1
                            # Immediately try to remove some redundant words
                            current = greedy_remove(current)
                            counts = compute_counts(current)
                            current_len = len(current)
                            current_set = set(current)
                            if current_len < best_len:
                                best = current[:]
                                best_len = current_len
        
        return best
    
    # Phase 1: Multiple greedy removal passes with different random orders
    phase1_deadline = deadline - 30.0
    while time.time() < phase1_deadline and best_len > 73:
        order = list(range(len(initial_indices)))
        rng.shuffle(order)
        result = greedy_remove(initial_indices, order)
        if len(result) < best_len:
            # Verify validity before accepting
            if is_valid_code(result):
                best_code = [list(idx_to_word(idx)) for idx in result]
                best_len = len(result)
    
    # Phase 2: Simulated annealing refinement from best solution
    if best_len > 73 and time.time() < deadline - 10.0:
        best_indices = [word_to_idx(cw) for cw in best_code]
        # Run SA multiple times
        sa_deadline = deadline - 5.0
        sa_runs = 0
        while time.time() < sa_deadline and best_len > 73:
            sa_result = simulated_anneal_removal(best_indices, max_iter=300)
            if len(sa_result) < best_len:
                # Verify validity before accepting
                if is_valid_code(sa_result):
                    best_code = [list(idx_to_word(idx)) for idx in sa_result]
                    best_len = len(sa_result)
                    best_indices = sa_result[:]
            sa_runs += 1
            # Update best_indices for next SA run
            if len(sa_result) < len(best_indices):
                best_indices = sa_result[:]
    
    # Phase 3: Final greedy polish from best solution
    if best_len > 73 and time.time() < deadline - 2.0:
        best_indices = [word_to_idx(cw) for cw in best_code]
        final_result = greedy_remove(best_indices)
        if len(final_result) < best_len:
            # Verify validity before accepting
            if is_valid_code(final_result):
                best_code = [list(idx_to_word(idx)) for idx in final_result]
                best_len = len(final_result)
    
    # Final verification: ensure the returned code is valid
    if not is_valid_code([word_to_idx(cw) for cw in best_code]):
        # Fallback: return a known valid code (the initial construction)
        best_code = initial_codewords[:]
    
    return best_code

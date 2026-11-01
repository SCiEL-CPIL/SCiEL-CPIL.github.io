import itertools
import random
import time
import numpy as np

def _popcount(x):
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count('1')

def construct(q, n, radius):
    # Specialized for K_3(6,1)
    if q != 3 or n != 6 or radius != 1:
        return [list(c) for c in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    total_words = 729
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute coverage as bitsets
    coverage = [0] * total_words
    for i, c in enumerate(all_words):
        mask = 1 << word_to_idx[c]
        for pos in range(6):
            for val in range(3):
                if val == c[pos]:
                    continue
                new_c = list(c)
                new_c[pos] = val
                mask |= 1 << word_to_idx[tuple(new_c)]
        coverage[i] = mask

    # Convert coverage to a boolean matrix for NumPy operations
    # coverage_matrix[i, j] is True if word j is covered by codeword i
    # We can store this as a list of sets or bitsets, but for the SA heuristic
    # we need fast deficit calculation.
    
    # Let's represent the coverage of a set S as a bitset union.
    # The "deficit" is the number of uncovered words.
    # deficit(S) = total_words - popcount(union_coverage(S))
    
    # For the SA heuristic, we want to minimize:
    # objective = |S| + lambda * deficit(S)
    # where lambda is a penalty weight.
    
    # We will use a simulated annealing approach where:
    # - State: a set of codewords S (represented as a list or set of indices)
    # - Move: add a random codeword, remove a random codeword, or swap
    # - Cost: |S| + penalty * deficit(S)
    # - Acceptance: Metropolis criterion with cooling
    
    # To make it fast, we need to update the union coverage incrementally.
    # However, maintaining a union of 729 bits in Python integers is fast.
    
    best_code = None
    best_size = total_words
    best_cost = float('inf')

    # Initial guess: Tetracode direct sum (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            w = tuple(h + list(tail))
            initial_indices.append(word_to_idx[w])
    
    initial_set = set(initial_indices)
    initial_union = 0
    for i in initial_indices:
        initial_union |= coverage[i]
    initial_deficit = total_words - _popcount(initial_union)
    initial_cost = len(initial_indices) + 10.0 * initial_deficit # penalty weight 10

    # If initial is already perfect (deficit 0), cost is 81.
    # We want to get to 73.

    # Simulated Annealing Parameters
    T = 100.0
    T_min = 0.01
    alpha = 0.995 # cooling rate
    max_steps_per_restart = 5000
    
    r = 0
    while time.time() < deadline:
        # Start from a perturbed version of the best known or the initial
        if best_code is not None and r > 0 and random.random() < 0.5:
            current_indices = list(best_code_indices)
        else:
            current_indices = list(initial_indices)
        
        random.shuffle(current_indices)
        current_set = set(current_indices)
        
        # Calculate initial union and deficit
        current_union = 0
        for i in current_indices:
            current_union |= coverage[i]
        current_deficit = total_words - _popcount(current_union)
        current_cost = len(current_indices) + 10.0 * current_deficit
        
        T = 100.0
        step = 0
        improved = False
        
        while T > T_min and step < max_steps_per_restart and time.time() < deadline:
            step += 1
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.3:
                # Remove a random codeword
                if not current_indices:
                    continue
                idx_to_remove = random.choice(current_indices)
                new_indices = [i for i in current_indices if i != idx_to_remove]
                new_set = current_set - {idx_to_remove}
                
                # Recalculate union efficiently?
                # Removing one word from union is hard with bitsets (need to recompute or track counts).
                # For n=729, recomputing union from scratch is O(|S| * 1) bitwise ORs.
                # |S| is ~70-80. 80 ORs of 729-bit ints is very fast.
                new_union = 0
                for i in new_indices:
                    new_union |= coverage[i]
                new_deficit = total_words - _popcount(new_union)
                new_cost = len(new_indices) + 10.0 * new_deficit
                
                delta_cost = new_cost - current_cost
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    current_indices = new_indices
                    current_set = new_set
                    current_union = new_union
                    current_deficit = new_deficit
                    current_cost = new_cost
                    if current_deficit == 0 and len(current_indices) < best_size:
                        best_size = len(current_indices)
                        best_code = [all_words[i] for i in current_indices]
                        best_code_indices = list(current_indices)
                        best_cost = current_cost
                        improved = True
                        
            elif move_type < 0.6:
                # Add a random codeword
                idx_to_add = random.randint(0, total_words - 1)
                if idx_to_add in current_set:
                    continue
                new_indices = current_indices + [idx_to_add]
                new_set = current_set | {idx_to_add}
                
                new_union = current_union | coverage[idx_to_add]
                new_deficit = total_words - _popcount(new_union)
                new_cost = len(new_indices) + 10.0 * new_deficit
                
                delta_cost = new_cost - current_cost
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    current_indices = new_indices
                    current_set = new_set
                    current_union = new_union
                    current_deficit = new_deficit
                    current_cost = new_cost
                    if current_deficit == 0 and len(current_indices) < best_size:
                        best_size = len(current_indices)
                        best_code = [all_words[i] for i in current_indices]
                        best_code_indices = list(current_indices)
                        best_cost = current_cost
                        improved = True

            else:
                # Swap: remove one, add one
                if not current_indices:
                    continue
                idx_to_remove = random.choice(current_indices)
                idx_to_add = random.randint(0, total_words - 1)
                if idx_to_add in current_set:
                    continue
                if idx_to_add == idx_to_remove:
                    continue
                    
                new_indices = [i for i in current_indices if i != idx_to_remove] + [idx_to_add]
                new_set = (current_set - {idx_to_remove}) | {idx_to_add}
                
                # Recompute union
                new_union = 0
                for i in new_indices:
                    new_union |= coverage[i]
                new_deficit = total_words - _popcount(new_union)
                new_cost = len(new_indices) + 10.0 * new_deficit
                
                delta_cost = new_cost - current_cost
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    current_indices = new_indices
                    current_set = new_set
                    current_union = new_union
                    current_deficit = new_deficit
                    current_cost = new_cost
                    if current_deficit == 0 and len(current_indices) < best_size:
                        best_size = len(current_indices)
                        best_code = [all_words[i] for i in current_indices]
                        best_code_indices = list(current_indices)
                        best_cost = current_cost
                        improved = True

            T *= alpha
            
            # If we hit a perfect code, try to minimize size further with local search
            if current_deficit == 0:
                # Try to remove redundant words
                changed = True
                while changed:
                    changed = False
                    for i in range(len(current_indices)):
                        idx = current_indices[i]
                        # Check if idx is redundant
                        others_union = 0
                        for j in range(len(current_indices)):
                            if j != i:
                                others_union |= coverage[current_indices[j]]
                        
                        if (coverage[idx] & ~others_union) == 0:
                            # Redundant
                            current_indices.pop(i)
                            current_set.discard(idx)
                            current_union = 0
                            for k in current_indices:
                                current_union |= coverage[k]
                            current_deficit = total_words - _popcount(current_union)
                            current_cost = len(current_indices) + 10.0 * current_deficit
                            
                            if current_deficit == 0 and len(current_indices) < best_size:
                                best_size = len(current_indices)
                                best_code = [all_words[k] for k in current_indices]
                                best_code_indices = list(current_indices)
                                best_cost = current_cost
                            
                            changed = True
                            break
                # If we improved, restart SA with lower T or break
                if improved:
                    break

        r += 1

    if best_code is None:
        # Fallback to tetracode
        return [list(h) + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in best_code]

# EVOLVE-BLOCK-START
# The code above is the complete implementation.
# EVOLVE-BLOCK-END

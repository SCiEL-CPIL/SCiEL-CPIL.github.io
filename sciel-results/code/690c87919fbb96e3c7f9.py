"""
Ternary covering code K_3(6,1) construction.
Uses Simulated Annealing with coverage count tracking to minimize codewords.
Falls back to the direct-sum construction (81 words) if time runs out.
"""
import time
import numpy as np
import random
import math

def _direct_sum_construction(q, n, radius):
    """Fallback: direct sum of ternary tetracode with all words on remaining coords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    Returns a list of codewords (each a list of n integers).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s
    
    total_words = q ** n
    
    # Generate all words
    import itertools
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor lists as lists of indices
    neighbor_lists = []
    for w in all_words:
        nbrs = [word_to_idx[w]]
        for i in range(n):
            for v in range(q):
                if v != w[i]:
                    nw = list(w)
                    nw[i] = v
                    nbrs.append(word_to_idx[tuple(nw)])
        neighbor_lists.append(nbrs)
        
    # Initial solution: Direct Sum Construction
    initial_code = _direct_sum_construction(q, n, radius)
    initial_indices = list(set(word_to_idx[tuple(cw)] for cw in initial_code))
    
    # State variables
    in_code = [False] * total_words
    cover_count = [0] * total_words
    uncovered_count = 0
    
    for idx in initial_indices:
        in_code[idx] = True
        for nbr in neighbor_lists[idx]:
            cover_count[nbr] += 1
            
    for c in cover_count:
        if c == 0:
            uncovered_count += 1
            
    best_indices = list(initial_indices)
    best_uncovered = uncovered_count
    best_size = len(best_indices)
    
    # Simulated Annealing
    # Cost = size + penalty * uncovered
    # We want to minimize cost.
    # If uncovered == 0, cost = size.
    # If uncovered > 0, cost is high.
    
    penalty = 5.0
    T = 10.0
    T_min = 0.01
    alpha = 0.9995
    
    current_indices = set(initial_indices)
    current_size = len(current_indices)
    current_uncovered = uncovered_count
    
    # Random seed for reproducibility in local runs, but random in eval
    random.seed(42)
    
    while time.time() < deadline:
        # Pick a move
        # Bias towards removal if valid, towards addition if invalid
        if current_uncovered == 0:
            # Valid state. Try to remove.
            if current_indices:
                idx = random.choice(list(current_indices))
                # Calculate delta
                delta_uncovered = 0
                for nbr in neighbor_lists[idx]:
                    if cover_count[nbr] == 1:
                        delta_uncovered += 1
                # Apply
                for nbr in neighbor_lists[idx]:
                    cover_count[nbr] -= 1
                    if cover_count[nbr] == 0:
                        current_uncovered += 1
                in_code[idx] = False
                current_indices.discard(idx)
                current_size -= 1
                
                delta_cost = -1 + penalty * (current_uncovered - (current_uncovered - delta_uncovered)) 
                # Wait, delta_uncovered calculated above is the increase.
                # Let's just compute cost difference directly.
                # Old cost: current_size + 1 + penalty * (current_uncovered - delta_uncovered)
                # New cost: current_size + penalty * current_uncovered
                # This is getting messy. Let's just track cost.
                
                # Re-calc cost change properly:
                # We already applied the change.
                # We need to know if we accept.
                # Let's store old state? No, too slow.
                # Let's calculate delta_cost BEFORE applying.
                
                # ROLLBACK for simplicity in logic, then apply if accepted.
                # Rollback
                in_code[idx] = True
                current_indices.add(idx)
                current_size += 1
                for nbr in neighbor_lists[idx]:
                    if cover_count[nbr] == 0:
                        current_uncovered -= 1
                    cover_count[nbr] += 1
                    
                # Calculate delta
                delta_uncovered_inc = 0
                for nbr in neighbor_lists[idx]:
                    if cover_count[nbr] == 1:
                        delta_uncovered_inc += 1
                delta_cost = -1 + penalty * delta_uncovered_inc
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    # Accept
                    for nbr in neighbor_lists[idx]:
                        cover_count[nbr] -= 1
                        if cover_count[nbr] == 0:
                            current_uncovered += 1
                    in_code[idx] = False
                    current_indices.discard(idx)
                    current_size -= 1
                    
        else:
            # Invalid state. Try to add a word that covers many uncovered words.
            # Pick a random uncovered word, and add a random neighbor of it?
            # Or pick a random candidate from all words.
            # To be efficient, pick a random word from the set of words that are NOT in code.
            # If code is large, this is expensive.
            # Better: Pick a random uncovered word u. Pick a random neighbor v of u. Add v.
            
            # Find an uncovered word
            # We don't have a list of uncovered words handy.
            # We can just pick a random index and check if it's uncovered.
            # Or maintain a list of uncovered indices.
            
            # Let's maintain a list of uncovered indices for fast sampling.
            # This requires updating the list when cover_count changes.
            # Since we are in the "invalid" branch, we assume we have a list.
            # But we didn't initialize it.
            
            # Let's just pick a random candidate from 0..total_words-1.
            # If it's in code, skip.
            # If it's not, try to add.
            
            idx = random.randint(0, total_words - 1)
            if in_code[idx]:
                continue
                
            # Calculate delta
            delta_uncovered_dec = 0
            for nbr in neighbor_lists[idx]:
                if cover_count[nbr] == 0:
                    delta_uncovered_dec += 1
            
            delta_cost = 1 - penalty * delta_uncovered_dec
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                for nbr in neighbor_lists[idx]:
                    if cover_count[nbr] == 0:
                        current_uncovered -= 1
                    cover_count[nbr] += 1
                in_code[idx] = True
                current_indices.add(idx)
                current_size += 1
        
        # Update best if valid
        if current_uncovered == 0:
            if current_size < best_size:
                best_size = current_size
                best_indices = list(current_indices)
                best_uncovered = 0
                
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart? Or just continue with T_min.
            # If stuck, maybe restart from best?
            # For now, just continue.
            
    # Return best found
    if best_uncovered == 0:
        return [list(all_words[idx]) for idx in best_indices]
    else:
        # Fallback to direct sum if SA didn't find a valid solution
        return _direct_sum_construction(q, n, radius)

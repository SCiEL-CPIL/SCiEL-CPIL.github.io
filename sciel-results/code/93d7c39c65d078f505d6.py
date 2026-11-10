"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocals as Fractions for exact arithmetic
    reciprocals = [Fraction(1, d) for d in denominators]
    
    def get_sum(indices):
        """Compute sum of reciprocals for given indices."""
        s = Fraction(0, 1)
        for i in indices:
            s += reciprocals[i]
        return s
    
    def deficit(indices):
        """Return 1 - sum(reciprocals), or None if sum >= 1."""
        s = get_sum(indices)
        if s >= 1:
            return None
        return 1 - s
    
    # Strategy: Multi-start local search with 2-swap moves
    # 1. Generate diverse initial solutions using different greedy strategies
    # 2. Apply local search (2-swap moves) to improve each
    # 3. Return the best solution found
    
    best_deficit = None
    best_indices = None
    
    def update_best(indices):
        nonlocal best_deficit, best_indices
        d = deficit(indices)
        if d is not None:
            if best_deficit is None or d < best_deficit:
                best_deficit = d
                best_indices = list(indices)
    
    # --- Initial solution: greedy by smallest reciprocal (largest denominator) ---
    # Start with empty set, add items that keep sum < 1, preferring those that
    # get us closest to 1 without exceeding it.
    
    # Method 1: Greedy - repeatedly add the item that minimizes deficit
    # while keeping sum < 1
    def greedy_fill():
        selected = set()
        current_sum = Fraction(0, 1)
        remaining = set(range(n))
        
        while remaining:
            best_item = None
            best_new_sum = None
            for i in remaining:
                new_sum = current_sum + reciprocals[i]
                if new_sum < 1:
                    if best_new_sum is None or new_sum > best_new_sum:
                        best_new_sum = new_sum
                        best_item = i
            if best_item is None:
                break
            selected.add(best_item)
            current_sum = best_new_sum
            remaining.remove(best_item)
        
        return list(selected)
    
    # Method 2: Greedy from largest denominators first (smallest reciprocals)
    def greedy_largest_first():
        selected = set()
        current_sum = Fraction(0, 1)
        # Sort by denominator descending (reciprocal ascending)
        order = sorted(range(n), key=lambda i: denominators[i], reverse=True)
        for i in order:
            new_sum = current_sum + reciprocals[i]
            if new_sum < 1:
                selected.add(i)
                current_sum = new_sum
        return list(selected)
    
    # Method 3: Greedy from smallest denominators first, but only if it fits
    def greedy_smallest_first():
        selected = set()
        current_sum = Fraction(0, 1)
        order = sorted(range(n), key=lambda i: denominators[i])
        for i in order:
            new_sum = current_sum + reciprocals[i]
            if new_sum < 1:
                selected.add(i)
                current_sum = new_sum
        return list(selected)
    
    # Method 4: Randomized greedy with different seeds
    def randomized_greedy(seed):
        rng = random.Random(seed)
        selected = set()
        current_sum = Fraction(0, 1)
        remaining = list(range(n))
        rng.shuffle(remaining)
        
        for i in remaining:
            new_sum = current_sum + reciprocals[i]
            if new_sum < 1:
                selected.add(i)
                current_sum = new_sum
        return list(selected)
    
    # Method 5: Start with a known good subset and try to improve
    # Known: {1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1/10} = 0.928968...
    # Let's use indices for denominators 3,4,5,6,7,8,9,10 -> indices 1,2,3,4,5,6,7
    def known_seed():
        # Denominators 3-10 are indices 1-7
        base = list(range(1, 8))
        return base
    
    # Generate initial candidates
    candidates = []
    candidates.append(greedy_fill())
    candidates.append(greedy_largest_first())
    candidates.append(greedy_smallest_first())
    candidates.append(known_seed())
    
    # Add several randomized starts
    for s in range(20):
        candidates.append(randomized_greedy(s))
    
    # Local search: 2-swap moves
    # Try swapping one selected item with one unselected item, or
    # adding/removing single items (1-swap), then 2-swaps
    
    def local_search(indices, max_time=None):
        """Improve a solution using local search moves."""
        if max_time is None:
            max_time = deadline
        selected = set(indices)
        current_sum = get_sum(indices)
        
        if current_sum >= 1:
            return indices  # Invalid, return as-is
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            best_move = None
            best_def = deficit(indices)
            
            # Try 1-swap: add an unselected item
            for i in range(n):
                if i not in selected:
                    new_sum = current_sum + reciprocals[i]
                    if new_sum < 1:
                        new_def = 1 - new_sum
                        if best_def is None or new_def < best_def:
                            best_def = new_def
                            best_move = ('add', i, None)
            
            # Try 1-swap: remove a selected item
            for i in list(selected):
                new_sum = current_sum - reciprocals[i]
                # Removing always keeps sum < 1 if it was < 1
                new_def = 1 - new_sum
                if best_def is None or new_def < best_def:
                    # Only accept if it actually improves (removing increases deficit)
                    # So this is only useful if we need to make room for a swap
                    pass  # Don't accept pure removals as improvements
            
            # Try 2-swap: remove one selected, add one unselected
            for i in list(selected):
                for j in range(n):
                    if j not in selected:
                        new_sum = current_sum - reciprocals[i] + reciprocals[j]
                        if new_sum < 1:
                            new_def = 1 - new_sum
                            if best_def is None or new_def < best_def:
                                best_def = new_def
                                best_move = ('swap', i, j)
            
            if best_move is not None:
                mtype, i, j = best_move
                if mtype == 'add':
                    selected.add(j)
                    current_sum += reciprocals[j]
                elif mtype == 'swap':
                    selected.remove(i)
                    selected.add(j)
                    current_sum = current_sum - reciprocals[i] + reciprocals[j]
                improved = True
        
        return list(selected)
    
    # Process each candidate with local search
    for idx, cand in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # First, ensure the candidate is valid
        d = deficit(cand)
        if d is None:
            # Try to fix by removing items until valid
            cand_set = set(cand)
            current_sum = get_sum(cand)
            # Remove largest reciprocals first until sum < 1
            while current_sum >= 1 and cand_set:
                # Remove the item with largest reciprocal
                max_recip = max(cand_set, key=lambda i: reciprocals[i])
                cand_set.remove(max_recip)
                current_sum -= reciprocals[max_recip]
            cand = list(cand_set)
        
        if not cand:
            continue
        
        # Apply local search with time limit
        time_limit = min(deadline, start_time + 3.0 + idx * 0.5)
        improved_cand = local_search(cand, max_time=time_limit)
        update_best(improved_cand)
        
        if time.time() > deadline:
            break
    
    # If we still have time, do more randomized restarts
    if time.time() < deadline - 2.0:
        rng = random.Random(42)
        for restart in range(50):
            if time.time() > deadline - 2.0:
                break
            
            # Random initial solution
            cand = []
            current_sum = Fraction(0, 1)
            for i in range(n):
                if rng.random() < 0.3:  # 30% chance to include
                    new_sum = current_sum + reciprocals[i]
                    if new_sum < 1:
                        cand.append(i)
                        current_sum = new_sum
            
            if not cand:
                continue
            
            # Quick local search
            time_limit = min(deadline - 2.0, start_time + 1.0 + restart * 0.2)
            improved_cand = local_search(cand, max_time=time_limit)
            update_best(improved_cand)
    
    # Final fallback: if no valid solution found, return a known valid one
    if best_indices is None or not best_indices:
        # {1/3, 1/4, 1/5, 1/6, 1/7, 1/8} = 0.758928... valid
        best_indices = [1, 2, 3, 4, 5, 6]
    
    return best_indices

# EVOLVE-BLOCK-END

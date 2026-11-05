import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    points = list(range(n))
    all_triples = list(combinations(points, 3))
    n_triples_total = len(all_triples)
    
    # Precompute bitmasks for all triples
    triple_masks = [0] * n_triples_total
    for i, t in enumerate(all_triples):
        triple_masks[i] = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        
    # Precompute conflict: conflicts[i] is a set of indices j such that triple i and j share >= 2 points
    # We can represent conflicts as a bitmask of indices? No, 560 bits is too big for int.
    # Use a set of integers.
    conflicts = [set() for _ in range(n_triples_total)]
    for i in range(n_triples_total):
        for j in range(i + 1, n_triples_total):
            common = triple_masks[i] & triple_masks[j]
            if bin(common).count('1') >= 2:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Precompute masks by size for alpha calculation
    # masks_by_size[k] contains all integer masks with popcount k
    masks_by_size = [[] for _ in range(n + 1)]
    for k in range(1, n + 1):
        for combo in combinations(points, k):
            mask = 0
            for p in combo:
                mask |= (1 << p)
            masks_by_size[k].append(mask)
            
    def compute_alpha(triple_indices):
        """
        Compute the independence number for a set of triple indices.
        Returns the largest k such that there exists a subset of size k containing no triple.
        """
        if not triple_indices:
            return n
            
        t_masks = [triple_masks[i] for i in triple_indices]
        
        for k in range(n, 0, -1):
            for mask in masks_by_size[k]:
                # Check if mask contains any triple
                is_independent = True
                for tm in t_masks:
                    if (mask & tm) == tm:
                        is_independent = False
                        break
                if is_independent:
                    return k
        return 0

    def is_valid_addition(triple_idx, current_set):
        """Check if triple_idx can be added to current_set (set of indices) without conflict."""
        for sel in current_set:
            if triple_idx in conflicts[sel]:
                return False
        return True

    def get_random_valid_triple(current_set):
        """Pick a random triple that can be added to current_set."""
        # Try a few random candidates
        for _ in range(20):
            idx = random.randint(0, n_triples_total - 1)
            if idx not in current_set and is_valid_addition(idx, current_set):
                return idx
        return None

    def get_random_swap(current_set):
        """
        Pick a random triple to remove and a random valid triple to add.
        Returns (remove_idx, add_idx) or (None, None).
        """
        if not current_set:
            return None, None
            
        remove_idx = random.choice(list(current_set))
        remaining = current_set - {remove_idx}
        
        # Find a valid triple to add to remaining
        for _ in range(20):
            add_idx = random.randint(0, n_triples_total - 1)
            if add_idx == remove_idx:
                continue
            if add_idx in remaining:
                continue
            if is_valid_addition(add_idx, remaining):
                return remove_idx, add_idx
        return None, None

    # Initial random packing
    def generate_initial_packing():
        current_set = set()
        # Add random valid triples
        while True:
            idx = get_random_valid_triple(current_set)
            if idx is None:
                break
            current_set.add(idx)
        return current_set

    best_alpha = n
    best_set = set()
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.995
    steps = 0
    max_steps = 50000 # Safety cap, but time limit is main constraint
    
    # Start with a few restarts or just one long run? 
    # Let's do one long run with restarts if stuck.
    current_set = generate_initial_packing()
    current_alpha = compute_alpha(current_set)
    
    if current_alpha < best_alpha:
        best_alpha = current_alpha
        best_set = current_set.copy()
        
    while time.time() < deadline - 5:
        if steps > max_steps:
            break
        steps += 1
        
        move_type = random.random()
        new_set = current_set.copy()
        
        if move_type < 0.4:
            # Add
            idx = get_random_valid_triple(new_set)
            if idx is not None:
                new_set.add(idx)
            else:
                continue # No valid move, skip
        elif move_type < 0.6:
            # Remove
            if new_set:
                remove_idx = random.choice(list(new_set))
                new_set.remove(remove_idx)
            else:
                continue
        else:
            # Swap
            rem, add = get_random_swap(new_set)
            if rem is not None:
                new_set.remove(rem)
                new_set.add(add)
            else:
                continue
                
        new_alpha = compute_alpha(new_set)
        
        # Acceptance criterion
        delta = new_alpha - current_alpha
        if delta < 0:
            # Improvement
            current_set = new_set
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_set = current_set.copy()
                if best_alpha <= 6:
                    break
        elif delta == 0:
            # Same score, accept with probability 0.5 to explore
            if random.random() < 0.5:
                current_set = new_set
                # current_alpha unchanged
        else:
            # Worse score, accept with probability exp(-delta / T)
            # Since delta is integer >= 1, and T starts at 5, this is reasonable
            prob = 2.71828 ** (-delta / T)
            if random.random() < prob:
                current_set = new_set
                current_alpha = new_alpha
                
        T *= cooling
        
        # Restart if stuck for a long time? 
        # Simple heuristic: if T is very low and no improvement for many steps, restart
        # For now, just let it cool down.

    # Convert best_set (indices) to list of triples
    result = [all_triples[i] for i in best_set]
    
    return result

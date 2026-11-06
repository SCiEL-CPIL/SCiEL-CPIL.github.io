"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n = 16
    # 1. Generate all possible triples (indices 0..559)
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # 2. Generate all K(2,2,2) configurations
    triple_to_ks = [[] for _ in range(num_triples)]
    all_ks_masks = []
    
    # Hardcoded 15 matchings of indices 0-5
    matchings = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)]
    ]
    
    # Precompute bits for each triple index
    triple_bits = [1 << i for i in range(num_triples)]
    
    for six_points in combinations(range(n), 6):
        for matching in matchings:
            p1 = (six_points[matching[0][0]], six_points[matching[0][1]])
            p2 = (six_points[matching[1][0]], six_points[matching[1][1]])
            p3 = (six_points[matching[2][0]], six_points[matching[2][1]])
            
            # Generate 8 triple indices
            t_indices = []
            mask = 0
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        t = tuple(sorted((x1, x2, x3)))
                        idx = triple_to_idx[t]
                        t_indices.append(idx)
                        mask |= triple_bits[idx]
            
            k_idx = len(all_ks_masks)
            all_ks_masks.append(mask)
            
            for ti in t_indices:
                triple_to_ks[ti].append(k_idx)
                
    # Helper to check if a triple can be added
    def can_add(idx, current_mask):
        bit = triple_bits[idx]
        for k_idx in triple_to_ks[idx]:
            k_mask = all_ks_masks[k_idx]
            if (k_mask & current_mask) == (k_mask ^ bit):
                return False
        return True

    # Helper to perform greedy fill
    def greedy_fill(indices, current_mask):
        for idx in indices:
            if can_add(idx, current_mask):
                current_mask |= triple_bits[idx]
        return current_mask

    best_mask = 0
    best_count = 0
    
    # Initial greedy restarts to find a good starting point
    while time.time() < deadline - 5.0:
        current_mask = 0
        indices = list(range(num_triples))
        random.shuffle(indices)
        current_mask = greedy_fill(indices, current_mask)
        
        count = current_mask.bit_count()
        if count > best_count:
            best_count = count
            best_mask = current_mask

    # Local search: Kick phase
    # Remove k random triples and try to re-fill
    while time.time() < deadline:
        if best_count == 0:
            break
            
        current_mask = best_mask
        
        # Choose a random kick size (e.g., 5 to 15)
        kick_size = random.randint(5, 15)
        
        # Get list of triples in current_mask
        selected_indices = [i for i in range(num_triples) if (current_mask & triple_bits[i])]
        
        if len(selected_indices) < kick_size:
            continue
            
        # Randomly pick kick_size triples to remove
        kick_indices = random.sample(selected_indices, kick_size)
        
        # Remove them
        new_mask = current_mask
        for idx in kick_indices:
            new_mask ^= triple_bits[idx]
            
        # Create a list of candidates: the removed ones + all not in new_mask
        # Actually, we just need to try to add back the removed ones and potentially
        # other ones that were previously blocked.
        # A simpler approach: take the new_mask, and run greedy on the set of
        # (removed indices + indices not in new_mask).
        # But "indices not in new_mask" is expensive to generate.
        
        # Better: 
        # 1. Start with new_mask.
        # 2. Create a list of "free" indices: those not in new_mask.
        #    This is O(N). N=560. This is fast.
        # 3. Shuffle the free indices, but prioritize the kick_indices?
        #    Or just shuffle all free indices.
        
        free_indices = [i for i in range(num_triples) if not (new_mask & triple_bits[i])]
        
        # To encourage recovering the kicked triples, we can put them at the front
        # or just shuffle. Shuffling is safer to avoid bias.
        random.shuffle(free_indices)
        
        # Greedy fill on the new state
        new_mask = greedy_fill(free_indices, new_mask)
        
        new_count = new_mask.bit_count()
        if new_count > best_count:
            best_count = new_count
            best_mask = new_mask

    # Convert best_mask back to list of triples
    result = []
    for i in range(num_triples):
        if best_mask & triple_bits[i]:
            result.append(list(all_triples[i]))
            
    return result
# EVOLVE-BLOCK-END

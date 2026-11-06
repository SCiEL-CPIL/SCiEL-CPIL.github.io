"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
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
    
    num_ks = len(all_ks_masks)
    
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

    # Efficient local search using incremental violation tracking
    
    # Initialize with multiple greedy restarts (limited time)
    best_mask = 0
    best_count = 0
    greedy_deadline = start_time + 20.0  # Spend 20s on greedy restarts
    
    while time.time() < greedy_deadline:
        current_mask = 0
        indices = list(range(num_triples))
        random.shuffle(indices)
        current_mask = greedy_fill(indices, current_mask)
        
        count = current_mask.bit_count()
        if count > best_count:
            best_count = count
            best_mask = current_mask
    
    # Now do efficient local search with incremental updates
    # Track: for each clause, how many of its 8 triples are selected
    # clause_count[k] = number of selected triples in clause k
    # A clause is violated if clause_count[k] == 8
    
    # Precompute clause membership efficiently
    # For each triple, we already have triple_to_ks[idx] = list of clause indices
    
    # Start from best_mask
    current_mask = best_mask
    
    # Compute initial clause counts
    clause_count = [0] * num_ks
    for i in range(num_triples):
        if current_mask & triple_bits[i]:
            for k_idx in triple_to_ks[i]:
                clause_count[k_idx] += 1
    
    # Track blocked triples: blocked[i] = number of clauses where adding i would make count == 8
    # i.e., clauses where clause_count[k] == 7
    blocked = [0] * num_triples
    for i in range(num_triples):
        if not (current_mask & triple_bits[i]):
            for k_idx in triple_to_ks[i]:
                if clause_count[k_idx] == 7:
                    blocked[i] += 1
    
    # Local search loop with time budget
    local_search_deadline = deadline - 5.0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < local_search_deadline and no_improve_count < max_no_improve:
        improved = False
        
        # Try to add an unblocked triple
        addable = [i for i in range(num_triples) if not (current_mask & triple_bits[i]) and blocked[i] == 0]
        
        if addable:
            # Pick a random addable triple
            idx = random.choice(addable)
            new_mask = current_mask | triple_bits[idx]
            
            # Update clause counts
            for k_idx in triple_to_ks[idx]:
                clause_count[k_idx] += 1
                # If this makes clause_count == 8, it's a violation - shouldn't happen if blocked[idx] == 0
                # But update blocked counts for other triples in this clause
                if clause_count[k_idx] == 8:
                    # This is a violation, shouldn't happen
                    pass
            
            # Update blocked counts for triples affected by this addition
            # For each clause containing idx, other triples in that clause may now be blocked
            for k_idx in triple_to_ks[idx]:
                # Get all triples in this clause
                k_mask = all_ks_masks[k_idx]
                # For each triple j in this clause that is not selected and j != idx
                # If clause_count[k_idx] == 7, then j is now blocked in this clause
                if clause_count[k_idx] == 7:
                    # Find other unselected triples in this clause
                    for j in range(num_triples):
                        if j != idx and not (current_mask & triple_bits[j]) and (k_mask & triple_bits[j]):
                            blocked[j] += 1
            
            current_mask = new_mask
            improved = True
            no_improve_count = 0
            continue
        
        # No addable triple - try a removal to unblock others
        # Find triples whose removal would unblock the most others
        # For simplicity, try removing a random selected triple
        
        selected_indices = [i for i in range(num_triples) if (current_mask & triple_bits[i])]
        if not selected_indices:
            break
        
        # Try removing a few random triples to create space
        kick_size = random.randint(1, 4)
        if len(selected_indices) <= kick_size:
            no_improve_count += 1
            continue
        
        kick_indices = random.sample(selected_indices, kick_size)
        
        # Remove them
        for idx in kick_indices:
            current_mask ^= triple_bits[idx]
            for k_idx in triple_to_ks[idx]:
                clause_count[k_idx] -= 1
                # Update blocked counts: clauses that were at 7 are now at 6, so unblock
                if clause_count[k_idx] == 6:
                    k_mask = all_ks_masks[k_idx]
                    for j in range(num_triples):
                        if j not in kick_indices and not (current_mask & triple_bits[j]) and (k_mask & triple_bits[j]):
                            if blocked[j] > 0:
                                blocked[j] -= 1
        
        # Re-greedy fill
        free_indices = [i for i in range(num_triples) if not (current_mask & triple_bits[i])]
        random.shuffle(free_indices)
        
        for idx in free_indices:
            if blocked[idx] == 0:
                current_mask |= triple_bits[idx]
                for k_idx in triple_to_ks[idx]:
                    clause_count[k_idx] += 1
                    if clause_count[k_idx] == 7:
                        k_mask = all_ks_masks[k_idx]
                        for j in range(num_triples):
                            if j != idx and not (current_mask & triple_bits[j]) and (k_mask & triple_bits[j]):
                                blocked[j] += 1
        
        new_count = current_mask.bit_count()
        if new_count > best_count:
            best_count = new_count
            best_mask = current_mask
            no_improve_count = 0
            improved = True
        else:
            no_improve_count += 1
    
    # Final polish: try to add any remaining valid triples
    for i in range(num_triples):
        if not (best_mask & triple_bits[i]):
            if can_add(i, best_mask):
                best_mask |= triple_bits[i]
                best_count = best_mask.bit_count()
    
    # Convert best_mask back to list of triples
    result = []
    for i in range(num_triples):
        if best_mask & triple_bits[i]:
            result.append(list(all_triples[i]))
            
    return result
# EVOLVE-BLOCK-END

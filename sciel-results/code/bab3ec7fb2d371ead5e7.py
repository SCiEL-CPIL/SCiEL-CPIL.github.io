import numpy as np
import time
import random
import math

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses Multi-Start Simulated Annealing with Bitset-based state representation.
    """
    m = 205
    
    # 1. Precompute forbidden differences and conflict masks
    forbidden = set(pow(x, 2, m) for x in range(1, m))
    forbidden.discard(0)
    
    # conflict_mask[i] has bit j set if i and j are adjacent (conflict)
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i != j and (i - j) % m in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    # Precompute all vertices list for random selection
    all_vertices = list(range(m))
    full_mask = (1 << m) - 1
    
    best_mask = 0
    best_size = 0
    
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    def get_size(mask):
        return mask.bit_count()
        
    def random_bit(mask):
        """Returns the index of a random set bit in mask, or -1 if mask is 0."""
        if mask == 0:
            return -1
        while True:
            idx = random.randint(0, m - 1)
            if mask & (1 << idx):
                return idx

    def random_unset_bit(mask):
        """Returns the index of a random unset bit in mask, or -1 if mask is full."""
        unset_mask = full_mask ^ mask
        if unset_mask == 0:
            return -1
        while True:
            idx = random.randint(0, m - 1)
            if unset_mask & (1 << idx):
                return idx

    def greedy_fill():
        """
        Creates a random initial independent set using randomized greedy.
        """
        curr_mask = 0
        order = all_vertices[:]
        random.shuffle(order)
        for v in order:
            if (curr_mask & conflict_mask[v]) == 0:
                curr_mask |= (1 << v)
        return curr_mask

    def polish(mask):
        """
        Greedy polish: try to add any vertex that doesn't conflict.
        Returns the polished mask.
        """
        improved = True
        while improved:
            improved = False
            # Iterate through vertices in random order to avoid bias
            order = all_vertices[:]
            random.shuffle(order)
            for v in order:
                if not (mask & (1 << v)) and (mask & conflict_mask[v]) == 0:
                    mask |= (1 << v)
                    improved = True
        return mask

    def simulate_annealing(initial_mask, t_start, t_end, cooling_rate, max_iter):
        """
        Runs SA on the given initial mask.
        Returns the best mask found during the run.
        """
        curr_mask = initial_mask
        curr_size = get_size(curr_mask)
        best_local_mask = curr_mask
        best_local_size = curr_size
        
        T = t_start
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Generate a neighbor
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a vertex
                v = random_unset_bit(curr_mask)
                if v == -1:
                    continue
                if (curr_mask & conflict_mask[v]) == 0:
                    new_mask = curr_mask | (1 << v)
                    new_size = curr_size + 1
                    curr_mask = new_mask
                    curr_size = new_size
                    if curr_size > best_local_size:
                        best_local_size = curr_size
                        best_local_mask = curr_mask
                else:
                    # If can't add, maybe swap?
                    u = random_bit(curr_mask)
                    if u == -1:
                        continue
                    temp_mask = curr_mask & ~(1 << u)
                    if (temp_mask & conflict_mask[v]) == 0:
                        new_mask = temp_mask | (1 << v)
                        curr_mask = new_mask
                        # Size unchanged

            elif move_type < 0.7:
                # Try 1-swap (remove u, add v)
                u = random_bit(curr_mask)
                if u == -1:
                    continue
                v = random_unset_bit(curr_mask)
                if v == -1:
                    continue
                temp_mask = curr_mask & ~(1 << u)
                if (temp_mask & conflict_mask[v]) == 0:
                    new_mask = temp_mask | (1 << v)
                    curr_mask = new_mask

            elif move_type < 0.9:
                # Try 2-swap (remove u1, u2, add v1, v2)
                # Pick u1, u2
                u1 = random_bit(curr_mask)
                if u1 == -1:
                    continue
                u2 = random_bit(curr_mask & ~(1 << u1))
                if u2 == -1:
                    continue
                
                temp_mask = curr_mask & ~(1 << u1) & ~(1 << u2)
                
                # Find valid v1, v2
                # v1, v2 must be in V \ temp_mask
                # And not conflict with temp_mask
                # And not conflict with each other
                
                # Compute conflict mask of temp_mask
                # This is expensive if done by summing. 
                # But we can check candidates directly.
                
                # Get list of unset bits in temp_mask
                unset_mask = full_mask ^ temp_mask
                if unset_mask == 0:
                    continue
                
                # Pick v1
                v1 = -1
                # We need v1 such that (temp_mask & conflict_mask[v1]) == 0
                # We can just pick random unset bits and check
                found_v1 = False
                while not found_v1:
                    # Pick random from unset_mask
                    # To avoid infinite loop if no valid v1, we limit attempts or check if any exist
                    # For now, simple rejection
                    v1 = random_unset_bit(temp_mask)
                    if v1 == -1:
                        break
                    if (temp_mask & conflict_mask[v1]) == 0:
                        found_v1 = True
                    else:
                        # If we keep failing, maybe no valid v1? 
                        # We can break after a few tries to save time
                        # But for m=205, density is high, so likely to find one quickly.
                        pass

                if v1 == -1:
                    continue
                
                # Pick v2
                # v2 must be in unset_mask, v2 != v1
                # v2 must not conflict with temp_mask
                # v2 must not conflict with v1
                valid_v2_mask = unset_mask & ~(1 << v1)
                # Exclude v2 that conflict with temp_mask
                # We can't easily compute this mask without iterating.
                # So we pick random v2 from valid_v2_mask and check.
                found_v2 = False
                v2 = -1
                # Limit attempts to prevent stalling
                for _ in range(10):
                    v2 = random_unset_bit(temp_mask) # This picks from full_mask ^ temp_mask
                    if v2 == -1 or v2 == v1:
                        continue
                    if (temp_mask & conflict_mask[v2]) == 0 and (conflict_mask[v2] & (1 << v1)) == 0:
                        found_v2 = True
                        break

                if found_v2:
                    new_mask = temp_mask | (1 << v1) | (1 << v2)
                    curr_mask = new_mask
                    # Size unchanged

            else:
                # Remove a vertex (worsening move, for escape)
                u = random_bit(curr_mask)
                if u == -1:
                    continue
                new_mask = curr_mask & ~(1 << u)
                new_size = curr_size - 1
                if random.random() < math.exp(-1.0 / T):
                    curr_mask = new_mask
                    curr_size = new_size
                    
            # Cool down
            T *= cooling_rate
            
        # Final polish
        best_local_mask = polish(best_local_mask)
        best_local_size = get_size(best_local_mask)
        
        return best_local_mask, best_local_size

    # Main Loop: Multi-Start SA
    while time.time() < deadline:
        init_mask = greedy_fill()
        
        t_start = 1.0
        t_end = 0.01
        cooling_rate = 0.995
        max_iter = 5000
        
        best_local_mask, best_local_size = simulate_annealing(init_mask, t_start, t_end, cooling_rate, max_iter)
        
        if best_local_size > best_size:
            best_size = best_local_size
            best_mask = best_local_mask
            
    result = []
    for i in range(m):
        if best_mask & (1 << i):
            result.append(i)
            
    return np.array(result, dtype=np.int64)

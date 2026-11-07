import random

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Precompute adjacency bitmasks for fast degree calculation
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    def get_degrees(mask):
        """Calculate degrees for all vertices in the subset represented by mask."""
        degrees = {}
        for v in range(n):
            if (mask >> v) & 1:
                # Count bits in (adj_mask[v] & mask)
                # v is in mask, but adj_mask[v] does not contain v (no self-loops), so this is correct
                degrees[v] = bin(adj_mask[v] & mask).count('1')
        return degrees

    def score_mask(mask):
        """Return the number of distinct degrees in the subset."""
        if mask == 0:
            return 0
        degrees = get_degrees(mask)
        return len(set(degrees.values()))

    best_score = -1
    best_mask = 0
    
    # Simulated Annealing
    # We want to maximize the number of distinct degrees.
    # Max possible distinct degrees is n (if all degrees are unique), but in a graph of n vertices,
    # degrees range from 0 to n-1. In an induced subgraph of size k, degrees range from 0 to k-1.
    # So max distinct degrees is k.
    
    # Strategy: Iterate over many random starting points and anneal.
    # Since n=17 is small, we can afford to do this many times.
    
    import math
    
    for restart in range(100):
        # Initialize with a random subset of size between min_vertices and n
        # Bias towards larger subsets as they allow for more distinct degrees?
        # Actually, smaller subsets might have easier to achieve distinct degrees if the graph is sparse?
        # Let's just sample uniformly.
        size = random.randint(min_vertices, n)
        current_vertices = random.sample(range(n), size)
        current_mask = 0
        for v in current_vertices:
            current_mask |= (1 << v)
            
        current_score = score_mask(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        max_iters = 50000
        stuck_count = 0
        max_stuck = 1000
        
        for i in range(max_iters):
            if T < min_T:
                break
                
            # Generate a neighbor
            # Moves: Add, Remove, Swap
            # We must maintain size >= min_vertices
            in_list = [v for v in range(n) if (current_mask >> v) & 1]
            out_list = [v for v in range(n) if not ((current_mask >> v) & 1)]
            
            move_type = random.random()
            
            if move_type < 0.3 and out_list:
                # Add
                v_add = random.choice(out_list)
                new_mask = current_mask | (1 << v_add)
            elif move_type < 0.6 and len(in_list) > min_vertices:
                # Remove
                v_rem = random.choice(in_list)
                new_mask = current_mask & ~(1 << v_rem)
            elif out_list and in_list:
                # Swap
                v_add = random.choice(out_list)
                v_rem = random.choice(in_list)
                new_mask = (current_mask & ~(1 << v_rem)) | (1 << v_add)
            else:
                # No valid move or forced to break
                if not out_list and len(in_list) <= min_vertices:
                    break
                # If we are here, we might have no out_list but can remove?
                # Or no in_list?
                # If out_list is empty, we can only remove.
                if not out_list:
                    if len(in_list) > min_vertices:
                        v_rem = random.choice(in_list)
                        new_mask = current_mask & ~(1 << v_rem)
                    else:
                        break
                else:
                    # in_list is empty? Impossible since we start with >= min_vertices
                    # But if min_vertices is 0? No, it's 8.
                    # So in_list is never empty if we maintain constraint.
                    # If we somehow end up with in_list empty, we must add.
                    v_add = random.choice(out_list)
                    new_mask = current_mask | (1 << v_add)
            
            new_score = score_mask(new_mask)
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_mask = new_mask
                current_score = new_score
                stuck_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
            else:
                stuck_count += 1
                if stuck_count > max_stuck:
                    break
            
            T *= cooling

    # Convert best_mask to list of ints
    result = [v for v in range(n) if (best_mask >> v) & 1]
    
    # Ensure type is list of ints
    return [int(v) for v in result]

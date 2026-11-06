import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 256
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # 1. Filter vertices that violate diagonal constraint (2*a is square)
    valid_vertices = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            valid_vertices.append(a)
            
    num_valid = len(valid_vertices)
    # Map vertex value to index for bit operations
    val_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 2. Build Conflict Bitmasks
    # conflict[i] is a bitmask where bit j is 1 if valid_vertices[i] + valid_vertices[j] is a square
    conflict = [0] * num_valid
    for i in range(num_valid):
        vi = valid_vertices[i]
        mask = 0
        for j in range(num_valid):
            if i == j:
                continue
            vj = valid_vertices[j]
            if (vi + vj) in squares:
                mask |= (1 << j)
        conflict[i] = mask
        
    # Helper to check if a vertex can be added to a set represented by a bitmask
    def can_add(vertex_idx, set_mask):
        # Check if vertex_idx is already in set
        if (set_mask >> vertex_idx) & 1:
            return False
        # Check if any neighbor of vertex_idx is in set
        if conflict[vertex_idx] & set_mask:
            return False
        return True

    # Helper to get a greedy initial solution
    def greedy_initial():
        # Order by degree (fewest conflicts first is often good for MIS? Or most?)
        # Usually, picking vertices with fewest conflicts first allows more vertices.
        # But for MIS, we want to pick vertices that don't conflict with each other.
        # A common heuristic is "Greedy by degree": pick vertex with min degree, remove it and its neighbors.
        # Here we just want to build a set.
        # Let's try multiple random orderings and pick the best.
        
        best_mask = 0
        best_size = 0
        
        # Try a few deterministic and random orderings
        indices = list(range(num_valid))
        
        # Strategy 1: Natural order
        # Strategy 2: Reverse
        # Strategy 3: Random shuffles
        
        orders = [
            indices,
            list(reversed(indices)),
        ]
        
        for _ in range(50):
            shuffled = indices[:]
            random.shuffle(shuffled)
            orders.append(shuffled)
            
        for order in orders:
            curr_mask = 0
            for idx in order:
                if can_add(idx, curr_mask):
                    curr_mask |= (1 << idx)
            
            size = bin(curr_mask).count('1')
            if size > best_size:
                best_size = size
                best_mask = curr_mask
                
        return best_mask, best_size

    # Local Search with Simulated Annealing
    def local_search(initial_mask):
        current_mask = initial_mask
        current_size = bin(current_mask).count('1')
        best_mask = current_mask
        best_size = current_size
        
        # Temperature parameters
        T = 10.0
        cooling_rate = 0.995
        min_T = 0.1
        
        # Precompute list of all valid indices for random selection
        all_indices = list(range(num_valid))
        
        while T > min_T:
            # Try a random move: 
            # 1. Add a random vertex if possible
            # 2. Remove a random vertex from set, then try to add as many as possible (or specific ones)
            
            move_type = random.choice([0, 1, 2])
            
            if move_type == 0:
                # Try adding a random vertex not in set
                # Find indices not in current_mask
                # Instead of generating list, pick random index and check
                new_mask = current_mask
                # Pick a few random candidates
                for _ in range(10):
                    idx = random.choice(all_indices)
                    if not (current_mask >> idx) & 1:
                        if can_add(idx, current_mask):
                            new_mask = current_mask | (1 << idx)
                            break
                if new_mask == current_mask:
                    continue
            elif move_type == 1:
                # Try removing a random vertex from set and see if we can add more
                # Pick a random vertex in set
                set_indices = [i for i in all_indices if (current_mask >> i) & 1]
                if not set_indices:
                    continue
                remove_idx = random.choice(set_indices)
                temp_mask = current_mask & ~(1 << remove_idx)
                
                # Try to add any vertex that was previously blocked by remove_idx
                # Or just try to add any vertex that fits in temp_mask
                # Let's try to add vertices that conflict with remove_idx, as they are now free
                # Neighbors of remove_idx:
                neighbors = conflict[remove_idx]
                # Also try some random vertices
                candidates = []
                # Add neighbors
                for i in all_indices:
                    if (neighbors >> i) & 1:
                        candidates.append(i)
                # Add some randoms
                for _ in range(5):
                    candidates.append(random.choice(all_indices))
                
                new_mask = temp_mask
                for idx in candidates:
                    if not (temp_mask >> idx) & 1:
                        if can_add(idx, temp_mask):
                            new_mask |= (1 << idx)
            else:
                # 2-flip: Remove 2, Add 2? Too complex. Let's stick to 1-flip for now but with better acceptance.
                # Or try "swap": Remove one, Add one.
                set_indices = [i for i in all_indices if (current_mask >> i) & 1]
                if not set_indices:
                    continue
                remove_idx = random.choice(set_indices)
                temp_mask = current_mask & ~(1 << remove_idx)
                
                # Find a vertex to add
                added = False
                for _ in range(10):
                    idx = random.choice(all_indices)
                    if not (temp_mask >> idx) & 1:
                        if can_add(idx, temp_mask):
                            new_mask = temp_mask | (1 << idx)
                            added = True
                            break
                if not added:
                    continue

            new_size = bin(new_mask).count('1')
            
            # Acceptance Criterion
            if new_size > current_size:
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            else:
                # Simulated Annealing acceptance
                delta = current_size - new_size # positive if worse
                # Probability of accepting worse solution
                # We want to maximize size, so delta is "energy increase"
                # If new_size < current_size, delta > 0
                if delta == 0:
                    accept_prob = 0.5
                else:
                    accept_prob = math.exp(-delta / T)
                
                if random.random() < accept_prob:
                    current_mask = new_mask
                    current_size = new_size
            
            T *= cooling_rate
            
            # Time check
            if time.time() - start_time > deadline - 5:
                break
        
        return best_mask, best_size

    # 3. Execution
    # Get initial solution
    init_mask, init_size = greedy_initial()
    
    # Run local search with a few restarts
    best_mask, best_size = local_search(init_mask)
    
    # If we have time, do a few more restarts with different initializations
    if time.time() - start_time < deadline - 10:
        for _ in range(5):
            if time.time() - start_time > deadline - 5:
                break
            init_mask, _ = greedy_initial()
            m, s = local_search(init_mask)
            if s > best_size:
                best_mask = m
                best_size = s

    # Convert bitmask back to list of values
    result = []
    for i in range(num_valid):
        if (best_mask >> i) & 1:
            result.append(valid_vertices[i])
            
    return result


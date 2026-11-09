import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build adjacency list
    # neighbors[i] is a list of j such that (i-j) % m is in forbidden
    neighbors = [[] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j: continue
            diff = (i - j) % m
            if diff in forbidden:
                neighbors[i].append(j)
                
    # Precompute degrees
    degrees = [len(neighbors[i]) for i in range(m)]
    
    def get_size(in_set):
        return np.sum(in_set)
        
    def is_valid_addition(u, in_set, conflict_count):
        # u can be added if it's not in set and has no conflicts with current set
        # conflict_count[u] tracks number of neighbors in set
        return not in_set[u] and conflict_count[u] == 0

    def add_vertex(u, in_set, conflict_count):
        in_set[u] = True
        for v in neighbors[u]:
            conflict_count[v] += 1
        # Note: u itself is not a neighbor of u in this graph construction (no self loops)
        # But u is now in set, so it blocks itself? 
        # The condition for adding v is conflict_count[v] == 0.
        # If v is in set, it's not a candidate.
        # If v is not in set, conflict_count[v] counts neighbors in set.
        # So we don't need to mark u as blocking u, just its neighbors.
        
    def remove_vertex(u, in_set, conflict_count):
        in_set[u] = False
        for v in neighbors[u]:
            conflict_count[v] -= 1

    def greedy_construct(order):
        in_set = np.zeros(m, dtype=bool)
        conflict_count = np.zeros(m, dtype=int)
        for u in order:
            if is_valid_addition(u, in_set, conflict_count):
                add_vertex(u, in_set, conflict_count)
        return in_set

    # Initial best solution
    best_in_set = None
    best_size = 0
    
    # Generate a few greedy starts
    start_times = []
    # Start with natural order
    best_in_set = greedy_construct(range(m))
    best_size = np.sum(best_in_set)
    
    # Randomized greedy starts
    for _ in range(50):
        order = list(range(m))
        random.shuffle(order)
        sol = greedy_construct(order)
        if np.sum(sol) > best_size:
            best_size = np.sum(sol)
            best_in_set = sol.copy()
            
    # Simulated Annealing
    # We will run SA from the best greedy solution found
    current_in_set = best_in_set.copy()
    current_conflict_count = np.zeros(m, dtype=int)
    for u in range(m):
        if current_in_set[u]:
            for v in neighbors[u]:
                current_conflict_count[v] += 1
                
    current_size = best_size
    T = 1.0
    cooling = 0.999
    min_T = 1e-5
    
    start_time = time.time()
    deadline = start_time + 590.0 # 10s safety margin
    
    # We can also do restarts if time permits
    # But SA from a good start is usually effective.
    # Let's run one long SA chain, and if time is left, try random restarts.
    
    while time.time() < deadline:
        # Pick a random move
        # Types: Add, Remove, Swap
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Find a candidate to add
            # Candidates are vertices not in set with conflict_count 0
            # We can just pick a random vertex and check
            u = random.randint(0, m - 1)
            if is_valid_addition(u, current_in_set, current_conflict_count):
                # Accept if it improves size (it always does by 1)
                add_vertex(u, current_in_set, current_conflict_count)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = current_in_set.copy()
            else:
                # Rejection: no change, but maybe cool down? 
                # Standard SA: if move is not possible, it's effectively a "null" move 
                # or we can treat it as a failed attempt. 
                # To keep temperature meaningful, we should only cool on actual state changes or valid attempts.
                # Let's just not cool on invalid moves to avoid getting stuck in high T.
                pass
                
        elif move_type == 'remove':
            # Pick a random vertex in the set to remove
            if current_size == 0: continue
            in_set_indices = np.where(current_in_set)[0]
            if len(in_set_indices) == 0: continue
            u = in_set_indices[random.randint(0, len(in_set_indices) - 1)]
            
            # Delta size is -1
            delta_size = -1
            # Accept with probability exp(delta_size / T)
            if random.random() < np.exp(delta_size / T):
                remove_vertex(u, current_in_set, current_conflict_count)
                current_size -= 1
                # Note: this makes solution worse, but might allow better later
                # We don't update best_size here
            else:
                pass
            T *= cooling
            if T < min_T:
                T = min_T
                
        elif move_type == 'swap':
            # Pick u to remove (from set) and v to add (not in set)
            if current_size == 0: continue
            in_set_indices = np.where(current_in_set)[0]
            if len(in_set_indices) == 0: continue
            u = in_set_indices[random.randint(0, len(in_set_indices) - 1)]
            
            # Try to find a valid v to add if u is removed
            # Temporarily remove u
            remove_vertex(u, current_in_set, current_conflict_count)
            
            # Now try to pick a random v not in set
            v = random.randint(0, m - 1)
            if is_valid_addition(v, current_in_set, current_conflict_count):
                # Swap is valid (size remains same)
                # Actually, we want to increase size. 
                # If we just swap, size is constant.
                # But it might lead to a state where we can add more later.
                # However, pure swaps don't increase size.
                # It's better to do "Remove then Add" as separate steps or 
                # "Add then Remove" which changes size by +1 or -1.
                # Let's stick to Add/Remove for size optimization.
                # Swap is useful for escaping local minima in constant-size landscape, 
                # but since we want max size, Add/Remove is primary.
                # If we swap, we accept it always? Or with prob 1?
                # Let's accept swap if it doesn't decrease size (it doesn't).
                # But does it help? Maybe.
                add_vertex(v, current_in_set, current_conflict_count)
                # Size unchanged
            else:
                # Swap failed, put u back
                add_vertex(u, current_in_set, current_conflict_count)
            T *= cooling
            if T < min_T:
                T = min_T
                
        # Check time more frequently
        if time.time() > deadline:
            break

    # Convert best_in_set to array of indices
    result = np.where(best_in_set)[0]
    return result.astype(np.int64)

# EVOLVE-BLOCK-END

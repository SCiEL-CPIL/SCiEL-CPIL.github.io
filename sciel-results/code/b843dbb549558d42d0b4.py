import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    # Compute dot products
    dots = vectors_v4 @ vectors_v4.T
    
    # Conflict if dot > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to list of neighbors for faster iteration
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    
    # State variables
    selected = np.zeros(N4, dtype=bool)
    block_count = np.zeros(N4, dtype=np.int32)
    blocked = np.zeros(N4, dtype=bool)
    
    def add_node(u):
        selected[u] = True
        blocked[u] = True # Mark as blocked to prevent re-adding
        for v in neighbors[u]:
            block_count[v] += 1
            if block_count[v] == 1:
                blocked[v] = True

    def remove_node(u):
        selected[u] = False
        blocked[u] = False # Free it
        for v in neighbors[u]:
            block_count[v] -= 1
            if block_count[v] == 0:
                blocked[v] = False

    def get_available():
        # Nodes that are not selected and not blocked
        return np.where(~selected & ~blocked)[0]

    def try_add_one():
        available = get_available()
        if len(available) > 0:
            # Pick a random one or the first one
            # Random is better for diversity
            u = available[np.random.randint(len(available))]
            add_node(u)
            return True
        return False

    def try_swap_one_for_two():
        # Pick a random selected node to remove
        sel_indices = np.where(selected)[0]
        if len(sel_indices) == 0:
            return False
        
        u = sel_indices[np.random.randint(len(sel_indices))]
        
        # Temporarily remove u
        remove_node(u)
        
        # Try to find 2 non-conflicting nodes from available
        available = get_available()
        if len(available) < 2:
            # Restore u
            add_node(u)
            return False
            
        # Shuffle available to try different pairs
        np.random.shuffle(available)
        
        found = False
        # Try to pick 2
        # Simple heuristic: pick first available, then find second that doesn't conflict
        v1 = available[0]
        # Find v2 in available[1:] that doesn't conflict with v1
        # v2 must not be a neighbor of v1
        v1_neighbors = set(neighbors[v1])
        
        for v2 in available[1:]:
            if v2 not in v1_neighbors:
                # Found v1, v2
                add_node(v1)
                add_node(v2)
                found = True
                break
                
        if not found:
            # Restore u
            add_node(u)
            
        return found

    def try_swap_one_for_three():
        sel_indices = np.where(selected)[0]
        if len(sel_indices) == 0:
            return False
        
        u = sel_indices[np.random.randint(len(sel_indices))]
        remove_node(u)
        
        available = get_available()
        if len(available) < 3:
            add_node(u)
            return False
            
        np.random.shuffle(available)
        
        found = False
        v1 = available[0]
        v1_neighbors = set(neighbors[v1])
        
        # Find v2
        v2_found = None
        for v2 in available[1:]:
            if v2 not in v1_neighbors:
                v2_found = v2
                break
                
        if v2_found is None:
            add_node(u)
            return False
            
        v2_neighbors = set(neighbors[v2_found])
        
        # Find v3
        v3_found = None
        for v3 in available[1:]:
            if v3 == v2_found: continue
            if v3 not in v1_neighbors and v3 not in v2_neighbors:
                v3_found = v3
                break
                
        if v3_found is None:
            add_node(u)
            return False
            
        add_node(v1)
        add_node(v2_found)
        add_node(v3_found)
        return True

    # Initialize with Greedy Min-Degree
    degrees = np.array([len(n) for n in neighbors])
    alive = np.ones(N4, dtype=bool)
    
    while np.any(alive):
        if time.time() > deadline - 10:
            break
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[np.random.randint(len(cands))]
        
        add_node(chosen)
        
        to_remove = np.unique(np.concatenate(([chosen], neighbors[chosen])))
        alive[to_remove] = False
        rem = np.where(alive)[0]
        if len(rem) > 0:
            # Update degrees for alive nodes
            # This is slow. Just recompute or ignore for greedy init?
            # For greedy init, we can just skip degree update or do it simply.
            pass

    best_size = np.sum(selected)
    best_selected = selected.copy()
    
    # Local Search
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5:
        improved = False
        
        # Try to add one
        if try_add_one():
            improved = True
        
        # Try to swap 1 for 2
        if not improved and try_swap_one_for_two():
            improved = True
            
        # Try to swap 1 for 3 (less frequent)
        if not improved and rng.random() < 0.1:
            if try_swap_one_for_three():
                improved = True
                
        if improved:
            current_size = np.sum(selected)
            if current_size > best_size:
                best_size = current_size
                best_selected = selected.copy()
        else:
            # Perturb
            # Remove a random selected node
            sel_indices = np.where(selected)[0]
            if len(sel_indices) > 0:
                u = sel_indices[rng.integers(len(sel_indices))]
                remove_node(u)
            # Or swap a random node
            if rng.random() < 0.5 and len(sel_indices) > 0:
                 # We already removed one, let's add a random available one if possible
                 available = get_available()
                 if len(available) > 0:
                     v = available[rng.integers(len(available))]
                     add_node(v)

    result_v4 = vectors_v4[best_selected]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

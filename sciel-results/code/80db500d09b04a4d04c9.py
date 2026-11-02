import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Generate all integer vectors with norm squared = 4
    # Type 1: +/- 2 e_i (11 * 2 = 22 vectors)
    # Type 2: Four +/- 1s (C(11,4) * 2^4 = 330 * 16 = 5280 vectors)
    
    vectors = []
    
    # Type 1
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    # Type 2
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            vectors.append(v)
            
    vectors = np.array(vectors, dtype=np.int64)
    N = len(vectors)
    
    # Compute dot products
    dots = vectors @ vectors.T
    
    # For norm-squared = 4 vectors, dist_sq[i,j] = 4 + 4 - 2*dots[i,j] = 8 - 2*dots[i,j]
    # We need dist_sq >= 4 (max norm is 2, so squared max norm is 4)
    # 8 - 2*dots[i,j] >= 4  =>  4 >= 2*dots[i,j]  =>  dots[i,j] <= 2
    # Conflict if dots[i,j] > 2
    
    conflicts_mask = dots > 2
    np.fill_diagonal(conflicts_mask, False)
    
    # Precompute neighbor lists for fast conflict checking
    neighbors = [set() for _ in range(N)]
    for i in range(N):
        neigh = np.where(conflicts_mask[i])[0]
        neighbors[i] = set(neigh.tolist())
    
    # --- Strategy: Simulated Annealing on the conflict graph ---
    # Objective: Maximize number of selected vertices (independent set size)
    # Penalty: For each conflict (edge) where both endpoints are selected, subtract a large penalty
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    best_size = 0
    best_indices = []
    
    # Initial solution: random subset or greedy
    def get_greedy_init():
        available = set(range(N))
        selected = []
        while available:
            if time.time() > deadline:
                break
            # Pick random from available
            v = rng.choice(list(available))
            selected.append(v)
            available.discard(v)
            for n in neighbors[v]:
                if n in available:
                    available.discard(n)
        return selected
    
    # Start with a greedy solution
    current_selected = get_greedy_init()
    current_set = set(current_selected)
    
    # Compute initial penalty
    def compute_penalty(selected_set):
        pen = 0
        for v in selected_set:
            for n in neighbors[v]:
                if n in selected_set and v < n:  # Count each edge once
                    pen += 1
        return pen
    
    current_penalty = compute_penalty(current_set)
    current_score = len(current_selected) - 100 * current_penalty
    
    if len(current_selected) > best_size:
        best_size = len(current_selected)
        best_indices = list(current_selected)
    
    # Simulated Annealing
    T = 100.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 0.1
    max_iter = 50000
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        if T < min_T:
            break
            
        # Random move: add a random vertex, remove a random vertex, or swap
        move_type = rng.integers(0, 3)
        
        if move_type == 0:
            # Add a random vertex
            candidates_to_add = [i for i in range(N) if i not in current_set]
            if not candidates_to_add:
                continue
            v = rng.choice(candidates_to_add)
            # Calculate delta penalty
            new_conflicts = len(neighbors[v] & current_set)
            delta_pen = new_conflicts  # Increase in penalty edges
            
            delta_score = 1 - 100 * delta_pen
            
            if delta_score > 0 or rng.random() < np.exp(delta_score / T):
                current_set.add(v)
                current_penalty += delta_pen
                current_score += delta_score
                
        elif move_type == 1:
            # Remove a random vertex
            if not current_set:
                continue
            v = rng.choice(list(current_set))
            # Calculate delta penalty
            conflicts_removed = len(neighbors[v] & current_set)
            delta_pen = -conflicts_removed  # Decrease in penalty edges
            
            delta_score = -1 - 100 * delta_pen
            
            if delta_score > 0 or rng.random() < np.exp(delta_score / T):
                current_set.discard(v)
                current_penalty += delta_pen
                current_score += delta_score
                
        else:
            # Swap: remove one, add one
            if not current_set:
                continue
            v_remove = rng.choice(list(current_set))
            candidates_to_add = [i for i in range(N) if i not in current_set]
            if not candidates_to_add:
                continue
            v_add = rng.choice(candidates_to_add)
            
            # Remove v_remove
            conf_removed = len(neighbors[v_remove] & (current_set - {v_remove}))
            # Add v_add
            conf_added = len(neighbors[v_add] & (current_set - {v_remove}))
            
            delta_pen = conf_added - conf_removed
            delta_score = 0 - 100 * delta_pen  # Size stays same
            
            if delta_score > 0 or rng.random() < np.exp(delta_score / T):
                current_set.discard(v_remove)
                current_set.add(v_add)
                current_penalty += delta_pen
                current_score += delta_score
                
        # Track best
        if current_penalty == 0 and len(current_set) > best_size:
            best_size = len(current_set)
            best_indices = list(current_set)
            
        T *= cooling_rate
        
        # Occasionally restart if stuck
        if it % 5000 == 0 and it > 0:
            # Check if we've made progress recently
            if len(current_set) <= best_size:
                # Restart with a new greedy
                current_selected = get_greedy_init()
                current_set = set(current_selected)
                current_penalty = compute_penalty(current_set)
                current_score = len(current_selected) - 100 * current_penalty
                T = 100.0 * (0.5 + 0.5 * rng.random())
                
    # Final cleanup: remove any remaining conflicts from best_indices
    if best_indices:
        # Verify and clean
        final_set = set(best_indices)
        # Remove conflicts greedily
        cleaned = []
        for v in best_indices:
            if all(n not in set(cleaned) for n in neighbors[v]):
                cleaned.append(v)
        if len(cleaned) > 0:
            best_indices = cleaned
            best_size = len(cleaned)
            
    if best_size == 0:
        # Fallback
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
        
    return vectors[best_indices]

# EVOLVE-BLOCK-END

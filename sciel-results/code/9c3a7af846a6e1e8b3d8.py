import time
import random
import itertools
import numpy as np

V = 16
N_SUBSETS = 1 << V

# Precompute all triples and their masks
ALL_TRIPLES = []
TRIPLE_MASKS = []
PAIRS_IN_TRIPLE = {}
for t in itertools.combinations(range(V), 3):
    mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
    ALL_TRIPLES.append(t)
    TRIPLE_MASKS.append(mask)
    pairs = set()
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = t[i], t[j]
            if a > b:
                a, b = b, a
            pairs.add((a, b))
    PAIRS_IN_TRIPLE[t] = pairs

# Precompute subset sizes for numpy
SUBSET_MASKS = np.arange(N_SUBSETS, dtype=np.int32)
SUBSET_SIZES = np.array([bin(i).count('1') for i in range(N_SUBSETS)], dtype=np.int8)

# Precompute for each triple index, the mask
TRIPLE_MASKS_ARR = np.array(TRIPLE_MASKS, dtype=np.int32)

# Precompute supersets for each triple
# triple_supersets[t_idx] is a boolean array of size 2^16
# triple_supersets[t_idx][s] is True if subset s contains triple t_idx
TRIPLE_SUPERSETS = []
for mask in TRIPLE_MASKS_ARR:
    # Superset condition: (S & mask) == mask
    # We can vectorize this for all subsets at once
    supersets = (SUBSET_MASKS & mask) == mask
    TRIPLE_SUPERSETS.append(supersets)

# Convert to a 2D array for fast slicing if needed, or keep as list
# List of arrays is fine, we can do np.all(np.stack(...), axis=0) or similar
# But simpler: just loop and AND. With 40 triples, 40 ANDs on 65k booleans is ~fast.
# Even better: precompute nothing more, just use the list.

def compute_alpha(edge_indices):
    """
    Compute independence number given a list of triple indices.
    """
    if not edge_indices:
        return V
    
    # Start with all subsets as independent
    independent = np.ones(N_SUBSETS, dtype=bool)
    
    # For each edge, AND with the precomputed supersets (inverted)
    # We want subsets that do NOT contain the triple.
    # So independent &= ~TRIPLE_SUPERSETS[idx]
    
    for idx in edge_indices:
        independent &= ~TRIPLE_SUPERSETS[idx]
        
    # Find max size among independent sets
    sizes = SUBSET_SIZES[independent]
    return int(sizes.max())

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    # Phase 1: Greedy construction to get a good starting point
    # We want a large packing (max number of triples) to minimize alpha.
    # Greedy: randomly shuffle triples, add if valid.
    
    # We will run multiple restarts to find a good starting configuration.
    best_alpha = V
    best_edges = []
    triple_to_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}
    
    # Try a few greedy starts
    for _ in range(10):
        edges = []
        used_pairs = set()
        shuffled = ALL_TRIPLES[:]
        random.shuffle(shuffled)
        
        for t in shuffled:
            valid = True
            for p in PAIRS_IN_TRIPLE[t]:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                edges.append(t)
                for p in PAIRS_IN_TRIPLE[t]:
                    used_pairs.add(p)
        
        # Convert to indices for compute_alpha
        edge_indices = [triple_to_idx[t] for t in edges]
        alpha = compute_alpha(edge_indices)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_edges = edges[:]
            
        if time.time() > deadline - 100:
            break

    # Phase 2: Local Search / Simulated Annealing
    # State: list of triple indices
    # Moves: Add, Remove, Swap
    
    # Initialize current state from best greedy
    current_edges = best_edges[:]
    current_indices = [triple_to_idx[t] for t in current_edges]
    current_alpha = best_alpha
    
    # Track best
    best_alpha = current_alpha
    best_indices = current_indices[:]
    
    # Temperature
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    
    # Precompute list of all triple indices for random selection
    all_indices = list(range(len(ALL_TRIPLES)))
    
    # For fast validity checking, maintain a set of used pairs
    used_pairs = set()
    for t in current_edges:
        for p in PAIRS_IN_TRIPLE[t]:
            used_pairs.add(p)
    
    # Main loop
    # We use a "try/undo" strategy to avoid copying large structures
    while time.time() < deadline:
        move_type = random.random()
        move_applied = False
        
        if move_type < 0.5:
            # Add a triple
            idx = random.choice(all_indices)
            t = ALL_TRIPLES[idx]
            valid = True
            for p in PAIRS_IN_TRIPLE[t]:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                # Apply move
                current_indices.append(idx)
                for p in PAIRS_IN_TRIPLE[t]:
                    used_pairs.add(p)
                move_applied = True
                # Store info to undo if needed
                undo_action = ('add', idx, t)
                
        elif move_type < 0.8:
            # Remove a triple
            if not current_indices:
                continue
            pos = random.randint(0, len(current_indices) - 1)
            idx = current_indices[pos]
            t = ALL_TRIPLES[idx]
            # Apply move
            current_indices.pop(pos)
            for p in PAIRS_IN_TRIPLE[t]:
                used_pairs.remove(p)
            move_applied = True
            undo_action = ('remove', idx, t, pos)
            
        else:
            # Swap: remove one, add another
            if not current_indices:
                continue
            pos = random.randint(0, len(current_indices) - 1)
            idx_old = current_indices[pos]
            t_old = ALL_TRIPLES[idx_old]
            
            # Remove old
            current_indices.pop(pos)
            for p in PAIRS_IN_TRIPLE[t_old]:
                used_pairs.remove(p)
            
            # Try to add new
            idx_new = random.choice(all_indices)
            t_new = ALL_TRIPLES[idx_new]
            valid = True
            for p in PAIRS_IN_TRIPLE[t_new]:
                if p in used_pairs:
                    valid = False
                    break
            
            if valid:
                # Apply add part of swap
                current_indices.append(idx_new)
                for p in PAIRS_IN_TRIPLE[t_new]:
                    used_pairs.add(p)
                move_applied = True
                undo_action = ('swap', idx_old, t_old, pos, idx_new, t_new)
            else:
                # Undo removal of old
                current_indices.insert(pos, idx_old)
                for p in PAIRS_IN_TRIPLE[t_old]:
                    used_pairs.add(p)
                continue

        if not move_applied:
            continue

        # Evaluate
        new_alpha = compute_alpha(current_indices)
        
        if new_alpha < current_alpha:
            # Accept
            current_alpha = new_alpha
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_indices = current_indices[:]
                
        elif new_alpha == current_alpha:
            # Accept with some probability to explore plateaus
            if random.random() < 0.3:
                pass # Keep state
            else:
                # Undo
                if undo_action[0] == 'add':
                    current_indices.pop()
                    for p in PAIRS_IN_TRIPLE[undo_action[2]]:
                        used_pairs.remove(p)
                elif undo_action[0] == 'remove':
                    current_indices.insert(undo_action[3], undo_action[1])
                    for p in PAIRS_IN_TRIPLE[undo_action[2]]:
                        used_pairs.add(p)
                elif undo_action[0] == 'swap':
                    current_indices.pop()
                    for p in PAIRS_IN_TRIPLE[undo_action[5]]:
                        used_pairs.remove(p)
                    current_indices.insert(undo_action[3], undo_action[1])
                    for p in PAIRS_IN_TRIPLE[undo_action[2]]:
                        used_pairs.add(p)
            T = max(T_min, T * cooling)
            
        else:
            # Worse
            delta = new_alpha - current_alpha
            # Accept with probability exp(-delta / T)
            if T > 0:
                prob = np.exp(-delta / T)
            else:
                prob = 0
            if random.random() < prob:
                current_alpha = new_alpha
            else:
                # Undo
                if undo_action[0] == 'add':
                    current_indices.pop()
                    for p in PAIRS_IN_TRIPLE[undo_action[2]]:
                        used_pairs.remove(p)
                elif undo_action[0] == 'remove':
                    current_indices.insert(undo_action[3], undo_action[1])
                    for p in PAIRS_IN_TRIPLE[undo_action[2]]:
                        used_pairs.add(p)
                elif undo_action[0] == 'swap':
                    current_indices.pop()
                    for p in PAIRS_IN_TRIPLE[undo_action[5]]:
                        used_pairs.remove(p)
                    current_indices.insert(undo_action[3], undo_action[1])
                    for p in PAIRS_IN_TRIPLE[undo_action[2]]:
                        used_pairs.add(p)
            T = max(T_min, T * cooling)

    # Return the best edges found
    # Convert indices back to triples
    best_edges = [ALL_TRIPLES[i] for i in best_indices]
    return best_edges

# Quick test
# if __name__ == "__main__":
#     result = construct()
#     # Validate
#     edges = result
#     pairs = set()
#     for e in edges:
#         for i in range(3):
#             for j in range(i+1, 3):
#                 a, b = e[i], e[j]
#                 if a > b: a, b = b, a
#                 if (a,b) in pairs:
#                     print("Invalid: repeated pair", (a,b))
#                 pairs.add((a,b))
#     # Compute alpha
#     indices = [ALL_TRIPLES.index(e) for e in edges]
#     print("Alpha:", compute_alpha(indices))

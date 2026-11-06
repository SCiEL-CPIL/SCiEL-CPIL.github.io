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

def compute_alpha(edge_indices):
    """
    Compute independence number given a list of triple indices.
    """
    if not edge_indices:
        return V
    
    # Get masks of edges
    edge_masks = TRIPLE_MASKS_ARR[edge_indices]
    
    # independent[i] is True if subset i does NOT contain any edge
    independent = np.ones(N_SUBSETS, dtype=bool)
    
    # For each edge, mark all supersets as not independent
    # Superset condition: (S & e) == e
    # Vectorized: independent &= ((SUBSET_MASKS & e) != e)
    
    # To speed up, we can process edges in batches or just loop.
    # 560 triples max, but typical packing is ~30-40.
    # 40 iterations of vectorized ops on 65536 elements is fast.
    
    for e in edge_masks:
        independent &= ((SUBSET_MASKS & e) != e)
        
        # Optional early exit if independent has no True values?
        # Unlikely for small edge sets.
        
    # Find max size among independent sets
    # If no independent sets, alpha = 0 (but empty set is always independent)
    # Wait, empty set is always independent. So max is at least 0.
    # Actually, the empty set has size 0.
    # If all sets are bad, that's impossible since empty set contains no triple.
    # So independent[0] is always True.
    
    # Get sizes of independent sets
    sizes = SUBSET_SIZES[independent]
    return int(sizes.max())

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    # Phase 1: Greedy construction to get a good starting point
    # We want a large packing (max number of triples) to minimize alpha.
    # Greedy: randomly shuffle triples, add if valid.
    
    # We will run multiple restarts to find a good starting configuration.
    best_alpha = V
    best_edges = []
    
    # Try a few greedy starts
    for _ in range(5):
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
        # We need a mapping from triple to index
        # Let's build it once
        # Actually, we can just pass the masks directly to a helper or use the index list.
        # Let's create a dict for fast lookup in the main loop
        triple_to_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}
        
        edge_indices = [triple_to_idx[t] for t in edges]
        alpha = compute_alpha(edge_indices)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_edges = edges[:]
            
        if time.time() > deadline - 130:
            break

    # If we found alpha 3, we can't do better (theoretically min is 3? or 4?).
    # For v=16, max packing size is 36 or 37?
    # Steiner Triple System exists for v=6,7,10,13,16...
    # STS(16) does not exist (16 % 6 != 1,3).
    # So we can't have a perfect packing where every pair is in exactly one triple.
    # Max number of triples in a linear triple system on 16 points:
    # Each triple uses 3 pairs. Total pairs = 120.
    # Max triples = floor(120/3) = 40.
    # But constraints of linearity might reduce this.
    # If we can get close to 40, alpha should be small.
    
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
    # And a mapping from pair to list of triple indices? No, just set of pairs.
    used_pairs = set()
    for t in current_edges:
        for p in PAIRS_IN_TRIPLE[t]:
            used_pairs.add(p)
    
    # Main loop
    while time.time() < deadline:
        move_type = random.random()
        
        # Copy state
        new_indices = current_indices[:]
        new_used_pairs = used_pairs.copy()
        
        if move_type < 0.5:
            # Add a triple
            idx = random.choice(all_indices)
            t = ALL_TRIPLES[idx]
            valid = True
            for p in PAIRS_IN_TRIPLE[t]:
                if p in new_used_pairs:
                    valid = False
                    break
            if valid:
                new_indices.append(idx)
                for p in PAIRS_IN_TRIPLE[t]:
                    new_used_pairs.add(p)
            else:
                # If we can't add, skip this iteration to save time?
                # Or try another?
                # Let's just skip.
                continue
                
        elif move_type < 0.8:
            # Remove a triple
            if not new_indices:
                continue
            pos = random.randint(0, len(new_indices) - 1)
            idx = new_indices[pos]
            t = ALL_TRIPLES[idx]
            for p in PAIRS_IN_TRIPLE[t]:
                new_used_pairs.remove(p)
            new_indices.pop(pos)
            
        else:
            # Swap: remove one, add another
            if not new_indices:
                continue
            pos = random.randint(0, len(new_indices) - 1)
            idx_old = new_indices[pos]
            t_old = ALL_TRIPLES[idx_old]
            for p in PAIRS_IN_TRIPLE[t_old]:
                new_used_pairs.remove(p)
            new_indices.pop(pos)
            
            # Try to add a random new one
            idx_new = random.choice(all_indices)
            t_new = ALL_TRIPLES[idx_new]
            valid = True
            for p in PAIRS_IN_TRIPLE[t_new]:
                if p in new_used_pairs:
                    valid = False
                    break
            if valid:
                new_indices.append(idx_new)
                for p in PAIRS_IN_TRIPLE[t_new]:
                    new_used_pairs.add(p)
            else:
                # If swap fails, revert removal?
                # Or just skip?
                # Skipping is safer to avoid degradation without benefit.
                continue

        # Evaluate
        new_alpha = compute_alpha(new_indices)
        
        if new_alpha < current_alpha:
            # Accept
            current_alpha = new_alpha
            current_indices = new_indices
            used_pairs = new_used_pairs
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_indices = current_indices[:]
                
                # If we hit 3, we can stop? 
                # Theoretical min for v=16 is likely 3 or 4.
                # If we hit 3, that's excellent.
                # Let's continue for a bit to see if we can hit 2? Unlikely.
                # But 3 is a strong score.
                
        elif new_alpha == current_alpha:
            # Accept with some probability to explore plateaus
            if random.random() < 0.3:
                current_indices = new_indices
                used_pairs = new_used_pairs
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
                current_indices = new_indices
                used_pairs = new_used_pairs
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

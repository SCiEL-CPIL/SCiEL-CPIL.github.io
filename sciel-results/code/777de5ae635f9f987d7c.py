"""Construct a large triple system on 16 points avoiding K(2,2,2)."""
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 135  # safety margin before 150s kill
    n = 16
    points = list(range(n))
    
    # Generate all triples
    all_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                all_triples.append((i, j, k))
    
    triple_set = set(all_triples)
    triple_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)  # 560
    
    # Precompute all K(2,2,2) configurations
    # A K(2,2,2) is defined by 6 points partitioned into 3 pairs.
    # The 8 transversal triples are all ways to pick one from each pair.
    # We enumerate: choose 6 points, then partition into 3 pairs.
    
    k222_configs = []  # list of tuples of 8 triple indices
    seen_configs = set()
    
    for six_pts in combinations(range(n), 6):
        # Partition 6 points into 3 unordered pairs
        # Fix: pair 0 with something, then pair 1 with something, etc.
        # Standard way: enumerate all perfect matchings on 6 labeled points
        p0, p1, p2, p3, p4, p5 = six_pts
        # All ways to partition {p0..p5} into 3 pairs:
        # Fix p0's partner, then fix p1's partner (if not p0's partner), rest determined
        pairings = []
        # p0 pairs with p1: remaining {p2,p3,p4,p5} -> 3 pairings
        pairings.append(((p0,p1),(p2,p3),(p4,p5)))
        pairings.append(((p0,p1),(p2,p4),(p3,p5)))
        pairings.append(((p0,p1),(p2,p5),(p3,p4)))
        # p0 pairs with p2: remaining {p1,p3,p4,p5} -> 3 pairings
        pairings.append(((p0,p2),(p1,p3),(p4,p5)))
        pairings.append(((p0,p2),(p1,p4),(p3,p5)))
        pairings.append(((p0,p2),(p1,p5),(p3,p4)))
        # p0 pairs with p3: remaining {p1,p2,p4,p5} -> 3 pairings
        pairings.append(((p0,p3),(p1,p2),(p4,p5)))
        pairings.append(((p0,p3),(p1,p4),(p2,p5)))
        pairings.append(((p0,p3),(p1,p5),(p2,p4)))
        # p0 pairs with p4: remaining {p1,p2,p3,p5} -> 3 pairings
        pairings.append(((p0,p4),(p1,p2),(p3,p5)))
        pairings.append(((p0,p4),(p1,p3),(p2,p5)))
        pairings.append(((p0,p4),(p1,p5),(p2,p3)))
        # p0 pairs with p5: remaining {p1,p2,p3,p4} -> 3 pairings
        pairings.append(((p0,p5),(p1,p2),(p3,p4)))
        pairings.append(((p0,p5),(p1,p3),(p2,p4)))
        pairings.append(((p0,p5),(p1,p4),(p2,p3)))
        
        for pairing in pairings:
            # Generate the 8 transversal triples
            (pa, pb, pc) = pairing
            triples_in_config = []
            for xa in pa:
                for xb in pb:
                    for xc in pc:
                        t = tuple(sorted((xa, xb, xc)))
                        triples_in_config.append(t)
            config_key = frozenset(triples_in_config)
            if config_key not in seen_configs:
                seen_configs.add(config_key)
                k222_configs.append(tuple(triple_idx[t] for t in triples_in_config))
    
    num_configs = len(k222_configs)
    
    # For each triple, store list of config indices it belongs to
    triple_to_configs = [[] for _ in range(num_triples)]
    for ci, config in enumerate(k222_configs):
        for ti in config:
            triple_to_configs[ti].append(ci)
    
    # Conflict count: for each config, how many of its 8 triples are selected
    config_counts = [0] * num_configs
    selected = set()
    
    def can_add(triple_idx_val):
        """Check if adding this triple would complete any K(2,2,2)."""
        for ci in triple_to_configs[triple_idx_val]:
            if config_counts[ci] == 7:
                return False
        return True
    
    def add_triple(triple_idx_val):
        selected.add(triple_idx_val)
        for ci in triple_to_configs[triple_idx_val]:
            config_counts[ci] += 1
    
    def remove_triple(triple_idx_val):
        selected.remove(triple_idx_val)
        for ci in triple_to_configs[triple_idx_val]:
            config_counts[ci] -= 1
    
    def try_swap(remove_idx, add_idx):
        """Try swapping: remove one triple, add another. Returns True if successful."""
        # Check if removing remove_idx and adding add_idx is valid
        # After removal, config counts decrease for configs containing remove_idx
        # After addition, config counts increase for configs containing add_idx
        # Net effect: for configs in both, count unchanged
        # For configs only in remove_idx: count decreases by 1 (fine)
        # For configs only in add_idx: count increases by 1 (must not reach 8)
        
        remove_configs = set(triple_to_configs[remove_idx])
        add_configs = set(triple_to_configs[add_idx])
        
        # Check configs that gain: those in add_configs but not in remove_configs
        for ci in add_configs - remove_configs:
            if config_counts[ci] >= 7:
                return False
        
        # Perform the swap
        remove_triple(remove_idx)
        add_triple(add_idx)
        return True
    
    # Greedy construction: start with a random subset, then add greedily
    # Strategy: random initialization with varying density, then greedy fill, then local search
    
    best_result = []
    best_count = 0
    
    start_time = time.time()
    
    num_attempts = 0
    while time.time() < deadline - 5:
        num_attempts += 1
        # Vary the initial seed density
        if num_attempts <= 20:
            # Start with star-like structure
            init_density = 0.0
        elif num_attempts <= 40:
            init_density = 0.2
        elif num_attempts <= 60:
            init_density = 0.4
        else:
            init_density = random.uniform(0.0, 0.5)
        
        # Initialize
        selected.clear()
        config_counts = [0] * num_configs
        
        # Random initial set
        if init_density > 0:
            all_indices = list(range(num_triples))
            random.shuffle(all_indices)
            num_init = int(num_triples * init_density)
            for idx in all_indices[:num_init]:
                if can_add(idx):
                    add_triple(idx)
        
        # Greedy fill: repeatedly try to add a random unselected triple
        remaining = [i for i in range(num_triples) if i not in selected]
        random.shuffle(remaining)
        
        for idx in remaining:
            if time.time() > deadline - 10:
                break
            if can_add(idx):
                add_triple(idx)
        
        # Local search: add moves + swap moves
        improved = True
        no_improve_rounds = 0
        while improved and time.time() < deadline - 10:
            improved = False
            no_improve_rounds += 1
            
            # Add moves: try adding random unselected triples
            unselected = [i for i in range(num_triples) if i not in selected]
            random.shuffle(unselected)
            
            for idx in unselected:
                if time.time() > deadline - 10:
                    break
                if can_add(idx):
                    add_triple(idx)
                    improved = True
            
            # Swap moves: try removing one and adding another
            selected_list = list(selected)
            random.shuffle(selected_list)
            for rem_idx in selected_list:
                if time.time() > deadline - 10:
                    break
                # Try to find an add_idx that works
                candidates = [i for i in range(num_triples) if i not in selected]
                random.shuffle(candidates)
                for add_idx in candidates:
                    if time.time() > deadline - 10:
                        break
                    if try_swap(rem_idx, add_idx):
                        improved = True
                        break  # one swap per removal attempt
        
        # Record result
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = [list(all_triples[i]) for i in sorted(selected)]
        
        # If we've been running a while and not improving, break early
        if num_attempts > 30 and len(selected) <= best_count:
            no_improve_rounds += 1
            if no_improve_rounds > 5:
                break
    
    # Final check: verify validity
    final_selected = set(best_result)
    for config in k222_configs:
        count = 0
        for ti in config:
            if all_triples[ti] in final_selected:
                count += 1
        if count == 8:
            # Invalid! Remove one triple from this config
            for ti in config:
                t = all_triples[ti]
                if t in best_result:
                    best_result.remove(t)
                    break
    
    return best_result

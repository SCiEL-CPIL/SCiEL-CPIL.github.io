import time
import random
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    n = 16
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute the conflict structures
    # A forbidden configuration is defined by 3 disjoint pairs.
    # We need to track for each configuration how many of its 8 transversals are present.
    
    # Generate all unique configurations (sets of 3 disjoint pairs)
    # There are C(16,6) * 15 = 8008 * 15 = 120120 such configurations.
    
    configs = []
    config_to_idx = {}
    
    # Helper to generate all 15 perfect matchings on 6 vertices
    def get_matchings(v6):
        matchings = []
        # v0 pairs with v1..v5
        for i in range(1, 6):
            rem = [v for j, v in enumerate(v6) if j != 0 and j != i]
            # rem has 4 vertices. Matchings on 4 vertices: 3
            for j in range(1, 4):
                rem2 = [v for k, v in enumerate(rem) if k != 0 and k != j]
                p1 = tuple(sorted((v6[0], v6[i])))
                p2 = tuple(sorted((rem[0], rem[j])))
                p3 = tuple(sorted((rem2[0], rem2[1])))
                pairs = sorted([p1, p2, p3])
                matchings.append(tuple(pairs))
        return matchings

    for v6 in itertools.combinations(range(n), 6):
        for m in get_matchings(v6):
            if m not in config_to_idx:
                config_to_idx[m] = len(configs)
                configs.append(m)
    
    num_configs = len(configs)
    
    # Map each triple to the list of config indices it belongs to
    # A triple belongs to a config if it is one of the 8 transversals of that config.
    triple_to_configs = [[] for _ in range(num_triples)]
    
    for cfg_idx, config in enumerate(configs):
        p1, p2, p3 = config
        a, b = p1
        c, d = p2
        e, f = p3
        
        transversals = [
            (a, c, e), (a, c, f), (a, d, e), (a, d, f),
            (b, c, e), (b, c, f), (b, d, e), (b, d, f)
        ]
        
        for t in transversals:
            t_sorted = tuple(sorted(t))
            if t_sorted in triple_to_idx:
                triple_to_configs[triple_to_idx[t_sorted]].append(cfg_idx)
    
    # State variables
    selected = [False] * num_triples
    config_counts = [0] * num_configs
    
    def is_valid_add(t_idx):
        """Check if adding triple t_idx would violate any constraint."""
        for cfg_idx in triple_to_configs[t_idx]:
            if config_counts[cfg_idx] == 7:
                return False
        return True

    def add_triple(t_idx):
        """Add triple t_idx to the system, updating state."""
        selected[t_idx] = True
        for cfg_idx in triple_to_configs[t_idx]:
            config_counts[cfg_idx] += 1

    def remove_triple(t_idx):
        """Remove triple t_idx from the system, updating state."""
        selected[t_idx] = False
        for cfg_idx in triple_to_configs[t_idx]:
            config_counts[cfg_idx] -= 1
            
    def try_add(t_idx):
        """Try to add a triple. Returns True if added."""
        if is_valid_add(t_idx):
            add_triple(t_idx)
            return True
        return False

    best_system = None
    best_count = 0
    random.seed(42)
    
    # Precompute transversals for each config for faster access in local search
    config_transversals = []
    for config in configs:
        p1, p2, p3 = config
        a, b = p1
        c, d = p2
        e, f = p3
        transversals = [
            (a, c, e), (a, c, f), (a, d, e), (a, d, f),
            (b, c, e), (b, c, f), (b, d, e), (b, d, f)
        ]
        # Store as list of triple indices
        t_indices = []
        for t in transversals:
            t_sorted = tuple(sorted(t))
            t_indices.append(triple_to_idx[t_sorted])
        config_transversals.append(t_indices)

    def local_search(state_selected, state_counts, current_count):
        """Perform local search on the current state. Returns (improved_count, improved_selected, improved_counts)."""
        # Make copies to work on
        sel = state_selected[:]
        cnt = state_counts[:]
        count = current_count
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            
            # Sweep 1: Try to add any unselected triple that is now valid
            for t_idx in range(num_triples):
                if not sel[t_idx]:
                    # Check validity
                    valid = True
                    for cfg_idx in triple_to_configs[t_idx]:
                        if cnt[cfg_idx] == 7:
                            valid = False
                            break
                    if valid:
                        sel[t_idx] = True
                        for cfg_idx in triple_to_configs[t_idx]:
                            cnt[cfg_idx] += 1
                        count += 1
                        improved = True
                        
            if not improved:
                # No simple additions possible. Try swaps.
                unselected = [i for i in range(num_triples) if not sel[i]]
                if not unselected:
                    break
                    
                # Try random swaps
                swap_found = False
                for _ in range(100): # Increase swap attempts
                    if time.time() > deadline - 3:
                        break
                    t_add = random.choice(unselected)
                    
                    # Check if valid (should be invalid if sweep 1 didn't add it)
                    valid = True
                    blocking_configs = []
                    for cfg_idx in triple_to_configs[t_add]:
                        if cnt[cfg_idx] == 7:
                            valid = False
                            blocking_configs.append(cfg_idx)
                    
                    if valid:
                        # Should not happen, but if it does, add it
                        sel[t_idx] = True
                        for cfg_idx in triple_to_configs[t_add]:
                            cnt[cfg_idx] += 1
                        count += 1
                        improved = True
                        break
                    if not blocking_configs:
                        continue
                        
                    cfg_block = random.choice(blocking_configs)
                    
                    # Get selected transversals in this config
                    selected_in_cfg = []
                    for t_idx_t in config_transversals[cfg_block]:
                        if sel[t_idx_t]:
                            selected_in_cfg.append(t_idx_t)
                            
                    if not selected_in_cfg:
                        continue
                        
                    t_remove = random.choice(selected_in_cfg)
                    
                    # Simulate swap
                    # Remove t_remove
                    for cfg_idx in triple_to_configs[t_remove]:
                        cnt[cfg_idx] -= 1
                    sel[t_remove] = False
                    
                    # Check if t_add is now valid
                    valid_add = True
                    for cfg_idx in triple_to_configs[t_add]:
                        if cnt[cfg_idx] == 7:
                            valid_add = False
                            break
                    
                    if valid_add:
                        # Accept swap
                        sel[t_add] = True
                        for cfg_idx in triple_to_configs[t_add]:
                            cnt[cfg_idx] += 1
                        # Count stays same for the swap itself, but we mark improved to restart sweep
                        improved = True
                        swap_found = True
                        break
                    else:
                        # Revert
                        sel[t_remove] = True
                        for cfg_idx in triple_to_configs[t_remove]:
                            cnt[cfg_idx] += 1
                        
                if swap_found:
                    continue
                else:
                    break
        
        return count, sel, cnt

    # Strategy: 
    # 1. Multiple randomized greedy starts with different orderings.
    # 2. Local search with 1-swaps.
    
    max_runs = 200 # Increased runs due to efficiency
    
    for run in range(max_runs):
        if time.time() > deadline - 10:
            break
            
        # Reset state
        for i in range(num_triples):
            if selected[i]:
                remove_triple(i)
        for i in range(num_configs):
            config_counts[i] = 0
            
        # 1. Randomized Greedy Initialization
        # Use different strategies for initialization
        if run % 3 == 0:
            # Random order
            order = list(range(num_triples))
            random.shuffle(order)
        elif run % 3 == 1:
            # Greedy by "potential" (number of configs it blocks)
            # Triples that appear in fewer configs are "safer"
            # Or maybe we want to pick ones that appear in many configs to maximize coverage?
            # Let's try sorting by number of configs (ascending) - safer picks first
            order = sorted(range(num_triples), key=lambda x: len(triple_to_configs[x]))
            random.shuffle(order) # Add some randomness to the sorted order
        else:
            # Random but biased towards points that are under-represented?
            order = list(range(num_triples))
            random.shuffle(order)
            
        for t_idx in order:
            try_add(t_idx)
            
        current_count = sum(selected)
        
        # 2. Local Search Phase
        # Pass copies to local_search to avoid modifying global state until we know it's better
        new_count, new_selected, new_counts = local_search(selected, config_counts, current_count)
        
        if new_count > current_count:
            # Apply the changes
            # Reset global state to match new state
            for i in range(num_triples):
                if selected[i] != new_selected[i]:
                    if selected[i]:
                        remove_triple(i)
                    else:
                        add_triple(i)
            # Ensure counts are correct (should be, but let's be safe)
            # Actually, updating counts individually is expensive.
            # Since we have new_counts, we can just copy them.
            # But we need to ensure selected and config_counts are consistent.
            # The local_search function returned consistent sel and cnt.
            # So we can just assign them.
            selected[:] = new_selected
            config_counts[:] = new_counts
            current_count = new_count
            
        # Update best
        if current_count > best_count:
            best_count = current_count
            best_system = [all_triples[i] for i in range(num_triples) if selected[i]]
        
    if best_system is None:
        # Fallback: just return a small valid set
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_system

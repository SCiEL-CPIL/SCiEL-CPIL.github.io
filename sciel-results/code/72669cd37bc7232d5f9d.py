import time
import random
import itertools
import math

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
    # selected: list of booleans, True if triple is in the system
    # config_counts: list of ints, count of transversals present for each config
    
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

    best_system = None
    best_count = 0
    random.seed(42)
    
    # Simulated Annealing Parameters
    # We run multiple independent annealing runs
    max_runs = 50
    steps_per_run = 5000
    initial_temp = 1.0
    cooling_factor = 0.99
    
    # Precompute list of unselected triples for faster random selection?
    # Maintaining a dynamic list is O(N) to update. 
    # Given N=560, random selection from range and checking boolean is fast enough.
    
    for run in range(max_runs):
        if time.time() > deadline - 5:
            break
            
        # Initialize state
        # Start with a random subset, then clean it up to be valid
        # This gives a good starting point
        selected = [False] * num_triples
        config_counts = [0] * num_configs
        
        # Randomly pick ~50% of triples to start
        initial_candidates = random.sample(range(num_triples), num_triples // 2)
        for t_idx in initial_candidates:
            if is_valid_add(t_idx):
                add_triple(t_idx)
        
        current_count = sum(selected)
        if current_count > best_count:
            best_count = current_count
            best_system = [all_triples[i] for i in range(num_triples) if selected[i]]
        
        # Simulated Annealing
        T = initial_temp
        no_improve_steps = 0
        max_no_improve = 1000
        
        for step in range(steps_per_run):
            if time.time() > deadline - 5:
                break
                
            if no_improve_steps > max_no_improve:
                break # Restart this run if stuck
            
            # Choose a move: Add or Remove
            # Probability of Add vs Remove
            # If we are far from maximum, Add is good. If stuck, Remove helps explore.
            # Let's say 50% Add, 50% Remove
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to Add a random unselected triple
                # Find an unselected triple
                # To avoid bias, pick a random index and check if it's selected
                # If selected, retry a few times? Or just pick from a precomputed unselected list?
                # Precomputing unselected list every step is O(N). N=560. 5000 steps * 50 runs = 2.5M ops. OK.
                # But we can just pick random indices until we find one, or maintain a list.
                # Let's maintain a list of unselected indices for efficiency in selection, 
                # but updating it is O(1) if we use swap-remove.
                
                # For simplicity and speed, let's just pick a random triple index.
                # If it's selected, we skip this step (or treat as no-op).
                # This might waste steps. 
                # Better: maintain `unselected_indices` list.
                
                # Let's just pick a random index. If selected, we do a Remove move instead? 
                # No, that changes the distribution.
                # Let's just pick a random index. If it's already selected, we try again?
                # No, that's inefficient.
                
                # Efficient approach: 
                # Keep a list of unselected triples.
                # When we add, remove from unselected list.
                # When we remove, add to unselected list.
                
                # Let's refactor to use lists for selected and unselected.
                # Actually, the boolean array `selected` is fine.
                # To pick a random unselected triple:
                # 1. Pick random index i.
                # 2. If selected[i], we can't add. We can either skip or force a remove.
                # Let's force a remove if we pick a selected one when we wanted to add? 
                # That's not right.
                
                # Let's maintain `unselected_list`.
                # Init: [i for i in range(num_triples) if not selected[i]]
                # Add: remove t_idx from unselected_list (swap with last, pop)
                # Remove: append t_idx to unselected_list
                
                # Let's implement this.
                
                # Note: I didn't initialize unselected_list in the loop above. 
                # I will do it now inside the run loop.
                
                pass # Placeholder, logic below
                
            else:
                # Try to Remove a random selected triple
                # Pick a random selected triple
                pass # Placeholder

        # Re-implementation of the inner loop with proper list management
        
    # Re-doing the main loop with correct state management for lists
    
    best_system = None
    best_count = 0
    
    for run in range(max_runs):
        if time.time() > deadline - 5:
            break
            
        selected = [False] * num_triples
        config_counts = [0] * num_configs
        
        # Initialize
        initial_candidates = random.sample(range(num_triples), num_triples // 2)
        for t_idx in initial_candidates:
            if is_valid_add(t_idx):
                add_triple(t_idx)
        
        # Build unselected list
        unselected_list = [i for i in range(num_triples) if not selected[i]]
        selected_list = [i for i in range(num_triples) if selected[i]]
        
        current_count = len(selected_list)
        if current_count > best_count:
            best_count = current_count
            best_system = [all_triples[i] for i in selected_list]
        
        T = initial_temp
        no_improve_steps = 0
        max_no_improve = 2000
        
        for step in range(steps_per_run):
            if time.time() > deadline - 5:
                break
            if no_improve_steps > max_no_improve:
                break
                
            # Choose move
            if random.random() < 0.5:
                # Try Add
                if not unselected_list:
                    continue
                t_idx = random.choice(unselected_list)
                
                if is_valid_add(t_idx):
                    # Accept
                    add_triple(t_idx)
                    # Update lists
                    # Remove t_idx from unselected_list
                    # Find index
                    # Since it's random, finding index is O(N). 
                    # To make it O(1), we need a map from idx to position in list.
                    # Or just use swap-remove if we keep a position map.
                    # For N=560, O(N) search is ~560 ops. 5000 steps * 50 runs = 2.5M * 560 = 1.4B ops. Too slow in Python.
                    
                    # Optimization: Use a set for unselected? 
                    # Random choice from set is O(1) if we convert to list? No.
                    # Random choice from list is O(1) for choice, O(N) for removal.
                    
                    # Alternative: Just use the boolean array and pick random indices.
                    # If we pick a selected one when wanting to add, we just skip.
                    # This is simpler and avoids list maintenance overhead.
                    # Let's revert to simple random index picking.
                    
                    # Re-logic:
                    # 1. Pick random t_idx.
                    # 2. If we want to Add and it's selected, skip.
                    # 3. If we want to Remove and it's unselected, skip.
                    # This wastes steps but is very fast per step.
                    
                    # Let's restart the loop logic with this simpler approach.
                    pass

    # Final Clean Implementation
    
    best_system = None
    best_count = 0
    
    for run in range(max_runs):
        if time.time() > deadline - 5:
            break
            
        selected = [False] * num_triples
        config_counts = [0] * num_configs
        
        # Initialize with random valid subset
        # Shuffle all triples and greedily add
        order = list(range(num_triples))
        random.shuffle(order)
        for t_idx in order:
            if is_valid_add(t_idx):
                add_triple(t_idx)
        
        current_count = sum(selected)
        if current_count > best_count:
            best_count = current_count
            best_system = [all_triples[i] for i in range(num_triples) if selected[i]]
        
        T = initial_temp
        no_improve_steps = 0
        max_no_improve = 5000
        
        for step in range(steps_per_run):
            if time.time() > deadline - 5:
                break
            if no_improve_steps > max_no_improve:
                break
                
            # Pick random triple
            t_idx = random.randrange(num_triples)
            
            if selected[t_idx]:
                # Currently in system. Move: Remove
                # Removing is always valid (relaxation)
                # Acceptance: Always accept? 
                # In SA, we accept worse moves with prob exp(-delta/T).
                # Here delta = 1 (count decreases by 1).
                # Prob = exp(-1/T).
                if random.random() < math.exp(-1.0 / T):
                    remove_triple(t_idx)
                    current_count -= 1
                    no_improve_steps += 1 # It's a "worse" move
                else:
                    # Rejected
                    no_improve_steps += 1
            else:
                # Not in system. Move: Add
                if is_valid_add(t_idx):
                    # Adding is an improvement (count increases by 1)
                    # Always accept
                    add_triple(t_idx)
                    current_count += 1
                    no_improve_steps = 0
                    if current_count > best_count:
                        best_count = current_count
                        best_system = [all_triples[i] for i in range(num_triples) if selected[i]]
                else:
                    # Invalid add, cannot do it
                    no_improve_steps += 1
            
            # Cool down
            T *= cooling_factor
            if T < 0.01:
                T = 0.01 # Floor
    
    if best_system is None:
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_system

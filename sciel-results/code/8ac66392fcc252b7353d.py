import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560

    # Precompute index for O(1) lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all K(2,2,2) configurations (clauses)
    # A K(2,2,2) is defined by 3 disjoint pairs. The 8 transversal triples are the products.
    # We want to avoid having all 8 present. So the clause is: NOT (t1 AND t2 AND ... AND t8)
    # In MaxSAT, this is a clause of size 8: (t1' OR t2' OR ... OR t8') where ti' is NOT ti.
    # But we are maximizing the number of TRUE variables (triples included).
    # So we want to maximize sum(x_i) subject to: for each config c, sum(x_i for i in c) <= 7.
    # This is equivalent to minimizing the number of violated constraints.
    # WalkSAT on this formulation:
    # Variables: x_i (True if triple i is included)
    # Clauses: For each config c, the clause is violated if all 8 variables are True.
    # Objective: Maximize sum(x_i).
    
    configs = []
    config_id_map = {}
    triples_in_config = [] # List of lists of 8 triple indices

    points = range(N)
    # Generate all unique sets of 3 disjoint pairs
    # We can iterate over all ways to pick 6 points and partition them, but we must avoid duplicates.
    # A unique K(2,2,2) is determined by the set of 6 points and the pairing.
    # Total number of such configs: C(16,6) * 15 = 8008 * 15 = 120120.
    # This is manageable.
    
    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
        # 15 ways to partition 6 items into 3 pairs
        pairs = [
            [(p0, p1), (p2, p3), (p4, p5)],
            [(p0, p1), (p2, p4), (p3, p5)],
            [(p0, p1), (p2, p5), (p3, p4)],
            [(p0, p2), (p1, p3), (p4, p5)],
            [(p0, p2), (p1, p4), (p3, p5)],
            [(p0, p2), (p1, p5), (p3, p4)],
            [(p0, p3), (p1, p2), (p4, p5)],
            [(p0, p3), (p1, p4), (p2, p5)],
            [(p0, p3), (p1, p5), (p2, p4)],
            [(p0, p4), (p1, p2), (p3, p5)],
            [(p0, p4), (p1, p3), (p2, p5)],
            [(p0, p4), (p1, p5), (p2, p3)],
            [(p0, p5), (p1, p2), (p3, p4)],
            [(p0, p5), (p1, p3), (p2, p4)],
            [(p0, p5), (p1, p4), (p2, p3)]
        ]
        for pair_set in pairs:
            # Normalize pair order and pair internal order for unique ID
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)
                
                # Find the 8 transversal triples
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)
                
    num_configs = len(configs)
    
    # Build incidence: for each triple, which configs does it belong to?
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    # State
    # present[i] is True if triple i is in the current solution
    present = np.zeros(num_triples, dtype=bool)
    
    # For each config, count how many of its 8 triples are present.
    # If count == 8, the config is violated (all 8 present).
    config_counts = np.zeros(num_configs, dtype=np.int8)
    
    # We want to maximize the number of triples.
    # Let's use a WalkSAT-like approach.
    # Current state: a set of triples.
    # Objective: Maximize |S| such that no config has count 8.
    
    # Initialize with a random valid state? Or start empty and add?
    # Starting empty and adding greedily is fast but gets stuck.
    # Let's start with a random subset and then optimize.
    
    def get_score():
        return np.sum(present)
        
    def is_violated(cid):
        return config_counts[cid] == 8

    def flip(i, val):
        """Set variable i to val (True/False). Returns change in score."""
        old_val = present[i]
        if old_val == val:
            return 0
        present[i] = val
        delta_score = 1 if val else -1
        for cid in triple_to_configs[i]:
            if val:
                config_counts[cid] += 1
            else:
                config_counts[cid] -= 1
        return delta_score

    def run_walksat(seed_val, initial_density=0.5):
        random.seed(seed_val)
        present[:] = False
        config_counts[:] = 0
        
        # Initialize randomly
        for i in range(num_triples):
            if random.random() < initial_density:
                flip(i, True)
                
        # Ensure we start with a valid state? 
        # WalkSAT usually starts anywhere. But we need to return a VALID solution.
        # So we must repair violations at the end or during the process.
        # Actually, the constraint is "no config has 8". 
        # If we end with violations, we must remove triples to fix them.
        
        best_score = get_score()
        best_state = present.copy()
        
        # WalkSAT Loop
        # We want to maximize score. 
        # Standard WalkSAT minimizes unsatisfied clauses. 
        # Here, "unsatisfied" means violated configs (count=8).
        # But we also want to maximize the number of True variables.
        # This is a Max-SAT problem. 
        # Let's use a heuristic: 
        # 1. If there are violations, pick a violated config and flip one of its True variables to False.
        # 2. If no violations, try to flip a False variable to True if it doesn't create a violation.
        #    If that fails, pick a random variable and flip it (stochastic kick).
        
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() < deadline - 5:
            # Check for violations
            violated_configs = np.where(config_counts == 8)[0]
            
            if len(violated_configs) > 0:
                # Pick a random violated config
                cid = random.choice(violated_configs)
                # Pick a random triple in this config that is currently True
                tris_in_cid = triples_in_config[cid]
                true_tris = [t for t in tris_in_cid if present[t]]
                if not true_tris:
                    continue # Should not happen if count==8
                t_to_flip = random.choice(true_tris)
                flip(t_to_flip, False)
                no_improve = 0
            else:
                # No violations. Try to add a triple.
                # Find all False triples that can be added without creating a violation.
                # A triple i can be added if for all cid in triple_to_configs[i], config_counts[cid] < 7.
                # (If count is 7, adding i makes it 8).
                
                # This check is expensive if done naively for all 560 triples.
                # But 560 is small.
                addable = []
                for i in range(num_triples):
                    if not present[i]:
                        can_add = True
                        for cid in triple_to_configs[i]:
                            if config_counts[cid] >= 7:
                                can_add = False
                                break
                        if can_add:
                            addable.append(i)
                
                if addable:
                    t_to_flip = random.choice(addable)
                    flip(t_to_flip, True)
                    no_improve = 0
                else:
                    # Stuck. Perform a kick.
                    # Flip a random variable.
                    # Or remove a random triple to create space.
                    t_to_flip = random.randint(0, num_triples - 1)
                    flip(t_to_flip, not present[t_to_flip])
                    no_improve += 1
                    if no_improve > max_no_improve:
                        # Restart with new random seed or different density?
                        # Let's just break and restart
                        break
            
            current_score = get_score()
            if current_score > best_score:
                best_score = current_score
                best_state = present.copy()
                no_improve = 0
                
        # After walk, we might have violations. We must remove triples to satisfy constraints.
        # We want to keep as many as possible.
        # Greedy removal: while there are violations, remove a triple from a violated config.
        # To minimize removal, we should remove triples that are in the most violated configs?
        # Or just random.
        
        # Let's repair the best_state found during the walk.
        # Wait, best_state was copied when score was high, but it might have violations.
        # We need to repair best_state.
        
        present[:] = best_state
        # Recalculate config_counts for best_state
        config_counts[:] = 0
        for i in range(num_triples):
            if present[i]:
                for cid in triple_to_configs[i]:
                    config_counts[cid] += 1
                    
        # Repair
        while True:
            violated_configs = np.where(config_counts == 8)[0]
            if len(violated_configs) == 0:
                break
            cid = random.choice(violated_configs)
            tris_in_cid = triples_in_config[cid]
            true_tris = [t for t in tris_in_cid if present[t]]
            if not true_tris:
                break
            t_to_remove = random.choice(true_tris)
            flip(t_to_remove, False)
            
        return get_score(), present.copy()

    best_count = 0
    best_set = np.zeros(num_triples, dtype=bool)
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 5:
        seed = num_restarts
        # Vary initial density
        density = 0.5 + 0.1 * random.random()
        count, state = run_walksat(seed, density)
        if count > best_count:
            best_count = count
            best_set = state
        num_restarts += 1
        
    # Convert indices back to triples
    result = [all_triples[i] for i in range(num_triples) if best_set[i]]
    
    return result

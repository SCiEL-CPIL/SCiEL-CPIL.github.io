import itertools
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    points = list(range(n))
    
    # Generate all triples
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Generate all K(2,2,2) configurations
    # A K(2,2,2) is defined by 6 points and a partition into 3 pairs.
    configs = []
    # To speed up, we can just iterate over all sets of 6 points
    for six_points in itertools.combinations(points, 6):
        # Generate all partitions of 6 points into 3 pairs
        # Number of such partitions is 15
        # We can generate them by pairing the first element with any of the other 5,
        # then the next smallest with any of the remaining 3, etc.
        p = list(six_points)
        # Recursive generation or explicit loops
        # Let's use a helper to generate pairings
        def get_pairings(pts):
            if len(pts) == 0:
                yield []
                return
            first = pts[0]
            rest = pts[1:]
            for i in range(len(rest)):
                second = rest[i]
                remaining = rest[:i] + rest[i+1:]
                for sub_pairing in get_pairings(remaining):
                    yield [(first, second)] + sub_pairing
        
        for pairing in get_pairings(p):
            # The 8 transversals
            transversals = []
            for mask in range(8):
                triple = []
                for j in range(3):
                    a, b = pairing[j]
                    if (mask >> j) & 1:
                        triple.append(a)
                    else:
                        triple.append(b)
                triple.sort()
                transversals.append(tuple(triple))
            
            # Check if these 8 triples are distinct (they should be for distinct points)
            # And map to indices
            config_indices = []
            valid = True
            for t in transversals:
                if t in triple_to_idx:
                    config_indices.append(triple_to_idx[t])
                else:
                    valid = False
                    break
            if valid and len(set(config_indices)) == 8:
                configs.append(tuple(config_indices))
                
    num_configs = len(configs)
    
    # Build reverse mapping: triple index -> list of config indices
    triple_to_configs = [[] for _ in range(num_triples)]
    for c_idx, config in enumerate(configs):
        for t_idx in config:
            triple_to_configs[t_idx].append(c_idx)
            
    def greedy_solve(seed):
        rng = random.Random(seed)
        selected = set(range(num_triples))
        config_full = [True] * num_configs
        config_count = [8] * num_configs
        full_configs = set(range(num_configs))
        
        while full_configs:
            # Count frequency of each triple in full configs
            freq = {t: 0 for t in selected}
            for c_idx in full_configs:
                for t_idx in configs[c_idx]:
                    if t_idx in selected:
                        freq[t_idx] += 1
            
            # Find max frequency
            max_freq = max(freq.values())
            candidates = [t for t, f in freq.items() if f == max_freq]
            
            # Pick one randomly to allow exploration
            t_best = rng.choice(candidates)
            
            selected.remove(t_best)
            
            # Update configs
            for c_idx in triple_to_configs[t_best]:
                if config_full[c_idx]:
                    config_count[c_idx] -= 1
                    if config_count[c_idx] == 0:
                        config_full[c_idx] = False
                        full_configs.remove(c_idx)
                        
        return selected

    best_solution = None
    best_size = -1
    
    # Run multiple seeds
    # Estimate time per run
    # If 1 run takes 0.1s, we can do 1000 runs.
    # If 1 run takes 1s, we can do 100 runs.
    
    t_start = time.time()
    seed = 0
    while time.time() < deadline:
        sol = greedy_solve(seed)
        size = len(sol)
        if size > best_size:
            best_size = size
            best_solution = sol
        seed += 1
        
    if best_solution is None:
        # Fallback to seed behavior
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
        
    # Convert back to list of triples
    result = [list(all_triples[t]) for t in best_solution]
    return result


"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560

    # Precompute index for O(1) lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all K(2,2,2) configurations
    configs = []
    config_id_map = {}
    triples_in_config = []  # List of lists of 8 triple indices

    points = range(N)
    for pts in itertools.combinations(points, 6):
        # Generate all partitions of these 6 points into 3 pairs
        p0, p1, p2, p3, p4, p5 = pts
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
            [(p0, p5), (p1, p4), (p2, p3)],
        ]
        
        for pair_set in pairs:
            a1, a2 = pair_set[0]
            b1, b2 = pair_set[1]
            c1, c2 = pair_set[2]
            
            # The 8 transversal triples
            transversals = [
                (a1, b1, c1),
                (a1, b1, c2),
                (a1, b2, c1),
                (a1, b2, c2),
                (a2, b1, c1),
                (a2, b1, c2),
                (a2, b2, c1),
                (a2, b2, c2),
            ]
            
            triple_indices = []
            valid = True
            for t in transversals:
                idx = triple_to_idx.get(t)
                if idx is None:
                    valid = False
                    break
                triple_indices.append(idx)
            
            if not valid:
                continue
                
            # Create a canonical key to avoid duplicate configs
            key = tuple(sorted(triple_indices))
            if key not in config_id_map:
                config_id_map[key] = len(configs)
                configs.append(key)
                triples_in_config.append(triple_indices)

    num_configs = len(configs)

    # Precompute for each triple, which configs it belongs to
    triple_to_configs = [[] for _ in range(num_triples)]
    for c_idx, config in enumerate(triples_in_config):
        for t_idx in config:
            triple_to_configs[t_idx].append(c_idx)

    random.seed(42)
    
    best_solution = None
    best_count = -1

    def greedy_init():
        sol = [False] * num_triples
        counts = [0] * num_configs
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
        for idx in candidates:
            safe = True
            for c_idx in triple_to_configs[idx]:
                if counts[c_idx] == 7:
                    safe = False
                    break
            if safe:
                sol[idx] = True
                for c_idx in triple_to_configs[idx]:
                    counts[c_idx] += 1
        return sol, counts, sum(sol)

    def try_add_candidates(sol, counts, candidates):
        added = 0
        new_candidates = []
        for idx in candidates:
            if sol[idx]:
                continue 
            safe = True
            for c_idx in triple_to_configs[idx]:
                if counts[c_idx] == 7:
                    safe = False
                    break
            if safe:
                sol[idx] = True
                for c_idx in triple_to_configs[idx]:
                    counts[c_idx] += 1
                added += 1
            else:
                new_candidates.append(idx)
        return added, new_candidates

    # WalkSAT-style search with periodic fluctuation
    for restart in range(500):
        if time.time() > deadline:
            break
            
        sol, counts, current_count = greedy_init()
        
        if current_count > best_count:
            best_count = current_count
            best_solution = sol[:]
            
        candidates = [i for i in range(num_triples) if not sol[i]]
        random.shuffle(candidates)
        
        stagnation_counter = 0
        fluctuation_counter = 0
        step_count = 0
        
        while time.time() < deadline:
            step_count += 1
            
            # Periodic fluctuation: randomly remove 3-8 triples every 5000 steps
            if step_count % 5000 == 0:
                num_remove = random.randint(3, 8)
                active = [i for i in range(num_triples) if sol[i]]
                if len(active) > num_remove:
                    to_remove = random.sample(active, num_remove)
                    for idx in to_remove:
                        sol[idx] = False
                        for c_idx in triple_to_configs[idx]:
                            counts[c_idx] -= 1
                        current_count -= 1
                        candidates.append(idx)
                    random.shuffle(candidates)
                    stagnation_counter = 0
                    continue
            
            # Try to add from candidates
            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added
            
            if added > 0:
                stagnation_counter = 0
                if current_count > best_count:
                    best_count = current_count
                    best_solution = sol[:]
                continue
            
            # Stuck: candidates are all unsafe.
            # Perform a kick
            tight_triples = []
            for c_idx, cnt in enumerate(counts):
                if cnt == 7:
                    for t_idx in triples_in_config[c_idx]:
                        if sol[t_idx]:
                            tight_triples.append(t_idx)
            
            if tight_triples:
                kick_idx = random.choice(tight_triples)
            else:
                active = [i for i in range(num_triples) if sol[i]]
                if not active:
                    break
                kick_idx = random.choice(active)
            
            # Remove kick_idx
            sol[kick_idx] = False
            for c_idx in triple_to_configs[kick_idx]:
                counts[c_idx] -= 1
            current_count -= 1
            
            # Add kick_idx to candidates
            candidates.append(kick_idx)
            
            # Try to add from candidates
            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added
            
            if added > 0:
                stagnation_counter = 0
                if current_count > best_count:
                    best_count = current_count
                    best_solution = sol[:]
            else:
                stagnation_counter += 1
                if stagnation_counter > 20:
                    break  # Restart this search

    if best_solution is None:
        best_solution, _, _ = greedy_init()
        best_count = sum(best_solution)

    result = [list(all_triples[i]) for i in range(num_triples) if best_solution[i]]
    return result
# EVOLVE-BLOCK-END

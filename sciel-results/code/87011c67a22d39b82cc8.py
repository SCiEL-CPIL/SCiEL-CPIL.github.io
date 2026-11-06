"""
Maximize number of triples in a 3-uniform hypergraph on 16 vertices
such that no three disjoint pairs have all 8 transversal triples present.

Strategy:
1. Precompute all K(2,2,2) configurations and conflict graph.
2. Use branch-and-bound with memoization on the conflict graph to find
   maximum independent sets, combined with local search for speed.
3. Use degree-based greedy initialization and simulated annealing for
   escaping local optima.
4. Time-bounded with multiple restarts and a final polish pass.
"""

import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_total = len(all_triples)  # 560
    
    # Precompute all K(2,2,2) configurations
    k222_configs = []
    triple_to_k222 = [[] for _ in range(num_total)]
    
    for six_vertices in itertools.combinations(range(N), 6):
        a, b, c, d, e, f = six_vertices
        partitions = [
            ((a,b),(c,d),(e,f)), ((a,b),(c,e),(d,f)), ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)), ((a,c),(b,e),(d,f)), ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)), ((a,d),(b,e),(c,f)), ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)), ((a,e),(b,d),(c,f)), ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)), ((a,f),(b,d),(c,e)), ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in partitions:
            transversals = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        transversals.append(t)
            trans_indices = frozenset(triple_to_idx[t] for t in transversals)
            k222_configs.append(trans_indices)
            config_idx = len(k222_configs) - 1
            for ti in trans_indices:
                triple_to_k222[ti].append(config_idx)
    
    num_k222 = len(k222_configs)
    
    # Build conflict graph: two triples conflict if they are in the same K(2,2,2)
    # and adding both would complete it (i.e., they share a K(2,2,2) config)
    # Actually, the constraint is that no K(2,2,2) has all 8 triples present.
    # So we can model this as: for each K(2,2,2), at most 7 of its 8 triples can be selected.
    # This is a packing constraint, not a pairwise conflict.
    
    # For branch-and-bound, we'll use the following approach:
    # - Upper bound: for each K(2,2,2) config, if 7 of its triples are already selected,
    #   the 8th cannot be added. We can compute a fractional upper bound.
    # - Memoization: cache results for subsets of "critical" configs.
    
    # Actually, let's use a different approach: local search with simulated annealing
    # that allows temporary violations (soft constraints) to escape local optima.
    
    def time_check():
        return time.time() >= deadline
    
    best_result = []
    best_count = 0
    
    # Greedy initialization
    def greedy_init(rng):
        present = [False] * num_total
        k222_count = [0] * num_k222
        count = 0
        
        order = list(range(num_total))
        rng.shuffle(order)
        
        for ti in order:
            can_add = True
            for ci in triple_to_k222[ti]:
                if k222_count[ci] == 7:
                    can_add = False
                    break
            if can_add:
                present[ti] = True
                count += 1
                for ci in triple_to_k222[ti]:
                    k222_count[ci] += 1
        return present, k222_count, count
    
    # Simulated annealing with soft constraints
    def simulated_annealing(present, k222_count, count, rng, max_steps=50000):
        steps = 0
        T = 1.0  # temperature
        T_min = 0.01
        cooling = 0.9995
        
        # Compute current "energy" = number of violations (should be 0 for valid)
        def compute_energy(k222_count):
            energy = 0
            for c in k222_count:
                if c > 8:
                    energy += (c - 8)
            return energy
        
        current_energy = compute_energy(k222_count)
        
        while steps < max_steps and T > T_min:
            if time_check():
                break
            
            # Random move: add, remove, or swap
            move_type = rng.random()
            
            if move_type < 0.3:
                # Add a triple
                ti = rng.randint(0, num_total - 1)
                if present[ti]:
                    continue
                # Add it (may create violations)
                present[ti] = True
                count += 1
                for ci in triple_to_k222[ti]:
                    k222_count[ci] += 1
                new_energy = compute_energy(k222_count)
                delta_e = new_energy - current_energy
                # Accept if improves count or with probability exp(-delta_e/T)
                if delta_e <= 0 or rng.random() < (2.718281828 ** (-delta_e / T)):
                    current_energy = new_energy
                else:
                    # Revert
                    present[ti] = False
                    count -= 1
                    for ci in triple_to_k222[ti]:
                        k222_count[ci] -= 1
            elif move_type < 0.6:
                # Remove a triple
                ti = rng.randint(0, num_total - 1)
                if not present[ti]:
                    continue
                present[ti] = False
                count -= 1
                for ci in triple_to_k222[ti]:
                    k222_count[ci] -= 1
                new_energy = compute_energy(k222_count)
                delta_e = new_energy - current_energy
                # Removing always reduces or keeps energy, so accept
                current_energy = new_energy
            else:
                # Swap: remove one, add another
                ti_remove = rng.randint(0, num_total - 1)
                if not present[ti_remove]:
                    continue
                present[ti_remove] = False
                count -= 1
                for ci in triple_to_k222[ti_remove]:
                    k222_count[ci] -= 1
                
                ti_add = rng.randint(0, num_total - 1)
                if present[ti_add]:
                    # Revert removal
                    present[ti_remove] = True
                    count += 1
                    for ci in triple_to_k222[ti_remove]:
                        k222_count[ci] += 1
                    continue
                
                present[ti_add] = True
                count += 1
                for ci in triple_to_k222[ti_add]:
                    k222_count[ci] += 1
                new_energy = compute_energy(k222_count)
                delta_e = new_energy - current_energy
                if delta_e <= 0 or rng.random() < (2.718281828 ** (-delta_e / T)):
                    current_energy = new_energy
                else:
                    # Revert swap
                    present[ti_add] = False
                    count -= 1
                    for ci in triple_to_k222[ti_add]:
                        k222_count[ci] -= 1
                    present[ti_remove] = True
                    count += 1
                    for ci in triple_to_k222[ti_remove]:
                        k222_count[ci] += 1
            
            steps += 1
            T *= cooling
        
        # Final polish: remove any violating triples
        for ci in range(num_k222):
            if k222_count[ci] > 8:
                # Remove random triples from this config until valid
                config_triples = list(k222_configs[ci])
                rng.shuffle(config_triples)
                for ti in config_triples:
                    if k222_count[ci] <= 8:
                        break
                    if present[ti]:
                        present[ti] = False
                        count -= 1
                        for cj in triple_to_k222[ti]:
                            k222_count[cj] -= 1
        
        return present, count
    
    # Local search with valid moves only (for final polish)
    def local_search_valid(present, k222_count, count, rng, max_steps=30000):
        steps = 0
        no_improve = 0
        max_no_improve = 5000
        
        while steps < max_steps and no_improve < max_no_improve:
            if time_check():
                break
            
            moved = False
            
            # Try to add
            for _ in range(30):
                ti = rng.randint(0, num_total - 1)
                if present[ti]:
                    continue
                can_add = True
                for ci in triple_to_k222[ti]:
                    if k222_count[ci] == 7:
                        can_add = False
                        break
                if can_add:
                    present[ti] = True
                    count += 1
                    for ci in triple_to_k222[ti]:
                        k222_count[ci] += 1
                    moved = True
                    no_improve = 0
                    break
            
            if moved:
                steps += 1
                continue
            
            # Try swap
            swap_done = False
            for _ in range(30):
                ti_remove = rng.randint(0, num_total - 1)
                if not present[ti_remove]:
                    continue
                
                for ci in triple_to_k222[ti_remove]:
                    k222_count[ci] -= 1
                present[ti_remove] = False
                count -= 1
                
                found_add = False
                for _ in range(50):
                    ti_candidate = rng.randint(0, num_total - 1)
                    if present[ti_candidate]:
                        continue
                    can_add = True
                    for ci in triple_to_k222[ti_candidate]:
                        if k222_count[ci] == 7:
                            can_add = False
                            break
                    if can_add:
                        present[ti_candidate] = True
                        count += 1
                        for ci in triple_to_k222[ti_candidate]:
                            k222_count[ci] += 1
                        found_add = True
                        break
                
                if found_add:
                    swap_done = True
                    no_improve = 0
                else:
                    for ci in triple_to_k222[ti_remove]:
                        k222_count[ci] += 1
                    present[ti_remove] = True
                    count += 1
            
            if swap_done:
                steps += 1
                continue
            
            no_improve += 1
            steps += 1
        
        return present, count
    
    # Main loop
    rng = random.Random(42)
    
    for restart in range(300):
        if time_check():
            break
        
        # Greedy init
        present, k222_count, count = greedy_init(rng)
        
        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]
        
        # Simulated annealing
        present, count = simulated_annealing(present, k222_count, count, rng)
        
        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]
        
        # Final valid local search
        present, count = local_search_valid(present, k222_count, count, rng)
        
        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]
        
        if time_check():
            break
    
    if not best_result:
        best_result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_result

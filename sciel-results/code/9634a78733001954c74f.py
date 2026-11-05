import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block -> triples mapping
    block_triple_indices = [[] for _ in range(num_blocks)]
    triple_to_blocks = [[] for _ in range(num_triples)]
    block_triple_sets = [set() for _ in range(num_blocks)]
    
    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices[i] = tris
        block_triple_sets[i] = set(tris)

    best_sol = None
    best_energy = float('inf')
    
    seed_counter = 0
    
    while time.time() < deadline - 15:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        # Greedy Construction
        sol = greedy_construct_v2(rng, num_blocks, block_triple_indices, num_triples, deadline - 10)
        
        # Local Search
        time_left = deadline - 10 - time.time()
        if time_left < 2:
            break
            
        sol = local_search_v2(sol, num_blocks, block_triple_indices, block_triple_sets, triple_to_blocks, num_triples, time_left, rng)
        
        # Evaluate
        if sol:
            covered = np.zeros(num_triples, dtype=bool)
            for b_idx in sol:
                for t_idx in block_triple_indices[b_idx]:
                    covered[t_idx] = True
            
            uncovered_count = np.sum(~covered)
            
            if uncovered_count == 0:
                energy = len(sol)
                if energy < best_energy:
                    best_energy = energy
                    best_sol = list(sol)
                
                if energy <= 70:
                    break

    if best_sol is None:
        return []
        
    return [list(all_blocks[i]) for i in best_sol]

def greedy_construct_v2(rng, num_blocks, block_triple_indices, num_triples, deadline):
    uncovered_count = num_triples
    selected = []
    selected_set = set()
    
    while uncovered_count > 0 and time.time() < deadline:
        best_gain = -1
        candidates = []
        
        for i in range(num_blocks):
            if i in selected_set:
                continue
            
            gain = 0
            for t_idx in block_triple_indices[i]:
                gain += 1
            if gain > best_gain:
                best_gain = gain
                candidates = [i]
            elif gain == best_gain and gain > 0:
                candidates.append(i)
        
        if best_gain <= 0:
            break
            
        chosen = rng.choice(candidates)
        selected.append(chosen)
        selected_set.add(chosen)
        
        for t_idx in block_triple_indices[chosen]:
            uncovered_count -= 1 if uncovered_count > 0 else 0
            # Note: this is an approximation. We are not tracking exact counts per triple in this fast greedy.
            # This might lead to suboptimal greedy but faster.
            # Actually, let's do the exact one but maybe faster?
            # The exact one was slow. Let's stick to a slightly smarter greedy.
            
    # Refinement: Remove redundant blocks
    # Sort by size? All same size.
    # Just try removing each and see if it breaks coverage? Too slow.
    return selected

def local_search_v2(initial_sol, num_blocks, block_triple_indices, block_triple_sets, triple_to_blocks, num_triples, time_limit, rng):
    sol = list(initial_sol)
    sol_set = set(sol)
    
    # Coverage counts
    counts = [0] * num_triples
    for idx in sol:
        for t_idx in block_triple_indices[idx]:
            counts[t_idx] += 1
            
    current_uncovered = sum(1 for c in counts if c == 0)
    current_energy = len(sol) + 100 * current_uncovered
    best_sol = list(sol)
    best_energy = current_energy
    
    T = 5.0
    cooling = 0.99
    
    while time.time() < time_limit and T > 0.1:
        move_type = rng.random()
        
        if move_type < 0.4 and sol:
            # Removal
            to_remove = rng.choice(sol)
            
            # Check if redundant
            is_redundant = True
            for t_idx in block_triple_indices[to_remove]:
                if counts[t_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                sol.remove(to_remove)
                sol_set.discard(to_remove)
                for t_idx in block_triple_indices[to_remove]:
                    counts[t_idx] -= 1
                current_energy -= 1
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_sol = list(sol)
            else:
                # Calculate delta_uncovered
                delta_uncovered = 0
                for t_idx in block_triple_indices[to_remove]:
                    if counts[t_idx] == 1:
                        delta_uncovered += 1
                
                delta_e = 1 + 100 * delta_uncovered
                if rng.random() < math.exp(-delta_e / T):
                    sol.remove(to_remove)
                    sol_set.discard(to_remove)
                    for t_idx in block_triple_indices[to_remove]:
                        counts[t_idx] -= 1
                    current_uncovered += delta_uncovered
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)

        elif move_type < 0.7 and sol:
            # Swap
            to_remove = rng.choice(sol)
            
            # Find triples that will be uncovered
            lost_tris = []
            for t_idx in block_triple_indices[to_remove]:
                if counts[t_idx] == 1:
                    lost_tris.append(t_idx)
            
            if not lost_tris:
                continue # This is effectively a removal, handled above? Or neutral swap.
                         # If neutral, we can just swap to add diversity.
                         # But we want to find a replacement.
            
            # Find a block that covers all lost_tris
            # Intersect the list of blocks covering each lost triple
            candidate_sets = []
            for t_idx in lost_tris:
                # Filter triple_to_blocks[t_idx] to exclude sol_set
                candidates = [b for b in triple_to_blocks[t_idx] if b not in sol_set]
                if not candidates:
                    candidate_sets = []
                    break
                candidate_sets.append(candidates)
            
            if not candidate_sets:
                continue
            
            # Intersect
            common_candidates = set(candidate_sets[0])
            for s in candidate_sets[1:]:
                common_candidates.intersection_update(s)
                if not common_candidates:
                    break
            
            if common_candidates:
                to_add = rng.choice(list(common_candidates))
                
                # Perform swap
                sol.remove(to_remove)
                sol_set.discard(to_remove)
                for t_idx in block_triple_indices[to_remove]:
                    counts[t_idx] -= 1
                
                sol.append(to_add)
                sol_set.add(to_add)
                for t_idx in block_triple_indices[to_add]:
                    counts[t_idx] += 1
                    
                # Calculate new uncovered
                new_uncovered = sum(1 for c in counts if c == 0)
                new_energy = len(sol) + 100 * new_uncovered
                
                delta_e = new_energy - current_energy
                if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
                else:
                    # Undo
                    sol.remove(to_add)
                    sol_set.discard(to_add)
                    for t_idx in block_triple_indices[to_add]:
                        counts[t_idx] -= 1
                    sol.append(to_remove)
                    sol_set.add(to_remove)
                    for t_idx in block_triple_indices[to_remove]:
                        counts[t_idx] += 1

        else:
            # Add (if under budget? or just try and reject if energy increases too much)
            # Usually we want to minimize size. Adding increases size by 1.
            # Only accept if it reduces uncovered count significantly.
            
            # Pick a random block not in sol
            b = rng.randint(0, num_blocks - 1)
            if b in sol_set:
                continue
            
            gain = 0
            for t_idx in block_triple_indices[b]:
                if counts[t_idx] == 0:
                    gain += 1
            
            delta_e = 1 - 100 * gain
            
            if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                sol.append(b)
                sol_set.add(b)
                for t_idx in block_triple_indices[b]:
                    counts[t_idx] += 1
                current_uncovered -= gain
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_sol = list(sol)

        T *= cooling

    return best_sol

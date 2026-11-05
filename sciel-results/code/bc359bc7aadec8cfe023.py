import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 150s limit, leave margin
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute triple indices for each block
    block_triple_indices = [[] for _ in range(num_blocks)]
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices[i] = tris

    def greedy_construct(seed):
        rng = random.Random(seed)
        uncovered = [0] * num_triples # 1 if uncovered
        for i in range(num_triples):
            uncovered[i] = 1
        
        selected = []
        selected_set = set()
        
        while any(uncovered) and time.time() < deadline - 10:
            best_gain = -1
            candidates = []
            
            # Calculate gain for each block not in selected
            # Optimization: Only consider blocks that cover at least 1 uncovered triple
            for i in range(num_blocks):
                if i in selected_set:
                    continue
                gain = 0
                for tri_idx in block_triple_indices[i]:
                    if uncovered[tri_idx]:
                        gain += 1
                        if gain > best_gain:
                            break # Early exit if already better than current best? No, need exact gain for top-k
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if best_gain <= 0:
                break
                
            # Pick randomly from top candidates to diversify
            chosen = rng.choice(candidates)
            selected.append(chosen)
            selected_set.add(chosen)
            
            for tri_idx in block_triple_indices[chosen]:
                uncovered[tri_idx] = 0
                
        return selected

    def local_search_minimize(selected_blocks, time_limit):
        sol = list(selected_blocks)
        sol_set = set(sol)
        
        counts = [0] * num_triples
        for idx in sol:
            for tri_idx in block_triple_indices[idx]:
                counts[tri_idx] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            i = 0
            while i < len(sol):
                idx = sol[i]
                is_redundant = True
                for tri_idx in block_triple_indices[idx]:
                    if counts[tri_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    sol.pop(i)
                    sol_set.discard(idx)
                    for tri_idx in block_triple_indices[idx]:
                        counts[tri_idx] -= 1
                    improved = True
                else:
                    i += 1
        return sol

    def simulated_annealing(initial_sol, time_limit, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        sol_set = set(sol)
        
        counts = [0] * num_triples
        for idx in sol:
            for tri_idx in block_triple_indices[idx]:
                counts[tri_idx] += 1
        
        def get_uncovered_count():
            return sum(1 for c in counts if c == 0)
        
        current_energy = len(sol) + 100 * get_uncovered_count()
        best_sol = list(sol)
        best_energy = current_energy
        
        T = 5.0
        cooling = 0.99
        
        while time.time() < time_limit and T > 0.01:
            # Generate neighbor
            move_type = rng.random()
            
            if move_type < 0.7 and sol:
                # Try remove
                to_remove = rng.choice(sol)
                # Check if redundant
                redundant = True
                for tri_idx in block_triple_indices[to_remove]:
                    if counts[tri_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Always accept
                    sol.remove(to_remove)
                    sol_set.discard(to_remove)
                    for tri_idx in block_triple_indices[to_remove]:
                        counts[tri_idx] -= 1
                    current_energy -= 1
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
                else:
                    # Calculate delta energy if we removed it anyway (uncovered triples)
                    # This is a "bad" move
                    delta_uncovered = 0
                    for tri_idx in block_triple_indices[to_remove]:
                        if counts[tri_idx] == 1:
                            delta_uncovered += 1
                    delta_e = 1 + 100 * delta_uncovered
                    if rng.random() < pow(2.718, -delta_e / T):
                        # Accept
                        sol.remove(to_remove)
                        sol_set.discard(to_remove)
                        for tri_idx in block_triple_indices[to_remove]:
                            counts[tri_idx] -= 1
                        current_energy += delta_e
            elif move_type < 0.9:
                # Try swap
                if not sol: continue
                to_remove = rng.choice(sol)
                # Find uncovered triples if we remove to_remove
                uncovered_tris = []
                for tri_idx in block_triple_indices[to_remove]:
                    if counts[tri_idx] == 1:
                        uncovered_tris.append(tri_idx)
                
                if not uncovered_tris:
                    # This is effectively a removal
                    continue
                
                # Find a block that covers all uncovered_tris
                candidates = []
                for tri_idx in uncovered_tris:
                    for b_idx in triple_to_blocks[tri_idx]:
                        if b_idx in sol_set:
                            continue
                        # Check if b_idx covers all uncovered_tris
                        covers_all = True
                        for ut in uncovered_tris:
                            # Check if ut is in block_triple_indices[b_idx]
                            # This is O(N) check. Optimize with sets?
                            if ut not in block_triple_indices[b_idx]:
                                covers_all = False
                                break
                        if covers_all:
                            candidates.append(b_idx)
                            break # One candidate is enough for this move
                
                if candidates:
                    to_add = rng.choice(candidates)
                    # Perform swap
                    sol.remove(to_remove)
                    sol_set.discard(to_remove)
                    for tri_idx in block_triple_indices[to_remove]:
                        counts[tri_idx] -= 1
                    
                    sol.append(to_add)
                    sol_set.add(to_add)
                    for tri_idx in block_triple_indices[to_add]:
                        counts[tri_idx] += 1
                    
                    # Energy change: size same, uncovered change 0 (by construction)
                    # So delta_e = 0. Always accept?
                    # This allows exploration of the "same size" manifold.
                    current_energy = len(sol) + 100 * get_uncovered_count()
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
            else:
                # Try add (explore)
                if time.time() > time_limit: break
                # Pick random block
                to_add = rng.choice(range(num_blocks))
                if to_add in sol_set:
                    continue
                sol.append(to_add)
                sol_set.add(to_add)
                for tri_idx in block_triple_indices[to_add]:
                    counts[tri_idx] += 1
                
                # Simulate: just accept with prob
                # Re-calculate energy
                new_energy = len(sol) + 100 * get_uncovered_count()
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < pow(2.718, -delta / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
                else:
                    # Reject
                    sol.remove(to_add)
                    sol_set.discard(to_add)
                    for tri_idx in block_triple_indices[to_add]:
                        counts[tri_idx] -= 1
            
            T *= cooling
            
        return best_sol

    best_sol = None
    best_size = num_blocks
    
    # Run multiple iterations with different seeds
    # 1. Greedy
    # 2. Local Search
    # 3. SA
    
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        print(f"Restart {i}", file=sys.stderr)
        
        # Greedy
        g_sol = greedy_construct(1000 + i)
        
        # Local Search to minimize
        ls_sol = local_search_minimize(g_sol, deadline - 30)
        
        if len(ls_sol) < best_size:
            # Verify
            covered = set()
            for idx in ls_sol:
                for tri_idx in block_triple_indices[idx]:
                    covered.add(tri_idx)
            if len(covered) == num_triples:
                best_size = len(ls_sol)
                best_sol = ls_sol
                
        # SA
        if time.time() < deadline - 10:
            sa_sol = simulated_annealing(ls_sol, deadline - 5, 2000 + i)
            if len(sa_sol) < best_size:
                covered = set()
                for idx in sa_sol:
                    for tri_idx in block_triple_indices[idx]:
                        covered.add(tri_idx)
                if len(covered) == num_triples:
                    best_size = len(sa_sol)
                    best_sol = sa_sol
    
    if best_sol is None:
        # Fallback
        best_sol = greedy_construct(1)
        best_sol = local_search_minimize(best_sol, deadline - 5)
        
    return [all_blocks[i] for i in best_sol]

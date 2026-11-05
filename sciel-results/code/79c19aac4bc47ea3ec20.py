import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Precompute triple indices for each block and reverse mapping
    block_triple_indices = [[] for _ in range(num_blocks)]
    triple_to_blocks = [[] for _ in range(num_triples)]

    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices[i] = tris

    # Convert to numpy for fast operations where possible
    # block_triple_indices_np: list of arrays
    bti_arrays = [np.array(bi, dtype=np.int32) for bi in block_triple_indices]

    def greedy_construct(seed):
        rng = random.Random(seed)
        uncovered = np.ones(num_triples, dtype=np.int8)
        
        # gain[i] = number of uncovered triples in block i
        gains = np.array([len(bti) for bti in block_triple_indices], dtype=np.int32)
        
        selected = []
        selected_set = set()

        while time.time() < deadline - 10:
            # Find block with max gain
            # We need to ensure we don't pick already selected blocks
            # We can set gains of selected blocks to -1
            # But we need to restore them? No, we just need max among unselected.
            # Simple approach: mask or just check. 
            # Since num_blocks is ~177k, np.argmax is fast.
            # But we must ignore selected.
            
            # Optimization: Set gain of selected blocks to -1
            for idx in selected:
                gains[idx] = -1
                
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
                
            # Find all blocks with max gain
            # candidates = np.where(gains == max_gain)[0]
            # But we need to filter out selected ones if any had -1? 
            # We set them to -1, so they won't be max_gain (unless max_gain is -1, which we break on).
            
            candidates = np.where(gains == max_gain)[0]
            chosen = rng.choice(candidates)
            
            selected.append(chosen)
            selected_set.add(chosen)
            
            # Update gains for all other blocks
            # For each triple in chosen block, for each block sharing that triple,
            # decrement gain by 1 (since that triple is now covered)
            for tri_idx in block_triple_indices[chosen]:
                if uncovered[tri_idx]: # It should be 1
                    uncovered[tri_idx] = 0
                    for other_idx in triple_to_blocks[tri_idx]:
                        if other_idx != chosen:
                            gains[other_idx] -= 1

            if not uncovered.any():
                break

        return selected

    def local_search_minimize(selected_blocks, time_limit):
        sol = list(selected_blocks)
        sol_set = set(sol)

        counts = np.zeros(num_triples, dtype=np.int8)
        for idx in sol:
            counts[bti_arrays[idx]] += 1

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            i = 0
            while i < len(sol):
                idx = sol[i]
                # Check if redundant
                if (counts[bti_arrays[idx]] <= 1).all():
                    sol.pop(i)
                    sol_set.discard(idx)
                    counts[bti_arrays[idx]] -= 1
                    improved = True
                else:
                    i += 1
        return sol

    def simulated_annealing(initial_sol, time_limit, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        sol_set = set(sol)

        counts = np.zeros(num_triples, dtype=np.int8)
        for idx in sol:
            counts[bti_arrays[idx]] += 1

        def get_uncovered_count():
            return int(np.sum(counts == 0))

        current_energy = len(sol) + 100 * get_uncovered_count()
        best_sol = list(sol)
        best_energy = current_energy

        T = 5.0
        cooling = 0.99

        while time.time() < time_limit and T > 0.01:
            move_type = rng.random()

            if move_type < 0.7 and sol:
                # Try remove
                to_remove = rng.choice(sol)
                redundant = (counts[bti_arrays[to_remove]] <= 1).all()
                if redundant:
                    sol.remove(to_remove)
                    sol_set.discard(to_remove)
                    counts[bti_arrays[to_remove]] -= 1
                    current_energy -= 1
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
                else:
                    delta_uncovered = int(np.sum(counts[bti_arrays[to_remove]] == 1))
                    delta_e = 1 + 100 * delta_uncovered
                    if rng.random() < np.exp(-delta_e / T):
                        sol.remove(to_remove)
                        sol_set.discard(to_remove)
                        counts[bti_arrays[to_remove]] -= 1
                        current_energy += delta_e
            elif move_type < 0.9:
                # Try swap
                if not sol:
                    continue
                to_remove = rng.choice(sol)
                uncovered_tris = bti_arrays[to_remove][counts[bti_arrays[to_remove]] == 1]

                if len(uncovered_tris) == 0:
                    continue

                # Find a block that covers all uncovered_tris
                candidates = []
                for tri_idx in uncovered_tris:
                    for b_idx in triple_to_blocks[tri_idx]:
                        if b_idx in sol_set:
                            continue
                        # Check if b_idx covers all uncovered_tris
                        covers_all = np.all(counts[bti_arrays[b_idx][uncovered_tris]] > 0)
                        if covers_all:
                            candidates.append(b_idx)
                            break

                if candidates:
                    to_add = rng.choice(candidates)
                    sol.remove(to_remove)
                    sol_set.discard(to_remove)
                    counts[bti_arrays[to_remove]] -= 1

                    sol.append(to_add)
                    sol_set.add(to_add)
                    counts[bti_arrays[to_add]] += 1

                    current_energy = len(sol) + 100 * get_uncovered_count()
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
            else:
                # Try add
                if time.time() > time_limit:
                    break
                to_add = rng.choice(range(num_blocks))
                if to_add in sol_set:
                    continue
                sol.append(to_add)
                sol_set.add(to_add)
                counts[bti_arrays[to_add]] += 1

                new_energy = len(sol) + 100 * get_uncovered_count()
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
                else:
                    sol.remove(to_add)
                    sol_set.discard(to_add)
                    counts[bti_arrays[to_add]] -= 1

            T *= cooling

        return best_sol

    best_sol = None
    best_size = num_blocks

    num_restarts = 5

    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break

        print(f"Restart {i}", file=sys.stderr)

        g_sol = greedy_construct(1000 + i)
        ls_sol = local_search_minimize(g_sol, deadline - 30)

        if len(ls_sol) < best_size:
            covered = set()
            for idx in ls_sol:
                covered.update(block_triple_indices[idx])
            if len(covered) == num_triples:
                best_size = len(ls_sol)
                best_sol = ls_sol

        if time.time() < deadline - 10:
            sa_sol = simulated_annealing(ls_sol, deadline - 5, 2000 + i)
            if len(sa_sol) < best_size:
                covered = set()
                for idx in sa_sol:
                    covered.update(block_triple_indices[idx])
                if len(covered) == num_triples:
                    best_size = len(sa_sol)
                    best_sol = sa_sol

    if best_sol is None:
        best_sol = greedy_construct(1)
        best_sol = local_search_minimize(best_sol, deadline - 5)

    return [all_blocks[i] for i in best_sol]

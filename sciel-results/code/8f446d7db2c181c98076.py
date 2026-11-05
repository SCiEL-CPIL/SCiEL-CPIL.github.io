import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Generate all blocks
    blocks = list(itertools.combinations(points, k))
    num_blocks = len(blocks)
    
    # Map block index to list of triple indices it covers
    block_triples = []
    for b in blocks:
        ts = [triple_to_idx[triple] for triple in itertools.combinations(b, t)]
        block_triples.append(ts)
        
    # Map triple index to list of block indices covering it
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_blocks[t_idx].append(b_idx)

    def greedy_randomized(seed):
        random.seed(seed)
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        
        selected_blocks = []
        
        while is_uncovered.any():
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            candidates = np.where(gain == max_gain)[0]
            b_idx = random.choice(candidates.tolist())
            
            selected_blocks.append(b_idx)
            
            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    for other_b_idx in triple_blocks[t_idx]:
                        gain[other_b_idx] -= 1
                        
        return selected_blocks

    def reduce_cover(selected_blocks):
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1
        
        active = [True] * len(selected_blocks)
        
        changed = True
        while changed:
            changed = False
            for i, b_idx in enumerate(selected_blocks):
                if not active[i]:
                    continue
                redundant = True
                for t_idx in block_triples[b_idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
        
        return [b_idx for i, b_idx in enumerate(selected_blocks) if active[i]]

    def local_search_sa(cover, time_limit):
        """Simulated Annealing local search to minimize block count."""
        if not cover:
            return cover
        
        current_cover = set(cover)
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in current_cover:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1
        
        P = 1000 
        def get_cost(cover_set, counts):
            uncovered = np.sum(counts == 0)
            return len(cover_set) + P * uncovered

        current_cost = get_cost(current_cover, cover_count)
        best_cover = set(current_cover)
        best_cost = current_cost
        
        T = 10.0
        T_min = 0.1
        cooling = 0.995
        
        # Use a local random instance to avoid interfering with global state if needed, 
        # but re-seeding is fine.
        rng = random.Random(12345)
        
        while (time.time() - start_time) < time_limit and T > T_min:
            move_type = rng.random()
            
            if len(current_cover) == 0:
                # Must add
                b_new = rng.choice(range(num_blocks))
                if b_new in current_cover:
                    continue
                # Apply Add
                current_cover.add(b_new)
                delta_uncovered = 0
                for t_idx in block_triples[b_new]:
                    if cover_count[t_idx] == 0:
                        delta_uncovered -= 1
                    cover_count[t_idx] += 1
                delta_cost = 1 + P * delta_uncovered
                new_cost = current_cost + delta_cost
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_cover = set(current_cover)
                else:
                    prob = np.exp(-delta_cost / T)
                    if rng.random() < prob:
                        current_cost = new_cost
                    else:
                        # Revert
                        current_cover.remove(b_new)
                        for t_idx in block_triples[b_new]:
                            cover_count[t_idx] -= 1
            else:
                if move_type < 0.3:
                    # Remove
                    b_old = rng.choice(list(current_cover))
                    # Apply Remove
                    current_cover.remove(b_old)
                    delta_uncovered = 0
                    for t_idx in block_triples[b_old]:
                        cover_count[t_idx] -= 1
                        if cover_count[t_idx] == 0:
                            delta_uncovered += 1
                    delta_cost = -1 + P * delta_uncovered
                    new_cost = current_cost + delta_cost
                    
                    if new_cost < current_cost:
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_cover = set(current_cover)
                    else:
                        prob = np.exp(-delta_cost / T)
                        if rng.random() < prob:
                            current_cost = new_cost
                        else:
                            # Revert
                            current_cover.add(b_old)
                            for t_idx in block_triples[b_old]:
                                cover_count[t_idx] += 1
                elif move_type < 0.6:
                    # Add
                    b_new = rng.choice(range(num_blocks))
                    if b_new in current_cover:
                        continue
                    # Apply Add
                    current_cover.add(b_new)
                    delta_uncovered = 0
                    for t_idx in block_triples[b_new]:
                        if cover_count[t_idx] == 0:
                            delta_uncovered -= 1
                        cover_count[t_idx] += 1
                    delta_cost = 1 + P * delta_uncovered
                    new_cost = current_cost + delta_cost
                    
                    if new_cost < current_cost:
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_cover = set(current_cover)
                    else:
                        prob = np.exp(-delta_cost / T)
                        if rng.random() < prob:
                            current_cost = new_cost
                        else:
                            # Revert
                            current_cover.remove(b_new)
                            for t_idx in block_triples[b_new]:
                                cover_count[t_idx] -= 1
                else:
                    # Swap
                    b_old = rng.choice(list(current_cover))
                    b_new = rng.choice(range(num_blocks))
                    if b_new == b_old or b_new in current_cover:
                        continue
                    
                    # Apply Swap
                    current_cover.remove(b_old)
                    current_cover.add(b_new)
                    
                    delta_uncovered = 0
                    # Remove b_old
                    for t_idx in block_triples[b_old]:
                        cover_count[t_idx] -= 1
                        if cover_count[t_idx] == 0:
                            delta_uncovered += 1
                    # Add b_new
                    for t_idx in block_triples[b_new]:
                        if cover_count[t_idx] == 0:
                            delta_uncovered -= 1
                        cover_count[t_idx] += 1
                        
                    delta_cost = 0 + P * delta_uncovered # Size change is 0
                    new_cost = current_cost + delta_cost
                    
                    if new_cost < current_cost:
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_cover = set(current_cover)
                    else:
                        prob = np.exp(-delta_cost / T)
                        if rng.random() < prob:
                            current_cost = new_cost
                        else:
                            # Revert
                            current_cover.remove(b_new)
                            current_cover.add(b_old)
                            # Reverse counts
                            for t_idx in block_triples[b_new]:
                                cover_count[t_idx] -= 1
                            for t_idx in block_triples[b_old]:
                                cover_count[t_idx] += 1
            
            T *= cooling
            
        return list(best_cover)

    best_cover = None
    seed = 0
    while True:
        if time.time() - start_time > timeout:
            break
        
        # Run greedy
        cover = greedy_randomized(seed)
        # Reduce
        cover = reduce_cover(cover)
        
        if best_cover is None or len(cover) < len(best_cover):
            best_cover = list(cover)
        
        seed += 1
    
    # Final local search on best cover
    if best_cover is not None:
        time_remaining = timeout - (time.time() - start_time)
        if time_remaining > 5:
            # Use SA local search
            best_cover = local_search_sa(best_cover, start_time + timeout - 2)
            # Final reduction
            best_cover = reduce_cover(best_cover)
            
    if best_cover is None:
        cover = greedy_randomized(0)
        return [list(blocks[b_idx]) for b_idx in cover]
        
    return [list(blocks[b_idx]) for b_idx in best_cover]

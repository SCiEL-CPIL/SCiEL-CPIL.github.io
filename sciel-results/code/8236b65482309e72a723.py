import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using fast greedy multi-start
    followed by genetic algorithm refinement.
    """
    deadline = time.time() + 112.0  # Safety margin below 150s

    # Precompute t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute for each t-subset, the list of k-subsets that contain it
    tsub_to_ksubs = {}
    for ts in all_tsubs:
        ts_set = set(ts)
        remaining = [p for p in range(v) if p not in ts_set]
        ksubs = []
        for extra in itertools.combinations(remaining, k - t):
            block = tuple(sorted(ts + tuple(extra)))
            ksubs.append(block)
        tsub_to_ksubs[ts] = ksubs
    
    # Precompute for each k-subset, the list of t-subset indices it covers
    ksub_to_tsub_indices = {}
    
    def get_ksub_indices(block):
        block = tuple(block)
        if block not in ksub_to_tsub_indices:
            idxs = [tsub_to_idx[comb] for comb in itertools.combinations(block, t)]
            ksub_to_tsub_indices[block] = np.array(idxs, dtype=np.int32)
        return ksub_to_tsub_indices[block]
    
    def greedy_once(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = rng.choice(uncovered_indices.tolist())
            S = all_tsubs[start_idx]
            
            # Get precomputed k-subsets containing S
            candidates = tsub_to_ksubs[S]
            
            # Evaluate gain for each candidate
            best_block = None
            best_gain = -1
            
            # Shuffle for tie-breaking
            candidates = list(candidates)
            rng.shuffle(candidates)
            
            for B in candidates:
                idxs = get_ksub_indices(B)
                gain = np.count_nonzero(uncovered[idxs])
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = B
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Update uncovered mask
            idxs = get_ksub_indices(best_block)
            uncovered[idxs] = False
        
        return blocks

    def remove_redundant(blocks, max_iter=5):
        if not blocks:
            return blocks
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Count coverage of each t-subset
            cover_count = np.zeros(n_tsubs, dtype=np.int32)
            block_idx_map = []
            for b in blocks:
                idxs = get_ksub_indices(tuple(b))
                block_idx_map.append(idxs)
                cover_count[idxs] += 1
            
            valid_blocks = []
            removed_any = False
            
            for i, b in enumerate(blocks):
                if time.time() > deadline:
                    return blocks
                
                idxs = block_idx_map[i]
                if np.all(cover_count[idxs] > 1):
                    cover_count[idxs] -= 1
                    removed_any = True
                else:
                    valid_blocks.append(b)
            
            if not removed_any:
                break
            blocks = valid_blocks
        
        return blocks

    def validate_covering(blocks):
        if not blocks:
            return n_tsubs == 0
        covered = np.zeros(n_tsubs, dtype=bool)
        for b in blocks:
            idxs = get_ksub_indices(tuple(b))
            covered[idxs] = True
        return np.all(covered)

    def mutate_block(individual, rng):
        """
        Try to replace one block with a better one, reducing total block count
        if possible, or maintaining validity.
        """
        if len(individual) == 0:
            return individual[:]
        
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for b in individual:
            idxs = get_ksub_indices(tuple(b))
            cover_count[idxs] += 1
        
        # Find blocks that are "critical" (cover some t-subset exactly once)
        critical_indices = []
        for i, b in enumerate(individual):
            idxs = get_ksub_indices(tuple(b))
            if np.any(cover_count[idxs] == 1):
                critical_indices.append(i)
        
        if not critical_indices:
            return individual[:]
        
        # Pick a random critical block to try replacing
        replace_idx = rng.choice(critical_indices)
        old_block = individual[replace_idx]
        old_idxs = get_ksub_indices(tuple(old_block))
        old_critical = np.where(cover_count[old_idxs] == 1)[0]
        
        if len(old_critical) == 0:
            return individual[:]
        
        # Try to find a new block that covers all critical t-subsets
        # and is not the same as the old block
        target_tsubs = set(all_tsubs[i] for i in old_critical)
        
        candidates = tsub_to_ksubs[old_critical[0]]
        candidates = list(candidates)
        rng.shuffle(candidates)
        
        best_new_block = None
        best_score = -1
        
        for B in candidates:
            if B == tuple(old_block):
                continue
            idxs = set(get_ksub_indices(B))
            # Check if it covers all critical t-subsets
            if not all(i in idxs for i in old_critical):
                continue
            # Score: how many total t-subsets it covers
            score = len(idxs)
            if score > best_score:
                best_score = score
                best_new_block = B
        
        if best_new_block is None:
            return individual[:]
        
        new_individual = individual[:]
        new_individual[replace_idx] = best_new_block
        
        if validate_covering(new_individual):
            return new_individual
        else:
            return individual[:]

    def crossover(parent1, parent2, rng):
        """
        Crossover: merge two parents' blocks, then reduce to valid covering.
        """
        combined = list(parent1) + list(parent2)
        
        # Remove duplicates
        seen = set()
        unique_combined = []
        for b in combined:
            key = tuple(b)
            if key not in seen:
                seen.add(key)
                unique_combined.append(b)
        
        # Try to reduce
        reduced = remove_redundant(unique_combined, max_iter=3)
        
        if validate_covering(reduced):
            return reduced
        else:
            # Fallback: use the smaller parent
            if len(parent1) <= len(parent2):
                return parent1[:]
            else:
                return parent2[:]

    def ga_evolve(population, num_generations, deadline):
        best_individual = min(population, key=len)
        best_len = len(best_individual)
        rng = random.Random(98765)
        
        for gen in range(num_generations):
            if time.time() > deadline:
                break
            
            new_population = [best_individual[:]]
            
            while len(new_population) < len(population):
                if time.time() > deadline:
                    break
                
                # Select parents with tournament
                p1_idx = rng.randint(0, len(population) - 1)
                p2_idx = rng.randint(0, len(population) - 1)
                p1 = population[p1_idx]
                p2 = population[p2_idx]
                
                child = crossover(p1, p2, rng)
                
                # Mutation with some probability
                if rng.random() < 0.4:
                    child = mutate_block(child, rng)
                
                new_population.append(child)
            
            # Update best
            for ind in new_population:
                if len(ind) < best_len:
                    best_len = len(ind)
                    best_individual = ind[:]
            
            population = new_population
        
        return best_individual

    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multi-start greedy
    num_starts = 0
    greedy_deadline = deadline - 25.0  # Reserve 25s for GA
    
    while time.time() < greedy_deadline:
        seed = num_starts
        blocks = greedy_once(seed)
        
        if not blocks:
            num_starts += 1
            continue
            
        reduced_blocks = remove_redundant(blocks)
        
        if len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks[:]
        
        num_starts += 1
    
    if best_blocks is None:
        best_blocks = greedy_once(0)
        best_blocks = remove_redundant(best_blocks)
        best_len = len(best_blocks)
    
    # Phase 2: Genetic algorithm refinement
    if time.time() < deadline - 3.0:
        # Build population with diversity
        population = [best_blocks[:]]
        
        for _ in range(4):
            if time.time() > deadline - 8.0:
                break
            seed = num_starts
            num_starts += 1
            blocks = greedy_once(seed)
            if blocks:
                reduced = remove_redundant(blocks)
                population.append(reduced)
        
        # Run GA
        ga_deadline = deadline - 3.0
        ga_best = ga_evolve(population, num_generations=50, deadline=ga_deadline)
        
        if len(ga_best) < best_len:
            best_len = len(ga_best)
            best_blocks = ga_best[:]
    
    return best_blocks

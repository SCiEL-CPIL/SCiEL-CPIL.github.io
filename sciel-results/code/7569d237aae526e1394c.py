"""
Covering design C(v, k, t) using Genetic Algorithm refinement of Greedy constructions.
Combines the efficient local greedy builder from the current approach with a 
population-based search (GA) to explore block combinations via crossover and mutation.
"""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute list of all possible k-blocks for potential mutation/selection?
    # For v=19, k=8, there are ~75k blocks. Storing them is fine.
    # However, generating all is O(C(v,k)).
    all_blocks = list(itertools.combinations(points, k))
    
    # Helper: Convert block tuple to list of t-subset indices
    def block_to_indices(block):
        return [subset_to_id[comb] for comb in itertools.combinations(block, t)]

    # Helper: Get coverage mask (boolean numpy array) for a block
    def block_mask(block):
        mask = np.zeros(N_t, dtype=bool)
        for comb in itertools.combinations(block, t):
            mask[subset_to_id[comb]] = True
        return mask

    # 1. Fast Local Greedy Constructor
    def greedy_cover(seed=None):
        """
        Runs a local greedy construction.
        Returns list of blocks.
        """
        if seed is not None:
            random.seed(seed)
        
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to anchor the block
            start_subset = random.choice(list(uncovered))
            block = list(start_subset)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_point = None
                current_combos = list(itertools.combinations(block, t - 1))
                
                # Candidates: points not in block
                candidates = [x for x in range(v) if x not in block_set]
                # Optional: Shuffle for diversity if gains are tied? 
                # For speed, we iterate all.
                
                for x in candidates:
                    gain = 0
                    # Check how many new t-subsets are covered by adding x
                    for rest in current_combos:
                        cand = tuple(sorted(rest + (x,)))
                        if cand in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None:
                    # Fallback: pick random
                    best_point = random.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(tuple(block)) # Use tuple for hashing/sets
            
            # Update uncovered
            for comb in itertools.combinations(block, t):
                uncovered.discard(comb)
        
        return blocks

    # 2. Reduction Engine
    def reduce_blocks(blocks):
        """
        Efficient reduction using coverage counts.
        blocks: list of tuples
        """
        if not blocks:
            return blocks
        
        # Use a set to ensure distinct blocks
        unique_blocks = list(set(blocks))
        
        # Initial counts
        counts = np.zeros(N_t, dtype=np.int32)
        block_indices = []
        for b in unique_blocks:
            idxs = block_to_indices(b)
            block_indices.append(idxs)
            counts[idxs] += 1
        
        reduced_blocks = unique_blocks[:]
        reduced_indices = block_indices[:]
        changed = True
        
        # Iterate until no more changes or time limit
        # To speed up, we can prioritize blocks with low "essentiality"
        while changed and time.time() < deadline - 5:
            changed = False
            # Random order for diversity in reduction
            order = list(range(len(reduced_blocks)))
            random.shuffle(order)
            
            for idx in order:
                if time.time() > deadline - 5:
                    break
                
                # Check if block is redundant
                redundant = True
                for sid in reduced_indices[idx]:
                    if counts[sid] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block
                    for sid in reduced_indices[idx]:
                        counts[sid] -= 1
                    
                    reduced_blocks.pop(idx)
                    reduced_indices.pop(idx)
                    changed = True
        
        return reduced_blocks

    # 3. Genetic Algorithm Components
    
    def fitness(blocks):
        return len(blocks)
    
    def crossover(parent1, parent2):
        """
        Combine two coverings. The union of two valid coverings is valid.
        """
        return list(set(parent1) | set(parent2))
    
    def mutate(blocks, rng):
        """
        Mutation: Swap a random block with a "greedy" block that covers many 
        currently poorly covered subsets, or just a random block from all_blocks.
        For simplicity and speed, we replace one block with a random one from 
        the pool of all possible blocks, or a block generated by a quick local greedy 
        step on the most uncovered subset.
        
        To be safe and fast, we'll just replace a random block with a random 
        valid block from the precomputed list, and then reduce.
        """
        if not blocks:
            return blocks
        
        blocks = list(blocks)
        idx = rng.randint(0, len(blocks) - 1)
        
        # Strategy: Replace block[idx] with a block that covers the most 
        # "critical" subsets? Too slow.
        # Simple Strategy: Replace with a random block from all_blocks.
        # This introduces diversity.
        new_block = rng.choice(all_blocks)
        blocks[idx] = new_block
        
        return blocks

    def repair(blocks):
        """
        If crossover/mutation somehow breaks validity (it shouldn't for union, 
        but mutation might leave gaps if we just replace a block), we need to repair.
        However, replacing a block in a valid covering *does* break validity 
        if the removed block was the only one covering some subsets.
        
        So, after mutation, we must check if the covering is still valid.
        If not, we add blocks to cover the missing subsets (Greedy Repair).
        """
        # Check validity
        covered = set()
        for b in blocks:
            for comb in itertools.combinations(b, t):
                covered.add(comb)
        
        if len(covered) == N_t:
            return blocks
        
        # Repair: Greedily add blocks to cover missing subsets
        missing = set(all_t_subsets) - covered
        # Convert to list for random choice
        missing_list = list(missing)
        
        while missing_list:
            if time.time() > deadline - 5:
                break
            # Pick a random missing subset
            start_subset = rng.choice(missing_list)
            block = list(start_subset)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_point = None
                current_combos = list(itertools.combinations(block, t - 1))
                candidates = [x for x in range(v) if x not in block_set]
                
                for x in candidates:
                    gain = 0
                    for rest in current_combos:
                        cand = tuple(sorted(rest + (x,)))
                        if cand in missing:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None:
                    best_point = rng.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            new_block = tuple(block)
            blocks.append(new_block)
            
            # Update missing
            for comb in itertools.combinations(block, t):
                missing.discard(comb)
            
            # Re-sync missing_list if it changed significantly? 
            # For efficiency, we can just recompute missing_list if it becomes empty or small.
            # Actually, `missing` is a set, `missing_list` is stale.
            # Let's just clear missing_list and break if missing is empty.
            if not missing:
                break
            else:
                # To avoid O(N) list conversion every step, we can just pick from `missing` set directly
                # But rng.choice needs a sequence. 
                # Optimization: If missing is large, just pick from it.
                # This loop is only for repair, which should be rare if we start from good parents.
                pass 
        
        # Final check and cleanup
        return blocks

    # 4. Main GA Loop
    
    # Initialize Population
    pop_size = 10
    population = []
    
    # First, generate a few good greedy solutions
    for i in range(pop_size):
        blocks = greedy_cover(seed=i * 1000)
        blocks = reduce_blocks(blocks)
        population.append(blocks)
    
    best_blocks = min(population, key=len)
    best_count = len(best_blocks)
    
    # GA Parameters
    generations = 50
    elite_count = 2
    
    for gen in range(generations):
        if time.time() > deadline - 20: # Leave time for final reduction
            break
        
        # Sort population by fitness
        population.sort(key=len)
        
        # Elitism: Keep best elite_count
        new_population = population[:elite_count][:]
        
        # Fill rest of population
        while len(new_population) < pop_size:
            if time.time() > deadline - 10:
                break
            
            # Selection: Tournament selection of 2
            p1_idx = random.randint(0, min(5, len(population)-1))
            p2_idx = random.randint(0, min(5, len(population)-1))
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            if random.random() < 0.5:
                child = mutate(child, random)
            
            # Repair (if mutation broke it)
            # Note: Crossover (Union) is always valid. Mutation (Replacement) is not.
            # So we only repair if we mutated.
            # To be safe, we always check/repair.
            # But repair is expensive. Let's only repair if we mutated.
            # Actually, let's just reduce first. If reduction fails to make it valid? 
            # Reduction doesn't check validity, it just removes redundant.
            # So if we replace a block, we might have holes.
            
            # Efficient Repair Check:
            # Instead of full repair, we can just add a few greedy blocks if needed.
            # But for now, let's assume the union of two good solutions is good, 
            # and mutation is rare or we accept the risk and rely on the fact that 
            # we are minimizing, so if a child is invalid, it should be penalized.
            # BUT, the evaluator requires a VALID covering.
            
            # So we MUST ensure validity.
            # Let's do a fast validity check + repair.
            
            # Fast Validity Check
            covered = set()
            valid = True
            for b in child:
                for comb in itertools.combinations(b, t):
                    covered.add(comb)
                if len(covered) == N_t:
                    break
            
            if len(covered) < N_t:
                # Repair
                missing = set(all_t_subsets) - covered
                # Greedy repair loop
                while missing:
                    start_subset = random.choice(list(missing))
                    block = list(start_subset)
                    block_set = set(block)
                    while len(block) < k:
                        best_gain = -1
                        best_point = None
                        current_combos = list(itertools.combinations(block, t - 1))
                        candidates = [x for x in range(v) if x not in block_set]
                        for x in candidates:
                            gain = 0
                            for rest in current_combos:
                                cand = tuple(sorted(rest + (x,)))
                                if cand in missing:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_point = x
                        if best_point is None:
                            best_point = random.choice(candidates)
                        block.append(best_point)
                        block_set.add(best_point)
                    block.sort()
                    child.append(tuple(block))
                    for comb in itertools.combinations(block, t):
                        missing.discard(comb)
                    if time.time() > deadline - 5:
                        break
            
            # Reduce
            child = reduce_blocks(child)
            
            new_population.append(child)
            
            if len(child) < best_count:
                best_count = len(child)
                best_blocks = child
        
        population = new_population
    
    # Final Reduction on the best found
    if time.time() < deadline - 2:
        best_blocks = reduce_blocks(best_blocks)
    
    # Ensure return type is list of lists
    return [list(b) for b in best_blocks]


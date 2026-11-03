import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Precompute all t-subsets and map to indices
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Precompute for each k-subset, which triples it covers (as a bitmask or list)
    # For efficiency, we'll use a set of triple indices per block
    def block_triple_indices(block):
        """Return set of triple indices covered by a block."""
        return frozenset(triple_to_idx[tri] for tri in itertools.combinations(block, t))
    
    # Greedy construction with weighted scoring
    def greedy_construct(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        uncovered = set(range(n_triples))
        blocks = []
        
        while uncovered:
            # Weighted greedy: prioritize blocks that cover rare triples
            # Compute weight for each triple (how many blocks could cover it)
            # For simplicity, use uniform weight but pick pivot randomly
            pivot_idx = rng.choice(list(uncovered))
            pivot_tri = all_triples[pivot_idx]
            pivot_points = set(pivot_tri)
            other_points = [p for p in range(v) if p not in pivot_points]
            
            best_block = None
            best_gain = -1
            best_block_set = None
            
            # Sample a subset of candidates if too many
            n_candidates = len(other_points)
            k_other = k - t
            
            if n_candidates <= 20:
                candidates = list(itertools.combinations(other_points, k_other))
            else:
                # Sample candidates
                n_samples = min(500, len(list(itertools.combinations(other_points, k_other))))
                candidates = []
                for _ in range(n_samples):
                    combo = tuple(sorted(rng.sample(other_points, k_other)))
                    candidates.append(combo)
            
            for combo in candidates:
                block = tuple(sorted(pivot_points | set(combo)))
                gain = 0
                for tri in itertools.combinations(block, t):
                    idx = triple_to_idx[tri]
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_block_set = block_triple_indices(block)
            
            if best_block is not None:
                blocks.append(list(best_block))
                for idx in best_block_set:
                    if idx in uncovered:
                        uncovered.discard(idx)
        
        return blocks
    
    # Remove redundant blocks
    def remove_redundant(blocks):
        blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                remaining = blocks[:i] + blocks[i+1:]
                # Check if remaining covers all triples
                covered = set()
                for b in remaining:
                    covered.update(block_triple_indices(b))
                if len(covered) == n_triples:
                    blocks.pop(i)
                    changed = True
                    break
        return blocks
    
    # Local search: try to replace blocks with better ones
    def local_search(blocks, max_time=15):
        if time.time() > start_time + max_time:
            return blocks
        
        blocks = list(blocks)
        improved = True
        while improved and time.time() < start_time + max_time:
            improved = False
            n = len(blocks)
            
            # Try removing each block
            for i in range(n):
                if time.time() > start_time + max_time:
                    return blocks
                remaining = blocks[:i] + blocks[i+1:]
                covered = set()
                for b in remaining:
                    covered.update(block_triple_indices(b))
                if len(covered) == n_triples:
                    blocks.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping a block
            for i in range(len(blocks)):
                if time.time() > start_time + max_time:
                    return blocks
                
                # Find triples only covered by block i
                only_triples = set()
                for tri in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[tri]
                    count = 0
                    for j, b in enumerate(blocks):
                        if j != i:
                            if idx in block_triple_indices(b):
                                count += 1
                                break
                    if count == 0:
                        only_triples.add(idx)
                
                if not only_triples:
                    blocks.pop(i)
                    improved = True
                    break
                
                # Must include all points from only_triples
                must_include = set()
                for idx in only_triples:
                    must_include.update(all_triples[idx])
                
                if len(must_include) > k:
                    continue
                
                remaining_points = list(set(range(v)) - must_include)
                need = k - len(must_include)
                
                if need < 0 or need > len(remaining_points):
                    continue
                
                # Sample candidate replacements
                best_candidate = None
                best_score = -1
                
                n_samples = min(200, 1)
                for _ in range(n_samples):
                    if need == 0:
                        combo = ()
                    else:
                        combo = tuple(sorted(random.sample(remaining_points, need)))
                    candidate = tuple(sorted(must_include | set(combo)))
                    
                    # Score: how many triples does this cover that aren't covered by others?
                    cand_set = block_triple_indices(candidate)
                    gain = 0
                    for idx in cand_set:
                        covered_by_other = False
                        for j, b in enumerate(blocks):
                            if j != i:
                                if idx in block_triple_indices(b):
                                    covered_by_other = True
                                    break
                        if not covered_by_other:
                            gain += 1
                    
                    if gain > best_score:
                        best_score = gain
                        best_candidate = candidate
                
                if best_candidate is not None and best_score >= len(only_triples):
                    blocks[i] = list(best_candidate)
                    improved = True
                    break
        
        return remove_redundant(blocks)
    
    # Genetic algorithm
    def genetic_algorithm(pop_size=20, max_gen=50, max_time=60):
        if time.time() > start_time + max_time:
            return None
        
        rng = random.Random(int(time.time()))
        
        # Initialize population with greedy solutions
        population = []
        for i in range(pop_size):
            blocks = greedy_construct(seed=rng.randint(0, 2**31))
            blocks = remove_redundant(blocks)
            population.append(blocks)
        
        best_blocks = min(population, key=len)
        best_len = len(best_blocks)
        
        for gen in range(max_gen):
            if time.time() > start_time + max_time:
                break
            
            # Evaluate fitness
            def fitness(blocks):
                if not blocks:
                    return 0
                covered = set()
                for b in blocks:
                    covered.update(block_triple_indices(b))
                uncovered = n_triples - len(covered)
                # Penalize uncovered triples heavily, then block count
                return uncovered * 1000 + len(blocks)
            
            # Sort by fitness
            population.sort(key=fitness)
            
            if len(population[0]) < best_len:
                best_len = len(population[0])
                best_blocks = population[0]
            
            # Selection: keep top half
            elite = population[:pop_size//2]
            
            # Generate new population
            new_population = list(elite)
            
            while len(new_population) < pop_size:
                if time.time() > start_time + max_time:
                    break
                
                # Crossover: combine blocks from two parents
                p1 = rng.choice(elite)
                p2 = rng.choice(elite)
                
                # Take random subset of blocks from each parent
                n1 = rng.randint(1, len(p1))
                n2 = rng.randint(1, len(p2))
                child = rng.sample(p1, min(n1, len(p1))) + rng.sample(p2, min(n2, len(p2)))
                
                # Remove duplicates
                seen = set()
                unique_child = []
                for b in child:
                    key = tuple(sorted(b))
                    if key not in seen:
                        seen.add(key)
                        unique_child.append(list(b))
                child = unique_child
                
                # Mutation: randomly replace or add blocks
                if child and rng.random() < 0.3:
                    idx = rng.randint(0, len(child) - 1)
                    # Replace with a random block containing a random triple
                    tri = rng.choice(all_triples)
                    other = [p for p in range(v) if p not in tri]
                    combo = tuple(sorted(rng.sample(other, k - t)))
                    child[idx] = list(sorted(set(tri) | set(combo)))
                
                # If child doesn't cover all, complete it greedily
                covered = set()
                for b in child:
                    covered.update(block_triple_indices(b))
                uncovered = set(range(n_triples)) - covered
                
                if uncovered:
                    # Complete greedily
                    while uncovered:
                        pivot_idx = rng.choice(list(uncovered))
                        pivot_tri = all_triples[pivot_idx]
                        pivot_points = set(pivot_tri)
                        other_points = [p for p in range(v) if p not in pivot_points]
                        
                        best_block = None
                        best_gain = -1
                        for _ in range(50):
                            combo = tuple(sorted(rng.sample(other_points, k - t)))
                            block = tuple(sorted(pivot_points | set(combo)))
                            gain = 0
                            for tri2 in itertools.combinations(block, t):
                                idx2 = triple_to_idx[tri2]
                                if idx2 in uncovered:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_block = block
                        
                        if best_block:
                            child.append(list(best_block))
                            for tri2 in itertools.combinations(best_block, t):
                                idx2 = triple_to_idx[tri2]
                                if idx2 in uncovered:
                                    uncovered.discard(idx2)
                
                # Remove redundant
                child = remove_redundant(child)
                
                new_population.append(child)
            
            population = new_population
        
        return best_blocks
    
    # Main search loop
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multiple greedy runs
    n_greedy = 5
    for i in range(n_greedy):
        if time.time() > deadline - 30:
            break
        blocks = greedy_construct(seed=i * 12345 + 67890)
        blocks = remove_redundant(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
    
    # Phase 2: Local search on best
    if best_blocks and time.time() < deadline - 20:
        improved = local_search(best_blocks, max_time=15)
        if len(improved) < best_len:
            best_len = len(improved)
            best_blocks = improved
    
    # Phase 3: Genetic algorithm
    if time.time() < deadline - 10:
        ga_result = genetic_algorithm(pop_size=15, max_gen=30, max_time=50)
        if ga_result and len(ga_result) < best_len:
            best_len = len(ga_result)
            best_blocks = ga_result
    
    # Phase 4: Final local search
    if best_blocks and time.time() < deadline - 5:
        improved = local_search(best_blocks, max_time=8)
        if len(improved) < best_len:
            best_len = len(improved)
            best_blocks = improved
    
    # Fallback: deterministic greedy
    if best_blocks is None:
        uncovered = set(range(n_triples))
        blocks = []
        while uncovered:
            pivot_idx = min(uncovered)
            pivot_tri = all_triples[pivot_idx]
            pivot_points = set(pivot_tri)
            other_points = [p for p in range(v) if p not in pivot_points]
            best_block = None
            best_gain = -1
            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_points | set(combo)))
                gain = 0
                for tri in itertools.combinations(block, t):
                    idx = triple_to_idx[tri]
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            blocks.append(list(best_block))
            for tri in itertools.combinations(best_block, t):
                idx = triple_to_idx[tri]
                if idx in uncovered:
                    uncovered.discard(idx)
        best_blocks = remove_redundant(blocks)
    
    return best_blocks

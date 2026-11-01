import itertools
import time
import random
from collections import defaultdict

def _coverage_score(blocks, v, k, t):
    """Calculate coverage score: sum of 1/count for uncovered triples.
    Lower is better. 0 means complete coverage."""
    if not blocks:
        return float('inf')
    
    # Count how many blocks cover each triple
    triple_counts = defaultdict(int)
    for block in blocks:
        for tri in itertools.combinations(block, t):
            triple_counts[tri] += 1
    
    # Check coverage
    total_triples = 0
    uncovered_sum = 0.0
    for i in range(v - t + 1):
        for j in range(i + 1, v - t + 2):
            for l in range(j + 1, v - t + 3):
                tri = (i, j, l)
                total_triples += 1
                if tri not in triple_counts:
                    uncovered_sum += 1.0
                else:
                    # Smooth surrogate: soft penalty for low coverage counts
                    # Use 1/count^0.5 to encourage redundancy while penalizing uncovered
                    uncovered_sum += 1.0 / triple_counts[tri]**0.5
    
    return uncovered_sum

def _greedy_fast(v, k, t, seed):
    """Fast greedy construction using precomputed triple indices."""
    rng = random.Random(seed)
    
    # Precompute all triples and map to indices
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # For each point, precompute which triples it covers
    point_triples = defaultdict(list)
    for tri in all_triples:
        for pt in tri:
            point_triples[pt].append(triple_to_idx[tri])
    
    uncovered = [True] * n_triples
    blocks = []
    
    while True:
        # Find uncovered triples
        uncovered_indices = [i for i, u in enumerate(uncovered) if u]
        if not uncovered_indices:
            break
        
        # Pick random uncovered triple
        start_idx = rng.choice(uncovered_indices)
        start_triple = all_triples[start_idx]
        block = list(start_triple)
        block_set = set(block)
        
        # Greedily add points
        while len(block) < k:
            best_gain = -1
            candidates = []
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Calculate gain: how many uncovered triples are covered by adding x
                # A triple is covered if it contains x and t-1 points from block
                gain = 0
                for comb in itertools.combinations(block, t - 1):
                    tri = tuple(sorted(list(comb) + [x]))
                    idx = triple_to_idx.get(tri)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [x]
                elif gain == best_gain:
                    candidates.append(x)
            
            if candidates:
                next_point = rng.choice(candidates)
            else:
                next_point = rng.choice([x for x in range(v) if x not in block_set])
            
            block.append(next_point)
            block_set.add(next_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            uncovered[idx] = False
    
    return blocks

def _local_search(blocks, v, k, t, deadline, seed):
    """Local search: try to delete blocks or swap points to improve coverage."""
    rng = random.Random(seed)
    blocks = [list(b) for b in blocks]
    
    # Precompute triple indices
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    def compute_coverage(block_list):
        """Return (uncovered_count, triple_counts array)"""
        triple_counts = [0] * n_triples
        for block in block_list:
            for tri in itertools.combinations(block, t):
                idx = triple_to_idx[tri]
                triple_counts[idx] += 1
        uncovered = sum(1 for c in triple_counts if c == 0)
        return uncovered, triple_counts
    
    current_uncovered, _ = compute_coverage(blocks)
    best_blocks = [list(b) for b in blocks]
    best_uncovered = current_uncovered
    
    while time.time() < deadline:
        # Try block deletion
        if len(blocks) > 1:
            for i in range(len(blocks)):
                trial = blocks[:i] + blocks[i+1:]
                unc, _ = compute_coverage(trial)
                if unc < best_uncovered:
                    best_uncovered = unc
                    best_blocks = [list(b) for b in trial]
                    blocks = trial
                    break
            else:
                # Try point swaps
                for i in range(len(blocks)):
                    for j in range(len(blocks[i])):
                        old_pt = blocks[i][j]
                        for new_pt in range(v):
                            if new_pt in blocks[i]:
                                continue
                            blocks[i][j] = new_pt
                            unc, _ = compute_coverage(blocks)
                            if unc < best_uncovered:
                                best_uncovered = unc
                                best_blocks = [list(b) for b in blocks]
                            blocks[i][j] = old_pt
                        if time.time() > deadline - 5:
                            return best_blocks
        
        # If no improvement, break
        if best_uncovered == 0:
            break
    
    return best_blocks

def _genetic_algorithm(v, k, t, deadline, seed):
    """Genetic algorithm evolving populations of block sets."""
    rng = random.Random(seed)
    
    # Precompute triple indices
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    def fitness(block_set):
        """Lower is better. Returns coverage score."""
        if not block_set:
            return float('inf')
        triple_counts = [0] * n_triples
        for block in block_set:
            for tri in itertools.combinations(block, t):
                idx = triple_to_idx[tri]
                triple_counts[idx] += 1
        score = 0.0
        for c in triple_counts:
            if c == 0:
                score += 1.0
            else:
                score += 1.0 / c**0.5
        return score
    
    def crossover(parent1, parent2):
        """Crossover: combine blocks from both parents, keep best subset."""
        combined = list(parent1) + list(parent2)
        # Remove duplicates
        seen = set()
        unique = []
        for b in combined:
            key = tuple(b)
            if key not in seen:
                seen.add(key)
                unique.append(b)
        
        # Sort by coverage contribution and keep top
        def block_coverage(b):
            return sum(1 for tri in itertools.combinations(b, t) if triple_to_idx.get(tri) is not None)
        
        unique.sort(key=block_coverage, reverse=True)
        
        # Keep a reasonable number
        max_blocks = min(len(unique), 80)
        return unique[:max_blocks]
    
    def mutate(block_set):
        """Mutation: swap a point in a random block, or add/remove a block."""
        block_set = [list(b) for b in block_set]
        
        if not block_set:
            # Create a random block
            block = sorted(rng.sample(range(v), k))
            return [block]
        
        op = rng.random()
        
        if op < 0.5 and len(block_set) > 1:
            # Remove a random block
            idx = rng.randrange(len(block_set))
            block_set.pop(idx)
        elif op < 0.75:
            # Swap a point in a random block
            block_idx = rng.randrange(len(block_set))
            pos = rng.randrange(k)
            old_pt = block_set[block_idx][pos]
            new_pt = rng.randrange(v)
            if new_pt not in block_set[block_idx]:
                block_set[block_idx][pos] = new_pt
                block_set[block_idx].sort()
        else:
            # Add a random block
            if len(block_set) < 100:
                block = sorted(rng.sample(range(v), k))
                block_set.append(block)
        
        return block_set
    
    # Initialize population
    pop_size = 20
    population = []
    
    # First few are greedy solutions
    for i in range(min(5, pop_size)):
        population.append(_greedy_fast(v, k, t, seed + i * 1000))
    
    # Rest are random or mutated
    while len(population) < pop_size:
        if population:
            base = rng.choice(population)
            population.append(mutate(base))
        else:
            # Random initial
            blocks = []
            for _ in range(70):
                blocks.append(sorted(rng.sample(range(v), k)))
            population.append(blocks)
    
    best_blocks = None
    best_fitness = float('inf')
    
    for block_set in population:
        f = fitness(block_set)
        if f < best_fitness:
            best_fitness = f
            best_blocks = [list(b) for b in block_set]
    
    # Evolution
    generation = 0
    while time.time() < deadline - 10:
        generation += 1
        
        # Evaluate fitness
        scored = [(fitness(bs), bs) for bs in population]
        scored.sort(key=lambda x: x[0])
        
        # Update best
        if scored[0][0] < best_fitness:
            best_fitness = scored[0][0]
            best_blocks = [list(b) for b in scored[0][1]]
        
        # If we found a complete covering, we can still try to minimize block count
        if best_fitness < 0.01:
            # Try to reduce block count via local search
            candidate = _local_search(best_blocks, v, k, t, min(time.time() + 5, deadline - 5), seed + generation)
            if len(candidate) < len(best_blocks):
                best_blocks = candidate
        
        # Create next generation
        next_pop = []
        
        # Elitism: keep top 20%
        elite_count = max(2, pop_size // 5)
        for i in range(elite_count):
            next_pop.append([list(b) for b in scored[i][1]])
        
        # Fill rest with crossover and mutation
        while len(next_pop) < pop_size:
            if time.time() > deadline - 10:
                break
            
            p1 = rng.choice(scored[:pop_size//2])[1]
            p2 = rng.choice(scored[:pop_size//2])[1]
            
            child = crossover(p1, p2)
            child = mutate(child)
            next_pop.append(child)
        
        population = next_pop
    
    return best_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Quick greedy restarts to get a good starting point
    for seed in range(5):
        if time.time() > deadline - 60:
            break
        blocks = _greedy_fast(v, k, t, seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: single greedy
        best_blocks = _greedy_fast(v, k, t, 0)
        best_len = len(best_blocks)
    
    # Phase 2: Local search on best greedy
    if time.time() < deadline - 30:
        ls_blocks = _local_search(best_blocks, v, k, t, min(time.time() + 20, deadline - 30), 42)
        if len(ls_blocks) < best_len:
            best_len = len(ls_blocks)
            best_blocks = ls_blocks
    
    # Phase 3: Genetic algorithm if time remains
    if time.time() < deadline - 15:
        ga_blocks = _genetic_algorithm(v, k, t, deadline - 5, 12345)
        if ga_blocks and len(ga_blocks) < best_len:
            best_len = len(ga_blocks)
            best_blocks = ga_blocks
    
    # Final local search polish
    if time.time() < deadline - 3 and best_blocks:
        final = _local_search(best_blocks, v, k, t, deadline - 1, 99999)
        if len(final) < best_len:
            best_blocks = final
    
    return best_blocks

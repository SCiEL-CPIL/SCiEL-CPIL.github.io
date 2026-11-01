"""Covering design C(24,7,3): genetic algorithm with greedy seeding and local search."""
import random
import time
import itertools
import math
from collections import defaultdict
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# EVOLVE-BLOCK-START


def _block_triples(block, t):
    """Yield all t-subsets of a block."""
    return itertools.combinations(block, t)


def _is_valid(blocks, v, k, t):
    """Check that blocks form a valid covering of all t-subsets."""
    covered = set()
    for b in blocks:
        if len(b) != k or len(set(b)) != k:
            return False
        for x in b:
            if x < 0 or x >= v:
                return False
        covered.update(itertools.combinations(b, t))
    total = 1
    for i in range(v - t + 1, v + 1):
        total *= i
    for i in range(1, t + 1):
        total //= i
    return len(covered) == total


def _local_search_remove_redundant(blocks, v, k, t, all_tsub_set):
    """Iteratively remove blocks that are fully redundant (all their t-subsets covered by others).
    Uses numpy matrix for efficiency. Returns the reduced list of blocks."""
    if not blocks:
        return blocks

    n = len(blocks)
    if n == 0:
        return blocks

    # Build numpy incidence matrix: rows=blocks, cols=triples
    triple_list = list(all_tsub_set)
    triple_to_idx = {tr: i for i, tr in enumerate(triple_list)}
    n_triples = len(triple_list)

    # Build incidence matrix
    mat = np.zeros((n, n_triples), dtype=np.uint8)
    for i, b in enumerate(blocks):
        for tr in itertools.combinations(b, t):
            mat[i, triple_to_idx[tr]] = 1

    col_sums = mat.sum(axis=0)

    changed = True
    while changed:
        changed = False
        to_remove = []
        for i in range(len(blocks)):
            # Check if every triple in this block is covered by at least one other block
            row = mat[i]
            # For each triple this block covers, check if another block also covers it
            covered_triples = np.where(row > 0)[0]
            if len(covered_triples) == 0:
                continue
            # col_sums > 1 means at least 2 blocks cover that triple
            # But we need: for each triple in block i, col_sums[triple] > 1
            # i.e., no triple is uniquely covered by block i
            if np.all(col_sums[covered_triples] > 1):
                to_remove.append(i)

        if to_remove:
            remove_set = set(to_remove)
            keep = [i for i in range(len(blocks)) if i not in remove_set]
            blocks = [blocks[i] for i in keep]
            # Rebuild matrix
            n = len(blocks)
            mat = np.zeros((n, n_triples), dtype=np.uint8)
            for i, b in enumerate(blocks):
                for tr in itertools.combinations(b, t):
                    mat[i, triple_to_idx[tr]] = 1
            col_sums = mat.sum(axis=0)
            changed = True

    return blocks


def _try_swap_reduce(blocks, v, k, t, rng, all_tsub_set):
    """Try to swap a point in a block to create redundancy, then delete.
    Returns new blocks list if an improving swap is found, else None."""
    if len(blocks) < 2:
        return None

    n = len(blocks)
    triple_list = list(all_tsub_set)
    triple_to_idx = {tr: i for i, tr in enumerate(triple_list)}
    n_triples = len(triple_list)

    # Build numpy incidence matrix
    mat = np.zeros((n, n_triples), dtype=np.uint8)
    for i, b in enumerate(blocks):
        for tr in itertools.combinations(b, t):
            mat[i, triple_to_idx[tr]] = 1
    col_sums = mat.sum(axis=0)

    # Try random point swaps
    for _ in range(200):
        bi = rng.randrange(n)
        block = list(blocks[bi])
        p_idx = rng.randrange(k)
        p = block[p_idx]
        q = rng.randrange(v)
        if q == p or q in block:
            continue
        new_block = block[:p_idx] + [q] + block[p_idx + 1:]
        new_block.sort()

        # Check: does the new block still cover all t-subsets that were ONLY covered by old block?
        old_triples = set(itertools.combinations(block, t))
        new_triples = set(itertools.combinations(new_block, t))
        lost = old_triples - new_triples
        can_swap = True
        for tr in lost:
            if col_sums[triple_to_idx[tr]] < 2:
                can_swap = False
                break
        if not can_swap:
            continue

        # Perform swap and try to delete a block
        new_blocks = list(blocks)
        new_blocks[bi] = new_block

        # Recompute cover counts
        new_mat = np.zeros((n, n_triples), dtype=np.uint8)
        for i, b in enumerate(new_blocks):
            for tr in itertools.combinations(b, t):
                new_mat[i, triple_to_idx[tr]] = 1
        new_col_sums = new_mat.sum(axis=0)

        # Try to find a removable block
        for i in range(len(new_blocks)):
            row = new_mat[i]
            covered_triples = np.where(row > 0)[0]
            if len(covered_triples) == 0:
                continue
            if np.all(new_col_sums[covered_triples] > 1):
                return new_blocks[:i] + new_blocks[i + 1:]

    return None


def _fitness(blocks, v, k, t, all_tsub_set):
    """Compute fitness: number of blocks + penalty for uncovered triples.
    Lower is better."""
    if not blocks:
        return float('inf')
    
    covered = set()
    for b in blocks:
        for tr in itertools.combinations(b, t):
            covered.add(tr)
    
    uncovered_count = len(all_tsub_set) - len(covered)
    # Penalize uncovered triples heavily to ensure validity is prioritized
    return len(blocks) + 100.0 * uncovered_count


def _crossover(parent1, parent2, v, k, t, rng):
    """Crossover: take blocks from both parents, prefer blocks that cover rare triples."""
    # Compute coverage contribution of each block (how many unique triples it covers)
    def block_contribution(block, all_blocks):
        """Count triples in this block not covered by other blocks."""
        other_covered = set()
        for b in all_blocks:
            if b is not block:
                for tr in itertools.combinations(b, t):
                    other_covered.add(tr)
        unique = 0
        for tr in itertools.combinations(block, t):
            if tr not in other_covered:
                unique += 1
        return unique
    
    # Score each block by its unique contribution
    p1_scores = [(block_contribution(b, parent1), i, b) for i, b in enumerate(parent1)]
    p2_scores = [(block_contribution(b, parent2), i, b) for i, b in enumerate(parent2)]
    
    # Sort by contribution (descending)
    p1_scores.sort(key=lambda x: -x[0])
    p2_scores.sort(key=lambda x: -x[0])
    
    # Select top blocks from each parent
    child_blocks = []
    seen = set()
    
    # Alternate between parents, taking highest-contribution blocks first
    i1, i2 = 0, 0
    while (i1 < len(p1_scores) or i2 < len(p2_scores)) and len(child_blocks) < max(len(parent1), len(parent2)):
        # Pick from parent1 if it has higher remaining contribution
        if i1 < len(p1_scores) and (i2 >= len(p2_scores) or p1_scores[i1][0] >= p2_scores[i2][0]):
            block = p1_scores[i1][2]
            i1 += 1
        else:
            block = p2_scores[i2][2]
            i2 += 1
        
        block_key = tuple(sorted(block))
        if block_key not in seen:
            seen.add(block_key)
            child_blocks.append(list(block))
    
    return child_blocks


def _mutate(blocks, v, k, t, rng, all_tsub_set):
    """Mutation: randomly modify blocks by swapping points or adding/removing blocks."""
    new_blocks = [list(b) for b in blocks]
    
    # With some probability, add a new random block
    if rng.random() < 0.1 and new_blocks:
        new_block = sorted(rng.sample(range(v), k))
        new_blocks.append(new_block)
    
    # With some probability, remove a random block (if it makes the solution still valid or close)
    elif rng.random() < 0.05 and len(new_blocks) > 1:
        idx = rng.randrange(len(new_blocks))
        new_blocks.pop(idx)
    
    # Mutate individual blocks: swap points
    for _ in range(rng.randrange(1, 4)):
        if not new_blocks:
            break
        bi = rng.randrange(len(new_blocks))
        block = list(new_blocks[bi])
        p_idx = rng.randrange(k)
        p = block[p_idx]
        q = rng.randrange(v)
        if q != p and q not in block:
            block[p_idx] = q
            block.sort()
            new_blocks[bi] = block
    
    return new_blocks


def _genetic_search(initial_blocks, v, k, t, all_tsub_set, rng, deadline, pop_size=8, generations=50):
    """Genetic algorithm to evolve a population of block sets."""
    # Initialize population
    population = [list(initial_blocks)]
    for _ in range(pop_size - 1):
        # Create variants of initial blocks
        variant = _mutate(initial_blocks, v, k, t, rng, all_tsub_set)
        variant = _local_search_remove_redundant(variant, v, k, t, all_tsub_set)
        population.append(variant)
    
    best_individual = min(population, key=lambda b: (len(b), _fitness(b, v, k, t, all_tsub_set)))
    best_fitness = _fitness(best_individual, v, k, t, all_tsub_set)
    
    for gen in range(generations):
        if time.time() > deadline:
            break
        
        # Evaluate fitness
        scored_pop = [( _fitness(ind, v, k, t, all_tsub_set), ind) for ind in population]
        scored_pop.sort(key=lambda x: x[0])
        
        # Elitism: keep top 2
        new_population = [scored_pop[0][1], scored_pop[1][1]] if len(scored_pop) >= 2 else [scored_pop[0][1]]
        
        # Selection and reproduction
        while len(new_population) < pop_size:
            # Tournament selection
            def select():
                candidates = rng.sample(scored_pop, min(3, len(scored_pop)))
                return min(candidates, key=lambda x: x[0])[1]
            
            p1 = select()
            p2 = select()
            
            # Crossover
            child = _crossover(p1, p2, v, k, t, rng)
            
            # Mutation
            child = _mutate(child, v, k, t, rng, all_tsub_set)
            
            # Local search
            child = _local_search_remove_redundant(child, v, k, t, all_tsub_set)
            
            # Try swap-reduce (multiple attempts)
            for _attempt in range(3):
                if len(child) > 1:
                    swapped = _try_swap_reduce(child, v, k, t, rng, all_tsub_set)
                    if swapped is not None:
                        child = swapped
                    else:
                        break
            
            new_population.append(child)
        
        population = new_population
        
        # Track best
        current_best = min(population, key=lambda b: (len(b), _fitness(b, v, k, t, all_tsub_set)))
        current_fitness = _fitness(current_best, v, k, t, all_tsub_set)
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_individual = current_best
    
    return best_individual


def _randomized_greedy(v, k, t, rng, deadline, all_tsub_set):
    """Build a covering using randomized greedy with randomization in point selection."""
    all_points = list(range(v))
    uncovered = set(all_tsub_set)
    blocks = []

    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset to start the block
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)

        while len(block) < k:
            # For each candidate point, compute how many new t-subsets it covers
            best_gain = -1
            best_candidates = []
            block_set = set(block)
            for x in all_points:
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    tr = tuple(sorted(rest + (x,)))
                    if tr in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(x)
            if not best_candidates:
                # No point helps; pick random
                remaining = [x for x in all_points if x not in block_set]
                if not remaining:
                    break
                block.append(rng.choice(remaining))
            else:
                # Randomize among top candidates
                block.append(rng.choice(best_candidates))
            block_set = set(block)

        block.sort()
        blocks.append(block)
        # Update uncovered
        for tr in itertools.combinations(block, t):
            uncovered.discard(tr)

    return blocks


def _deterministic_greedy(v, k, t, all_tsub_set):
    """Deterministic greedy baseline."""
    uncovered = set(all_tsub_set)
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        block_set = set(block)
        while len(block) < k:
            best_gain = -1
            best_x = None
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    tr = tuple(sorted(rest + (x,)))
                    if tr in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_set = set(block)
        block.sort()
        blocks.append(block)
        for tr in itertools.combinations(block, t):
            uncovered.discard(tr)
    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using genetic algorithm with greedy seeding."""
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    all_tsub_set = set(itertools.combinations(range(v), t))

    # Phase 1: Deterministic greedy baseline
    best_blocks = _deterministic_greedy(v, k, t, all_tsub_set)
    best_count = len(best_blocks)

    # Phase 2: Randomized greedy to build initial population seeds
    seed_blocks_list = []
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        rng = random.Random(seed * 7919 + 1)
        blocks = _randomized_greedy(v, k, t, rng, deadline - 30, all_tsub_set)
        if blocks:
            blocks = _local_search_remove_redundant(blocks, v, k, t, all_tsub_set)
            seed_blocks_list.append(blocks)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks

    # Phase 3: Genetic algorithm on the best seed
    if time.time() < deadline - 5:
        rng_ga = random.Random(42)
        ga_result = _genetic_search(best_blocks, v, k, t, all_tsub_set, rng_ga, deadline - 5, 
                                     pop_size=6, generations=30)
        if ga_result and len(ga_result) < best_count:
            best_blocks = ga_result
            best_count = len(best_blocks)

    # Phase 4: Additional local search refinement
    current = list(best_blocks)
    no_improve_count = 0
    max_no_improve = 30

    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False

        # Try local search removal (batch)
        reduced = _local_search_remove_redundant(current, v, k, t, all_tsub_set)
        if len(reduced) < len(current):
            current = reduced
            improved = True
            no_improve_count = 0
            continue

        # Try point swaps to enable deletion
        rng_ls = random.Random(int(time.time() * 1000000))
        new_blocks = _try_swap_reduce(current, v, k, t, rng_ls, all_tsub_set)
        if new_blocks is not None:
            current = new_blocks
            improved = True
            no_improve_count = 0
            continue

        no_improve_count += 1

    # Final validation
    if not _is_valid(current, v, k, t):
        return best_blocks

    return current
# EVOLVE-BLOCK-END

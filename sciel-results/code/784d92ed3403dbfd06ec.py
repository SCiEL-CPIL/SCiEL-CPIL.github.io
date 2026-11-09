import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a genetic algorithm with
    greedy initialization, crossover, mutation (block swap), and redundancy removal.
    """
    deadline = time.time() + 130.0  # Safety margin (150s limit)

    # Precompute t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute t-subset indices for a block (cached)
    block_tsub_cache = {}

    def get_block_tsubs(block):
        if block not in block_tsub_cache:
            idxs = np.array([tsub_to_idx[comb] for comb in itertools.combinations(block, t)], dtype=np.int32)
            block_tsub_cache[block] = idxs
        return block_tsub_cache[block]

    def compute_cover_count(blocks):
        """Compute how many times each t-subset is covered."""
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            idxs = get_block_tsubs(b)
            cover_count[idxs] += 1
        return cover_count

    def is_valid_covering(blocks):
        """Check if blocks cover all t-subsets."""
        if not blocks:
            return n_tsubs == 0
        cover_count = compute_cover_count(blocks)
        return np.all(cover_count > 0)

    def remove_redundant(blocks, max_iters=10):
        """Remove redundant blocks iteratively."""
        if not blocks:
            return blocks

        current_blocks = list(blocks)
        changed = True
        iters = 0
        while changed and iters < max_iters:
            if time.time() > deadline - 2.0:
                break
            changed = False
            iters += 1

            cover_count = compute_cover_count(current_blocks)

            new_blocks = []
            for i in range(len(current_blocks)):
                if time.time() > deadline - 2.0:
                    break
                b = current_blocks[i]
                idxs = get_block_tsubs(b)
                if np.all(cover_count[idxs] > 1):
                    cover_count[idxs] -= 1
                    changed = True
                else:
                    new_blocks.append(b)

            current_blocks = new_blocks

        return current_blocks

    def greedy_once(seed):
        """Fast greedy construction."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []

        while np.any(uncovered):
            if time.time() > deadline - 5.0:
                break

            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = rng.choice(uncovered_indices)
            S = all_tsubs[start_idx]
            S_set = set(S)

            remaining_points = [p for p in range(v) if p not in S_set]
            candidates = []
            for extra in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(S + tuple(extra)))
                candidates.append(block)

            best_block = None
            best_gain = -1

            rng.shuffle(candidates)

            for B in candidates:
                idxs = get_block_tsubs(B)
                gain = np.sum(uncovered[idxs])
                if gain > best_gain:
                    best_gain = gain
                    best_block = B

            if best_block is None:
                break

            blocks.append(best_block)
            idxs = get_block_tsubs(best_block)
            uncovered[idxs] = False

        return blocks

    def crossover(parent1, parent2, rng):
        """Combine two parent coverings: take union, then reduce."""
        if time.time() > deadline - 3.0:
            return None

        # Union of both sets
        combined = list(parent1) + list(parent2)

        # Remove duplicates
        seen = set()
        unique_blocks = []
        for b in combined:
            if b not in seen:
                seen.add(b)
                unique_blocks.append(b)

        # Reduce
        reduced = remove_redundant(unique_blocks, max_iters=5)

        if is_valid_covering(reduced):
            return reduced
        return None

    def mutate(blocks, rng, num_swaps=3):
        """Swap some blocks for better alternatives."""
        if time.time() > deadline - 3.0:
            return None

        current_blocks = list(blocks)
        cover_count = compute_cover_count(current_blocks)

        for _ in range(num_swaps):
            if time.time() > deadline - 3.0:
                break

            # Pick a random block to try replacing
            i = rng.randrange(len(current_blocks))
            old_block = current_blocks[i]
            old_tsubs = get_block_tsubs(old_block)

            # Find t-subsets that are only covered by this block (critical)
            critical = np.where(cover_count[old_tsubs] == 1)[0]
            if len(critical) > 0:
                # This block is critical; try to find a replacement that covers all critical tsubs
                # and ideally more
                critical_tsubs = [all_tsubs[old_tsubs[c]] for c in critical]

                # Try random k-subsets that cover all critical tsubs
                best_new_block = None
                best_gain = -1

                for attempt in range(50):
                    # Pick a random critical tsub and build a block around it
                    crit_tsub = rng.choice(critical_tsubs)
                    crit_set = set(crit_tsub)
                    remaining = [p for p in range(v) if p not in crit_set]
                    extra = rng.sample(remaining, k - t)
                    new_block = tuple(sorted(crit_tsub + tuple(extra)))

                    # Check if it covers all critical tsubs
                    new_tsubs = get_block_tsubs(new_block)
                    new_set = set(new_tsubs)
                    if all(old_tsubs[c] in new_set for c in critical):
                        # Compute gain: how many new tsubs it covers
                        gain = np.sum((cover_count[new_tsubs] == 0).astype(np.int32))
                        if gain > best_gain:
                            best_gain = gain
                            best_new_block = new_block

                if best_new_block is not None and best_new_block not in current_blocks:
                    # Replace
                    cover_count[old_tsubs] -= 1
                    new_tsubs = get_block_tsubs(best_new_block)
                    cover_count[new_tsubs] += 1
                    current_blocks[i] = best_new_block
            else:
                # Block is redundant or non-critical; try to replace with something better
                # Just try a random replacement
                for attempt in range(20):
                    crit_tsub = rng.choice(all_tsubs)
                    crit_set = set(crit_tsub)
                    remaining = [p for p in range(v) if p not in crit_set]
                    extra = rng.sample(remaining, k - t)
                    new_block = tuple(sorted(crit_tsub + tuple(extra)))

                    new_tsubs = get_block_tsubs(new_block)
                    gain = np.sum((cover_count[new_tsubs] == 0).astype(np.int32))
                    if gain > 0 and new_block not in current_blocks:
                        cover_count[old_tsubs] -= 1
                        cover_count[new_tsubs] += 1
                        current_blocks[i] = new_block
                        break

        # Verify validity
        if is_valid_covering(current_blocks):
            return remove_redundant(current_blocks, max_iters=3)
        return None

    # Main genetic algorithm
    rng = random.Random(42)

    # Initialize population with greedy solutions
    population = []
    pop_size = 8
    for i in range(pop_size):
        if time.time() > deadline - 10.0:
            break
        blocks = greedy_once(i)
        if blocks:
            reduced = remove_redundant(blocks, max_iters=5)
            if is_valid_covering(reduced):
                population.append(reduced)

    if not population:
        # Fallback
        blocks = greedy_once(0)
        return remove_redundant(blocks, max_iters=5)

    best_solution = min(population, key=len)
    best_len = len(best_solution)

    # Genetic algorithm loop
    generation = 0
    while time.time() < deadline - 10.0:
        generation += 1

        # Select parents (tournament selection)
        offspring = []
        for _ in range(pop_size // 2):
            if time.time() > deadline - 5.0:
                break

            # Tournament selection
            p1_idx = rng.randrange(len(population))
            p2_idx = rng.randrange(len(population))

            # Crossover
            child = crossover(population[p1_idx], population[p2_idx], rng)
            if child is None:
                continue

            # Mutation
            mutated = mutate(child, rng, num_swaps=2)
            if mutated is not None:
                offspring.append(mutated)
            else:
                offspring.append(child)

        # Update population: keep best, replace worst
        if offspring:
            # Combine population and offspring
            all_solutions = population + offspring

            # Sort by length
            all_solutions.sort(key=len)

            # Keep top pop_size
            population = all_solutions[:pop_size]

            # Update best
            if len(population[0]) < best_len:
                best_len = len(population[0])
                best_solution = population[0][:]

    return best_solution


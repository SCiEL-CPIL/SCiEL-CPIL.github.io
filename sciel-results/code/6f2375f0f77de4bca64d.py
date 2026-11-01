import time
import itertools
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    num_triples = len(all_t_subsets)
    triple_set = set(all_t_subsets)

    # Precompute: for each pair (a,b), list of triples containing that pair
    # This speeds up gain calculation
    pair_to_triples = {}
    for triple in all_t_subsets:
        for i in range(t):
            for j in range(i+1, t):
                pair = (triple[i], triple[j])
                if pair not in pair_to_triples:
                    pair_to_triples[pair] = []
                pair_to_triples[pair].append(triple)

    def random_greedy_cover(rng):
        """Fast randomized greedy covering."""
        uncovered = set(all_t_subsets)
        blocks = []

        while uncovered:
            # Pick a random uncovered triple
            seed_triple = rng.choice(tuple(uncovered))
            block = list(seed_triple)
            block_set = set(block)

            while len(block) < k:
                # Compute gain for each candidate point
                best_gain = -1
                best_points = []

                # Precompute pairs in current block
                block_pairs = list(itertools.combinations(block, t - 1))

                for x in all_points:
                    if x in block_set:
                        continue

                    gain = 0
                    for pair in block_pairs:
                        # Count triples (x, pair[0], pair[1]) in uncovered
                        for triple in pair_to_triples.get(pair, []):
                            if x in triple and triple in uncovered:
                                gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)

                if best_points:
                    chosen = rng.choice(best_points)
                else:
                    available = [x for x in all_points if x not in block_set]
                    if available:
                        chosen = rng.choice(available)
                    else:
                        break

                block.append(chosen)
                block_set.add(chosen)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for triple in itertools.combinations(block, t):
                uncovered.discard(triple)

        return blocks

    def build_coverage_map(blocks):
        """Build triple -> list of block indices covering it."""
        coverage = {triple: [] for triple in all_t_subsets}
        for i, block in enumerate(blocks):
            for triple in itertools.combinations(block, t):
                coverage[triple].append(i)
        return coverage

    def local_search_remove_redundant(blocks, max_iter=50):
        """Remove redundant blocks iteratively."""
        if not blocks:
            return blocks

        coverage = build_coverage_map(blocks)
        changed = True
        iterations = 0

        while changed and iterations < max_iter:
            changed = False
            to_remove = []

            for i, block in enumerate(blocks):
                is_redundant = True
                for triple in itertools.combinations(block, t):
                    if len(coverage[triple]) == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.append(i)

            if to_remove:
                remove_set = set(to_remove)
                blocks = [b for i, b in enumerate(blocks) if i not in remove_set]
                coverage = build_coverage_map(blocks)
                changed = True
                iterations += 1

        return blocks

    def local_search_swap(blocks, time_limit=5.0):
        """Try to replace blocks with better ones to reduce total count."""
        if not blocks:
            return blocks

        start = time.time()
        coverage = build_coverage_map(blocks)

        # Try removing each block and see if we can cover the gaps with fewer new blocks
        for i in range(len(blocks)):
            if time.time() - start > time_limit:
                break

            # Temporarily remove block i
            remaining = [b for j, b in enumerate(blocks) if j != i]
            if not remaining:
                continue

            # Find triples only covered by block i
            only_covered = []
            for triple in itertools.combinations(blocks[i], t):
                if len(coverage[triple]) == 1:
                    only_covered.append(triple)

            if not only_covered:
                # Block i is fully redundant - remove it
                blocks = remaining
                coverage = build_coverage_map(blocks)
                continue

            # Try to cover only_covered with 1 new block
            if len(only_covered) <= 35:  # C(7,3) = 35
                # Find a block that covers all of only_covered
                # For each triple in only_covered, find candidate points
                # This is complex; skip for now and just try random swaps
                pass

            # Try random swap: replace block i with a random block
            # and check if coverage is maintained
            if random.random() < 0.3:
                new_block = list(random.sample(all_points, k))
                new_block.sort()
                # Check if new block covers all only_covered triples
                new_block_set = set(new_block)
                all_covered = True
                for triple in only_covered:
                    if not all(x in new_block_set for x in triple):
                        all_covered = False
                        break

                if all_covered:
                    blocks[i] = new_block
                    coverage = build_coverage_map(blocks)

        return blocks

    def genetic_algorithm(initial_blocks, population_size=10, generations=20, time_limit=30.0):
        """GA to evolve block sets."""
        start = time.time()
        rng = random.Random()

        # Fitness: number of blocks, penalized if not a valid covering
        def fitness(blocks):
            if not blocks:
                return (1000, 1000)  # Very bad
            # Check coverage
            covered = set()
            for block in blocks:
                for triple in itertools.combinations(block, t):
                    covered.add(triple)
            uncovered_count = num_triples - len(covered)
            return (uncovered_count, len(blocks))

        # Initialize population
        population = []
        for i in range(population_size):
            if i == 0:
                population.append(list(initial_blocks))
            else:
                # Mutate the initial blocks
                mutated = list(initial_blocks)
                for _ in range(rng.randint(1, 5)):
                    if mutated and rng.random() < 0.5:
                        # Remove a random block
                        idx = rng.randrange(len(mutated))
                        mutated.pop(idx)
                    elif rng.random() < 0.5:
                        # Add a random block
                        new_block = list(rng.sample(all_points, k))
                        new_block.sort()
                        mutated.append(new_block)
                population.append(mutated)

        best = min(population, key=fitness)
        best_fit = fitness(best)

        for gen in range(generations):
            if time.time() - start > time_limit:
                break

            # Evaluate fitness
            fits = [fitness(ind) for ind in population]

            # Selection: tournament
            def select():
                a, b = rng.sample(range(len(population)), 2)
                return population[a] if fits[a] < fits[b] else population[b]

            # Create new generation
            new_population = []
            for _ in range(population_size):
                parent1 = select()
                parent2 = select()

                # Crossover: take union of blocks, then trim
                child_blocks = list(set(tuple(b) for b in parent1) | set(tuple(b) for b in parent2))
                child_blocks = [list(b) for b in child_blocks]

                # Mutation
                for _ in range(rng.randint(0, 3)):
                    if child_blocks and rng.random() < 0.4:
                        idx = rng.randrange(len(child_blocks))
                        child_blocks.pop(idx)
                    if rng.random() < 0.3:
                        new_block = list(rng.sample(all_points, k))
                        new_block.sort()
                        child_blocks.append(new_block)

                # Local cleanup: remove redundant
                child_blocks = local_search_remove_redundant(child_blocks, max_iter=10)

                new_population.append(child_blocks)

            # Replace worst individuals
            for i in range(len(new_population)):
                if fitness(new_population[i]) < fitness(population[i]):
                    population[i] = new_population[i]

            # Update best
            for ind in population:
                f = fitness(ind)
                if f < best_fit:
                    best_fit = f
                    best = ind

        return best

    # Main loop
    best_blocks = None
    best_len = float('inf')
    rng = random.Random()

    # Phase 1: Randomized greedy restarts with local search
    greedy_count = 0
    while time.time() < deadline:
        blocks = random_greedy_cover(rng)
        blocks = local_search_remove_redundant(blocks)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

        greedy_count += 1

        # After some greedy runs, try GA on the best solution
        if greedy_count % 10 == 0 and best_blocks is not None and time.time() < deadline - 40:
            ga_blocks = genetic_algorithm(best_blocks, population_size=8, generations=15, time_limit=25.0)
            ga_blocks = local_search_remove_redundant(ga_blocks)
            if len(ga_blocks) < best_len:
                best_len = len(ga_blocks)
                best_blocks = ga_blocks

        # Also try swap-based local search occasionally
        if greedy_count % 5 == 0 and best_blocks is not None and time.time() < deadline - 10:
            swap_blocks = local_search_swap(list(best_blocks), time_limit=3.0)
            swap_blocks = local_search_remove_redundant(swap_blocks)
            if len(swap_blocks) < best_len:
                best_len = len(swap_blocks)
                best_blocks = swap_blocks

    return best_blocks if best_blocks is not None else []

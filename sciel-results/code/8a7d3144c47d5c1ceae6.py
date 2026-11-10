import random
import time
import numpy as np

N = 256

def _build_aps():
    """Build list of all 4-term arithmetic progressions in 0..255."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _build_pos_to_aps(aps):
    """For each position, store (ap_index) for all APs it belongs to."""
    pos_to_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_to_aps[a].append(idx)
        pos_to_aps[b].append(idx)
        pos_to_aps[c].append(idx)
        pos_to_aps[d].append(idx)
    return pos_to_aps

def _score_full(aps, colors):
    """Count monochromatic 4-APs (full recount)."""
    total = 0
    for a, b, c, d in aps:
        if colors[a] == colors[b] == colors[c] == colors[d]:
            total += 1
    return total

def _local_search_incremental(aps, pos_to_aps, colors, max_passes=200, deadline=None):
    """
    Greedy local search using incremental delta scoring.
    When flipping position i, only APs containing i need re-evaluation.
    """
    colors = list(colors)
    n = len(colors)
    num_aps = len(aps)

    # Track which APs are monochromatic
    mono_flags = np.zeros(num_aps, dtype=np.int8)
    for idx in range(num_aps):
        a, b, c, d = aps[idx]
        mono_flags[idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
    current_score = int(mono_flags.sum())

    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            old = colors[i]
            new = 1 - old
            colors[i] = new

            # Compute delta: for each AP containing i, check if mono status changed
            delta = 0
            for ap_idx in pos_to_aps[i]:
                a, b, c, d = aps[ap_idx]
                was_mono = mono_flags[ap_idx]
                now_mono = (colors[a] == colors[b] == colors[c] == colors[d])
                if was_mono and not now_mono:
                    delta -= 1
                elif (not was_mono) and now_mono:
                    delta += 1
                mono_flags[ap_idx] = 1 if now_mono else 0

            if delta < 0:
                current_score += delta
                improved = True
            else:
                # Revert
                colors[i] = old
                for ap_idx in pos_to_aps[i]:
                    a, b, c, d = aps[ap_idx]
                    mono_flags[ap_idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0

        if not improved:
            break

    return colors, current_score

def _ga_population_init(pop_size):
    """Initialize diverse GA population."""
    population = []

    # 1. Random
    for _ in range(pop_size // 4):
        population.append([random.randint(0, 1) for _ in range(N)])

    # 2. Block patterns
    for block_size in [2, 3, 4, 5, 8, 16, 32]:
        population.append([(i // block_size) % 2 for i in range(N)])

    # 3. Thue-Morse
    population.append([bin(i).count('1') % 2 for i in range(N)])

    # 4. Striped patterns
    for period in [3, 5, 7, 11, 13]:
        pattern = []
        for i in range(N):
            pos_in_period = i % period
            pattern.append(0 if pos_in_period < period // 2 else 1)
        population.append(pattern)

    # 5. Quadratic residue-like
    for p in [3, 5, 7, 11]:
        pop = []
        for i in range(N):
            is_res = any((x * x) % p == i % p for x in range(p))
            pop.append(1 if is_res else 0)
        population.append(pop)

    # Pad with random
    while len(population) < pop_size:
        population.append([random.randint(0, 1) for _ in range(N)])

    return population[:pop_size]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    random.seed(42)

    aps = _build_aps()
    pos_to_aps = _build_pos_to_aps(aps)

    best_score = float('inf')
    best_pop = None

    pop_size = 60
    population = _ga_population_init(pop_size)

    # Evaluate initial population
    scores = [_score_full(aps, pop) for pop in population]
    order = sorted(range(len(population)), key=lambda i: scores[i])
    population = [population[i] for i in order]
    scores = [scores[i] for i in order]

    best_score = scores[0]
    best_pop = population[0][:]

    # Local search on top candidates
    for i in range(min(10, len(population))):
        if time.time() > deadline - 30:
            break
        improved_pop, s = _local_search_incremental(
            aps, pos_to_aps, population[i], max_passes=100, deadline=deadline - 30
        )
        if s < best_score:
            best_score = s
            best_pop = improved_pop[:]

    # Main GA loop with incremental local search
    mutation_rate = 0.08
    elitism = 3
    gen = 0

    while time.time() < deadline - 20:
        gen += 1

        def tournament_select():
            idx1 = random.randint(0, len(population) - 1)
            idx2 = random.randint(0, len(population) - 1)
            return population[idx1][:] if scores[idx1] <= scores[idx2] else population[idx2][:]

        new_population = []

        # Elitism
        for i in range(elitism):
            new_population.append(population[i][:])

        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 15:
                break
            p1 = tournament_select()
            p2 = tournament_select()

            # Uniform crossover
            child = [p1[i] if random.random() < 0.5 else p2[i] for i in range(N)]

            # Mutation
            for i in range(N):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]

            new_population.append(child)

        # Evaluate new population
        new_scores = [_score_full(aps, pop) for pop in new_population]

        # Combine and select
        order = sorted(range(len(new_population)), key=lambda i: new_scores[i])
        population = [new_population[i] for i in order]
        scores = [new_scores[i] for i in order]

        if scores[0] < best_score:
            best_score = scores[0]
            best_pop = population[0][:]

        # Local search on top few each generation
        for i in range(min(5, len(population))):
            if time.time() > deadline - 10:
                break
            improved_pop, s = _local_search_incremental(
                aps, pos_to_aps, population[i], max_passes=50, deadline=deadline - 10
            )
            if s < best_score:
                best_score = s
                best_pop = improved_pop[:]

        # Periodically inject diversity
        if gen % 10 == 0 and time.time() < deadline - 25:
            for _ in range(5):
                new_ind = [random.randint(0, 1) for _ in range(N)]
                s = _score_full(aps, new_ind)
                if s < scores[-1]:
                    population[-1] = new_ind
                    scores[-1] = s

    # Final local search on best individual
    final_pop, final_score = _local_search_incremental(
        aps, pos_to_aps, best_pop, max_passes=500, deadline=deadline
    )
    if final_score < best_score:
        best_pop = final_pop
        best_score = final_score

    return best_pop

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END

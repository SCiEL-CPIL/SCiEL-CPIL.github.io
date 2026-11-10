import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))

    total_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)

    # Precompute neighbors for fast delta calculation
    neighbors = [[] for _ in range(N)]
    for i1, i2, i3, i4 in aps:
        neighbors[i1].append((i2, i3, i4))
        neighbors[i2].append((i1, i3, i4))
        neighbors[i3].append((i1, i2, i4))
        neighbors[i4].append((i1, i2, i3))

    # Convert to numpy arrays for faster access
    neighbors_np = []
    for i in range(N):
        if neighbors[i]:
            neighbors_np.append(np.array(neighbors[i], dtype=np.int32))
        else:
            neighbors_np.append(np.empty((0, 3), dtype=np.int32))

    def count_mono(colors):
        """Count monochromatic APs using numpy vectorization"""
        c0 = colors[aps_arr[:, 0]]
        c1 = colors[aps_arr[:, 1]]
        c2 = colors[aps_arr[:, 2]]
        c3 = colors[aps_arr[:, 3]]
        return int(np.sum((c0 == c1) & (c0 == c2) & (c0 == c3)))

    def fast_sa(colors, T_start, T_end, max_iter, deadline):
        """Fast simulated annealing with neighbor-based delta."""
        colors = colors.copy()
        current_score = count_mono(colors)
        best_score = current_score
        best_colors = colors.copy()

        T = T_start
        cooling = (T_end / T_start) ** (1.0 / max(1, max_iter))

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            i = random.randint(0, N - 1)
            old_c = colors[i]
            new_c = 1 - old_c

            others = neighbors_np[i]
            if len(others) == 0:
                delta = 0
            else:
                cols = colors[others]
                c0, c1, c2 = cols[:, 0], cols[:, 1], cols[:, 2]
                all_eq = (c0 == c1) & (c1 == c2)
                val = c0
                was_mono = all_eq & (val == old_c)
                becomes_mono = all_eq & (val == new_c)
                delta = int(np.sum(becomes_mono) - np.sum(was_mono))

            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-9)):
                colors[i] = new_c
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()

            T *= cooling

        return best_colors, best_score

    def generate_lfsr_initialization(seed_val):
        """Generate LFSR-based structured coloring."""
        rng = np.random.RandomState(seed_val)
        # Use LFSR-like pattern: XOR-shift based initialization
        colors = np.zeros(N, dtype=np.int32)
        state = seed_val
        for i in range(N):
            # Simple LFSR step (Galois form)
            state ^= (state << 13) & 0xFFFFFFFF
            state ^= (state >> 17)
            state ^= (state << 5) & 0xFFFFFFFF
            colors[i] = state & 1
        return colors

    def generate_diverse_initializations():
        """Generate diverse initializations including LFSR-based ones."""
        inits = []

        # 1. Pure random
        inits.append(np.random.randint(0, 2, size=N))

        # 2. Block pattern with period 2
        inits.append(np.array([(i // 2) % 2 for i in range(N)], dtype=np.int32))

        # 3. Block pattern with period 4
        inits.append(np.array([(i // 4) % 2 for i in range(N)], dtype=np.int32))

        # 4. Striped pattern
        inits.append(np.array([i % 2 for i in range(N)], dtype=np.int32))

        # 5. Thue-Morse-like pattern
        tm = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int32)
        inits.append(tm)

        # 6. Block pattern with period 8
        inits.append(np.array([(i // 8) % 2 for i in range(N)], dtype=np.int32))

        # 7. Block pattern with period 16
        inits.append(np.array([(i // 16) % 2 for i in range(N)], dtype=np.int32))

        # 8-11. LFSR-based structured initializations with different seeds
        for seed in [42, 123, 456, 789]:
            inits.append(generate_lfsr_initialization(seed))

        # 12. Another random
        inits.append(np.random.randint(0, 2, size=N))

        return inits

    # Evaluate initializations and build population
    init_list = generate_diverse_initializations()
    init_scores = []
    for init in init_list:
        score = count_mono(init)
        init_scores.append((score, init))

    init_scores.sort(key=lambda x: x[0])

    # Build population from top initializations + random perturbations
    pop_size = 16
    population = []
    scores = []

    # Take top 8 initializations
    for score, init in init_scores[:8]:
        population.append(init.copy())
        scores.append(score)

    # Add 8 more with small random perturbations of best ones
    for _ in range(8):
        base = population[random.randint(0, 7)].copy()
        # Flip a few random bits
        for _ in range(random.randint(2, 6)):
            i = random.randint(0, N - 1)
            base[i] = 1 - base[i]
        s = count_mono(base)
        population.append(base)
        scores.append(s)

    best_score = min(scores)
    best_coloring = population[scores.index(best_score)].copy()

    # Genetic algorithm loop with SA exploitation
    gen = 0
    while time.time() < deadline - 15.0:
        gen += 1

        # Tournament selection and crossover to create new population
        new_pop = []
        new_scores = []

        for _ in range(pop_size):
            # Tournament selection
            idxs = random.sample(range(pop_size), min(3, pop_size))
            best_idx = min(idxs, key=lambda i: scores[i])
            p1 = population[best_idx]

            idxs2 = random.sample(range(pop_size), min(3, pop_size))
            best_idx2 = min(idxs2, key=lambda i: scores[i])
            p2 = population[best_idx2]

            # Uniform crossover
            mask = np.random.randint(0, 2, size=N)
            child = np.where(mask == 1, p1, p2)

            # Mutation
            for i in range(N):
                if random.random() < 0.03:
                    child[i] = 1 - child[i]

            new_pop.append(child)

        # Run SA on each new individual
        for i in range(pop_size):
            if time.time() > deadline - 15.0:
                break
            # Determine SA parameters based on generation
            T_start = max(0.5, 2.0 - gen * 0.1)
            T_end = max(0.01, 0.1 - gen * 0.005)
            max_iter = max(200, 1000 - gen * 20)

            sa_colors, sa_score = fast_sa(
                new_pop[i], T_start, T_end, max_iter, deadline - 15.0
            )
            new_pop[i] = sa_colors
            new_scores.append(sa_score)

        # Combine old and new populations, keep best
        all_candidates = [(c, s) for c, s in zip(population, scores)] + \
                         [(c, s) for c, s in zip(new_pop, new_scores)]
        all_candidates.sort(key=lambda x: x[1])

        population = [c for c, s in all_candidates[:pop_size]]
        scores = [s for c, s in all_candidates[:pop_size]]

        # Track global best
        if scores[0] < best_score:
            best_score = scores[0]
            best_coloring = population[0].copy()

    # Final intensive SA on the best solution
    if time.time() < deadline - 2.0:
        final_colors, final_score = fast_sa(
            best_coloring, 1.0, 0.01, 5000, deadline - 1.0
        )
        if final_score < best_score:
            best_score = final_score
            best_coloring = final_colors

    return best_coloring.tolist()

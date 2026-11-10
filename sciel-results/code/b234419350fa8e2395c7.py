import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))

    aps_arr = np.array(aps, dtype=np.int32)

    # 2. Precompute neighbors for fast delta calculation
    neighbors = [[] for _ in range(N)]
    for i1, i2, i3, i4 in aps:
        neighbors[i1].append((i2, i3, i4))
        neighbors[i2].append((i1, i3, i4))
        neighbors[i3].append((i1, i2, i4))
        neighbors[i4].append((i1, i2, i3))

    neighbors_np = []
    for i in range(N):
        if neighbors[i]:
            neighbors_np.append(np.array(neighbors[i], dtype=np.int32))
        else:
            neighbors_np.append(np.empty((0, 3), dtype=np.int32))

    def count_mono(colors):
        c1 = colors[aps_arr[:, 0]]
        c2 = colors[aps_arr[:, 1]]
        c3 = colors[aps_arr[:, 2]]
        c4 = colors[aps_arr[:, 3]]
        return int(np.sum((c1 == c2) & (c2 == c3) & (c3 == c4)))

    def flip_delta(colors, i):
        """Compute delta in mono count if we flip bit i."""
        old_c = colors[i]
        new_c = 1 - old_c
        others = neighbors_np[i]
        if len(others) == 0:
            return 0
        cols = colors[others]
        c0, c1, c2 = cols[:, 0], cols[:, 1], cols[:, 2]
        all_eq = (c0 == c1) & (c1 == c2)
        val = c0
        was_mono = all_eq & (val == old_c)
        becomes_mono = all_eq & (val == new_c)
        return int(np.sum(becomes_mono) - np.sum(was_mono))

    def polish(colors, score, time_limit=8.0):
        """Exact local search: only accept improving single flips."""
        polish_start = time.time()
        improved = True
        while improved and time.time() < polish_start + time_limit:
            improved = False
            for i in range(N):
                if time.time() > polish_start + time_limit:
                    break
                delta = flip_delta(colors, i)
                if delta < 0:
                    colors[i] = 1 - colors[i]
                    score += delta
                    improved = True
        return score

    # 3. Base patterns for diversity
    patterns = [
        np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8),
        np.array([(i // 3) % 2 for i in range(N)], dtype=np.int8),
        np.array([(i % 4) < 2 for i in range(N)], dtype=np.int8),
        np.array([1 - ((i // 2) % 2) for i in range(N)], dtype=np.int8),
        np.array([(i // 5) % 2 for i in range(N)], dtype=np.int8),
        np.array([(i % 8) < 4 for i in range(N)], dtype=np.int8),
    ]

    def random_perturb(colors, num_perturb):
        idxs = random.sample(range(N), num_perturb)
        for idx in idxs:
            colors[idx] = 1 - colors[idx]
        return colors

    def uniform_crossover(p1, p2):
        mask = (np.random.random(N) < 0.5).astype(np.int8)
        return p1 * (1 - mask) + p2 * mask

    def mutate(colors, rate=0.02):
        num_mut = int(N * rate)
        if num_mut > 0:
            idxs = random.sample(range(N), num_mut)
            for idx in idxs:
                colors[idx] = 1 - colors[idx]
        return colors

    best_score = float('inf')
    best_coloring = None

    def update_best(colors, score):
        nonlocal best_score, best_coloring
        if score < best_score:
            best_score = score
            best_coloring = colors.copy()
            return True
        return False

    # 4. Simulated Annealing on a single coloring
    def sa_run(colors, score, time_budget):
        T = 1.5
        alpha = 0.9996
        min_T = 0.03
        sa_end = time.time() + time_budget
        stagnation = 0
        last_improve = score

        while T > min_T and time.time() < sa_end:
            r = random.random()
            if stagnation > 60:
                if r < 0.12:
                    move = 'kick'
                elif r < 0.40:
                    move = 'two_bit'
                else:
                    move = 'one_bit'
            else:
                if r < 0.02:
                    move = 'kick'
                elif r < 0.25:
                    move = 'two_bit'
                else:
                    move = 'one_bit'

            if move == 'kick':
                ksize = random.randint(2, 5)
                kidx = random.sample(range(N), ksize)
                old = colors[kidx].copy()
                colors[kidx] = 1 - old
                new_score = count_mono(colors)
                delta = new_score - score
                if delta < 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                    score = new_score
                    if score < last_improve:
                        last_improve = score
                        stagnation = 0
                        update_best(colors, score)
                else:
                    colors[kidx] = old
                    stagnation += 1
            elif move == 'two_bit':
                i, j = random.sample(range(N), 2)
                old_i, old_j = colors[i], colors[j]
                di = flip_delta(colors, i)
                colors[i] = 1 - old_i
                dj = flip_delta(colors, j)
                colors[j] = 1 - old_j
                total = di + dj
                if total < 0 or random.random() < np.exp(-total / max(T, 0.01)):
                    score += total
                    if score < last_improve:
                        last_improve = score
                        stagnation = 0
                        update_best(colors, score)
                else:
                    colors[i] = old_i
                    colors[j] = old_j
                    stagnation += 1
            else:
                i = random.randint(0, N - 1)
                delta = flip_delta(colors, i)
                if delta < 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                    colors[i] = 1 - colors[i]
                    score += delta
                    if score < last_improve:
                        last_improve = score
                        stagnation = 0
                        update_best(colors, score)
                else:
                    stagnation += 1
            T *= alpha
        return score

    # 5. Genetic Algorithm with population
    def ga_run(pop_size=16, generations=20, time_budget=60.0):
        ga_end = time.time() + time_budget

        # Initialize population with diverse patterns + perturbations
        population = []
        scores = []
        for i in range(pop_size):
            pat = patterns[i % len(patterns)].copy()
            random_perturb(pat, random.randint(3, 10))
            s = count_mono(pat)
            population.append(pat)
            scores.append(s)
            update_best(pat, s)

        # Evolve
        gen = 0
        while gen < generations and time.time() < ga_end:
            # Sort by fitness (lower is better)
            order = sorted(range(pop_size), key=lambda x: scores[x])

            # Elitism: keep top 4
            elite = [(population[order[k]].copy(), scores[order[k]]) for k in range(4)]

            # Selection: tournament of 3
            def tournament():
                a, b, c = random.sample(range(pop_size), 3)
                return min(a, b, c, key=lambda x: scores[x])

            # Create new generation
            new_pop = []
            new_scores = []

            # Add elites
            for ec, es in elite:
                new_pop.append(ec)
                new_scores.append(es)

            # Fill rest with crossover + mutation
            while len(new_pop) < pop_size:
                p1_idx = tournament()
                p2_idx = tournament()
                child = uniform_crossover(population[p1_idx], population[p2_idx])
                mutate(child, rate=0.015)
                cs = count_mono(child)
                new_pop.append(child)
                new_scores.append(cs)
                update_best(child, cs)

            # Short SA polish on the best few children
            for k in range(min(3, len(new_pop))):
                # Find the best in new_pop
                best_k = min(range(len(new_pop)), key=lambda x: new_scores[x])
                if time.time() < ga_end - 2.0:
                    sa_time = min(3.0, max(0.5, (ga_end - time.time()) / max(1, generations - gen)))
                    new_scores[best_k] = sa_run(new_pop[best_k], new_scores[best_k], sa_time)

            population = new_pop
            scores = new_scores
            gen += 1

        return population, scores

    # 6. Main search loop: alternate SA restarts and GA phases
    phase = 'sa'
    num_restarts = 0
    ga_count = 0

    while time.time() < deadline - 15.0:
        if phase == 'sa':
            # SA restart with diverse initialization
            base_idx = num_restarts % len(patterns)
            colors = patterns[base_idx].copy()
            random_perturb(colors, random.randint(3, 12))
            current_score = count_mono(colors)
            update_best(colors, current_score)

            # Time budget for this SA run
            remaining = deadline - 15.0 - time.time()
            sa_budget = max(4.0, min(15.0, remaining / 4.0))
            current_score = sa_run(colors, current_score, sa_budget)

            num_restarts += 1

            # Switch to GA every 3 SA restarts
            if num_restarts % 3 == 0:
                phase = 'ga'
            else:
                phase = 'sa'

        elif phase == 'ga':
            # GA phase
            remaining = deadline - 15.0 - time.time()
            if remaining > 20.0:
                ga_budget = min(50.0, remaining * 0.6)
                ga_run(pop_size=14, generations=15, time_budget=ga_budget)
            else:
                # Not enough time for a full GA, do a quick one
                ga_run(pop_size=8, generations=5, time_budget=max(5.0, remaining - 15.0))

            ga_count += 1
            phase = 'sa'

    # 7. Final polish on best coloring
    if best_coloring is not None:
        final_colors = best_coloring.copy()
        final_score = polish(final_colors, best_score, time_limit=10.0)
        if final_score < best_score:
            best_score = final_score
            best_coloring = final_colors

    return best_coloring.tolist()
# EVOLVE-BLOCK-END

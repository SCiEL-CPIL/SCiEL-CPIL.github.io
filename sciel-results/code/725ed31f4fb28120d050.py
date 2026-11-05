"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    total_words = q ** n

    # Precompute all ball index lists for every possible codeword
    all_balls = []
    for idx in range(total_words):
        word = []
        tmp = idx
        for _ in range(n):
            word.append(tmp % q)
            tmp //= q
        word.reverse()
        ball = [idx]
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    delta = (new_val - orig) * (q ** (n - 1 - pos))
                    ball.append(idx + delta)
        all_balls.append(tuple(ball))

    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    def build_coverage_count(cw_list):
        counts = np.zeros(total_words, dtype=np.int32)
        for cw in cw_list:
            b = all_balls[word_to_idx(cw)]
            counts[np.asarray(b, dtype=np.int64)] += 1
        return counts

    def is_redundant(cw, counts):
        b = all_balls[word_to_idx(cw)]
        return bool(np.all(counts[np.asarray(b, dtype=np.int64)] > 1))

    # Initial code: direct sum of ternary tetracode with all tails
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # --- Phase 1: fast greedy removal ---
    counts = build_coverage_count(codewords)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                return codewords
            if is_redundant(codewords[i], counts):
                b = np.asarray(all_balls[word_to_idx(codewords[i])], dtype=np.int64)
                counts[b] -= 1
                codewords.pop(i)
                improved = True
                break

    # --- Phase 2: swap-based local search ---
    if time.time() < deadline:
        counts = build_coverage_count(codewords)
        balls_arr = [np.asarray(all_balls[word_to_idx(cw)], dtype=np.int64) for cw in codewords]
        cw_set = set(tuple(c) for c in codewords)

        improved_swap = True
        while improved_swap and time.time() < deadline:
            improved_swap = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                cw_old_t = tuple(codewords[i])
                for pos in range(n):
                    if time.time() > deadline:
                        break
                    orig = codewords[i][pos]
                    for new_val in range(q):
                        if new_val == orig:
                            continue
                        new_cw = list(codewords[i])
                        new_cw[pos] = new_val
                        new_cw_t = tuple(new_cw)
                        if new_cw_t in cw_set:
                            continue
                        new_idx = word_to_idx(new_cw)
                        ball_new_arr = np.asarray(all_balls[new_idx], dtype=np.int64)
                        old_ball_arr = balls_arr[i]
                        mask = np.isin(old_ball_arr, ball_new_arr)
                        uncovered = old_ball_arr[~mask]
                        if len(uncovered) > 0 and np.any(counts[uncovered] <= 1):
                            continue
                        counts[old_ball_arr] -= 1
                        counts[ball_new_arr] += 1
                        codewords[i] = new_cw
                        balls_arr[i] = ball_new_arr
                        cw_set.discard(cw_old_t)
                        cw_set.add(new_cw_t)
                        improved_swap = True
                        break
                    if improved_swap:
                        break
                if improved_swap:
                    break
            if not improved_swap:
                # Two-position swaps
                cw_set = set(tuple(c) for c in codewords)
                for i in range(len(codewords)):
                    if time.time() > deadline:
                        break
                    cw_old_t = tuple(codewords[i])
                    for pos1 in range(n):
                        if time.time() > deadline:
                            break
                        for pos2 in range(pos1 + 1, n):
                            if time.time() > deadline:
                                break
                            orig1 = codewords[i][pos1]
                            orig2 = codewords[i][pos2]
                            for v1 in range(q):
                                if v1 == orig1:
                                    continue
                                for v2 in range(q):
                                    if v2 == orig2:
                                        continue
                                    new_cw = list(codewords[i])
                                    new_cw[pos1] = v1
                                    new_cw[pos2] = v2
                                    new_cw_t = tuple(new_cw)
                                    if new_cw_t in cw_set:
                                        continue
                                    new_idx = word_to_idx(new_cw)
                                    ball_new_arr = np.asarray(all_balls[new_idx], dtype=np.int64)
                                    old_ball_arr = balls_arr[i]
                                    mask = np.isin(old_ball_arr, ball_new_arr)
                                    uncovered = old_ball_arr[~mask]
                                    if len(uncovered) > 0 and np.any(counts[uncovered] <= 1):
                                        continue
                                    counts[old_ball_arr] -= 1
                                    counts[ball_new_arr] += 1
                                    codewords[i] = new_cw
                                    balls_arr[i] = ball_new_arr
                                    cw_set.discard(cw_old_t)
                                    cw_set.add(new_cw_t)
                                    improved_swap = True
                                    break
                                if improved_swap:
                                    break
                            if improved_swap:
                                break
                        if improved_swap:
                            break
                    if improved_swap:
                        break

    # --- Phase 3: final greedy removal pass ---
    if time.time() < deadline:
        counts = build_coverage_count(codewords)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() > deadline:
                    return codewords
                if is_redundant(codewords[i], counts):
                    b = np.asarray(all_balls[word_to_idx(codewords[i])], dtype=np.int64)
                    counts[b] -= 1
                    codewords.pop(i)
                    improved = True
                    break

    # --- Phase 4: random-restart greedy removal ---
    if time.time() < deadline:
        rng = random.Random(12345)
        best = list(codewords)
        best_len = len(best)
        while time.time() < deadline - 2.0:
            trial = list(codewords)
            rng.shuffle(trial)
            counts = build_coverage_count(trial)
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                for i in range(len(trial) - 1, -1, -1):
                    if is_redundant(trial[i], counts):
                        b = np.asarray(all_balls[word_to_idx(trial[i])], dtype=np.int64)
                        counts[b] -= 1
                        trial.pop(i)
                        improved = True
                        break
            if len(trial) < best_len:
                best = trial
                best_len = len(trial)

    # --- Phase 5: Genetic algorithm with fitness-based redundancy removal ---
    if time.time() < deadline - 5.0:
        rng = random.Random(42)
        best = list(codewords)
        best_len = len(best)
        
        # Population-based search
        pop_size = 20
        population = [list(best) for _ in range(pop_size)]
        
        # Initialize with some diversity
        for p in range(1, pop_size):
            trial = list(best)
            # Randomly perturb a few codewords
            for _ in range(rng.randint(1, 3)):
                idx = rng.randint(0, len(trial) - 1)
                pos = rng.randint(0, n - 1)
                old_val = trial[idx][pos]
                new_val = rng.randint(0, q - 1)
                if new_val != old_val:
                    trial[idx][pos] = new_val
                    # Check validity after swap
                    counts = build_coverage_count(trial)
                    if np.all(counts >= 1):
                        pass
                    else:
                        trial[idx][pos] = old_val
            population[p] = trial
        
        # Fitness: negative sum of squared coverage counts (lower is better)
        def fitness(cw_list):
            if len(cw_list) == 0:
                return float('inf')
            counts = build_coverage_count(cw_list)
            if np.any(counts < 1):
                return float('inf')  # Invalid
            return -np.sum(counts ** 2)
        
        # Crossover: combine two valid codes
        def crossover(parent1, parent2, rng):
            # Take elements from both parents, ensuring no duplicates
            child = []
            used = set()
            for cw in parent1:
                t = tuple(cw)
                if t not in used:
                    child.append(list(cw))
                    used.add(t)
            # Add some from parent2 if not already present
            for cw in parent2:
                t = tuple(cw)
                if t not in used and len(child) < len(parent1) + 2:
                    child.append(list(cw))
                    used.add(t)
            return child
        
        # Mutation: swap a random codeword to a neighbor
        def mutate(cw_list, rng):
            trial = [list(cw) for cw in cw_list]
            if len(trial) == 0:
                return trial
            idx = rng.randint(0, len(trial) - 1)
            pos = rng.randint(0, n - 1)
            old_val = trial[idx][pos]
            new_val = rng.randint(0, q - 1)
            if new_val != old_val:
                trial[idx][pos] = new_val
            return trial
        
        # GA loop
        ga_iterations = 0
        while time.time() < deadline - 5.0 and ga_iterations < 100:
            ga_iterations += 1
            
            # Evaluate fitness
            fit_scores = []
            for ind in population:
                f = fitness(ind)
                fit_scores.append(f)
            
            # Find best
            min_fit = min(fit_scores)
            best_idx = fit_scores.index(min_fit)
            if min_fit < float('inf') and len(population[best_idx]) < best_len:
                best = [list(cw) for cw in population[best_idx]]
                best_len = len(best)
            
            # Selection: tournament
            def tournament_select(pop, scores, k=3):
                candidates = rng.sample(range(len(pop)), k)
                best_cand = min(candidates, key=lambda i: scores[i])
                return pop[best_cand]
            
            # Generate next generation
            new_pop = []
            # Elitism: keep best
            new_pop.append([list(cw) for cw in population[best_idx]])
            
            while len(new_pop) < pop_size:
                p1 = tournament_select(population, fit_scores)
                p2 = tournament_select(population, fit_scores)
                
                if rng.random() < 0.5:
                    child = crossover(p1, p2, rng)
                else:
                    child = mutate(p1, rng)
                
                # Validate
                if len(child) > 0:
                    counts = build_coverage_count(child)
                    if np.all(counts >= 1):
                        new_pop.append(child)
                    else:
                        new_pop.append([list(cw) for cw in p1])
                else:
                    new_pop.append([list(cw) for cw in p1])
            
            population = new_pop
        
        # Final greedy removal on best
        counts = build_coverage_count(best)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(len(best) - 1, -1, -1):
                if is_redundant(best[i], counts):
                    b = np.asarray(all_balls[word_to_idx(best[i])], dtype=np.int64)
                    counts[b] -= 1
                    best.pop(i)
                    improved = True
                    break

    return best
# EVOLVE-BLOCK-END

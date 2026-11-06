"""Combined: randomized K-color greedy initialization + genetic algorithm with structure-preserving crossover."""
import time
import random
import math


def construct():
    n = 128
    deadline = time.time() + 85.0

    # Precompute all nonconstant 4-term APs (0-indexed positions)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))

    # Map each position to the APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    # APs where this position is the last element (for incremental greedy checks)
    aps_ending_at = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        aps_ending_at[ap[3]].append(idx)

    def count_violations(colors):
        """Count number of APs with fewer than 3 distinct colors."""
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c0 == c1 == c2 == c3:
                v += 1
            elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                v += 1
            elif c0 == c1 and c2 == c3:
                v += 1
            elif c0 == c2 and c1 == c3:
                v += 1
            elif c0 == c3 and c1 == c2:
                v += 1
        return v

    def is_ap_valid(colors, ap):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        distinct = 1
        if c1 != c0:
            distinct += 1
        if c2 != c0 and c2 != c1:
            distinct += 1
        if c3 != c0 and c3 != c1 and c3 != c2:
            distinct += 1
        return distinct >= 3

    def fitness(colors):
        """Lower is better: violations dominate, then color count."""
        v = count_violations(colors)
        c = len(set(colors))
        return v * 1000 + c

    def randomized_greedy(K, rng):
        """Try to build a valid coloring using only colors 0..K-1 via randomized greedy."""
        colors = [-1] * n
        for idx in range(n):
            order = list(range(K))
            rng.shuffle(order)
            chosen = None
            for c in order:
                good = True
                for ap_idx in aps_ending_at[idx]:
                    ap = aps[ap_idx]
                    c0 = colors[ap[0]]
                    c1 = colors[ap[1]]
                    c2 = colors[ap[2]]
                    # c3 is the candidate
                    distinct = 1
                    if c1 != c0:
                        distinct += 1
                    if c2 != c0 and c2 != c1:
                        distinct += 1
                    if c != c0 and c != c1 and c != c2:
                        distinct += 1
                    if distinct < 3:
                        good = False
                        break
                if good:
                    chosen = c
                    break
            if chosen is None:
                return None
            colors[idx] = chosen
        return colors

    def local_search_refine(colors, time_limit):
        """Simulated annealing to fix violations and minimize color count."""
        colors = colors[:]
        current_violations = count_violations(colors)
        color_counts = [0] * 128
        for c in colors:
            color_counts[c] += 1
        current_colors_used = sum(1 for c in range(128) if color_counts[c] > 0)

        best_colors = colors[:]
        best_violations = current_violations
        best_colors_used = current_colors_used
        best_fit = fitness(best_colors)

        temp = 3.0
        cooling = 0.997
        min_temp = 0.01

        while time.time() < time_limit and temp > min_temp:
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            used = [c for c in range(128) if color_counts[c] > 0]
            candidates = used[:]
            for _ in range(2):
                new_c = random.randint(0, 127)
                if new_c not in candidates:
                    candidates.append(new_c)

            best_new_color = old_color
            best_delta_v = 0
            best_delta_c = 0
            best_score = 0

            affected_aps = pos_to_aps[pos]
            curr_invalid = 0
            for ap_idx in affected_aps:
                ap = aps[ap_idx]
                if not is_ap_valid(colors, ap):
                    curr_invalid += 1

            for new_color in candidates:
                if new_color == old_color:
                    continue
                colors[pos] = new_color
                new_invalid = 0
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    if not is_ap_valid(colors, ap):
                        new_invalid += 1
                delta_v = new_invalid - curr_invalid
                delta_c = 0
                if color_counts[old_color] == 1:
                    delta_c -= 1
                if color_counts[new_color] == 0:
                    delta_c += 1
                score = delta_v * 100 + delta_c
                if score < best_score:
                    best_score = score
                    best_delta_v = delta_v
                    best_delta_c = delta_c
                    best_new_color = new_color
                elif score == best_score and random.random() < 0.3:
                    best_new_color = new_color
                    best_delta_v = delta_v
                    best_delta_c = delta_c
            colors[pos] = old_color

            accept = False
            if best_delta_v < 0 or (best_delta_v == 0 and best_delta_c < 0):
                accept = True
            elif best_delta_v == 0 and best_delta_c == 0:
                accept = True
            else:
                energy = best_delta_v * 100 + best_delta_c
                if energy > 0 and temp > 0:
                    prob = math.exp(-energy / temp)
                    accept = random.random() < prob

            if accept:
                colors[pos] = best_new_color
                color_counts[old_color] -= 1
                color_counts[best_new_color] += 1
                current_violations += best_delta_v
                current_colors_used = sum(1 for c in range(128) if color_counts[c] > 0)
                if current_violations < best_violations or \
                   (current_violations == best_violations and current_colors_used < best_colors_used):
                    best_violations = current_violations
                    best_colors_used = current_colors_used
                    best_colors = colors[:]
                    best_fit = fitness(best_colors)
                if current_violations == 0 and current_colors_used <= 3:
                    break
            temp *= cooling

        return best_colors, best_violations, best_colors_used

    def try_reduce_colors(colors, max_time):
        """Reduce distinct colors by remapping one color onto another."""
        colors = colors[:]
        best_colors = colors[:]
        best_count = len(set(colors))

        while time.time() < max_time:
            used_colors = sorted(set(best_colors))
            if len(used_colors) <= best_count:
                break
            if best_count <= 3:
                break

            improved = False
            for color_to_remove in used_colors:
                if time.time() > max_time:
                    break
                positions = [i for i in range(n) if best_colors[i] == color_to_remove]
                if not positions:
                    continue
                for target_color in used_colors:
                    if target_color == color_to_remove:
                        continue
                    old_vals = {pos: best_colors[pos] for pos in positions}
                    for pos in positions:
                        best_colors[pos] = target_color
                    if count_violations(best_colors) == 0:
                        best_count = len(set(best_colors))
                        improved = True
                        break
                    else:
                        for pos in positions:
                            best_colors[pos] = old_vals[pos]
                if improved:
                    break
            if not improved:
                break

        return best_colors, best_count

    # Step 1: Initialize population with randomized greedy colorings
    rng = random.Random(42)
    population = []
    
    # Try various K values
    for K in [4, 5, 6, 7, 8]:
        for attempt in range(200):
            if time.time() > deadline - 30:
                break
            coloring = randomized_greedy(K, rng)
            if coloring is not None:
                population.append(coloring)
                if len(population) >= 50:
                    break
        if len(population) >= 50:
            break

    # If greedy didn't find enough, use local search from random starts
    if len(population) < 20:
        for _ in range(50):
            if time.time() > deadline - 30:
                break
            colors = [random.randint(0, 7) for _ in range(n)]
            refined, _, _ = local_search_refine(colors, time.time() + 0.5)
            if count_violations(refined) == 0:
                population.append(refined)

    if not population:
        # Fallback: use identity mapping
        return list(range(n))

    # Step 2: Genetic algorithm with tournament selection and crossover
    pop_size = min(len(population), 30)
    if pop_size < 5:
        # Pad with local search results
        for _ in range(10 - pop_size):
            colors = [random.randint(0, 7) for _ in range(n)]
            refined, _, _ = local_search_refine(colors, time.time() + 0.3)
            population.append(refined)
        pop_size = min(len(population), 30)

    # Sort initial population by fitness
    population.sort(key=fitness)
    best_individual = population[0][:]
    best_fit = fitness(best_individual)

    # Precompute constraint weights for crossover: how many APs each position pair shares
    # For crossover, we swap color assignments between positions that share high-constraint APs
    # Build a list of position pairs with shared APs
    shared_ap_count = [[0] * n for _ in range(n)]
    for ap in aps:
        for i in range(4):
            for j in range(i + 1, 4):
                shared_ap_count[ap[i]][ap[j]] += 1

    # Genetic algorithm
    gen = 0
    max_gens = 200
    while time.time() < deadline - 10 and gen < max_gens:
        gen += 1
        new_population = []

        # Elitism: keep top 3
        for i in range(min(3, len(population))):
            new_population.append(population[i][:])

        # Generate offspring via tournament selection and crossover
        while len(new_population) < pop_size and time.time() < deadline - 10:
            # Tournament selection
            def tournament_select():
                t = random.sample(range(len(population)), min(3, len(population)))
                winner = t[0]
                for idx in t[1:]:
                    if fitness(population[idx]) < fitness(population[winner]):
                        winner = idx
                return population[winner]

            parent1 = tournament_select()
            parent2 = tournament_select()

            # Crossover: swap color assignments between positions with high shared constraint
            child = parent1[:]
            
            # Identify positions where parents differ
            diff_positions = [i for i in range(n) if parent1[i] != parent2[i]]
            
            if diff_positions:
                # Sort by shared constraint with other differing positions
                # For simplicity, randomly select a subset of differing positions to swap
                num_swaps = random.randint(1, min(10, len(diff_positions)))
                swap_positions = random.sample(diff_positions, num_swaps)
                for pos in swap_positions:
                    child[pos] = parent2[pos]

            # Mutation: flip a few positions to random colors
            if random.random() < 0.3:
                for _ in range(random.randint(1, 3)):
                    pos = random.randint(0, n - 1)
                    child[pos] = random.randint(0, 127)

            new_population.append(child)

        # Local search on a few of the new individuals
        for i in range(min(5, len(new_population))):
            if time.time() > deadline - 10:
                break
            refined, _, _ = local_search_refine(new_population[i], time.time() + 0.5)
            new_population[i] = refined

        # Evaluate and sort
        new_population.sort(key=fitness)
        
        # Update best
        if fitness(new_population[0]) < best_fit:
            best_fit = fitness(new_population[0])
            best_individual = new_population[0][:]

        # Replace population
        population = new_population[:pop_size]

    # Step 3: Final refinement on best individual
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        best_individual, _, _ = local_search_refine(best_individual, deadline - 2.0)

    # Step 4: Try to reduce colors further
    if count_violations(best_individual) == 0:
        best_individual, _ = try_reduce_colors(best_individual, deadline - 1.0)

    # Ensure valid solution
    if count_violations(best_individual) > 0:
        return list(range(n))

    return best_individual

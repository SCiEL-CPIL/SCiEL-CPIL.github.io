import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row

    # Lower bound: M(M+1)/2 >= total_diffs_needed
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs_needed)) / 2)

    def row_from_gaps(gaps):
        """Convert gap vector to row (starting at 0)."""
        row = [0]
        for g in gaps:
            row.append(row[-1] + g)
        return row

    def get_scope(rows):
        return max(row[-1] for row in rows)

    def is_valid(rows):
        """Check all differences are distinct across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_construct(max_scope):
        """Deterministic greedy: pick smallest valid next element."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if x > max_scope:
                    return None
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows

    def randomized_greedy(max_scope, rng):
        """Randomized greedy with bias toward smaller values."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                # Try candidates from row[-1]+1 upward, with random selection
                candidates = []
                x = row[-1] + 1
                while x <= min(max_scope, row[-1] + 60):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                    x += 1

                if not candidates:
                    # Extend search range
                    x = row[-1] + 61
                    while x <= max_scope:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                        x += 1

                if not candidates:
                    return None

                # Weighted random selection favoring smaller values
                weights = [1.0 / (c - row[-1]) ** 0.7 for c in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                chosen = candidates[-1]
                for c, w in zip(candidates, weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen = c
                        break

                row.append(chosen)
                for a in row[:-1]:
                    used.add(chosen - a)
            rows.append(row)
        return rows

    def differential_evolution(gaps_list, max_scope, rng, time_limit, population_size=30):
        """
        Differential evolution on gap vectors.
        gaps_list: list of n_rows gap vectors, each of length k, all positive ints.
        Objective: minimize max scope while keeping all differences distinct.
        """
        # Initialize population
        population = []
        for g in gaps_list:
            population.append([list(row) for row in g])

        # Add random valid-ish individuals
        for _ in range(population_size - n_rows):
            individual = []
            for _ in range(n_rows):
                gaps = [max(1, rng.randint(1, max(2, max_scope // (k * 3)))) for _ in range(k)]
                # Scale to fit within max_scope
                total = sum(gaps)
                if total > max_scope:
                    scale = max_scope / total
                    gaps = [max(1, int(g * scale)) for g in gaps]
                    # Ensure sum <= max_scope
                    while sum(gaps) > max_scope:
                        idx = rng.randint(0, k - 1)
                        if gaps[idx] > 1:
                            gaps[idx] -= 1
                    while sum(gaps) < max_scope and rng.random() < 0.5:
                        idx = rng.randint(0, k - 1)
                        gaps[idx] += 1
                individual.append(gaps)
            population.append(individual)

        def evaluate(individual):
            """Return (scope, num_conflicts). Lower is better."""
            rows = [row_from_gaps(gaps) for gaps in individual]
            scope = max(row[-1] for row in rows)
            if scope > max_scope:
                return (scope, n_rows * diffs_per_row)  # Heavy penalty
            used = set()
            conflicts = 0
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in used:
                            conflicts += 1
                        else:
                            used.add(d)
            return (scope, conflicts)

        # Evaluate initial population
        scores = [evaluate(ind) for ind in population]

        best_idx = min(range(len(population)), key=lambda i: scores[i])
        best_individual = [list(row) for row in population[best_idx]]
        best_score = scores[best_idx]

        generation = 0
        while time.time() < time_limit:
            generation += 1
            new_population = [list(row) for row in population]

            for i in range(len(population)):
                if time.time() > time_limit:
                    break

                # DE/rand/1 mutation
                if len(population) < 4:
                    continue
                a, b, c = rng.sample([j for j in range(len(population)) if j != i], 3)

                # F (scale factor) with slight randomization
                F = 0.5 + 0.5 * rng.random()

                trial = []
                valid = True
                for row_idx in range(n_rows):
                    trial_row = []
                    for pos in range(k):
                        mutant = (population[a][row_idx][pos]
                                  + F * (population[b][row_idx][pos] - population[c][row_idx][pos]))
                        val = max(1, int(round(mutant)))
                        # Clip to reasonable range
                        val = min(val, max(1, max_scope // k + 10))
                        trial_row.append(val)
                    trial.append(trial_row)

                # Check if trial is valid (positive gaps)
                if all(g >= 1 for row in trial for g in row):
                    trial_score = evaluate(trial)
                    if trial_score < scores[i]:
                        new_population[i] = trial
                        scores[i] = trial_score
                        if trial_score < best_score:
                            best_score = trial_score
                            best_individual = [list(row) for row in trial]

            population = new_population

            # Elitism: keep best
            pop_best_idx = min(range(len(population)), key=lambda i: scores[i])
            if scores[pop_best_idx] < best_score:
                best_score = scores[pop_best_idx]
                best_individual = [list(row) for row in population[pop_best_idx]]

        return best_individual

    def local_search_gaps(gaps_list, max_scope, rng, time_limit):
        """
        Local search on gap vectors: perturb individual gaps to reduce scope/conflicts.
        """
        current = [list(row) for row in gaps_list]
        current_score = None

        def score_gaps(gaps):
            rows = [row_from_gaps(g) for g in gaps]
            scope = max(row[-1] for row in rows)
            if scope > max_scope:
                return (scope, n_rows * diffs_per_row)
            used = set()
            conflicts = 0
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in used:
                            conflicts += 1
                        else:
                            used.add(d)
            return (scope, conflicts)

        current_score = score_gaps(current)
        best = [list(row) for row in current]
        best_score = current_score

        no_improve = 0
        max_no_improve = 50

        while time.time() < time_limit and no_improve < max_no_improve:
            # Pick a random gap to perturb
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(0, k - 1)
            delta = rng.choice([-2, -1, -1, 1, 1, 2])
            new_val = current[row_idx][pos] + delta
            if new_val < 1:
                new_val = 1

            trial = [list(row) for row in current]
            trial[row_idx][pos] = new_val

            trial_score = score_gaps(trial)

            if trial_score <= current_score:
                current = trial
                current_score = trial_score
                no_improve = 0
                if trial_score < best_score:
                    best_score = trial_score
                    best = [list(row) for row in current]
            else:
                no_improve += 1

            # Occasionally try a larger perturbation
            if rng.random() < 0.1:
                row_idx = rng.randint(0, n_rows - 1)
                pos = rng.randint(0, k - 1)
                new_val = max(1, current[row_idx][pos] + rng.randint(-5, 5))
                trial = [list(row) for row in current]
                trial[row_idx][pos] = new_val
                trial_score = score_gaps(trial)
                if trial_score <= current_score:
                    current = trial
                    current_score = trial_score
                    no_improve = 0
                    if trial_score < best_score:
                        best_score = trial_score
                        best = [list(row) for row in current]

        return best

    # === Main algorithm ===

    rng = random.Random()

    # Phase 1: Find initial solution
    best_gaps = None
    best_scope = float('inf')

    # Try greedy first
    greedy_sol = greedy_construct(200)
    if greedy_sol:
        scope = get_scope(grey_sol) if False else get_scope(greedy_sol)
        if is_valid(greedy_sol):
            best_scope = scope
            best_gaps = []
            for row in greedy_sol:
                gaps = [row[i+1] - row[i] for i in range(k)]
                best_gaps.append(gaps)

    # Try randomized greedy with multiple restarts
    for attempt in range(30):
        if time.time() > deadline - 40:
            break
        sol = randomized_greedy(200, rng)
        if sol and is_valid(sol):
            scope = get_scope(sol)
            if scope < best_scope:
                best_scope = scope
                best_gaps = []
                for row in sol:
                    gaps = [row[i+1] - row[i] for i in range(k)]
                    best_gaps.append(gaps)
                if best_scope <= lower_bound:
                    break

    if best_gaps is None:
        # Fallback: pure greedy without scope limit
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows

    # Phase 2: Differential Evolution to improve
    if time.time() < deadline - 20:
        de_time = deadline - 20
        improved_gaps = differential_evolution(
            best_gaps, best_scope, rng, de_time, population_size=25
        )
        if improved_gaps:
            rows = [row_from_gaps(g) for g in improved_gaps]
            if is_valid(rows):
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_gaps = improved_gaps

    # Phase 3: Local search refinement
    if time.time() < deadline - 3:
        ls_time = deadline - 3
        refined_gaps = local_search_gaps(best_gaps, best_scope, rng, ls_time)
        if refined_gaps:
            rows = [row_from_gaps(g) for g in refined_gaps]
            if is_valid(rows):
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_gaps = refined_gaps

    # Convert back to rows
    final_rows = [row_from_gaps(g) for g in best_gaps]

    # Final validation
    if not is_valid(final_rows):
        # Fallback to greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows

    return final_rows

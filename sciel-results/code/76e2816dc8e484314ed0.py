"""Combined approach: structural random-hypergraph seeding + greedy conflict-driven local search."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Start with the known valid n=9 coloring
    def seed_coloring(n):
        """Return a valid coloring for n=9 as baseline."""
        triples = list(combinations(range(n), 3))
        return [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]

    best_n = 9
    best_colors = seed_coloring(9)

    def build_indices(n):
        """Precompute K4 and K5 triple-index tuples and reverse indices."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))

        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))

        # Reverse indices
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)
        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)

        return triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s

    def count_violations(colors, k4s, k5s):
        """Count red K4s and blue K5s."""
        red_count = 0
        for idxs in k4s:
            if all(colors[i] == 0 for i in idxs):
                red_count += 1
        blue_count = 0
        for idxs in k5s:
            if all(colors[i] == 1 for i in idxs):
                blue_count += 1
        return red_count, blue_count

    def greedy_local_search(n, time_limit, seed=0):
        """Greedy local search prioritizing red K4 removal, then blue K5."""
        random.seed(seed)
        triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s = build_indices(n)

        # Random initialization with p=0.5
        colors = [random.randint(0, 1) for _ in range(num_triples)]

        # Track violation counts per triple for fast delta updates
        # red_k4_count[t] = number of red K4s containing triple t
        # blue_k5_count[t] = number of blue K5s containing triple t
        red_k4_count = [0] * num_triples
        blue_k5_count = [0] * num_triples

        def compute_initial_counts():
            nonlocal red_k4_count, blue_k5_count
            red_k4_count = [0] * num_triples
            blue_k5_count = [0] * num_triples
            for idxs in k4s:
                if all(colors[i] == 0 for i in idxs):
                    for t in idxs:
                        red_k4_count[t] += 1
            for idxs in k5s:
                if all(colors[i] == 1 for i in idxs):
                    for t in idxs:
                        blue_k5_count[t] += 1

        compute_initial_counts()

        total_red = sum(red_k4_count) // 4  # each K4 counted 4 times
        total_blue = sum(blue_k5_count) // 10  # each K5 counted 10 times

        best_colors_local = colors[:]
        best_total = total_red + total_blue

        no_improve = 0
        max_no_improve = 2000

        while time.time() < time_limit:
            if total_red + total_blue == 0:
                return colors, True

            # Find the most conflicting triple
            # Priority: red K4 first (rarer), then blue K5
            # Score a triple by how many violations it participates in
            best_t = -1
            best_score = -1

            # Sample a subset of triples to find conflicts (for speed on large n)
            if num_triples > 5000:
                candidates = random.sample(range(num_triples), min(1000, num_triples))
            else:
                candidates = range(num_triples)

            for t in candidates:
                # Score: weight red K4 participation higher
                score = red_k4_count[t] * 10 + blue_k5_count[t]
                if score > best_score:
                    best_score = score
                    best_t = t

            if best_t < 0 or best_score <= 0:
                # No conflicts found in sample, try full scan if small
                if num_triples <= 5000:
                    for t in range(num_triples):
                        score = red_k4_count[t] * 10 + blue_k5_count[t]
                        if score > best_score:
                            best_score = score
                            best_t = t
                if best_t < 0 or best_score <= 0:
                    break

            old_color = colors[best_t]
            new_color = 1 - old_color

            # Compute delta: how many violations are removed/added
            delta_red = 0
            delta_blue = 0

            # Update K4s containing best_t
            for k4_idx in triple_to_k4s[best_t]:
                idxs = k4s[k4_idx]
                red_before = sum(1 for i in idxs if colors[i] == 0)
                if old_color == 0:
                    red_after = red_before - 1
                else:
                    red_after = red_before + 1

                if red_before == 4 and red_after < 4:
                    delta_red -= 1
                    # Update counts for other triples in this K4
                    for i in idxs:
                        if i != best_t:
                            red_k4_count[i] -= 1
                elif red_before < 4 and red_after == 4:
                    delta_red += 1
                    for i in idxs:
                        if i != best_t:
                            red_k4_count[i] += 1

            # Update K5s containing best_t
            for k5_idx in triple_to_k5s[best_t]:
                idxs = k5s[k5_idx]
                blue_before = sum(1 for i in idxs if colors[i] == 1)
                if old_color == 1:
                    blue_after = blue_before - 1
                else:
                    blue_after = blue_before + 1

                if blue_before == 10 and blue_after < 10:
                    delta_blue -= 1
                    for i in idxs:
                        if i != best_t:
                            blue_k5_count[i] -= 1
                elif blue_before < 10 and blue_after == 10:
                    delta_blue += 1
                    for i in idxs:
                        if i != best_t:
                            blue_k5_count[i] += 1

            # Update the triple's own count
            red_k4_count[best_t] += delta_red
            blue_k5_count[best_t] += delta_blue

            total_red += delta_red
            total_blue += delta_blue

            # Accept if it reduces total violations
            if delta_red + delta_blue < 0:
                colors[best_t] = new_color
                no_improve = 0
                if total_red + total_blue < best_total:
                    best_total = total_red + total_blue
                    best_colors_local = colors[:]
                    if best_total == 0:
                        return colors, True
            else:
                # Don't accept worsening moves in greedy mode
                no_improve += 1
                if no_improve > max_no_improve:
                    # Random restart
                    colors = [random.randint(0, 1) for _ in range(num_triples)]
                    compute_initial_counts()
                    total_red = sum(red_k4_count) // 4
                    total_blue = sum(blue_k5_count) // 10
                    no_improve = 0

        return best_colors_local, (best_total == 0)

    # Try increasing n values
    for n in range(10, 35):
        if time.time() > deadline - 5:
            break

        # Allocate time budget
        if n <= 15:
            time_budget = min(3.0, (deadline - time.time()) * 0.3)
        elif n <= 25:
            time_budget = min(5.0, (deadline - time.time()) * 0.4)
        else:
            time_budget = min(8.0, (deadline - time.time()) * 0.5)

        if time_budget < 1.0:
            break

        found = False
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            colors, valid = greedy_local_search(n, deadline - 5, seed=n * 100 + attempt)
            if valid:
                # Verify
                triples, num_triples, k4s, k5s, _, _ = build_indices(n)
                red_count, blue_count = count_violations(colors, k4s, k5s)
                if red_count == 0 and blue_count == 0:
                    best_n = n
                    best_colors = colors
                    found = True
                    break

        if not found and time.time() > deadline - 10:
            break

    return {'n': best_n, 'colors': best_colors}

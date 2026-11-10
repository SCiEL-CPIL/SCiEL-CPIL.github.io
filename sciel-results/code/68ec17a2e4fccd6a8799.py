import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s

    n = 16
    all_points = list(range(n))

    # Precompute all triples as masks
    all_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                mask = (1 << i) | (1 << j) | (1 << k)
                all_triples.append(mask)

    num_triples = len(all_triples)  # 560

    # Precompute conflicts: for each triple, which other triples conflict?
    # Conflict if intersection size >= 2
    conflicts = [[] for _ in range(num_triples)]
    for i in range(num_triples):
        for j in range(i + 1, num_triples):
            inter = all_triples[i] & all_triples[j]
            if bin(inter).count('1') >= 2:
                conflicts[i].append(j)
                conflicts[j].append(i)

    # Precompute for each point, which triples contain it
    point_to_triples = [[] for _ in range(n)]
    for t_idx in range(num_triples):
        mask = all_triples[t_idx]
        for p in range(n):
            if mask & (1 << p):
                point_to_triples[p].append(t_idx)

    # Precompute masks by size for alpha calculation
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(1 << n):
        k = bin(mask).count('1')
        masks_by_size[k].append(mask)

    def calc_alpha(triples_indices):
        """Calculate the independence number (largest set containing no triple)."""
        if not triples_indices:
            return n

        t_masks = [all_triples[i] for i in triples_indices]

        # Check sizes descending
        for k in range(n, 0, -1):
            for s in masks_by_size[k]:
                is_ind = True
                for t in t_masks:
                    if (s & t) == t:
                        is_ind = False
                        break
                if is_ind:
                    return k
        return 0

    def is_valid_add(triple_idx, current_triples_set):
        """Check if triple_idx can be added without conflict."""
        for c in conflicts[triple_idx]:
            if c in current_triples_set:
                return False
        return True

    def greedy_pack():
        """Greedy construction with random ordering."""
        current_triples = set()
        available = list(range(num_triples))
        random.shuffle(available)

        for idx in available:
            if is_valid_add(idx, current_triples):
                current_triples.add(idx)

        return list(current_triples)

    def local_search(initial_triples, max_time=10.0):
        """Simulated annealing local search."""
        current_triples = set(initial_triples)
        current_alpha = calc_alpha(list(current_triples))
        best_triples = list(current_triples)
        best_alpha = current_alpha

        T = 2.0
        cooling_rate = 0.995
        iterations = 0
        max_iterations = 20000

        while time.time() - start_time < max_time and time.time() < deadline:
            new_triples = set(current_triples)
            move_type = random.random()

            if move_type < 0.4:
                # Add a random valid triple
                valid_adds = []
                for idx in range(num_triples):
                    if idx not in new_triples and is_valid_add(idx, new_triples):
                        valid_adds.append(idx)
                if valid_adds:
                    new_triples.add(random.choice(valid_adds))
                elif new_triples:
                    new_triples.remove(random.choice(list(new_triples)))

            elif move_type < 0.7:
                # Remove a random triple
                if new_triples:
                    new_triples.remove(random.choice(list(new_triples)))

            else:
                # Swap: remove one, add another
                if new_triples:
                    to_remove = random.choice(list(new_triples))
                    new_triples.remove(to_remove)
                    valid_adds = []
                    for idx in range(num_triples):
                        if idx not in new_triples and is_valid_add(idx, new_triples):
                            valid_adds.append(idx)
                    if valid_adds:
                        new_triples.add(random.choice(valid_adds))

            if new_triples == current_triples:
                continue

            new_alpha = calc_alpha(list(new_triples))
            delta = new_alpha - current_alpha

            if delta <= 0 or (delta > 0 and random.random() < (0.3 ** delta)):
                current_triples = new_triples
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
                    T = max(T, 1.5)

            iterations += 1
            T *= cooling_rate
            if T < 0.05:
                T = 0.05

        return best_triples, best_alpha

    best_alpha = n
    best_triples = None

    # Phase 1: Multiple greedy starts
    for _ in range(30):
        triples = greedy_pack()
        alpha = calc_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            if alpha <= 5:
                break

    # Phase 2: Local search from best and some random starts
    if best_triples is not None:
        # Local search from the best greedy result
        ls_triples, ls_alpha = local_search(best_triples, max_time=30.0)
        if ls_alpha < best_alpha:
            best_alpha = ls_alpha
            best_triples = ls_triples

        # Additional local searches from different greedy starts
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            triples = greedy_pack()
            alpha = calc_alpha(triples)
            if alpha <= best_alpha + 1:
                ls_triples, ls_alpha = local_search(triples, max_time=10.0)
                if ls_alpha < best_alpha:
                    best_alpha = ls_alpha
                    best_triples = ls_triples

    # Phase 3: More aggressive local search if time remains
    if best_triples is not None and time.time() < deadline - 5:
        # Try to improve further with longer local search
        ls_triples, ls_alpha = local_search(best_triples, max_time=deadline - time.time() - 2)
        if ls_alpha < best_alpha:
            best_alpha = ls_alpha
            best_triples = ls_triples

    if best_triples is None:
        # Fallback: simple construction
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    # Convert indices back to points
    result = []
    for idx in best_triples:
        mask = all_triples[idx]
        pts = []
        for i in range(n):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)

    return result

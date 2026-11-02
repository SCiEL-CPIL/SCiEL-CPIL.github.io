import time
import numpy as np
import random
from collections import defaultdict
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)

    # Fallback for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    # Precompute balls efficiently
    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Build balls array: balls[w] = [w, neighbors...]
    balls = np.zeros((total, ball_size), dtype=np.int32)

    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1

    # Build inverse mapping
    inv_lists = [None] * total
    for c in range(total):
        for w in balls[c]:
            if inv_lists[w] is None:
                inv_lists[w] = []
            inv_lists[w].append(c)

    best_size = total
    best_set = None

    np.random.seed(42)
    random.seed(42)

    def latin_hypercube_init(num_points):
        """
        Generate a diverse initial set using Latin Hypercube sampling.
        For each position, ensure all digit values are represented roughly equally.
        """
        selected = set()
        # For each position, create a permutation of digits and assign
        # This ensures balanced coverage across positions
        remaining = num_points
        while remaining > 0:
            # Create a random word with balanced properties
            # Use a random permutation for each position to ensure diversity
            word = []
            for pos in range(n):
                # Random digit but biased toward under-represented values
                word.append(random.randint(0, 2))
            w = 0
            for i, d in enumerate(word):
                w += d * weights[i]
            if w not in selected:
                selected.add(w)
                remaining -= 1
            if time.time() > deadline - 100.0:
                break
        return list(selected)

    def structured_linear_init():
        """
        Generate a structured initial set based on linear code structure.
        Use a parity-check matrix approach to generate codewords.
        """
        # Start with a random systematic generator matrix
        # For a [8,4] code over GF(3), we need 4 information positions
        # Use positions 0,1,2,3 as information, 4,5,6,7 as parity
        codewords = set()
        # Generate all 3^4 = 81 combinations for a [8,4] code
        # But we need 486 words, so use multiple parallel codes
        # Use 6 different parity-check structures
        for trial in range(6):
            # Random parity matrix
            P = np.random.randint(0, 3, size=(4, 4))
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        for d in range(3):
                            info = np.array([a, b, c, d], dtype=np.int64)
                            parity = (P @ info) % 3
                            word = np.concatenate([info, parity])
                            w = 0
                            for i, val in enumerate(word):
                                w += val * weights[i]
                            codewords.add(w)
        return list(codewords)

    def greedy_construct(init_set=None):
        """Randomized greedy set cover construction with optional initial set"""
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        uncovered_count = total

        if init_set:
            for c in init_set:
                for w in balls[c]:
                    if uncovered[w]:
                        uncovered_count -= 1
                        uncovered[w] = False
                    cover_count[w] += 1
                selected.append(c)

        # For each candidate center, track how many uncovered words it covers
        cand_count = np.zeros(total, dtype=np.int32)
        for c in range(total):
            if c in selected:
                continue
            cand_count[c] = np.sum(uncovered[balls[c]])

        active = np.ones(total, dtype=bool)
        for c in selected:
            active[c] = False

        while uncovered_count > 0:
            if time.time() > deadline - 10.0:
                break

            # Find the best candidate among active centers
            active_counts = np.where(active, cand_count, -1)
            max_count = np.max(active_counts)

            if max_count <= 0:
                break

            # Pick randomly among max-count active candidates for diversity
            candidates = np.where((active_counts == max_count) & active)[0]
            if len(candidates) == 0:
                break
            best_idx = candidates[np.random.randint(len(candidates))]

            selected.append(best_idx)
            active[best_idx] = False

            # Update cover counts and candidate counts
            for w in balls[best_idx]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                    uncovered[w] = False
                    # Decrement cand_count for all centers that cover w
                    for c in inv_lists[w]:
                        if active[c]:
                            cand_count[c] -= 1
                cover_count[w] += 1

        return selected, uncovered_count

    def dual_guided_set_cover(init_set=None):
        """
        Dual-guided set cover: uses shadow prices to prioritize hard-to-cover words.
        """
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        uncovered_count = total

        if init_set:
            for c in init_set:
                for w in balls[c]:
                    if uncovered[w]:
                        uncovered_count -= 1
                        uncovered[w] = False
                    cover_count[w] += 1
                selected.append(c)

        cand_count = np.zeros(total, dtype=np.int32)
        for c in range(total):
            if c in selected:
                continue
            cand_count[c] = np.sum(uncovered[balls[c]])

        active = np.ones(total, dtype=bool)
        for c in selected:
            active[c] = False

        while uncovered_count > 0:
            if time.time() > deadline - 10.0:
                break

            # Find uncovered words and compute their "difficulty"
            uncovered_words = np.where(uncovered)[0]
            if len(uncovered_words) == 0:
                break

            # Sample a subset of uncovered words for efficiency
            if len(uncovered_words) > 100:
                sampled = np.random.choice(uncovered_words, 100, replace=False)
            else:
                sampled = uncovered_words

            # For each sampled uncovered word, compute difficulty
            difficulties = np.zeros(len(sampled), dtype=np.float64)
            for i, w in enumerate(sampled):
                num_active_covering = sum(1 for c in inv_lists[w] if active[c])
                difficulties[i] = 1.0 / max(num_active_covering, 1)

            # Pick the hardest uncovered word
            hardest_idx = np.argmax(difficulties)
            hardest_word = sampled[hardest_idx]

            # Among active candidates that cover the hardest word, pick the one with highest cand_count
            candidates_covering = [c for c in inv_lists[hardest_word] if active[c]]
            if not candidates_covering:
                break

            best_candidate = max(candidates_covering, key=lambda c: cand_count[c])

            selected.append(best_candidate)
            active[best_candidate] = False

            # Update cover counts and candidate counts
            for w in balls[best_candidate]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                    uncovered[w] = False
                    for c in inv_lists[w]:
                        if active[c]:
                            cand_count[c] -= 1
                cover_count[w] += 1

        return selected, uncovered_count

    def prune_redundant(selected):
        """Remove codewords that are not needed"""
        if not selected:
            return selected

        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1

        changed = True
        iterations = 0
        while changed and time.time() < deadline - 3.0:
            changed = False
            iterations += 1
            if iterations > 50:
                break
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected

        return selected

    def local_search_swap(selected):
        """Swap-based local search optimization"""
        if not selected:
            return selected

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[balls[c]] += 1

            selected_set = set(selected)

            critical_candidates = []
            for c in selected:
                unique_words = [w for w in balls[c] if cover_count[w] == 1]
                if unique_words:
                    critical_candidates.append((c, unique_words))

            random.shuffle(critical_candidates)

            for c, unique_words in critical_candidates:
                if time.time() > deadline - 2.0:
                    break

                if not unique_words:
                    continue

                common_candidates = set(inv_lists[unique_words[0]])
                for w in unique_words[1:]:
                    common_candidates.intersection_update(inv_lists[w])

                for c_prime in common_candidates:
                    if c_prime not in selected_set:
                        selected.remove(c)
                        selected.append(c_prime)
                        selected_set.remove(c)
                        selected_set.add(c_prime)
                        improved = True
                        break
                if improved:
                    break

            if time.time() < deadline - 2.0:
                cover_count = np.zeros(total, dtype=np.int32)
                for c in selected:
                    cover_count[balls[c]] += 1

                new_selected = []
                for c in selected:
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                    else:
                        new_selected.append(c)

                if len(new_selected) < len(selected):
                    selected = new_selected
                    improved = True

        return selected

    def two_exchange_local_search(selected):
        """
        2-exchange local search: try removing 2 codewords and adding 1-2 others.
        This explores a larger neighborhood than single-swap.
        """
        if not selected:
            return selected

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[balls[c]] += 1

            selected_set = set(selected)
            n_selected = len(selected)

            if n_selected < 2:
                break

            # Try random pairs
            max_pairs = min(n_selected * (n_selected - 1) // 2, 200)
            pairs_tried = 0

            while pairs_tried < max_pairs and time.time() < deadline - 2.0:
                i = random.randint(0, n_selected - 1)
                j = random.randint(0, n_selected - 1)
                if i == j:
                    continue
                pairs_tried += 1

                c1, c2 = selected[i], selected[j]

                # Check if removing both would leave any word uncovered
                potentially_uncovered = set()
                for w in balls[c1]:
                    if cover_count[w] <= 1:
                        potentially_uncovered.add(w)
                for w in balls[c2]:
                    if cover_count[w] <= 1:
                        potentially_uncovered.add(w)

                if not potentially_uncovered:
                    # Both can be removed
                    selected.remove(c1)
                    selected.remove(c2)
                    selected_set.discard(c1)
                    selected_set.discard(c2)
                    improved = True
                    break

                # Try to find codewords that cover all potentially_uncovered words
                if len(potentially_uncovered) <= 3:
                    uw_list = list(potentially_uncovered)
                    common_candidates = set(inv_lists[uw_list[0]])
                    for w in uw_list[1:]:
                        common_candidates.intersection_update(inv_lists[w])

                    # Try single replacement
                    for c_prime in common_candidates:
                        if c_prime not in selected_set:
                            selected.remove(c1)
                            selected.remove(c2)
                            selected.append(c_prime)
                            selected_set.discard(c1)
                            selected_set.discard(c2)
                            selected_set.add(c_prime)
                            improved = True
                            break

                    if improved:
                        break

                    # Try double replacement
                    if len(common_candidates) >= 2:
                        cand_list = list(common_candidates)
                        random.shuffle(cand_list)
                        for idx1 in range(min(len(cand_list), 20)):
                            for idx2 in range(idx1 + 1, min(len(cand_list), 20)):
                                cp1, cp2 = cand_list[idx1], cand_list[idx2]
                                if cp1 not in selected_set and cp2 not in selected_set:
                                    selected.remove(c1)
                                    selected.remove(c2)
                                    selected.append(cp1)
                                    selected.append(cp2)
                                    selected_set.discard(c1)
                                    selected_set.discard(c2)
                                    selected_set.add(cp1)
                                    selected_set.add(cp2)
                                    improved = True
                                    break
                            if improved:
                                break

            # Final pruning
            if time.time() < deadline - 2.0:
                cover_count = np.zeros(total, dtype=np.int32)
                for c in selected:
                    cover_count[balls[c]] += 1

                new_selected = []
                for c in selected:
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                    else:
                        new_selected.append(c)

                if len(new_selected) < len(selected):
                    selected = new_selected
                    improved = True

        return selected

    # Main loop
    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1

        # Choose initialization strategy
        init_strategy = random.random()
        init_set = None

        if init_strategy < 0.4:
            # Latin Hypercube initialization
            init_size = random.randint(100, 200)
            init_set = latin_hypercube_init(init_size)
        elif init_strategy < 0.6:
            # Structured linear code initialization
            init_set = structured_linear_init()
        elif init_strategy < 0.8:
            # Small random initialization
            init_size = random.randint(50, 100)
            init_set = random.sample(range(total), init_size)
        # else: no initialization (pure greedy)

        # Choose construction strategy
        if random.random() < 0.5:
            selected, uncovered_count = greedy_construct(init_set)
        else:
            selected, uncovered_count = dual_guided_set_cover(init_set)

        # Check if valid
        if uncovered_count > 0:
            continue

        # Pruning
        selected = prune_redundant(selected)

        # Local Search: Swap-based optimization
        selected = local_search_swap(selected)

        # 2-exchange local search
        selected = two_exchange_local_search(selected)

        # Final pruning
        selected = prune_redundant(selected)

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

        # Early stop conditions
        if best_size <= 486 and time.time() > deadline - 15.0:
            break
        if time.time() > deadline - 8.0:
            break

    # Convert to list of digit lists
    if best_set is None:
        best_set = list(range(total))

    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)

    return result

import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline, start_time)

    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode

    # Generic fallback: return all words (valid format, worst case)
    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _construct_ternary_8_1_optimized(deadline, start_time):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    # Weights for positions (big-endian)
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)

    # Precompute balls for all words
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
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Precompute reverse balls: for each word, which codewords' balls contain it
    # This is the transpose of the balls matrix concept
    # ball_members[word] = list of codeword indices c such that word is in balls[c]
    ball_members = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            ball_members[w].append(c)

    # Precompute structured order
    structured_order = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    for e in range(3):
                        for f in range(3):
                            for g in range(3):
                                for h in range(3):
                                    w = (a * 3**7 + b * 3**6 + c * 3**5 + d * 3**4 +
                                         e * 3**3 + f * 3**2 + g * 3**1 + h)
                                    structured_order.append(w)

    def greedy_construct(uncovered_mask, order_hint=None):
        """Greedy set cover with optional order hint for tie-breaking."""
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        current_uncovered = uncovered_mask.copy()

        while np.any(current_uncovered):
            # Count uncovered neighbors for each word
            counts = np.sum(current_uncovered[balls], axis=1)
            max_count = np.max(counts)

            if max_count == 0:
                break

            # Get candidates with max count
            candidates = np.where(counts == max_count)[0]

            if order_hint is not None and len(candidates) > 1:
                order_arr = np.array(order_hint)
                candidate_positions = order_arr[candidates]
                best_pos = np.argmin(candidate_positions)
                best_idx = candidates[best_pos]
            else:
                best_idx = candidates[0]

            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)

        return selected, cover_count

    def prune(selected, cover_count):
        """Remove redundant codewords."""
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected, cover_count

    def prune_reverse(selected, cover_count):
        """Remove redundant codewords in reverse order."""
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in reversed(selected):
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = list(reversed(new_selected))
        return selected, cover_count

    def local_search(selected, cover_count, max_iter=50):
        """Swap-based local search."""
        selected_set = set(selected)
        improved = True
        iter_count = 0

        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            new_selected = list(selected)
            new_selected_set = set(new_selected)

            for i in range(len(new_selected)):
                c = new_selected[i]
                for w in balls[c]:
                    if w in new_selected_set:
                        continue

                    # Try swap: remove c, add w
                    cover_count[balls[c]] -= 1
                    if np.any(cover_count == 0):
                        cover_count[balls[c]] += 1
                        continue

                    cover_count[balls[w]] += 1

                    # Check if we can remove another codeword
                    removable_d = None
                    for j in range(len(new_selected)):
                        if j == i:
                            continue
                        d = new_selected[j]
                        if np.min(cover_count[balls[d]]) > 1:
                            removable_d = d
                            break

                    if removable_d is not None:
                        cover_count[balls[removable_d]] -= 1
                        new_selected[i] = w
                        new_selected_set.remove(c)
                        new_selected_set.add(w)
                        new_selected.remove(removable_d)
                        new_selected_set.remove(removable_d)
                        improved = True
                        selected = new_selected
                        selected_set = new_selected_set
                        break
                    else:
                        cover_count[balls[w]] -= 1
                        cover_count[balls[c]] += 1

                    if improved:
                        break
                if improved:
                    break

        return selected, cover_count

    def latin_hypercube_init(rng, target_size):
        """Generate diverse initial codewords using Latin hypercube-inspired sampling.
        
        For each position, ensure all 3 values appear roughly equally across the
        selected codewords. This creates a diverse, low-correlation initial set.
        """
        selected = set()
        # For each position, track how many times each value has been used
        pos_value_counts = [[0, 0, 0] for _ in range(n)]
        
        # Pre-shuffle positions for diversity
        pos_order = list(range(n))
        rng.shuffle(pos_order)
        
        while len(selected) < target_size:
            if time.time() > deadline - 30.0:
                break
            
            # Generate a random word
            word = []
            for pos in pos_order:
                # Pick value with least usage for this position
                counts = pos_value_counts[pos]
                min_count = min(counts)
                candidates = [v for v in range(3) if counts[v] == min_count]
                val = rng.choice(candidates)
                word.append(val)
                pos_value_counts[pos][val] += 1
            
            # Convert to integer
            w = 0
            for pos in range(n):
                w = w * 3 + word[pos]
            
            if w not in selected:
                selected.add(w)
        
        return list(selected)

    def vertex_cover_heuristic(uncovered_words, existing_selected, cover_count):
        """Greedy minimum vertex cover on the subgraph of uncovered words.
        
        For each uncovered word, find candidate codewords that cover it.
        Greedily pick the codeword that covers the most uncovered words.
        """
        selected = list(existing_selected)
        cover_count = cover_count.copy()
        uncovered = set(uncovered_words)
        
        while uncovered:
            if time.time() > deadline - 15.0:
                break
            
            # For each candidate, count how many uncovered words it covers
            best_candidate = None
            best_count = 0
            
            # Only consider candidates that are in ball_members of uncovered words
            candidate_counts = {}
            for u in uncovered:
                for c in ball_members[u]:
                    candidate_counts[c] = candidate_counts.get(c, 0) + 1
            
            if not candidate_counts:
                break
            
            # Find the candidate with max uncovered coverage
            max_cov = max(candidate_counts.values())
            top_candidates = [c for c, cnt in candidate_counts.items() if cnt == max_cov]
            
            # Pick the one that's already selected or closest to existing selected
            best_c = top_candidates[0]
            if len(top_candidates) > 1:
                # Prefer already-selected codewords
                sel_set = set(selected)
                for c in top_candidates:
                    if c in sel_set:
                        best_c = c
                        break
            
            selected.append(best_c)
            cover_count[balls[best_c]] += 1
            
            # Remove newly covered words from uncovered
            for w in balls[best_c]:
                if w in uncovered:
                    uncovered.discard(w)
        
        return selected, cover_count, uncovered

    # Phase 1: Multiple greedy constructions with different tie-breaking
    best_selected = None
    best_cover = None
    best_size = float('inf')

    rng = random.Random(42)
    full_order = list(range(total))

    # Try several different orderings
    num_greedy = 10
    for g in range(num_greedy):
        if time.time() > deadline - 40.0:
            break

        if g == 0:
            order = None  # Default (smallest index)
        elif g == 1:
            order = list(reversed(full_order))  # Largest index first
        elif g == 2:
            order = structured_order  # Algebraic structured order
        elif g == 3:
            order = list(reversed(structured_order))
        else:
            order = full_order[:]
            rng.shuffle(order)

        selected, cover_count = greedy_construct(np.ones(total, dtype=bool), order)
        selected, cover_count = prune(selected, cover_count)
        selected, cover_count = prune_reverse(selected, cover_count)

        if len(selected) < best_size:
            best_selected = list(selected)
            best_cover = cover_count.copy()
            best_size = len(selected)

    # Phase 2: Latin hypercube initialization + vertex cover heuristic
    if best_selected is not None and time.time() < deadline - 30.0:
        # Try multiple LH initializations
        num_lh = 5
        for lh in range(num_lh):
            if time.time() > deadline - 25.0:
                break
            
            # Target size based on best known
            target_size = max(400, best_size - 20)
            
            lh_initial = latin_hypercube_init(rng, target_size)
            
            # Compute cover count for LH initial
            lh_cover = np.zeros(total, dtype=np.int32)
            for c in lh_initial:
                lh_cover[balls[c]] += 1
            
            # Find uncovered words
            uncovered_mask = (lh_cover == 0)
            uncovered_words = np.where(uncovered_mask)[0]
            
            # Apply vertex cover heuristic to fill gaps
            lh_selected, lh_cover, remaining_uncovered = vertex_cover_heuristic(
                list(uncovered_words), lh_initial, lh_cover)
            
            # Prune
            lh_selected, lh_cover = prune(lh_selected, lh_cover)
            lh_selected, lh_cover = prune_reverse(lh_selected, lh_cover)
            
            if len(lh_selected) < best_size:
                best_selected = list(lh_selected)
                best_cover = lh_cover.copy()
                best_size = len(lh_selected)

    # Phase 3: Local search from best solution
    if best_selected is not None and time.time() < deadline - 15.0:
        selected, cover_count = local_search(best_selected, best_cover, max_iter=100)
        selected, cover_count = prune(selected, cover_count)
        selected, cover_count = prune_reverse(selected, cover_count)

        if len(selected) < best_size:
            best_selected = list(selected)
            best_cover = cover_count.copy()
            best_size = len(selected)

    # Phase 4: Perturbation and repair
    max_restarts = 15
    for r in range(max_restarts):
        if time.time() > deadline - 10.0:
            break

        selected = list(best_selected)
        cover_count = best_cover.copy()

        # Perturbation: remove k random codewords
        k = rng.randint(2, min(10, len(selected) // 8))
        if len(selected) > k:
            to_remove = rng.sample(selected, k)
            to_remove_set = set(to_remove)
            selected = [c for c in selected if c not in to_remove_set]

            # Recompute cover_count
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[balls[c]] += 1

            # Repair: add words to cover uncovered positions
            current_uncovered = (cover_count == 0)
            while np.any(current_uncovered):
                if time.time() > deadline - 12.0:
                    break
                counts = np.sum(current_uncovered[balls], axis=1)
                max_count = np.max(counts)
                if max_count == 0:
                    break
                candidates = np.where(counts == max_count)[0]
                if len(candidates) > 1:
                    best_idx = candidates[rng.randint(0, len(candidates) - 1)]
                else:
                    best_idx = candidates[0]
                selected.append(best_idx)
                cover_count[balls[best_idx]] += 1
                current_uncovered = (cover_count == 0)

        # Prune
        selected, cover_count = prune(selected, cover_count)
        selected, cover_count = prune_reverse(selected, cover_count)

        # Local search
        selected, cover_count = local_search(selected, cover_count, max_iter=30)
        selected, cover_count = prune(selected, cover_count)

        # Accept if better
        if len(selected) < best_size:
            best_selected = list(selected)
            best_cover = cover_count.copy()
            best_size = len(selected)

    # Final pruning
    selected, cover_count = prune(best_selected, best_cover)
    selected, cover_count = prune_reverse(selected, cover_count)

    # Final verification
    final_cover = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cover[balls[c]] += 1

    if np.any(final_cover == 0):
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            selected.append(m)
            final_cover[balls[m]] += 1

    # Convert to digit lists
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)

    return result

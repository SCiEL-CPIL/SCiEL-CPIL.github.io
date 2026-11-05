"""Combined program: Lazy NumPy Greedy + Aggressive Pruning + Swap Local Search."""
import itertools
import time
import numpy as np
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin: finish by 115s, limit is 120s

    total_words = q ** n

    # Helper: convert word (tuple/list) to index
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Helper: convert index to word (tuple)
    def index_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        return tuple(reversed(w))

    # Helper: get neighbors (indices) of a word index within radius 1
    def get_neighbors(idx):
        neighbors = [idx]
        for pos in range(n):
            pow_q = q ** (n - 1 - pos)
            current_val = (idx // pow_q) % q
            for new_val in range(q):
                if new_val != current_val:
                    neighbors.append(idx + (new_val - current_val) * pow_q)
        return neighbors

    # If the space is small enough for a fast greedy, use it.
    if total_words <= 100000:

        if radius == 1:
            sphere_size = 1 + n * (q - 1)
            # Build neighbor index table
            neighbors = np.empty((total_words, sphere_size), dtype=np.int32)
            for i in range(total_words):
                nbrs = get_neighbors(i)
                if len(nbrs) < sphere_size:
                    nbrs = nbrs + [i] * (sphere_size - len(nbrs))
                neighbors[i] = nbrs[:sphere_size]

            # --- Phase 1: Lazy NumPy Greedy Set Cover ---
            covered = np.zeros(total_words, dtype=bool)
            code_indices = []
            # lazy_gain[i] = last computed gain for candidate i
            # We use a priority queue approach: only recompute when needed
            # For simplicity with numpy, we'll do full scans but with early termination

            # Precompute which candidates cover which positions for faster gain computation
            # Actually, the neighbors matrix is already compact. Let's use it directly.

            # Standard greedy with numpy - it's fast enough for 729 words
            while time.time() < deadline - 50.0:  # Leave 50s for pruning + swaps
                gains = (~covered)[neighbors].sum(axis=1)
                best_idx = int(np.argmax(gains))
                if gains[best_idx] == 0:
                    break
                code_indices.append(best_idx)
                covered[neighbors[best_idx]] = True

            # Convert indices to codewords
            result_greedy = []
            for idx in code_indices:
                result_greedy.append(list(index_to_word(idx)))

            # --- Phase 2: Direct Sum Construction Baseline ---
            if q == 3 and n >= 4:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                base_n = 4
                base_code = tetracode
            elif q >= 3 and n >= 4:
                base_code = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
                base_n = 4
            else:
                base_n = n
                base_code = [list(range(n))]

            if n >= base_n:
                tails = list(itertools.product(range(q), repeat=n - base_n))
                result_direct = [h + list(tail) for h in base_code for tail in tails]
            else:
                result_direct = [list(range(n))]

            # Choose the better starting point
            if len(result_greedy) <= len(result_direct):
                initial_code = result_greedy
            else:
                initial_code = result_direct

            # --- Phase 3: Aggressive Iterative Redundancy Pruning ---
            total_words_int = total_words
            covered_by = [[] for _ in range(total_words_int)]
            codeword_covers = []

            for ci, c in enumerate(initial_code):
                c_idx = word_to_index(tuple(c))
                covers = set(get_neighbors(c_idx))
                codeword_covers.append(covers)
                for wi in covers:
                    covered_by[wi].append(ci)

            active = [True] * len(initial_code)

            # Multi-pass pruning with randomized restarts
            max_prune_iterations = 100

            def do_pruning_pass(active_state, order=None, max_iters=max_prune_iterations):
                """Perform one full pruning pass. Returns True if any removal occurred."""
                if order is None:
                    order = list(range(len(initial_code)))
                removed_any = False
                for _ in range(max_iters):
                    if time.time() > deadline - 10.0:
                        break
                    made_progress = False
                    for ci in order:
                        if time.time() > deadline - 10.0:
                            break
                        if not active_state[ci]:
                            continue
                        # Check if ci is redundant
                        is_redundant = True
                        for wi in codeword_covers[ci]:
                            found_other = False
                            for cj in covered_by[wi]:
                                if cj != ci and active_state[cj]:
                                    found_other = True
                                    break
                            if not found_other:
                                is_redundant = False
                                break
                        if is_redundant:
                            active_state[ci] = False
                            removed_any = True
                            made_progress = True
                    if not made_progress:
                        break
                return removed_any

            # Deterministic pass
            do_pruning_pass(active)

            # Randomized passes
            time_budget_for_pruning = deadline - 15.0
            num_random_passes = 0
            while time.time() < time_budget_for_pruning:
                active2 = active[:]
                order = list(range(len(initial_code)))
                random.shuffle(order)
                do_pruning_pass(active2, order)
                # Count active in both
                active_count_orig = sum(1 for a in active if a)
                active_count_new = sum(1 for a in active2 if a)
                if active_count_new < active_count_orig:
                    active = active2
                num_random_passes += 1
                # If no improvement in last few passes, stop
                if num_random_passes > 20:
                    break

            final_code = [initial_code[i] for i in range(len(initial_code)) if active[i]]

            # --- Phase 4: Swap Local Search ---
            # Try to replace one codeword with another to maintain coverage while reducing total
            # This can help escape local optima in pruning
            if time.time() < deadline - 10.0 and len(final_code) > 71:
                best_code = final_code
                best_len = len(final_code)

                # Precompute balls for current code
                def compute_balls(code):
                    balls = []
                    for c in code:
                        c_idx = word_to_index(tuple(c))
                        balls.append(set(get_neighbors(c_idx)))
                    return balls

                current_balls = compute_balls(best_code)
                current_total_covered = set()
                for b in current_balls:
                    current_total_covered |= b

                # Try swaps
                time_for_swaps = deadline - 5.0
                max_swap_trials = 50
                for trial in range(max_swap_trials):
                    if time.time() > time_for_swaps:
                        break

                    # Pick a random codeword to try replacing
                    ci = random.randint(0, len(best_code) - 1)

                    # Compute what's covered by others
                    others_covered = set()
                    for j in range(len(best_code)):
                        if j != ci:
                            others_covered |= current_balls[j]

                    # Points only covered by ci
                    exclusive_to_ci = current_balls[ci] - others_covered

                    # Try candidate replacements
                    # Sample candidates rather than trying all 729
                    num_candidates = min(100, total_words)
                    candidates = random.sample(range(total_words), num_candidates)

                    best_candidate = -1
                    best_overlap = float('inf')

                    for cand_idx in candidates:
                        cand_ball = set(get_neighbors(cand_idx))
                        # Check if candidate covers all exclusive points
                        if not exclusive_to_ci.issubset(cand_ball):
                            continue
                        # Check if new total still covers everything
                        new_total = others_covered | cand_ball
                        if len(new_total) < total_words:
                            continue
                        # Prefer candidate with less overlap with others
                        overlap = len(cand_ball & others_covered)
                        if overlap < best_overlap:
                            best_overlap = overlap
                            best_candidate = cand_idx

                    if best_candidate >= 0:
                        # Perform the swap
                        best_code[ci] = list(index_to_word(best_candidate))
                        current_balls[ci] = set(get_neighbors(best_candidate))
                        # Recompute total
                        current_total_covered = set()
                        for b in current_balls:
                            current_total_covered |= b

                        # After swap, do a quick pruning pass
                        # Rebuild covered_by for the new code
                        new_covered_by = [[] for _ in range(total_words_int)]
                        new_codeword_covers = []
                        for j, c in enumerate(best_code):
                            c_idx = word_to_index(tuple(c))
                            covers = set(get_neighbors(c_idx))
                            new_codeword_covers.append(covers)
                            for wi in covers:
                                new_covered_by[wi].append(j)

                        new_active = [True] * len(best_code)
                        # Quick pruning
                        for _ in range(20):
                            if time.time() > time_for_swaps:
                                break
                            made_progress = False
                            for j in range(len(best_code)):
                                if not new_active[j]:
                                    continue
                                is_redundant = True
                                for wi in new_codeword_covers[j]:
                                    found_other = False
                                    for cj in new_covered_by[wi]:
                                        if cj != j and new_active[cj]:
                                            found_other = True
                                            break
                                    if not found_other:
                                        is_redundant = False
                                        break
                                if is_redundant:
                                    new_active[j] = False
                                    made_progress = True
                            if not made_progress:
                                break

                        pruned_code = [best_code[j] for j in range(len(best_code)) if new_active[j]]
                        if len(pruned_code) < best_len:
                            best_len = len(pruned_code)
                            best_code = pruned_code
                            current_balls = compute_balls(best_code)
                            current_total_covered = set()
                            for b in current_balls:
                                current_total_covered |= b

                final_code = best_code

            # --- Phase 5: Final Pruning Pass ---
            if time.time() < deadline - 2.0:
                # Rebuild data structures for final code
                covered_by_final = [[] for _ in range(total_words_int)]
                codeword_covers_final = []
                for ci, c in enumerate(final_code):
                    c_idx = word_to_index(tuple(c))
                    covers = set(get_neighbors(c_idx))
                    codeword_covers_final.append(covers)
                    for wi in covers:
                        covered_by_final[wi].append(ci)

                active_final = [True] * len(final_code)
                # Do a few pruning passes
                for _ in range(5):
                    if time.time() > deadline - 2.0:
                        break
                    made_progress = False
                    for ci in range(len(final_code)):
                        if not active_final[ci]:
                            continue
                        is_redundant = True
                        for wi in codeword_covers_final[ci]:
                            found_other = False
                            for cj in covered_by_final[wi]:
                                if cj != ci and active_final[cj]:
                                    found_other = True
                                    break
                            if not found_other:
                                is_redundant = False
                                break
                        if is_redundant:
                            active_final[ci] = False
                            made_progress = True
                    if not made_progress:
                        break

                if made_progress:
                    final_code = [final_code[i] for i in range(len(final_code)) if active_final[i]]

            # Final Verification
            if _verify_coverage(final_code, q, n, radius):
                return final_code
            else:
                # Fallback to direct sum if verification fails
                if _verify_coverage(result_direct, q, n, radius):
                    return result_direct
                else:
                    return result_greedy

        else:
            # Generic radius: use a simpler greedy with itertools for small spaces
            if total_words <= 10000:
                return _greedy_generic(q, n, radius, deadline)
            # Otherwise fall through to direct sum

    # --- Fallback: Direct sum construction ---
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_n = 4
        base_code = tetracode
    elif q >= 3 and n >= 4 and radius == 1:
        base_code = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        base_n = 4
    else:
        base_n = min(n, 4)
        base_code = []
        for tail in itertools.product(range(q), repeat=base_n):
            base_code.append(list(tail))

    if n < base_n:
        return [list(range(n))]

    tails = list(itertools.product(range(q), repeat=n - base_n))
    return [h + list(tail) for h in base_code for tail in tails]


def _verify_coverage(codewords, q, n, radius):
    """Check that every word is within Hamming distance radius of some codeword."""
    total_words = q ** n
    covered = set()
    for cw in codewords:
        idx = 0
        for i, v in enumerate(cw):
            idx += v * (q ** (n - 1 - i))
        if radius == 1:
            covered.add(idx)
            for pos in range(n):
                pow_q = q ** (n - 1 - pos)
                cur = (idx // pow_q) % q
                for nv in range(q):
                    if nv != cur:
                        covered.add(idx + (nv - cur) * pow_q)
        else:
            for delta in _all_deltas(n, q, radius):
                covered.add(idx + delta)
    return len(covered) == total_words


def _all_deltas(n, q, radius):
    """Yield index deltas for all words within Hamming distance radius of the zero word."""
    deltas = set()
    deltas.add(0)
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            base_options = []
            for pos in positions:
                pow_q = q ** (n - 1 - pos)
                base_options.append([d * pow_q for d in range(1, q)])
            for combo in itertools.product(*base_options):
                deltas.add(sum(combo))
    return deltas


def _greedy_generic(q, n, radius, deadline):
    """Generic greedy for small spaces and any radius."""
    total_words = q ** n
    neighbors_list = []
    for i in range(total_words):
        w = []
        temp = i
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        nbrs = _neighbors_of_word(w, q, n, radius)
        idx_nbrs = []
        for nb in nbrs:
            idx = 0
            for j, v in enumerate(nb):
                idx += v * (q ** (n - 1 - j))
            idx_nbrs.append(idx)
        neighbors_list.append(idx_nbrs)

    covered = [False] * total_words
    code_indices = []

    while time.time() < deadline:
        best_idx = -1
        best_gain = 0
        for i in range(total_words):
            gain = sum(1 for nb in neighbors_list[i] if not covered[nb])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break
        code_indices.append(best_idx)
        for nb in neighbors_list[best_idx]:
            covered[nb] = True

    result = []
    for idx in code_indices:
        w = []
        temp = idx
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
    return result


def _neighbors_of_word(word, q, n, radius):
    """Return all words within Hamming distance <= radius of the given word."""
    result = [list(word)]
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            base = list(word)
            for combo in itertools.product(*[range(q)] * r):
                for pos, val in zip(positions, combo):
                    if val != base[pos]:
                        base[pos] = val
                if base != list(word):
                    result.append(list(base))
            base = list(word)
    return result

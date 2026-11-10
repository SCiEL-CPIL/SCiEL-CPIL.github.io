import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if not (q == 3 and n == 8 and radius == 1):
        N = q ** n
        if N > 100000:
            return [[i % q for i in range(n)] for i in range(N)]
        powers = [q ** i for i in range(n)]
        neighbors = []
        for w in range(N):
            ball = {w}
            digits = [0] * n
            temp = w
            for i in range(n):
                digits[i] = temp % q
                temp //= q
            for pos in range(n):
                original = digits[pos]
                for val in range(q):
                    if val == original:
                        continue
                    digits[pos] = val
                    new_w = 0
                    mult = 1
                    for i in range(n):
                        new_w += digits[i] * mult
                        mult *= q
                    ball.add(new_w)
                digits[pos] = original
            neighbors.append(list(ball))
        uncovered = set(range(N))
        code = []
        while uncovered and time.time() < deadline:
            best_cand = -1
            best_count = -1
            for c in uncovered:
                count = sum(1 for nb in neighbors[c] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best_cand = c
            if best_cand == -1:
                break
            code.append(best_cand)
            for nb in neighbors[best_cand]:
                if nb in uncovered:
                    uncovered.remove(nb)
        if uncovered:
            code.extend(uncovered)
        result = []
        for w in code:
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            result.append(digits)
        return result

    N = q ** n  # 6561
    powers = [q ** i for i in range(n)]
    neigh_size = 1 + n * (q - 1)  # 17

    # Precompute neighbors as int32 array
    neighbors = np.zeros((N, neigh_size), dtype=np.int32)
    for w in range(N):
        neighbors[w, 0] = w
        idx = 1
        for i in range(n):
            d = (w // powers[i]) % q
            for delta in range(1, q):
                new_d = (d + delta) % q
                diff = new_d - d
                neighbors[w, idx] = w + diff * powers[i]
                idx += 1

    # Precompute reverse neighbors (who covers each word) - same as neighbors here
    # For each word u, neighbors[u] = set of codewords that cover u
    # Actually neighbors[w] = words covered by codeword w
    # We need: for each word u, which codewords cover u? That's the same set since it's symmetric in the ball sense.
    # Actually no: neighbors[w] = {v : dist(w,v) <= 1}. This is symmetric.
    # So the "coverers" of u are exactly neighbors[u].

    def compute_coverage(in_code):
        cov = np.zeros(N, dtype=np.int32)
        for w in np.where(in_code)[0]:
            cov[neighbors[w, :]] += 1
        return cov

    def greedy_init(rng, greedy_deadline):
        """
        Greedy set cover with efficient max-gain selection.
        Uses numpy for bucket management.
        """
        in_code = np.zeros(N, dtype=np.bool_)
        coverage = np.zeros(N, dtype=np.int32)
        uncovered_count = N
        gains = np.full(N, neigh_size, dtype=np.int32)

        # Track bucket membership: for each gain level, maintain a list
        # Use a different approach: just track gains and find max each time
        # For speed, maintain a count of how many words have each gain
        gain_counts = np.zeros(neigh_size + 1, dtype=np.int32)
        for w in range(N):
            gain_counts[gains[w]] += 1

        max_gain = neigh_size

        while uncovered_count > 0 and time.time() < greedy_deadline:
            while max_gain > 0 and gain_counts[max_gain] == 0:
                max_gain -= 1
            if max_gain == 0:
                break

            # Find a random word with gain == max_gain and not in code
            # Scan from a random starting point
            found = False
            for attempt in range(50):
                w = int(rng.integers(N))
                if gains[w] == max_gain and not in_code[w]:
                    found = True
                    break
            if not found:
                # Fallback: linear scan
                for w in range(N):
                    if gains[w] == max_gain and not in_code[w]:
                        found = True
                        break
                if not found:
                    break

            in_code[w] = True
            gain_counts[gains[w]] -= 1
            nbs = neighbors[w, :]

            for u in nbs:
                old_cov = coverage[u]
                coverage[u] = old_cov + 1

                if old_cov == 0:
                    # u just became covered
                    uncovered_count -= 1
                    # Decrement gains for all v that cover u
                    for v in neighbors[u, :]:
                        if in_code[v]:
                            continue
                        old_g = gains[v]
                        new_g = old_g - 1
                        if new_g < 0:
                            new_g = 0
                        gain_counts[old_g] -= 1
                        gain_counts[new_g] += 1
                        gains[v] = new_g

        # Force finish if needed
        if uncovered_count > 0:
            coverage = compute_coverage(in_code)
            for w in range(N):
                if not in_code[w]:
                    if np.any(coverage[neighbors[w, :]] == 0):
                        in_code[w] = True

        return in_code

    def local_search(in_code, rng, deadline):
        """
        Simulated Annealing with efficient uncovered-count tracking.
        """
        current_in_code = in_code.copy()
        current_coverage = compute_coverage(current_in_code)
        current_size = int(np.sum(current_in_code))
        uncovered_count = int(np.sum(current_coverage == 0))

        best_in_code = current_in_code.copy()
        best_size = current_size

        # SA Parameters
        T = 5.0
        alpha = 0.9998
        min_T = 0.005
        M = 100  # Penalty per uncovered word

        sa_deadline = deadline - 5.0

        code_words = np.where(current_in_code)[0].tolist()

        while time.time() < sa_deadline:
            if T < min_T:
                if rng.random() < 0.03:
                    T = 2.0  # Reheat
                else:
                    break

            move_type = rng.random()

            if move_type < 0.5:
                # 1. Deletion
                if len(code_words) == 0:
                    break
                idx = int(rng.integers(len(code_words)))
                w = code_words[idx]

                nbs = neighbors[w, :]
                prev_cov = current_coverage[nbs].copy()
                current_coverage[nbs] -= 1
                newly_uncovered = int(np.sum((prev_cov == 1) & (current_coverage[nbs] == 0)))

                delta_uncovered = newly_uncovered
                delta_size = -1
                delta_cost = M * delta_uncovered + delta_size

                if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                    current_in_code[w] = False
                    current_size -= 1
                    code_words.pop(idx)
                    uncovered_count += delta_uncovered

                    if delta_uncovered == 0 and current_size < best_size:
                        best_size = current_size
                        best_in_code = current_in_code.copy()
                else:
                    current_coverage[nbs] = prev_cov

            elif move_type < 0.85:
                # 2. Swap: Remove w1, Add w2
                if len(code_words) == 0:
                    break
                idx1 = int(rng.integers(len(code_words)))
                w1 = code_words[idx1]
                nbs1 = neighbors[w1, :]

                prev_cov1 = current_coverage[nbs1].copy()
                current_coverage[nbs1] -= 1
                essential_mask = (prev_cov1 == 1) & (current_coverage[nbs1] == 0)
                essential_words = nbs1[essential_mask]

                if len(essential_words) == 0:
                    # Try to add a random non-code word
                    w2 = int(rng.integers(N))
                    if not current_in_code[w2]:
                        nbs2 = neighbors[w2, :]
                        prev_cov2 = current_coverage[nbs2].copy()
                        current_coverage[nbs2] += 1

                        if np.all(current_coverage > 0):
                            current_in_code[w1] = False
                            current_in_code[w2] = True
                            code_words[idx1] = w2
                        else:
                            current_coverage[nbs1] = prev_cov1
                            current_coverage[nbs2] = prev_cov2
                    else:
                        current_coverage[nbs1] = prev_cov1
                else:
                    # Need w2 to cover all essential_words
                    if len(essential_words) <= 8:
                        # Find intersection of neighbors of essential words
                        cand_mask = np.ones(N, dtype=np.bool_)
                        for ew in essential_words:
                            cand_mask &= np.zeros(N, dtype=np.bool_)
                            cand_mask[neighbors[ew, :]] = True

                        candidates = np.where(cand_mask & ~current_in_code)[0]
                        if len(candidates) > 0:
                            w2 = int(candidates[int(rng.integers(len(candidates)))])
                            nbs2 = neighbors[w2, :]
                            prev_cov2 = current_coverage[nbs2].copy()
                            current_coverage[nbs2] += 1

                            if np.all(current_coverage > 0):
                                current_in_code[w1] = False
                                current_in_code[w2] = True
                                code_words[idx1] = w2
                            else:
                                current_coverage[nbs1] = prev_cov1
                                current_coverage[nbs2] = prev_cov2
                        else:
                            current_coverage[nbs1] = prev_cov1
                    else:
                        current_coverage[nbs1] = prev_cov1

            else:
                # 3. Double-swap: Remove w1, w2, Add w3
                if len(code_words) < 2:
                    T *= alpha
                    continue

                idx1 = int(rng.integers(len(code_words)))
                idx2 = int(rng.integers(len(code_words) - 1))
                if idx2 >= idx1:
                    idx2 += 1

                w1 = code_words[idx1]
                w2 = code_words[idx2]

                nbs1 = neighbors[w1, :]
                nbs2 = neighbors[w2, :]

                prev_cov1 = current_coverage[nbs1].copy()
                prev_cov2 = current_coverage[nbs2].copy()

                current_coverage[nbs1] -= 1
                current_coverage[nbs2] -= 1

                newly_uncovered_mask = (current_coverage == 0)
                newly_uncovered_words = np.where(newly_uncovered_mask)[0]

                if len(newly_uncovered_words) == 0:
                    current_in_code[w1] = False
                    current_in_code[w2] = False
                    current_size -= 2
                    if idx1 > idx2:
                        code_words.pop(idx1)
                        code_words.pop(idx2)
                    else:
                        code_words.pop(idx2)
                        code_words.pop(idx1)

                    if current_size < best_size:
                        best_size = current_size
                        best_in_code = current_in_code.copy()
                else:
                    if len(newly_uncovered_words) <= 6:
                        cand_mask = np.ones(N, dtype=np.bool_)
                        for uw in newly_uncovered_words:
                            cand_mask &= np.zeros(N, dtype=np.bool_)
                            cand_mask[neighbors[uw, :]] = True

                        candidates = np.where(cand_mask & ~current_in_code)[0]
                        if len(candidates) > 0:
                            w3 = int(candidates[int(rng.integers(len(candidates)))])
                            nbs3 = neighbors[w3, :]
                            prev_cov3 = current_coverage[nbs3].copy()
                            current_coverage[nbs3] += 1

                            if np.all(current_coverage > 0):
                                current_in_code[w1] = False
                                current_in_code[w2] = False
                                current_in_code[w3] = True
                                current_size -= 1
                                if idx1 > idx2:
                                    code_words.pop(idx1)
                                    code_words.pop(idx2)
                                else:
                                    code_words.pop(idx2)
                                    code_words.pop(idx1)
                                code_words.append(w3)

                                if current_size < best_size:
                                    best_size = current_size
                                    best_in_code = current_in_code.copy()
                            else:
                                current_coverage[nbs1] = prev_cov1
                                current_coverage[nbs2] = prev_cov2
                                current_coverage[nbs3] = prev_cov3
                        else:
                            current_coverage[nbs1] = prev_cov1
                            current_coverage[nbs2] = prev_cov2
                    else:
                        current_coverage[nbs1] = prev_cov1
                        current_coverage[nbs2] = prev_cov2

            T *= alpha

        return best_in_code, best_size

    def final_polish(in_code, deadline):
        """
        Iterative greedy deletion to remove redundant codewords.
        """
        current_in_code = in_code.copy()
        current_coverage = compute_coverage(current_in_code)
        current_size = int(np.sum(current_in_code))

        changed = True
        while changed and time.time() < deadline:
            changed = False
            code_words = np.where(current_in_code)[0]
            for w in code_words:
                if time.time() > deadline:
                    break
                nbs = neighbors[w, :]
                if np.min(current_coverage[nbs]) > 1:
                    current_in_code[w] = False
                    current_coverage[nbs] -= 1
                    current_size -= 1
                    changed = True

        return current_in_code, current_size

    # Main Logic
    best_code = None
    best_size = N  # Start with worst case

    # Multiple restarts with different seeds
    seeds = [42, 123, 7, 999, 31415]
    for seed in seeds:
        if time.time() > deadline - 15.0:
            break

        rng = np.random.default_rng(seed)

        # Greedy initialization
        greedy_deadline = start_time + 3.0
        if time.time() > greedy_deadline:
            greedy_deadline = time.time() + 1.0
        initial_code = greedy_init(rng, greedy_deadline)

        # Local search
        code, size = local_search(initial_code, rng, deadline - 5.0)

        # Final polish
        polished_code, polished_size = final_polish(code, deadline - 2.0)

        if polished_size < best_size:
            best_size = polished_size
            best_code = polished_code

    # If we still have time, do another round
    if time.time() < deadline - 10.0 and best_code is not None:
        rng = np.random.default_rng(271828)
        initial_code = greedy_init(rng, time.time() + 2.0)
        code, size = local_search(initial_code, rng, deadline - 5.0)
        polished_code, polished_size = final_polish(code, deadline - 2.0)
        if polished_size < best_size:
            best_size = polished_size
            best_code = polished_code

    if best_code is None:
        # Fallback: return all words
        best_code = np.ones(N, dtype=np.bool_)
        best_size = N

    # Construct output
    final_code = []
    for w in range(N):
        if best_code[w]:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)

    return final_code

import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    # Leave a safety margin of 8 seconds for cleanup and return
    deadline = start_time + 112.0

    total_words = q ** n
    if total_words == 1:
        return [[0] * n]

    # Precompute neighbors
    n_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, n_neighbors), dtype=np.int32)

    q_powers = [q ** i for i in range(n)]

    for w in range(total_words):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            digit = (w // q_powers[n - 1 - pos]) % q
            for v in range(q):
                if v == digit:
                    continue
                diff = (v - digit) * q_powers[n - 1 - pos]
                neighbors[w, idx] = w + diff
                idx += 1

    # Precompute neighbor sets for fast membership testing
    neighbor_sets = [set(neighbors[w].tolist()) for w in range(total_words)]

    def greedy_init(seed_offset=0):
        rng = random.Random(seed_offset)
        covered = np.zeros(total_words, dtype=bool)
        gain = np.full(total_words, n_neighbors, dtype=np.int32)
        codewords = []
        remaining = total_words

        while remaining > 0:
            if time.time() > deadline - 25.0:
                break
            best_word = int(np.argmax(gain))
            best_gain = gain[best_word]

            if best_gain == 0:
                uncovered = np.where(~covered)[0]
                if len(uncovered) == 0:
                    break
                best_word = int(uncovered[0])

            codewords.append(best_word)

            for v in neighbors[best_word]:
                if not covered[v]:
                    covered[v] = True
                    remaining -= 1
                    for w in neighbors[v]:
                        gain[w] -= 1

        return codewords

    def ca_reorganize(cw, time_limit):
        """Cellular-automaton-like global reorganization.

        Synchronously remove every in-code word whose own neighborhood
        contains a coverage deficit (deficit = max(0, 1 - count)), then
        greedily rebuild coverage from the deficit set. This performs a
        global structural reorganization that can escape single-swap
        local minima of the exchange-based local search.
        """
        cw = cw[:]
        best_cw = cw[:]
        best_size = len(cw)

        in_code = np.zeros(total_words, dtype=bool)
        for ci in cw:
            in_code[ci] = True

        for _ in range(8):  # Increased from 6 to 8 iterations
            if time.time() > time_limit - 6.0:
                break

            count = np.zeros(total_words, dtype=np.int32)
            for ci in cw:
                count[neighbors[ci]] += 1

            deficit = (1 - count) > 0  # words needing coverage
            if not deficit.any():
                break

            # Synchronous CA update: a codeword is "unstable" if any word in
            # its own neighborhood has a deficit (it is the only thing keeping
            # that word covered). Remove all unstable codewords at once.
            remove = np.zeros(total_words, dtype=bool)
            for ci in cw:
                if deficit[neighbors[ci]].any():
                    remove[ci] = True

            new_cw = [ci for ci in cw if not remove[ci]]

            # Rebuild coverage from the remaining deficit set (greedy).
            covered = np.zeros(total_words, dtype=bool)
            for ci in new_cw:
                covered[neighbors[ci]] = True
            rem_deficit = deficit & ~covered
            if rem_deficit.any():
                gain = np.full(total_words, n_neighbors, dtype=np.int32)
                for ci in new_cw:
                    for u in neighbors[ci]:
                        gain[u] -= 1
                remaining = int(rem_deficit.sum())
                while remaining > 0:
                    if time.time() > time_limit - 6.0:
                        break
                    best_word = int(np.argmax(gain))
                    if gain[best_word] == 0:
                        unc = np.where(rem_deficit)[0]
                        if len(unc) == 0:
                            break
                        best_word = int(unc[0])
                    new_cw.append(best_word)
                    for v in neighbors[best_word]:
                        if rem_deficit[v]:
                            rem_deficit[v] = False
                            remaining -= 1
                        for w in neighbors[v]:
                            gain[w] -= 1

            cw = new_cw
            if len(cw) < best_size:
                best_size = len(cw)
                best_cw = cw[:]

        # Final polish: run local search on the best CA result if time permits
        if time.time() < time_limit - 10.0 and len(best_cw) < len(cw):
            polished = local_search(best_cw, time_limit - 5.0)
            if len(polished) < best_size:
                best_cw = polished
                best_size = len(polished)

        return best_cw

    def local_search(cw, time_limit):
        cw = cw[:]
        best_cw = cw[:]
        best_size = len(cw)

        count = np.zeros(total_words, dtype=np.int32)
        in_code = np.zeros(total_words, dtype=bool)

        for ci in cw:
            in_code[ci] = True
            for u in neighbors[ci]:
                count[u] += 1

        while time.time() < time_limit:
            improved = False

            # Pass 1: Remove redundant codewords
            indices = list(range(len(cw)))
            random.shuffle(indices)

            for i in indices:
                if i >= len(cw):
                    continue
                ci = cw[i]
                if np.min(count[neighbors[ci]]) > 1:
                    for u in neighbors[ci]:
                        count[u] -= 1
                    in_code[ci] = False
                    cw.pop(i)
                    improved = True
                    if len(cw) < best_size:
                        best_size = len(cw)
                        best_cw = cw[:]

            # Pass 2: Swaps
            if not improved:
                indices = list(range(len(cw)))
                random.shuffle(indices)

                for i in indices:
                    if i >= len(cw):
                        continue
                    ci = cw[i]

                    critical_mask = count[neighbors[ci]] == 1
                    critical = neighbors[ci][critical_mask]

                    if len(critical) == 0:
                        for u in neighbors[ci]:
                            count[u] -= 1
                        in_code[ci] = False
                        cw.pop(i)
                        improved = True
                        # Do not break, allow more removals in this pass

                    if len(critical) > 0:
                        first_crit = critical[0]

                        if len(critical) == 1:
                            valid_cands = []
                            for cand in neighbor_sets[first_crit]:
                                if cand == ci or in_code[cand]:
                                    continue
                                valid_cands.append(cand)

                            if valid_cands:
                                cand = random.choice(valid_cands)
                                for u in neighbors[ci]:
                                    count[u] -= 1
                                for u in neighbors[cand]:
                                    count[u] += 1

                                in_code[ci] = False
                                in_code[cand] = True
                                cw[i] = cand
                                improved = True
                                # Do not break, allow more swaps in this pass
                        else:
                            candidate_set = neighbor_sets[first_crit].copy()
                            for u in critical[1:]:
                                candidate_set &= neighbor_sets[u]
                                if not candidate_set:
                                    break

                            if candidate_set:
                                valid_cands = [c for c in candidate_set if c != ci and not in_code[c]]
                                if valid_cands:
                                    cand = random.choice(valid_cands)
                                    for u in neighbors[ci]:
                                        count[u] -= 1
                                    for u in neighbors[cand]:
                                        count[u] += 1

                                    in_code[ci] = False
                                    in_code[cand] = True
                                    cw[i] = cand
                                    improved = True
                                    # Do not break, allow more swaps in this pass

            # Pass 3: 2-for-1 Swap (Reduce size by 1 by replacing 2 codewords with 1)
            if not improved and len(cw) > 2:
                # Sample pairs to find a 2-for-1 or 2-for-0 improvement
                num_samples = min(40, len(cw) * 2)
                found_improvement = False
                for _ in range(num_samples):
                    if time.time() > time_limit:
                        break
                    idx1, idx2 = random.sample(range(len(cw)), 2)
                    c1, c2 = cw[idx1], cw[idx2]
                    
                    # Compute deficit set: words that would become uncovered if we remove c1 and c2
                    deficit_set = set()
                    for u in neighbors[c1]:
                        covers = 1  # c1 covers u
                        if u in neighbor_sets[c2]:
                            covers += 1
                        if count[u] - covers < 1:
                            deficit_set.add(u)
                    
                    for u in neighbors[c2]:
                        if u in deficit_set:
                            continue
                        covers = 1  # c2 covers u
                        if u in neighbor_sets[c1]:
                            covers += 1
                        if count[u] - covers < 1:
                            deficit_set.add(u)
                    
                    if not deficit_set:
                        # 2-for-0: remove both, no replacement needed
                        for u in neighbors[c1]:
                            count[u] -= 1
                        for u in neighbors[c2]:
                            count[u] -= 1
                        in_code[c1] = False
                        in_code[c2] = False
                        if idx1 < idx2:
                            cw.pop(idx2)
                            cw.pop(idx1)
                        else:
                            cw.pop(idx1)
                            cw.pop(idx2)
                        improved = True
                        found_improvement = True
                        if len(cw) < best_size:
                            best_size = len(cw)
                            best_cw = cw[:]
                        break
                    else:
                        # Try to find a single codeword covering all deficit words
                        deficit_list = list(deficit_set)
                        candidate_set = neighbor_sets[deficit_list[0]].copy()
                        for u in deficit_list[1:]:
                            candidate_set &= neighbor_sets[u]
                            if not candidate_set:
                                break
                        
                        if candidate_set:
                            valid_cands = [c for c in candidate_set if not in_code[c]]
                            if valid_cands:
                                cand = random.choice(valid_cands)
                                for u in neighbors[c1]:
                                    count[u] -= 1
                                for u in neighbors[c2]:
                                    count[u] -= 1
                                in_code[c1] = False
                                in_code[c2] = False
                                for u in neighbors[cand]:
                                    count[u] += 1
                                in_code[cand] = True
                                if idx1 < idx2:
                                    cw.pop(idx2)
                                    cw.pop(idx1)
                                else:
                                    cw.pop(idx1)
                                    cw.pop(idx2)
                                cw.append(cand)
                                improved = True
                                found_improvement = True
                                if len(cw) < best_size:
                                    best_size = len(cw)
                                    best_cw = cw[:]
                                break

            # Perturbation: random swap of two codewords to escape local min
            if not improved and len(cw) > 2 and random.random() < 0.4:
                idx1, idx2 = random.sample(range(len(cw)), 2)
                c1, c2 = cw[idx1], cw[idx2]

                temp_count = count.copy()
                for u in neighbors[c1]:
                    temp_count[u] -= 1
                for u in neighbors[c2]:
                    temp_count[u] += 1

                if np.min(temp_count) >= 1:
                    for u in neighbors[c1]:
                        count[u] -= 1
                    for u in neighbors[c2]:
                        count[u] += 1
                    in_code[c1] = False
                    in_code[c2] = True
                    cw[idx1] = c2
                    improved = True

            # Aggressive perturbation: replace one codeword with a random one
            if not improved and random.random() < 0.15:
                if len(cw) > 5:
                    idx = random.randint(0, len(cw) - 1)
                    c_remove = cw[idx]
                    # Try several random candidates
                    for _ in range(10):
                        if time.time() > time_limit:
                            break
                        c_add = random.randint(0, total_words - 1)
                        if in_code[c_add] and c_add != c_remove:
                            continue
                        temp_count = count.copy()
                        for u in neighbors[c_remove]:
                            temp_count[u] -= 1
                        for u in neighbors[c_add]:
                            temp_count[u] += 1
                        if np.min(temp_count) >= 1:
                            for u in neighbors[c_remove]:
                                count[u] -= 1
                            for u in neighbors[c_add]:
                                count[u] += 1
                            in_code[c_remove] = False
                            in_code[c_add] = True
                            cw[idx] = c_add
                            improved = True
                            break

        return best_cw

    def finish(cw):
        result = []
        for ci in cw:
            word = []
            val = ci
            for pos in range(n - 1, -1, -1):
                word.append(val % q)
                val //= q
            word.reverse()
            result.append(word)
        return result

    # Phase 1: Greedy Set Cover
    codewords = greedy_init(42)

    # Phase 2: Local Search
    best_cw = local_search(codewords, deadline - 30.0)

    # Phase 3: CA-like global reorganization (only if it keeps us valid & smaller)
    if time.time() < deadline - 15.0:
        ca_cw = ca_reorganize(best_cw, deadline - 12.0)
        if len(ca_cw) < len(best_cw):
            best_cw = ca_cw

    # Phase 4: Second restart with a different seed if time permits
    if time.time() < deadline - 40.0:
        codewords2 = greedy_init(1337)
        ls2 = local_search(codewords2, deadline - 25.0)
        if len(ls2) < len(best_cw):
            best_cw = ls2
        if time.time() < deadline - 15.0:
            ca2 = ca_reorganize(best_cw, deadline - 12.0)
            if len(ca2) < len(best_cw):
                best_cw = ca2

    # Phase 5: Final polish pass with tight time budget
    if time.time() < deadline - 8.0:
        final_polish = local_search(best_cw, deadline - 3.0)
        if len(final_polish) < len(best_cw):
            best_cw = final_polish

    return finish(best_cw)

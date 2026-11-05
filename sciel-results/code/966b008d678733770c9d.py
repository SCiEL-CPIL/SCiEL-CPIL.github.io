"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0  # safety margin before the 150s kill

    # ---- mapping helpers ----
    def word_to_int(w):
        val = 0
        for x in w:
            val = val * q + x
        return val

    def int_to_word(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return w

    total = q ** n

    # ---- precompute radius balls for every word as integer lists ----
    neighbor_lists = []
    for i in range(total):
        w = int_to_word(i)
        ball = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w[pos] = new_val
                    ball.append(word_to_int(w))
            w[pos] = orig
        neighbor_lists.append(ball)

    # ---- small-n fallback (exhaustive greedy) ----
    if n <= 4:
        all_words = list(itertools.product(range(q), repeat=n))
        uncovered = set(all_words)
        code = []
        for w in all_words:
            if not uncovered:
                break
            if w in uncovered:
                code.append(list(w))
                for u in list(uncovered):
                    if sum(1 for i in range(n) if u[i] != w[i]) <= radius:
                        uncovered.remove(u)
        return code

    # ---- initial code: direct sum of the perfect ternary tetracode ----
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_tails = list(itertools.product(range(q), repeat=n - 4))
        code = [h + list(t) for h in tetracode for t in base_tails]
    else:
        # Fallback: greedy initial code
        uncovered = set(range(total))
        code = []
        for i in range(total):
            if not uncovered:
                break
            if i in uncovered:
                code.append(int_to_word(i))
                for nb in neighbor_lists[i]:
                    if nb in uncovered:
                        uncovered.discard(nb)

    code_indices = [word_to_int(c) for c in code]
    code_set = set(code_indices)

    counts = np.zeros(total, dtype=np.int32)
    for ci in code_indices:
        for nb in neighbor_lists[ci]:
            counts[nb] += 1

    # ---- smart greedy removal with smooth surrogate (log-sum-exp of slack) ----
    def smart_remove_pass():
        """Remove the most-redundant removable codeword each step using smooth objective."""
        while time.time() < deadline - 5.0:
            best_idx = -1
            best_score = 0
            m = len(code_indices)
            for idx in range(m):
                ci = code_indices[idx]
                ball = neighbor_lists[ci]
                if not np.all(counts[ball] >= 2):
                    continue
                # Smooth surrogate: log-sum-exp of (counts-1) over ball
                # This rewards codewords whose ball has high uniform redundancy
                slacks = counts[ball].astype(np.float64) - 1.0
                # log-sum-exp for numerical stability
                max_s = np.max(slacks)
                if max_s > 50:
                    score = max_s + np.log(np.sum(np.exp(slacks - max_s)))
                else:
                    score = np.log(np.sum(np.exp(slacks))) + len(ball) * 0.1
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx < 0:
                break
            ci = code_indices[best_idx]
            ball = neighbor_lists[ci]
            for nb in ball:
                counts[nb] -= 1
            code_indices.pop(best_idx)
            code_set.discard(ci)

    smart_remove_pass()

    # ---- swap-based local search with simulated annealing ----
    rng = random.Random(54321)

    def swap_search(max_iters=5000):
        """Swap two codewords: remove one, add another. Accept if coverage maintained or
        with SA probability. This escapes local minima of pure removal."""
        nonlocal counts
        m = len(code_indices)
        if m < 2:
            return
        temp = 5.0
        cooling = 0.995
        for it in range(max_iters):
            if time.time() > deadline - 15.0:
                return
            # Pick a random codeword to potentially remove
            idx_remove = rng.randrange(m)
            ci_remove = code_indices[idx_remove]
            ball_remove = neighbor_lists[ci_remove]
            
            # Check if removing it would leave uncovered words
            would_uncover = [nb for nb in ball_remove if counts[nb] == 1]
            
            if not would_uncover:
                # Safe to remove - just do it
                for nb in ball_remove:
                    counts[nb] -= 1
                code_indices.pop(idx_remove)
                code_set.discard(ci_remove)
                m = len(code_indices)
                temp *= cooling
                continue
            
            # Need to find a replacement that covers the would_uncover words
            # Sample candidate positions that cover some of the critical words
            candidates = []
            for _ in range(15):
                # Pick a random critical word and find words at distance <= radius from it
                crit = rng.choice(would_uncover)
                # Find codeword candidates near crit
                cand_val = rng.randrange(total)
                if cand_val not in code_set:
                    candidates.append(cand_val)
            
            # Also try positions that are neighbors of the removed codeword
            for nb in ball_remove[:5]:
                if nb not in code_set:
                    candidates.append(nb)
            
            best_cand = None
            best_penalty = float('inf')
            
            for cand in candidates:
                if cand in code_set:
                    continue
                cand_ball = neighbor_lists[cand]
                # Check how many of would_uncover are covered by cand
                covers_crit = sum(1 for w in would_uncover if w in cand_ball)
                # Penalty: number of would_uncover NOT covered
                penalty = len(would_uncover) - covers_crit
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_cand = cand
                if penalty == 0:
                    break
            
            if best_cand is None:
                temp *= cooling
                continue
            
            # Evaluate the swap
            if best_penalty == 0:
                # Perfect swap: remove idx_remove, add best_cand
                for nb in ball_remove:
                    counts[nb] -= 1
                for nb in neighbor_lists[best_cand]:
                    counts[nb] += 1
                code_indices[idx_remove] = best_cand
                code_set.discard(ci_remove)
                code_set.add(best_cand)
                temp *= cooling
            else:
                # Imperfect swap: accept with SA probability
                # Penalty is the number of uncovered words after swap
                accept_prob = np.exp(-best_penalty / max(temp, 0.01))
                if rng.random() < accept_prob:
                    for nb in ball_remove:
                        counts[nb] -= 1
                    for nb in neighbor_lists[best_cand]:
                        counts[nb] += 1
                    code_indices[idx_remove] = best_cand
                    code_set.discard(ci_remove)
                    code_set.add(best_cand)
                temp *= cooling
        
        # After SA, do a cleanup pass
        smart_remove_pass()

    swap_search(max_iters=3000)

    # ---- add-remove local search to escape local minima ----
    def add_remove_search(max_rounds=20):
        """Try adding a new codeword then removing two, to escape local minima."""
        nonlocal counts
        for _round in range(max_rounds):
            if time.time() > deadline - 12.0:
                return
            if len(code_indices) < 2:
                return
            # Find words with count==1 (critical)
            critical_words = np.where(counts == 1)[0].tolist()
            if not critical_words:
                continue
            # Try to find a new codeword that covers critical words and allows removal of 2
            for _ in range(10):
                if time.time() > deadline - 12.0:
                    return
                cand_val = rng.randrange(total)
                if cand_val in code_set:
                    continue
                cand_ball = neighbor_lists[cand_val]
                new_counts = counts.copy()
                for nb in cand_ball:
                    new_counts[nb] += 1
                # Find removable pairs
                m = len(code_indices)
                removed = False
                for i in range(m):
                    if time.time() > deadline - 12.0:
                        return
                    ci_i = code_indices[i]
                    ball_i = neighbor_lists[ci_i]
                    if not np.all(new_counts[ball_i] >= 2):
                        continue
                    temp_counts = new_counts.copy()
                    for nb in ball_i:
                        temp_counts[nb] -= 1
                    for j in range(i + 1, m):
                        if time.time() > deadline - 12.0:
                            return
                        ci_j = code_indices[j]
                        ball_j = neighbor_lists[ci_j]
                        if np.all(temp_counts[ball_j] >= 2):
                            for nb in ball_j:
                                temp_counts[nb] -= 1
                            counts = temp_counts
                            code_indices.pop(j)
                            code_indices.pop(i)
                            code_set.discard(ci_i)
                            code_set.discard(ci_j)
                            code_indices.append(cand_val)
                            code_set.add(cand_val)
                            removed = True
                            break
                    if removed:
                        break
                if removed:
                    smart_remove_pass()
                    break

    add_remove_search(max_rounds=15)

    # ---- double/triple removal B&B ----
    def do_double_triple_removal():
        """Try double and triple removals in a loop until no more can be found."""
        nonlocal counts
        progress = True
        while progress and time.time() < deadline - 8.0:
            progress = False
            m = len(code_indices)
            if m <= 1:
                break
            # Try double removals
            found_double = False
            for i in range(m):
                if time.time() > deadline - 8.0:
                    return
                ci_i = code_indices[i]
                ball_i = neighbor_lists[ci_i]
                if not np.all(counts[ball_i] >= 2):
                    continue
                temp_counts = counts.copy()
                for nb in ball_i:
                    temp_counts[nb] -= 1
                for j in range(i + 1, m):
                    if time.time() > deadline - 8.0:
                        return
                    ci_j = code_indices[j]
                    ball_j = neighbor_lists[ci_j]
                    if np.all(temp_counts[ball_j] >= 2):
                        for nb in ball_j:
                            temp_counts[nb] -= 1
                        counts = temp_counts
                        code_indices.pop(j)
                        code_indices.pop(i)
                        code_set.discard(ci_i)
                        code_set.discard(ci_j)
                        found_double = True
                        progress = True
                        break
                if found_double:
                    break
            if found_double:
                continue
            # Try triple removals
            found_triple = False
            for i in range(m):
                if time.time() > deadline - 8.0:
                    return
                ci_i = code_indices[i]
                ball_i = neighbor_lists[ci_i]
                if not np.all(counts[ball_i] >= 2):
                    continue
                temp_counts = counts.copy()
                for nb in ball_i:
                    temp_counts[nb] -= 1
                for j in range(i + 1, m):
                    if time.time() > deadline - 8.0:
                        return
                    ci_j = code_indices[j]
                    ball_j = neighbor_lists[ci_j]
                    if not np.all(temp_counts[ball_j] >= 2):
                        continue
                    temp2 = temp_counts.copy()
                    for nb in ball_j:
                        temp2[nb] -= 1
                    for k in range(j + 1, m):
                        if time.time() > deadline - 8.0:
                            return
                        ci_k = code_indices[k]
                        ball_k = neighbor_lists[ci_k]
                        if np.all(temp2[ball_k] >= 2):
                            for nb in ball_k:
                                temp2[nb] -= 1
                            counts = temp2
                            code_indices.pop(k)
                            code_indices.pop(j)
                            code_indices.pop(i)
                            code_set.discard(ci_i)
                            code_set.discard(ci_j)
                            code_set.discard(ci_k)
                            found_triple = True
                            progress = True
                            break
                    if found_triple:
                        break
                if found_triple:
                    break

    do_double_triple_removal()

    # ---- Final smart removal pass ----
    smart_remove_pass()

    # ---- Final verification and return ----
    return [int_to_word(ci) for ci in code_indices]
# EVOLVE-BLOCK-END

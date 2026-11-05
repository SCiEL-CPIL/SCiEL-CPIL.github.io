import time
import numpy as np
import random
from collections import defaultdict

def construct(q, n, radius):
    t_start = time.time()
    deadline = 118.0  # safety margin before 150s kill

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute balls
    nb_count = 1 + n * (q - 1)
    balls = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        balls[idx] = nb_list

    # For each word, precompute which codewords cover it (reverse mapping)
    # coverers[word_idx] = list of codeword indices whose ball contains word_idx
    coverers = [[] for _ in range(space_size)]
    for c in range(space_size):
        for w_idx in balls[c]:
            coverers[w_idx].append(c)

    # --- Fast greedy initial solution ---
    uncovered = np.ones(space_size, dtype=np.int8)
    code_indices = []
    selected_mask = np.zeros(space_size, dtype=bool)

    while np.sum(uncovered) > 0 and (time.time() - t_start) < deadline - 60.0:
        # Score each candidate by how many uncovered words it covers
        scores = np.zeros(space_size, dtype=np.int32)
        for c in range(space_size):
            if selected_mask[c]:
                continue
            s = 0
            for w_idx in balls[c]:
                if uncovered[w_idx]:
                    s += 1
            scores[c] = s
        best_idx = int(np.argmax(scores))
        if scores[best_idx] <= 0:
            break
        code_indices.append(best_idx)
        selected_mask[best_idx] = True
        for w_idx in balls[best_idx]:
            uncovered[w_idx] = 0

    # Greedy removal
    cov = np.zeros(space_size, dtype=np.int32)
    for c_idx in code_indices:
        cov[balls[c_idx]] += 1

    active = [True] * len(code_indices)
    changed = True
    while changed and (time.time() - t_start) < deadline - 55.0:
        changed = False
        for i in range(len(code_indices)):
            if not active[i]:
                continue
            c_idx = code_indices[i]
            if np.min(cov[balls[c_idx]]) >= 2:
                active[i] = False
                cov[balls[c_idx]] -= 1
                changed = True

    best_code = [code_indices[i] for i in range(len(code_indices)) if active[i]]
    best_size = len(best_code)

    # --- Max-SAT / CDCL-style branch and bound ---
    # Variables: x[c] for c in 0..space_size-1 (1 = select codeword c)
    # Clauses: for each word w, OR of x[c] for c in coverers[w]  (must be satisfied)
    # Objective: minimize sum(x[c])

    # We'll use a simpler but effective approach:
    # Iterated local search with Max-SAT-inspired scoring
    # Key insight: use the coverage structure to guide swaps optimally

    def compute_cov(indices):
        c = np.zeros(space_size, dtype=np.int32)
        for idx in indices:
            c[balls[idx]] += 1
        return c

    def is_valid(cov):
        return np.min(cov) >= 1

    def greedy_remove(cov, indices):
        active = [True] * len(indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(indices)):
                if not active[i]:
                    continue
                idx = indices[i]
                if np.min(cov[balls[idx]]) >= 2:
                    active[i] = False
                    cov[balls[idx]] -= 1
                    changed = True
        return [indices[i] for i in range(len(indices)) if active[i]]

    # --- CDCL-inspired local search ---
    # Use variable activity scores and clause (constraint) learning
    # "Clauses" = uncovered words; "conflicts" = words with cov==0

    cur_indices = best_code[:]
    cur_cov = compute_cov(cur_indices)
    cur_size = len(cur_indices)

    # Variable activity (how often a codeword is involved in conflicts)
    var_activity = np.zeros(space_size, dtype=np.float64)
    # Clause (word) activity (how often a word is uncovered)
    clause_activity = np.zeros(space_size, dtype=np.float64)
    # Decay factor
    decay = 0.95

    rng = np.random.default_rng(42)

    sa_budget = deadline - (time.time() - t_start) - 10.0
    if sa_budget < 15.0:
        sa_budget = 15.0

    T_start = 5.0
    T_end = 0.05
    stall_count = 0
    max_stall = 200
    step = 0

    while (time.time() - t_start) < deadline - 10.0:
        elapsed_frac = 1.0 - (deadline - 10.0 - (time.time() - t_start)) / sa_budget
        elapsed_frac = max(0.0, min(1.0, elapsed_frac))
        T = T_start * (T_end / T_start) ** elapsed_frac

        # Decay activities
        var_activity *= decay
        clause_activity *= decay

        # Identify conflicts (uncovered words)
        unc_mask = (cur_cov == 0).astype(np.int8)
        num_unc = int(np.sum(unc_mask))

        if num_unc == 0:
            # Valid solution: try to delete
            new_indices = greedy_remove(cur_cov.copy(), cur_indices)
            if len(new_indices) < cur_size:
                cur_indices = new_indices
                cur_cov = compute_cov(cur_indices)
                cur_size = len(cur_indices)
                stall_count = 0
                if cur_size < best_size:
                    best_size = cur_size
                    best_code = cur_indices[:]
                step += 1
                continue

        # Update clause activities for uncovered words
        unc_list = np.where(unc_mask > 0)[0]
        for w_idx in unc_list:
            clause_activity[w_idx] += 1.0

        # Choose strategy based on temperature and state
        r = rng.random()

        if r < 0.40:
            # Targeted fix: pick the highest-activity uncovered word,
            # add a codeword that covers it (chosen by activity), then remove a redundant one
            if len(unc_list) == 0:
                step += 1
                continue
            # Weighted random selection of uncovered word
            w_probs = clause_activity[unc_list]
            w_sum = w_probs.sum()
            if w_sum <= 0:
                w_probs = np.ones(len(unc_list))
                w_sum = 1.0
            w_probs = w_probs / w_sum
            target = int(unc_list[int(np.searchsorted(np.cumsum(w_probs), rng.random()))])

            # Find codewords that cover target, weighted by var_activity
            candidates = coverers[target]
            sel_set = set(cur_indices)
            candidates = [c for c in candidates if c not in sel_set]
            if not candidates:
                step += 1
                continue

            # Weight by activity
            c_probs = var_activity[candidates]
            c_sum = c_probs.sum()
            if c_sum <= 0:
                c_probs = np.ones(len(candidates))
                c_sum = float(len(candidates))
            c_probs = c_probs / c_sum
            new_c = int(candidates[int(np.searchsorted(np.cumsum(c_probs), rng.random()))])

            # Find best codeword to remove (least damage)
            if len(cur_indices) < 2:
                step += 1
                continue
            best_rem = None
            best_damage = float('inf')
            for rc in cur_indices:
                b = balls[rc]
                damage = int(np.sum(cur_cov[b] == 1))
                if damage < best_damage:
                    best_damage = damage
                    best_rem = rc

            # Tentative swap
            new_cov = cur_cov.copy()
            new_cov[balls[best_rem]] -= 1
            new_cov[balls[new_c]] += 1

            new_unc = int(np.sum(new_cov == 0))
            old_unc = num_unc

            delta = (new_unc - old_unc) * 3.0
            if delta <= 0 or (T > 0.01 and rng.random() < np.exp(-delta / T)):
                new_indices = [c for c in cur_indices if c != best_rem]
                new_indices.append(new_c)
                cur_indices = new_indices
                cur_cov = new_cov
                cur_size = len(cur_indices)
                # Update activity
                var_activity[new_c] += 1.0
                var_activity[best_rem] += 0.5
                stall_count = 0
                if cur_size < best_size and int(np.sum(cur_cov == 0)) == 0:
                    best_size = cur_size
                    best_code = cur_indices[:]

        elif r < 0.70:
            # 2-opt swap: remove two, add two (chosen by activity)
            if cur_size < 4:
                step += 1
                continue

            # Pick two to remove (lowest activity among those with low damage)
            remove_candidates = list(cur_indices)
            # Score by damage
            damages = []
            for rc in remove_candidates:
                b = balls[rc]
                d = int(np.sum(cur_cov[b] == 1))
                damages.append(d)
            damages = np.array(damages)
            # Sort by damage (ascending) and pick from top
            order = np.argsort(damages)
            pool = [remove_candidates[order[i]] for i in range(min(6, len(order)))]
            if len(pool) < 2:
                step += 1
                continue
            rem_a, rem_b = rng.choice(pool, size=2, replace=False).tolist()

            # Find two new codewords that cover the most uncovered words
            # After removing rem_a, rem_b, what's uncovered?
            temp_cov = cur_cov.copy()
            temp_cov[balls[rem_a]] -= 1
            temp_cov[balls[rem_b]] -= 1
            temp_unc = (temp_cov == 0).astype(np.int8)
            temp_unc_count = int(np.sum(temp_unc))

            if temp_unc_count == 0:
                # Just removal works
                new_indices = [c for c in cur_indices if c != rem_a and c != rem_b]
                cur_indices = new_indices
                cur_cov = temp_cov
                cur_size = len(cur_indices)
                stall_count = 0
                if cur_size < best_size:
                    best_size = cur_size
                    best_code = cur_indices[:]
                step += 1
                continue

            # Greedily pick two best codewords to cover temp_unc
            sel_set = set(cur_indices) - {rem_a, rem_b}
            scores = np.zeros(space_size, dtype=np.int32)
            for c in range(space_size):
                if c in sel_set:
                    continue
                s = 0
                for w_idx in balls[c]:
                    if temp_unc[w_idx]:
                        s += 1
                scores[c] = s

            # Pick top 2
            top_candidates = np.argsort(-scores)[:10]
            top_candidates = [int(c) for c in top_candidates if scores[c] > 0]
            if len(top_candidates) < 2:
                step += 1
                continue

            new_a = top_candidates[0]
            # For second, recompute after adding first
            temp_cov2 = temp_cov.copy()
            temp_cov2[balls[new_a]] += 1
            temp_unc2 = (temp_cov2 == 0).astype(np.int8)
            scores2 = np.zeros(space_size, dtype=np.int32)
            for c in top_candidates:
                if c == new_a:
                    continue
                s = 0
                for w_idx in balls[c]:
                    if temp_unc2[w_idx]:
                        s += 1
                scores2[c] = s
            new_b = int(np.argmax(scores2))

            new_cov = temp_cov.copy()
            new_cov[balls[new_a]] += 1
            new_cov[balls[new_b]] += 1

            new_unc = int(np.sum(new_cov == 0))
            delta = (new_unc - num_unc) * 3.0
            if delta <= 0 or (T > 0.01 and rng.random() < np.exp(-delta / T)):
                new_indices = [c for c in cur_indices if c != rem_a and c != rem_b]
                new_indices.extend([new_a, new_b])
                cur_indices = new_indices
                cur_cov = new_cov
                cur_size = len(cur_indices)
                var_activity[new_a] += 1.0
                var_activity[new_b] += 1.0
                stall_count = 0
                if cur_size < best_size and int(np.sum(cur_cov == 0)) == 0:
                    best_size = cur_size
                    best_code = cur_indices[:]

        elif r < 0.85:
            # Kick: remove k random, re-greedy
            k = max(1, min(4, cur_size // 6))
            remove_set = set(rng.choice(cur_indices, size=k, replace=False).tolist())
            remaining = [c for c in cur_indices if c not in remove_set]
            rem_cov = np.zeros(space_size, dtype=np.int32)
            for c in remaining:
                rem_cov[balls[c]] += 1

            # Re-greedy
            unc = (rem_cov == 0).astype(np.int8)
            sel_mask = np.zeros(space_size, dtype=bool)
            for c in remaining:
                sel_mask[c] = True
            new_indices = list(remaining)
            while np.sum(unc) > 0 and (time.time() - t_start) < deadline - 12.0:
                scores = np.zeros(space_size, dtype=np.int32)
                for c in range(space_size):
                    if sel_mask[c]:
                        continue
                    s = 0
                    for w_idx in balls[c]:
                        if unc[w_idx]:
                            s += 1
                    scores[c] = s
                best = int(np.argmax(scores))
                if scores[best] <= 0:
                    break
                new_indices.append(best)
                sel_mask[best] = True
                for w_idx in balls[best]:
                    unc[w_idx] = 0

            # Greedy remove
            new_cov = compute_cov(new_indices)
            new_indices = greedy_remove(new_cov, new_indices)
            new_cov = compute_cov(new_indices)

            if len(new_indices) < cur_size:
                cur_indices = new_indices
                cur_cov = new_cov
                cur_size = len(cur_indices)
                stall_count = 0
                if cur_size < best_size and int(np.sum(cur_cov == 0)) == 0:
                    best_size = cur_size
                    best_code = cur_indices[:]

        else:
            # Pure deletion pass
            new_indices = greedy_remove(cur_cov.copy(), cur_indices)
            if len(new_indices) < cur_size:
                cur_indices = new_indices
                cur_cov = compute_cov(cur_indices)
                cur_size = len(cur_indices)
                stall_count = 0
                if cur_size < best_size and int(np.sum(cur_cov == 0)) == 0:
                    best_size = cur_size
                    best_code = cur_indices[:]

        if stall_count > 0:
            stall_count += 1
        if stall_count > max_stall:
            # Force restart from best with perturbation
            cur_indices = best_code[:]
            # Perturb: remove 2-3 random, re-greedy
            k = max(1, min(3, cur_size // 8))
            if k > 0 and cur_size > k:
                remove_set = set(rng.choice(cur_indices, size=k, replace=False).tolist())
                remaining = [c for c in cur_indices if c not in remove_set]
                rem_cov = np.zeros(space_size, dtype=np.int32)
                for c in remaining:
                    rem_cov[balls[c]] += 1
                unc = (rem_cov == 0).astype(np.int8)
                sel_mask = np.zeros(space_size, dtype=bool)
                for c in remaining:
                    sel_mask[c] = True
                new_indices = list(remaining)
                while np.sum(unc) > 0 and (time.time() - t_start) < deadline - 12.0:
                    scores = np.zeros(space_size, dtype=np.int32)
                    for c in range(space_size):
                        if sel_mask[c]:
                            continue
                        s = 0
                        for w_idx in balls[c]:
                            if unc[w_idx]:
                                s += 1
                        scores[c] = s
                    best = int(np.argmax(scores))
                    if scores[best] <= 0:
                        break
                    new_indices.append(best)
                    sel_mask[best] = True
                    for w_idx in balls[best]:
                        unc[w_idx] = 0
                new_cov = compute_cov(new_indices)
                new_indices = greedy_remove(new_cov, new_indices)
                cur_indices = new_indices
                cur_cov = compute_cov(cur_indices)
                cur_size = len(cur_indices)
            stall_count = 0

        step += 1

    # --- Final cleanup ---
    final_indices = best_code[:]
    final_cov = compute_cov(final_indices)

    if int(np.sum(final_cov == 0)) > 0:
        # Re-greedy to ensure validity
        unc = (final_cov == 0).astype(np.int8)
        sel_mask = np.zeros(space_size, dtype=bool)
        for c in final_indices:
            sel_mask[c] = True
        while np.sum(unc) > 0:
            scores = np.zeros(space_size, dtype=np.int32)
            for c in range(space_size):
                if sel_mask[c]:
                    continue
                s = 0
                for w_idx in balls[c]:
                    if unc[w_idx]:
                        s += 1
                scores[c] = s
            best = int(np.argmax(scores))
            if scores[best] <= 0:
                break
            final_indices.append(best)
            sel_mask[best] = True
            for w_idx in balls[best]:
                unc[w_idx] = 0
        final_cov = compute_cov(final_indices)

    final_indices = greedy_remove(final_cov, final_indices)
    final_cov = compute_cov(final_indices)

    # Verify
    if int(np.sum(final_cov == 0)) > 0:
        # Fallback
        final_indices = list(range(space_size))

    # Ensure distinctness
    seen = set()
    unique = []
    for c in final_indices:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return [idx_to_word(i) for i in unique]

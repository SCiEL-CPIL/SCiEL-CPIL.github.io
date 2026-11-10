import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115

    rng = random.Random(777)

    # Precompute all triples
    all_triples = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                all_triples.append((a, b, c))

    n_triples = len(all_triples)  # 560

    # Triple masks
    triple_masks = []
    for t in all_triples:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(m)

    # Popcount
    popcount = [0] * (1 << 16)
    for i in range(1, 1 << 16):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # Compatibility as bitsets (560 bits each, use Python int)
    # compat[i] is a 560-bit integer where bit j is 1 if triple i and j are compatible
    compat = [0] * n_triples
    for i in range(n_triples):
        mi = triple_masks[i]
        for j in range(i + 1, n_triples):
            mj = triple_masks[j]
            inter = mi & mj
            cnt = bin(inter).count('1')
            if cnt <= 1:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)

    # For a set of selected triples S, the compatible candidates are:
    # intersection of compat[k] for all k in S, minus S itself
    # We can compute this incrementally

    # Fast alpha computation using superset marking
    def compute_alpha(selected_set):
        if not selected_set:
            return 16
        bad = bytearray(1 << 16)
        for idx in selected_set:
            tm = triple_masks[idx]
            remaining = ((1 << 16) - 1) ^ tm
            sub = remaining
            while True:
                bad[tm | sub] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & remaining
        best = 0
        for m in range(1 << 16):
            if not bad[m]:
                c = popcount[m]
                if c > best:
                    best = c
        return best

    # Greedy construction
    def greedy_construct():
        available_mask = (1 << n_triples) - 1
        selected = set()
        while available_mask:
            # Pick random available triple
            bits = bin(available_mask).count('1')
            if bits == 0:
                break
            idx_in_bits = rng.randrange(bits)
            tmp = available_mask
            for _ in range(idx_in_bits):
                tmp &= tmp - 1
            bit = tmp & (-tmp)
            i = bit.bit_length() - 1

            selected.add(i)
            # Update available: keep only triples compatible with i
            available_mask &= compat[i]
            available_mask &= ~(1 << i)
        return selected

    # Local search with fast alpha
    def local_search(initial_set, time_limit):
        current = set(initial_set)
        current_alpha = compute_alpha(current)
        best = set(current)
        best_alpha = current_alpha
        t_start = time.time()

        while time.time() - t_start < time_limit:
            improved = False
            # Try single swaps
            for i in list(current):
                S = current - {i}
                # Find candidates: intersection of compat[k] for k in S, minus current
                if not S:
                    candidates = (1 << n_triples) - 1
                else:
                    candidates = (1 << n_triples) - 1
                    for k in S:
                        candidates &= compat[k]
                candidates &= ~((1 << n_triples) - 1)  # no-op
                # Remove current triples
                for k in current:
                    candidates &= ~(1 << k)

                if candidates == 0:
                    continue

                # Try a few random candidates
                bits = bin(candidates).count('1')
                if bits == 0:
                    continue
                # Try up to 3 random candidates
                tried = 0
                while tried < 3 and candidates:
                    idx_in_bits = rng.randrange(bin(candidates).count('1'))
                    tmp = candidates
                    for _ in range(idx_in_bits):
                        tmp &= tmp - 1
                    bit = tmp & (-tmp)
                    j = bit.bit_length() - 1

                    new_set = S | {j}
                    new_alpha = compute_alpha(new_set)
                    if new_alpha < current_alpha:
                        current = new_set
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best = set(current)
                        improved = True
                        break
                    candidates &= ~(1 << j)
                    tried += 1

                if improved:
                    break

            if not improved:
                # Random perturbation
                if current:
                    i = rng.choice(list(current))
                    S = current - {i}
                    if not S:
                        candidates = (1 << n_triples) - 1
                    else:
                        candidates = (1 << n_triples) - 1
                        for k in S:
                            candidates &= compat[k]
                    for k in current:
                        candidates &= ~(1 << k)
                    if candidates:
                        bits = bin(candidates).count('1')
                        idx_in_bits = rng.randrange(bits)
                        tmp = candidates
                        for _ in range(idx_in_bits):
                            tmp &= tmp - 1
                        bit = tmp & (-tmp)
                        j = bit.bit_length() - 1
                        current = S | {j}
                        current_alpha = compute_alpha(current)
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best = set(current)

        return best, best_alpha

    # Multi-restart
    best_system = None
    best_alpha = 16

    # Initial greedy constructions
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        greedy = greedy_construct()
        alpha = compute_alpha(greedy)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = set(greedy)

        # Local search
        remaining = deadline - time.time()
        if remaining < 1:
            break
        ls_time = min(0.3, remaining * 0.02)
        if ls_time > 0.05:
            ls_best, ls_alpha = local_search(greedy, ls_time)
            if ls_alpha < best_alpha:
                best_alpha = ls_alpha
                best_system = set(ls_best)

    # Continue with more restarts
    while time.time() < deadline - 3:
        if best_alpha <= 7:
            break
        greedy = greedy_construct()
        alpha = compute_alpha(greedy)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = set(greedy)

        remaining = deadline - time.time()
        if remaining < 1:
            break
        ls_time = min(0.2, remaining * 0.015)
        if ls_time > 0.03:
            ls_best, ls_alpha = local_search(greedy, ls_time)
            if ls_alpha < best_alpha:
                best_alpha = ls_alpha
                best_system = set(ls_best)

    if best_system is None:
        return []

    return [list(all_triples[i]) for i in sorted(best_system)]

# EVOLVE-BLOCK-END

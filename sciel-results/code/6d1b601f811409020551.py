"""Probabilistic covering design with smooth-objective simulated annealing for C(19,8,5)."""
import random
import itertools
import time
import math

def construct(v, k, t):
    """
    Construct covering design C(v,k,t) using greedy init + SA on smooth objective.
    Smooth objective: sum_i exp(-cov[i]/tau) approximates uncovered count smoothly.
    """
    start_time = time.time()
    timeout = 115

    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    def block_tsubs(block):
        return [t_idx[ts] for ts in itertools.combinations(block, t)]

    def greedy():
        cov = [0] * num_t
        blocks = []
        uncovered = set(range(num_t))
        while uncovered:
            best_b, best_g = None, -1
            for _ in range(200):
                unc = random.choice(list(uncovered))
                seed = list(all_t_subsets[unc])
                random.shuffle(seed)
                b = seed[:]
                while len(b) < k:
                    cands = [x for x in range(v) if x not in b]
                    random.shuffle(cands)
                    best_x, best_lg = None, -1
                    for x in cands[:15]:
                        g = 0
                        for rest in itertools.combinations(b, t-1):
                            ts = tuple(sorted(rest + (x,)))
                            idx = t_idx.get(ts)
                            if idx is not None and idx in uncovered:
                                g += 1
                        if g > best_lg:
                            best_lg, best_x = g, x
                    if best_x is not None:
                        b.append(best_x)
                b.sort()
                bt = tuple(b)
                g = sum(1 for ts in itertools.combinations(bt, t) if t_idx[ts] in uncovered)
                if g > best_g:
                    best_g, best_b = g, bt
            if best_b is None:
                break
            blocks.append(best_b)
            for ts in itertools.combinations(best_b, t):
                idx = t_idx[ts]
                cov[idx] += 1
                if cov[idx] == 1:
                    uncovered.discard(idx)
        return blocks

    blocks = greedy()
    block_ts = [block_tsubs(b) for b in blocks]
    cov = [0] * num_t
    for ts_list in block_ts:
        for i in ts_list:
            cov[i] += 1

    # Smooth objective with tau parameter
    tau = 1.5

    def obj_value(cov_arr):
        s = 0.0
        for c in cov_arr:
            if c <= 0:
                s += 1.0
            else:
                s += math.exp(-c / tau)
        return s

    current_obj = obj_value(cov)
    best_obj = current_obj
    best_blocks = list(blocks)

    temp = 5.0
    cooling = 0.9995
    min_temp = 0.05

    while temp > min_temp:
        if time.time() - start_time > timeout:
            break

        idx = random.randint(0, len(blocks) - 1)
        old_b = blocks[idx]
        old_ts = block_ts[idx]

        # Compute old contribution to objective
        old_contrib = 0.0
        for i in old_ts:
            old_contrib += math.exp(-cov[i] / tau) if cov[i] > 0 else 1.0

        # Remove old block
        for i in old_ts:
            cov[i] -= 1

        # Generate new block
        low_cov = [i for i, c in enumerate(cov) if c <= 1]
        if low_cov:
            target = random.choice(low_cov)
            seed = list(all_t_subsets[target])
        else:
            seed = list(old_b)
        random.shuffle(seed)
        nb = seed[:]
        while len(nb) < k:
            cands = [x for x in range(v) if x not in nb]
            random.shuffle(cands)
            best_x, best_lg = None, -1
            for x in cands[:12]:
                g = 0
                for rest in itertools.combinations(nb, t-1):
                    ts = tuple(sorted(rest + (x,)))
                    ii = t_idx.get(ts)
                    if ii is not None and cov[ii] <= 1:
                        g += 1
                if g > best_lg:
                    best_lg, best_x = g, x
            if best_x is not None:
                nb.append(best_x)
        nb.sort()
        nbt = tuple(nb)

        # Check duplicate
        dup = False
        for j, b in enumerate(blocks):
            if j != idx and b == nbt:
                dup = True
                break
        if dup:
            for i in old_ts:
                cov[i] += 1
            continue

        nts = block_tsubs(nbt)
        for i in nts:
            cov[i] += 1

        # Compute new contribution
        new_contrib = 0.0
        for i in nts:
            new_contrib += math.exp(-cov[i] / tau) if cov[i] > 0 else 1.0

        delta = new_contrib - old_contrib

        if delta <= 0 or random.random() < math.exp(-delta / temp):
            blocks[idx] = nbt
            block_ts[idx] = nts
            current_obj += delta
            if current_obj < best_obj:
                best_obj = current_obj
                best_blocks = list(blocks)
        else:
            for i in nts:
                cov[i] -= 1
            for i in old_ts:
                cov[i] += 1

        temp *= cooling

    return [list(b) for b in best_blocks]

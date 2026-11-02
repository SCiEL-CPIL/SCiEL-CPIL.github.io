import itertools
import time
import random
import math


def construct(v, k, t):
    deadline = time.time() + 115
    rng = random.Random(42)

    all_t_subsets = list(itertools.combinations(range(v), t))
    all_t_set = set(all_t_subsets)

    def block_triples(block):
        return set(itertools.combinations(block, t))

    def compute_coverage(block_list):
        covered = set()
        for b in block_list:
            covered |= block_triples(b)
        return covered

    def greedy_cover(rnd=None):
        if rnd is None:
            rnd = rng
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered and time.time() < deadline:
            target = rnd.choice(list(uncovered))
            block = set(target)
            while len(block) < k and time.time() < deadline:
                best_x = -1
                best_score = -1
                for x in range(v):
                    if x in block:
                        continue
                    score = 0
                    for rest in itertools.combinations(sorted(block), t - 1):
                        if tuple(sorted(rest + (x,))) in uncovered:
                            score += 1
                    score += rnd.random() * 0.05
                    if score > best_score:
                        best_score = score
                        best_x = x
                if best_x == -1:
                    break
                block.add(best_x)
            block = sorted(block)
            if block not in blocks:
                blocks.append(block)
            uncovered -= block_triples(block)
        return blocks

    def remove_redundant(blocks_in):
        blocks_out = list(blocks_in)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks_out)):
                covered = set()
                for j in range(len(blocks_out)):
                    if j != i:
                        covered |= block_triples(blocks_out[j])
                if covered == all_t_set:
                    blocks_out.pop(i)
                    changed = True
                    break
        return blocks_out

    blocks = greedy_cover()
    blocks = remove_redundant(blocks)
    best_blocks = list(blocks)
    best_count = len(blocks)

    def local_search(start, time_budget=12.0):
        cur = list(start)
        best = list(cur)
        best_n = len(cur)
        T = 2.0
        T_min = 0.01
        cooling = 0.995
        t0 = time.time()

        while time.time() - t0 < time_budget and time.time() < deadline:
            n = len(cur)
            if n < 2:
                break
            move = rng.random()

            if move < 0.3:
                idx = rng.randint(0, n - 1)
                trial = cur[:idx] + cur[idx + 1:]
                cov = compute_coverage(trial)
                missing = all_t_set - cov
                if not missing:
                    cur = trial
                    cur = remove_redundant(cur)
                    if len(cur) < best_n:
                        best = list(cur)
                        best_n = len(cur)
                else:
                    delta = len(missing)
                    if rng.random() < math.exp(-delta / T):
                        cur = trial
            elif move < 0.7:
                idx = rng.randint(0, n - 1)
                new_block = sorted(rng.sample(range(v), k))
                if new_block == cur[idx]:
                    continue
                trial = cur[:idx] + [new_block] + cur[idx + 1:]
                cov = compute_coverage(trial)
                missing = all_t_set - cov
                if not missing:
                    cur = trial
                    cur = remove_redundant(cur)
                    if len(cur) < best_n:
                        best = list(cur)
                        best_n = len(cur)
                else:
                    delta = len(missing)
                    if rng.random() < math.exp(-delta / T):
                        cur = trial
            elif move < 0.9:
                if n >= 2:
                    idx1, idx2 = rng.sample(range(n), 2)
                    cov_others = set()
                    for i in range(n):
                        if i != idx1 and i != idx2:
                            cov_others |= block_triples(cur[i])
                    need = all_t_set - cov_others
                    for _ in range(30):
                        cand = sorted(rng.sample(range(v), k))
                        if block_triples(cand) >= need:
                            trial = [b for i, b in enumerate(cur) if i != idx1 and i != idx2] + [cand]
                            if compute_coverage(trial) == all_t_set:
                                cur = trial
                                cur = remove_redundant(cur)
                                if len(cur) < best_n:
                                    best = list(cur)
                                    best_n = len(cur)
                            break
            else:
                new_block = sorted(rng.sample(range(v), k))
                if new_block not in cur:
                    cur = cur + [new_block]

            T = max(T * cooling, T_min)

        return best

    while time.time() < deadline:
        result = local_search(best_blocks, time_budget=12.0)
        if len(result) < best_count:
            best_count = len(result)
            best_blocks = result
            print(f"Improved to {best_count}", flush=True)
        else:
            for s in range(3):
                if time.time() >= deadline:
                    break
                nb = greedy_cover(rnd=random.Random(2000 + s))
                nb = remove_redundant(nb)
                if len(nb) < best_count:
                    best_count = len(nb)
                    best_blocks = nb
                    print(f"Greedy improved to {best_count}", flush=True)

    return best_blocks

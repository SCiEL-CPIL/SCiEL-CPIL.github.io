import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    max_time = 140

    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    block_tsubs = []
    for block in all_blocks:
        tsubs = [tsub_index[ts] for ts in itertools.combinations(block, t)]
        block_tsubs.append(tsubs)

    tsub_blocks = defaultdict(list)
    for j, tsubs in enumerate(block_tsubs):
        for ti in tsubs:
            tsub_blocks[ti].append(j)

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = [True] * n_tsubs
        uncovered_count = n_tsubs
        selected = []
        selected_set = set()

        gain = [len(block_tsubs[j]) for j in range(n_blocks)]
        active = list(range(n_blocks))

        while uncovered_count > 0:
            if time.time() - start_time > max_time:
                break

            best_gain = 0
            best_blocks = []
            for j in active:
                g = gain[j]
                if g > best_gain:
                    best_gain = g
                    best_blocks = [j]
                elif g == best_gain and g > 0:
                    best_blocks.append(j)

            if best_gain == 0:
                break

            chosen = rng.choice(best_blocks)
            selected.append(all_blocks[chosen])
            selected_set.add(chosen)

            for ti in block_tsubs[chosen]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    uncovered_count -= 1
                    for j in tsub_blocks[ti]:
                        if j not in selected_set:
                            gain[j] -= 1

            if chosen in active:
                active.remove(chosen)
            active = [j for j in active if gain[j] > 0]

        return selected

    def local_search(covering):
        if not covering:
            return covering
        result = list(covering)
        changed = True
        max_iters = 200
        while changed and max_iters > 0:
            changed = False
            max_iters -= 1
            if time.time() - start_time > max_time:
                break
            tsub_covers = defaultdict(list)
            for i, block in enumerate(result):
                for ts in itertools.combinations(block, t):
                    tsub_covers[ts].append(i)
            for i in range(len(result) - 1, -1, -1):
                can_remove = True
                for ts in itertools.combinations(result[i], t):
                    if len(tsub_covers[ts]) <= 1:
                        can_remove = False
                        break
                if can_remove:
                    result.pop(i)
                    changed = True
                    break
        return result

    best_covering = None
    best_size = float('inf')
    seeds = list(range(20))
    random.shuffle(seeds)
    for seed in seeds:
        if time.time() - start_time > max_time - 10:
            break
        covering = greedy_cover(seed)
        if not covering:
            continue
        covering = local_search(covering)
        if len(covering) < best_size:
            best_size = len(covering)
            best_covering = covering
        if best_size <= 69:
            break

    if best_covering is None:
        best_covering = greedy_cover(0)

    covered = set()
    for block in best_covering:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    if len(covered) != n_tsubs:
        best_covering = greedy_cover(0)

    return [list(b) for b in best_covering]

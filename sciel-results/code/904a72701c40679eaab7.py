"""Covering design C(19,7,4): greedy with lazy-heap gain + local-search swap polish."""
import itertools
import time
import random
import heapq


def construct(v, k, t):
    deadline = time.time() + 120.0
    rng = random.Random(20240517)

    def tsubs(block):
        return list(itertools.combinations(block, t))

    def covered_count(block):
        return sum(1 for s in tsubs(block) if s in uncovered)

    def greedy(seed_block, max_blocks=None):
        nonlocal uncovered
        blocks = []
        if seed_block is not None:
            blocks.append(tuple(sorted(seed_block)))
            uncovered.difference_update(tsubs(blocks[-1]))
        while uncovered:
            if max_blocks is not None and len(blocks) >= max_blocks:
                return None
            if time.time() > deadline - 3.0:
                return None
            base = min(uncovered)
            block = list(base)
            # lazy max-heap over gain of adding point x to current block
            heap = []
            for x in range(v):
                if x not in block:
                    g = sum(1 for r in itertools.combinations(block, t - 1)
                            if tuple(sorted(r + (x,))) in uncovered)
                    heapq.heappush(heap, (-g, x))
            while len(block) < k:
                while heap:
                    ng, x = heapq.heappop(heap)
                    if x in block:
                        continue
                    g = sum(1 for r in itertools.combinations(block, t - 1)
                            if tuple(sorted(r + (x,))) in uncovered)
                    if g == -ng:
                        block.append(x)
                        break
                    heapq.heappush(heap, (-g, x))
                else:
                    break
            if len(block) < k:
                # pad arbitrarily (rare)
                for x in range(v):
                    if x not in block:
                        block.append(x)
                    if len(block) == k:
                        break
            block = tuple(sorted(block))
            blocks.append(block)
            uncovered.difference_update(tsubs(block))
        return blocks

    def local_search(blocks, max_iters=30000):
        nonlocal uncovered
        blocks = [list(b) for b in blocks]
        n = len(blocks)
        inb = [set(b) for b in blocks]
        for _ in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline - 3.0:
                    break
                b = blocks[i]
                inb_i = inb[i]
                for j in range(n):
                    if j == i:
                        continue
                    for x in b:
                        for y in inb[j]:
                            if y in inb_i:
                                continue
                            nb = set(b)
                            nb.discard(x)
                            nb.add(y)
                            newcov = sum(1 for s in itertools.combinations(nb, t)
                                         if s in uncovered)
                            if newcov > covered_count(b):
                                b.remove(x)
                                b.append(y)
                                b.sort()
                                inb_i.discard(x)
                                inb_i.add(y)
                                inb[i] = inb_i
                                uncovered.difference_update(itertools.combinations(b, t))
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return [tuple(b) for b in blocks]

    # baseline greedy
    uncovered = set(itertools.combinations(range(v), t))
    base = greedy(None)
    if base is None:
        # fallback: simple greedy to guarantee validity
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            for x in range(v):
                if len(block) == k:
                    break
                if x not in block:
                    block.append(x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks

    best = local_search(base)
    best_n = len(best)

    # restarts: seed a new greedy with one random block, then polish
    for _ in range(200):
        if time.time() > deadline - 4.0:
            break
        if best_n <= 131:  # at known lower bound; stop
            break
        seed = tuple(sorted(rng.sample(range(v), k)))
        uncovered = set(itertools.combinations(range(v), t))
        cand = greedy(seed)
        if cand is None:
            continue
        if len(cand) < best_n:
            cand = local_search(cand)
            if len(cand) < best_n:
                best = cand
                best_n = len(cand)

    # final polish on best
    uncovered = set(itertools.combinations(range(v), t))
    final = local_search(best)
    if len(final) < len(best):
        best = final

    return [list(b) for b in best]

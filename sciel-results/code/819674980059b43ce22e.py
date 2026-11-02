import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin before 150s wall-clock kill

    # ---- Precomputation ----
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    t_subsets_per_block = len(list(itertools.combinations(range(k), t)))

    # For each block, the indices of the t-subsets it covers
    BlockTsets = np.zeros((num_blocks, t_subsets_per_block), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        cnt = 0
        for sub in itertools.combinations(block, t):
            BlockTsets[i, cnt] = tset_to_idx[sub]
            cnt += 1

    # Inverted index: for each t-set, the blocks that cover it
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for i in range(num_blocks):
        for ts in BlockTsets[i]:
            tset_to_blocks[ts].append(i)

    # ---- Core operators ----
    def polish(blocks):
        """Remove redundant blocks (every t-set they cover is covered >=2 times)."""
        if not blocks:
            return blocks
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[BlockTsets[b]] += 1
        i = len(blocks) - 1
        while i >= 0:
            b = blocks[i]
            if np.all(cover_count[BlockTsets[b]] >= 2):
                cover_count[BlockTsets[b]] -= 1
                blocks.pop(i)
            else:
                i -= 1
        return blocks

    def greedy_construct(explore_prob=0.0, noise=0.0):
        """Vectorized greedy set cover with optional exploration / gain noise."""
        uncovered = np.ones(num_tsets, dtype=bool)
        blocks = []
        while uncovered.any():
            if time.time() > deadline - 5.0:
                break
            gains = uncovered[BlockTsets].sum(axis=1)
            if noise:
                gains = gains + np.random.normal(0.0, noise, size=num_blocks)
            max_gain = gains.max()
            if max_gain <= 0:
                break
            if explore_prob > 0 and random.random() < explore_prob:
                thr = max_gain * 0.8
                cands = np.where(gains >= thr)[0]
            else:
                cands = np.where(gains == max_gain)[0]
            b = int(cands[np.random.randint(len(cands))])
            blocks.append(b)
            uncovered[BlockTsets[b]] = False
        return blocks

    def local_search(blocks, max_iters=200):
        """1-exchange: remove redundant blocks and swap blocks to reduce count."""
        n = len(blocks)
        if n < 2:
            return blocks, 0
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[BlockTsets[b]] += 1
        block_set = set(blocks)
        improved = 0
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            n = len(blocks)
            if n < 2:
                break
            i = np.random.randint(n)
            b_old = blocks[i]
            bts_old = BlockTsets[b_old]
            unique_ts = bts_old[cover_count[bts_old] == 1]
            if len(unique_ts) == 0:
                cover_count[bts_old] -= 1
                blocks.pop(i)
                block_set.discard(b_old)
                improved += 1
                continue
            # candidate blocks covering every uniquely-covered t-set
            cand = set(tset_to_blocks[int(unique_ts[0])])
            for ts in unique_ts[1:]:
                cand &= set(tset_to_blocks[int(ts)])
            cand -= block_set
            cand.discard(b_old)
            if not cand:
                continue
            # choose the candidate adding the fewest new t-sets
            best = -1
            best_score = float('inf')
            for b_new in cand:
                bts_new = BlockTsets[b_new]
                new_unique = int(np.sum(cover_count[bts_new] == 0))
                if new_unique < best_score:
                    best_score = new_unique
                    best = b_new
            if best == -1:
                continue
            cover_count[bts_old] -= 1
            cover_count[BlockTsets[best]] += 1
            blocks[i] = best
            block_set.discard(b_old)
            block_set.add(best)
            improved += 1
        return blocks, improved

    def hill_climbing(blocks, max_rounds=30):
        for _ in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            blocks, imp = local_search(blocks)
            if not imp:
                break
            blocks = polish(blocks)
        return blocks

    def fitness(blocks):
        """Multi-objective: block count plus a penalty for uncovered t-sets."""
        if not blocks:
            return num_tsets * 1000.0
        cover = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover[BlockTsets[b]] += 1
        uncovered = int(np.sum(cover == 0))
        return len(blocks) + uncovered * 1000.0

    def crossover(sol_a, sol_b):
        """Merge two coverings, then polish + local search."""
        if time.time() > deadline - 5.0:
            return sol_a
        merged = list(sol_a)
        for b in sol_b:
            if b not in merged:
                merged.append(b)
        merged = polish(merged)
        merged = hill_climbing(merged)
        return merged

    def mutate(sol):
        """Swap one block for a high-gain block; keep only if objective improves."""
        if time.time() > deadline - 5.0:
            return sol
        n = len(sol)
        if n < 2:
            return sol
        base = fitness(sol)
        i = int(np.random.randint(n))
        b_old = sol[i]
        bts_old = BlockTsets[b_old]
        cover = np.zeros(num_tsets, dtype=np.int32)
        for b in sol:
            cover[BlockTsets[b]] += 1
        unique_ts = bts_old[cover[bts_old] == 1]
        if len(unique_ts) == 0:
            # redundant: just drop it
            new_sol = sol[:i] + sol[i+1:]
            return new_sol
        cand = set(tset_to_blocks[int(unique_ts[0])])
        for ts in unique_ts[1:]:
            cand &= set(tset_to_blocks[int(ts)])
        cand -= set(sol)
        if not cand:
            return sol
        best = -1
        best_score = float('inf')
        for b_new in cand:
            bts_new = BlockTsets[b_new]
            s = int(np.sum(cover[bts_new] == 0))
            if s < best_score:
                best_score = s
                best = b_new
        if best == -1:
            return sol
        new_sol = sol[:]
        new_sol[i] = best
        new_sol = polish(new_sol)
        if fitness(new_sol) <= base:
            return new_sol
        return sol

    # ---- Population-based search ----
    best_blocks = None
    best_len = float('inf')
    best_fit = float('inf')

    def consider(blocks):
        nonlocal best_blocks, best_len, best_fit
        blocks = polish(blocks)
        f = fitness(blocks)
        if f < best_fit:
            best_fit = f
            best_len = len(blocks)
            best_blocks = blocks[:]

    # Seed the population with several distinct initial solutions
    population = []
    strategies = [
        lambda: greedy_construct(explore_prob=0.0, noise=0.0),
        lambda: greedy_construct(explore_prob=0.3, noise=0.0),
        lambda: greedy_construct(explore_prob=0.0, noise=0.5),
        lambda: greedy_construct(explore_prob=0.5, noise=0.5),
    ]
    for _ in range(4):
        if time.time() > deadline - 15.0:
            break
        s = strategies[len(population) % len(strategies)]()
        s = hill_climbing(s)
        population.append(s)
        consider(s)

    # Evolve the population
    gen = 0
    while time.time() < deadline - 8.0:
        gen += 1
        # Rank population by fitness (best first)
        population.sort(key=lambda s: fitness(s))
        # Keep the elite, drop the worst
        elite = population[:max(1, len(population) - 1)]
        # Generate offspring
        offspring = []
        while time.time() < deadline - 8.0:
            if len(offspring) >= 2:
                break
            a = elite[0]
            b = elite[1] if len(elite) > 1 else elite[0]
            if random.random() < 0.5:
                child = crossover(a, b)
            else:
                child = mutate(random.choice(elite))
            child = hill_climbing(child)
            offspring.append(child)
            consider(child)
        # New population: elites + offspring, capped in size
        population = (elite + offspring)[:4]

    if best_blocks is None:
        # Fallback: a single greedy solution
        s = greedy_construct()
        s = hill_climbing(s)
        best_blocks = s

    return [list(all_blocks[b]) for b in best_blocks]

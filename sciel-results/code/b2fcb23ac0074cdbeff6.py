import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 118
    rng = random.Random(987654321)

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}

    # For each block, list of t-subset indices it covers
    block_tsubs = []
    for blk in all_blocks:
        subs = []
        for sub in itertools.combinations(blk, t):
            subs.append(tsub_index[sub])
        block_tsubs.append(np.array(subs, dtype=np.int32))

    # Greedy initial solution
    def greedy_init():
        uncovered = np.ones(n_tsubs, dtype=np.int8)
        chosen = []
        for _ in range(300):  # cap
            if uncovered.sum() == 0:
                break
            best_gain = 0
            best_bi = -1
            for bi in range(n_blocks):
                if bi in chosen:
                    continue
                gain = uncovered[block_tsubs[bi]].sum()
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    if best_gain == uncovered.sum():
                        break
            if best_bi < 0:
                break
            chosen.append(best_bi)
            uncovered[block_tsubs[best_bi]] = 0
        return chosen

    # Prune redundant blocks
    def prune(chosen):
        if not chosen:
            return chosen
        counts = np.zeros(n_tsubs, dtype=np.int32)
        for bi in chosen:
            counts[block_tsubs[bi]] += 1
        pruned = []
        for bi in chosen:
            subs = block_tsubs[bi]
            if np.any(counts[subs] == 1):
                pruned.append(bi)
                counts[subs] -= 1
        return pruned

    # Start with greedy, then prune
    best = prune(greedy_init())
    best_len = len(best)

    # SA: fixed-length representation, 1-for-1 swaps
    # We'll work with a list of block indices, possibly with duplicates initially, but we'll avoid duplicates in practice.
    # Actually, for SA, let's use a set-based approach but with a fixed "population" of swaps.
    # Better: use a list, and allow duplicates in the "genome" but evaluate unique blocks.
    # To keep it simple and fast, let's use a list of unique blocks, and do 1-out/1-in swaps.

    def evaluate(chosen):
        """Return (n_uncovered, n_blocks)."""
        if not chosen:
            return n_tsubs, 0
        counts = np.zeros(n_tsubs, dtype=np.int32)
        for bi in chosen:
            counts[block_tsubs[bi]] += 1
        n_uncovered = int(np.sum(counts == 0))
        return n_uncovered, len(chosen)

    # SA parameters
    T_start = 10.0
    T_end = 0.01
    cooling = 0.995
    max_iter = 50000

    current = best[:]
    current_uncovered, current_nblocks = evaluate(current)
    current_fitness = current_uncovered * 1000 + current_nblocks

    best_sol = current[:]
    best_fitness = current_fitness
    best_uncovered = current_uncovered
    best_nblocks = current_nblocks

    T = T_start
    rng_np = np.random.RandomState(42)

    for it in range(max_iter):
        if time.time() > deadline - 5:
            break

        # Random 1-out/1-in swap
        if len(current) == 0:
            # Add a random block
            bi = rng_np.randint(0, n_blocks)
            new_current = current + [bi]
        else:
            # Pick a block to remove
            out_idx = rng_np.randint(0, len(current))
            bi_out = current[out_idx]
            # Pick a block to add (different from bi_out and not already in current)
            # To avoid expensive checks, just pick random and retry if duplicate
            bi_in = rng_np.randint(0, n_blocks)
            # Check if bi_in is already in current (excluding the one we're removing)
            # For speed, we'll allow duplicates in the list but evaluate unique set
            # Actually, let's just do the swap and evaluate
            new_current = current[:]
            new_current[out_idx] = bi_in

        new_uncovered, new_nblocks = evaluate(new_current)
        new_fitness = new_uncovered * 1000 + new_nblocks

        delta = new_fitness - current_fitness
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            current = new_current
            current_uncovered = new_uncovered
            current_nblocks = new_nblocks
            current_fitness = new_fitness

            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_uncovered = current_uncovered
                best_nblocks = current_nblocks
                best_sol = current[:]

        T *= cooling
        if T < T_end:
            T = T_end

    # Prune the best solution
    best_sol = prune(best_sol)
    best_uncovered, best_nblocks = evaluate(best_sol)

    # If SA didn't improve, try a few more greedy restarts
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            cand = prune(greedy_init())
            cu, cn = evaluate(cand)
            if cn < best_nblocks:
                best_sol = cand
                best_nblocks = cn
                best_uncovered = cu

    # Final prune
    best_sol = prune(best_sol)

    return [list(all_blocks[bi]) for bi in best_sol]

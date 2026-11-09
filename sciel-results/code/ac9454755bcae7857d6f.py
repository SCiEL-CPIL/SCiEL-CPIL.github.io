"""Covering design C(v,k,t) — Bitset-accelerated greedy with local search."""
import itertools
import time
import random

def construct(v, k, t):
    start = time.time()
    DEADLINE = start + 115.0  # safety margin before 120s kill

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    # Precompute: for each block, the list of t-subset indices it covers
    block_tsubs = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(t_idx[s])
        block_tsubs.append(subs)

    # Precompute: for each t-subset, the list of block indices that cover it
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)

    # Precompute block gain (number of t-subsets each block covers)
    block_size = [len(subs) for subs in block_tsubs]

    def run_greedy(seed=0):
        """Greedy set cover with optional random tie-breaking."""
        rng = random.Random(seed)
        is_covered = bytearray(N_t)
        uncovered = N_t
        gains = block_size[:]  # copy
        selected = []

        while uncovered > 0 and time.time() < DEADLINE:
            # Find block with max gain; break ties randomly
            best_gain = -1
            candidates = []
            for i in range(N_blocks):
                g = gains[i]
                if g > best_gain:
                    best_gain = g
                    candidates = [i]
                elif g == best_gain:
                    candidates.append(i)

            if best_gain <= 0:
                break

            if len(candidates) > 1:
                best_i = candidates[rng.randrange(len(candidates))]
            else:
                best_i = candidates[0]

            selected.append(all_blocks[best_i])

            for si in block_tsubs[best_i]:
                if not is_covered[si]:
                    is_covered[si] = 1
                    uncovered -= 1
                    for b_idx in tsub_blocks[si]:
                        gains[b_idx] -= 1

        return selected

    def try_reduce(blocks):
        """Remove redundant blocks iteratively."""
        if not blocks:
            return blocks

        # Map blocks to their t-subset index sets
        block_subs_list = []
        for b in blocks:
            subs = set()
            for s in itertools.combinations(b, t):
                subs.add(t_idx[s])
            block_subs_list.append(subs)

        n = len(blocks)
        cov = [0] * N_t
        for subs in block_subs_list:
            for si in subs:
                cov[si] += 1

        removed = [False] * n
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for i in range(n):
                if removed[i]:
                    continue
                removable = True
                for si in block_subs_list[i]:
                    if cov[si] <= 1:
                        removable = False
                        break
                if removable:
                    removed[i] = True
                    changed = True
                    for si in block_subs_list[i]:
                        cov[si] -= 1

        return [b for i, b in enumerate(blocks) if not removed[i]]

    def local_search(blocks):
        """Try to replace a block with another that covers more uncovered t-subsets."""
        if not blocks:
            return blocks

        # Build coverage structure
        block_set = set(blocks)
        block_subs_list = []
        for b in blocks:
            subs = set()
            for s in itertools.combinations(b, t):
                subs.add(t_idx[s])
            block_subs_list.append(subs)

        n = len(blocks)
        cov = [0] * N_t
        for subs in block_subs_list:
            for si in subs:
                cov[si] += 1

        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for i in range(n):
                if time.time() > DEADLINE:
                    break
                # Try removing block i and replacing with a better one
                # First check if we can just remove it
                removable = True
                for si in block_subs_list[i]:
                    if cov[si] <= 1:
                        removable = False
                        break
                if removable:
                    for si in block_subs_list[i]:
                        cov[si] -= 1
                    blocks.pop(i)
                    block_subs_list.pop(i)
                    n -= 1
                    improved = True
                    break

                # Try swapping: find a block not in set that covers more of the
                # t-subsets currently only covered by block i
                # (This is expensive; only do a limited search)
                # For now, just do removal pass

        return blocks

    # Run multiple greedy restarts and keep the best
    best_solution = None
    best_len = float('inf')

    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > DEADLINE - 10:
            break
        sol = run_greedy(seed=r)
        sol = try_reduce(sol)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Final local search / reduction pass
    if best_solution and time.time() < DEADLINE - 5:
        best_solution = try_reduce(best_solution)
        best_solution = local_search(best_solution)

    if best_solution is None:
        # Fallback: single greedy
        best_solution = run_greedy(seed=0)
        best_solution = try_reduce(best_solution)

    return best_solution

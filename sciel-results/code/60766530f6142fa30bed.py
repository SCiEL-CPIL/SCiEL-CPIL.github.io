import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # 1. Precomputation
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)

    # Map t-subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    # block_subsets[i] = set of indices of t-subsets covered by block i (use sets for O(1) lookup)
    block_subsets = [set() for _ in range(n_blocks)]
    # subset_to_blocks[j] = list of block indices covering t-subset j
    subset_to_blocks = [[] for _ in range(n_t_subsets)]

    for i, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            block_subsets[i].add(idx)
            subset_to_blocks[idx].append(i)

    # 2. Greedy Construction
    in_block = [False] * n_blocks
    cover_count = [0] * n_t_subsets
    uncovered_count = n_t_subsets
    solution = []

    while uncovered_count > 0:
        best_block = -1
        max_gain = 0
        for i in range(n_blocks):
            if in_block[i]:
                continue
            gain = 0
            for s_idx in block_subsets[i]:
                if cover_count[s_idx] == 0:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = i
                if gain == uncovered_count:  # Can't get better
                    break

        if best_block == -1 or max_gain == 0:
            break

        in_block[best_block] = True
        solution.append(best_block)
        for s_idx in block_subsets[best_block]:
            if cover_count[s_idx] == 0:
                uncovered_count -= 1
            cover_count[s_idx] += 1

    # 3. Helper: Remove redundant blocks
    def remove_redundants():
        nonlocal in_block, cover_count, solution
        changed = True
        while changed:
            changed = False
            for b in list(solution):
                if not in_block[b]:
                    continue
                redundant = True
                for s_idx in block_subsets[b]:
                    if cover_count[s_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    in_block[b] = False
                    solution.remove(b)
                    for s_idx in block_subsets[b]:
                        cover_count[s_idx] -= 1
                    changed = True
                    break

    remove_redundants()

    best_count = len(solution)
    best_solution_snapshot = list(solution)

    # 4. Local Search Loop
    while time.time() < deadline:
        if not solution:
            break
        b_out = random.choice(solution)

        # Find critical subsets for b_out (those with cover_count == 1)
        criticals = [s_idx for s_idx in block_subsets[b_out] if cover_count[s_idx] == 1]

        if not criticals:
            # b_out is redundant, remove it
            in_block[b_out] = False
            solution.remove(b_out)
            for s_idx in block_subsets[b_out]:
                cover_count[s_idx] -= 1
            remove_redundants()
            if len(solution) < best_count:
                best_count = len(solution)
                best_solution_snapshot = list(solution)
            continue

        # Find candidate b_in that covers all criticals
        first_crit = criticals[0]
        best_gain = -1
        best_b_in = -1

        for b_in in subset_to_blocks[first_crit]:
            if in_block[b_in]:
                continue
            # Check if b_in covers all criticals
            covers_all = all(s_idx in block_subsets[b_in] for s_idx in criticals)
            if not covers_all:
                continue
            # Calculate net gain: how many new subsets does b_in cover that aren't already covered?
            # Actually, for a swap, we want to minimize the increase in redundancy or maximize coverage
            # Simpler: just accept any valid swap, then remove redundants
            # To prefer better swaps, compute how many subsets would have cover_count decrease to 0 if we removed b_out
            # and how many would increase
            # For simplicity, accept first valid swap
            best_b_in = b_in
            break

        if best_b_in != -1:
            # Perform swap: remove b_out, add b_in
            in_block[b_out] = False
            solution.remove(b_out)
            for s_idx in block_subsets[b_out]:
                cover_count[s_idx] -= 1

            in_block[best_b_in] = True
            solution.append(best_b_in)
            for s_idx in block_subsets[best_b_in]:
                cover_count[s_idx] += 1

            # Remove any newly redundant blocks
            remove_redundants()

            if len(solution) < best_count:
                best_count = len(solution)
                best_solution_snapshot = list(solution)

    # Return the best solution found
    return [list(all_blocks[i]) for i in best_solution_snapshot]

import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin for 150s kill

    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)

    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Precompute: for each block, list of subset indices it covers
    block_to_subsets = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(subset_to_idx[s])
        block_to_subsets.append(subs)

    # For each subset, list of block indices that cover it
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx, subs in enumerate(block_to_subsets):
        for s_idx in subs:
            subset_to_blocks[s_idx].append(b_idx)

    # NumPy representation for fast operations
    # block_to_subsets_np: list of arrays
    block_to_subsets_np = [np.array(s, dtype=np.int32) for s in block_to_subsets]

    # For greedy: precompute the number of subsets each block covers (static)
    block_sizes = np.array([len(s) for s in block_to_subsets], dtype=np.int32)

    def time_left():
        return deadline - time.time()

    def greedy_standard():
        """Standard greedy set cover using NumPy for speed."""
        coverage = np.zeros(num_subsets, dtype=np.int32)
        solution = []
        uncovered_mask = np.ones(num_subsets, dtype=bool)
        uncovered_count = num_subsets

        while uncovered_count > 0 and time_left() > 2.0:
            # Compute score for each block: how many uncovered subsets it covers
            # Use numpy: for each block, count how many of its subsets are uncovered
            scores = np.zeros(num_blocks, dtype=np.int32)
            for b_idx in range(num_blocks):
                subs = block_to_subsets_np[b_idx]
                scores[b_idx] = np.sum(uncovered_mask[subs])

            max_score = np.max(scores)
            if max_score == 0:
                break

            # Randomize tie-breaking among top candidates
            top_candidates = np.where(scores == max_score)[0]
            best_b = int(random.choice(top_candidates))

            solution.append(best_b)

            # Update coverage
            subs = block_to_subsets_np[best_b]
            newly_covered = uncovered_mask[subs]
            uncovered_count -= np.sum(newly_covered)
            uncovered_mask[subs] = False

        return solution if len(solution) > 0 else None

    def greedy_critical():
        """Greedy that prioritizes the most 'critical' (hard to cover) uncovered subset."""
        coverage = np.zeros(num_subsets, dtype=np.int32)
        solution = []
        uncovered_mask = np.ones(num_subsets, dtype=bool)
        uncovered_count = num_subsets

        while uncovered_count > 0 and time_left() > 2.0:
            # Find the uncovered subset with the fewest covering blocks (most constrained)
            # Among uncovered, find min len(subset_to_blocks[s])
            best_s = -1
            best_count = float('inf')
            # Sample a random subset of uncovered for speed
            if uncovered_count < 200:
                candidates_s = np.where(uncovered_mask)[0]
            else:
                candidates_s = np.random.choice(num_subsets, size=200, replace=False)
                candidates_s = candidates_s[uncovered_mask[candidates_s]]

            for s in candidates_s:
                cnt = len(subset_to_blocks[s])
                if cnt < best_count:
                    best_count = cnt
                    best_s = s

            if best_s < 0:
                # Fallback to standard greedy step
                scores = np.zeros(num_blocks, dtype=np.int32)
                for b_idx in range(num_blocks):
                    subs = block_to_subsets_np[b_idx]
                    scores[b_idx] = np.sum(uncovered_mask[subs])
                max_score = np.max(scores)
                if max_score == 0:
                    break
                top_candidates = np.where(scores == max_score)[0]
                best_b = int(random.choice(top_candidates))
            else:
                # Pick a random block that covers best_s
                candidates_b = subset_to_blocks[best_s]
                best_b = random.choice(candidates_b)

            solution.append(best_b)

            subs = block_to_subsets_np[best_b]
            newly_covered = uncovered_mask[subs]
            uncovered_count -= np.sum(newly_covered)
            uncovered_mask[subs] = False

        return solution if len(solution) > 0 else None

    def local_search(solution):
        """Multi-phase local search: remove redundant, 1-swaps, then 2-for-3 exchanges."""
        if solution is None:
            return None

        # Track coverage counts
        coverage = np.zeros(num_subsets, dtype=np.int32)
        for b in solution:
            subs = block_to_subsets_np[b]
            coverage[subs] += 1

        # Phase 1: Remove redundant blocks
        changed = True
        while changed and time_left() > 5.0:
            changed = False
            new_solution = []
            for b in solution:
                subs = block_to_subsets_np[b]
                # Check if any subset has coverage == 1 (i.e., this block is the only cover)
                if np.min(coverage[subs]) > 1:
                    # Redundant, remove
                    coverage[subs] -= 1
                    changed = True
                else:
                    new_solution.append(b)
            solution = new_solution

        # Phase 2: 1-swap (replace one block with another)
        for _iteration in range(50):
            if time_left() < 10.0:
                break
            improved = False
            sol_set = set(solution)
            n = len(solution)

            for i in range(n):
                if time_left() < 10.0:
                    break
                b = solution[i]
                subs_b = block_to_subsets_np[b]

                # Find critical subsets (coverage == 1)
                critical = subs_b[coverage[subs_b] == 1]
                if len(critical) == 0:
                    # Redundant, remove
                    coverage[subs_b] -= 1
                    solution.pop(i)
                    improved = True
                    break

                # Find blocks that cover ALL critical subsets
                # Start with blocks covering the first critical subset
                if len(critical) > 0:
                    cand_blocks = set(subset_to_blocks[int(critical[0])])
                    for s in critical[1:]:
                        cand_blocks.intersection_update(subset_to_blocks[int(s)])
                        if not cand_blocks:
                            break

                    if cand_blocks:
                        # Filter: not in solution
                        valid = [c for c in cand_blocks if c not in sol_set]
                        if valid:
                            # Pick one that maximizes coverage of currently uncovered or low-coverage subsets
                            best_new = None
                            best_gain = -1
                            for new_b in valid[:20]:  # Limit to 20 for speed
                                subs_new = block_to_subsets_np[new_b]
                                # Gain: how many subsets with coverage 1 this new block covers (beyond critical)
                                gain = np.sum((coverage[subs_new] == 1) & ~np.isin(subs_new, critical))
                                if gain > best_gain:
                                    best_gain = gain
                                    best_new = new_b

                            if best_new is not None:
                                # Perform swap
                                coverage[subs_b] -= 1
                                subs_new = block_to_subsets_np[best_new]
                                coverage[subs_new] += 1
                                solution[i] = best_new
                                sol_set.discard(b)
                                sol_set.add(best_new)
                                improved = True
                                break

            if not improved:
                break

        # Phase 3: 2-for-3 exchange (remove 2 blocks, add 3)
        # This can escape local minima where no single swap helps
        for _iteration in range(20):
            if time_left() < 15.0:
                break

            n = len(solution)
            if n < 3:
                break

            improved_23 = False
            # Try random pairs of blocks to remove
            for _trial in range(100):
                if time_left() < 15.0:
                    break

                if len(solution) < 3:
                    break

                i, j = random.sample(range(len(solution)), 2)
                b1 = solution[i]
                b2 = solution[j]
                subs1 = block_to_subsets_np[b1]
                subs2 = block_to_subsets_np[b2]

                # Simulate removal
                # Find subsets that would become uncovered (coverage == 1 for both)
                # We need to find 3 blocks that cover all subsets that lose their last cover
                # Subsets with coverage == 1 in subs1, or coverage == 1 in subs2, or coverage == 2 in intersection

                # Coverage after removing b1 and b2:
                # For each subset, new_coverage = coverage - (1 if in subs1 else 0) - (1 if in subs2 else 0)
                # Uncovered = new_coverage < 0, i.e., coverage < 1 + (in subs2) + (in subs1)

                # More precisely: subsets that become uncovered are those where:
                # coverage[s] == 1 and s in subs1 and s not in subs2
                # coverage[s] == 1 and s in subs2 and s not in subs1
                # coverage[s] == 2 and s in both subs1 and subs2

                uncovered_after = set()
                for s in subs1:
                    if coverage[s] == 1:
                        if s not in subs2:
                            uncovered_after.add(int(s))
                for s in subs2:
                    if coverage[s] == 1:
                        if s not in subs1:
                            uncovered_after.add(int(s))
                # For subsets in both:
                for s in subs1:
                    if s in subs2 and coverage[s] == 2:
                        uncovered_after.add(int(s))

                if not uncovered_after:
                    # Removing both doesn't uncover anything - they're redundant!
                    # Just remove them
                    new_sol = []
                    for idx, b in enumerate(solution):
                        if idx != i and idx != j:
                            new_sol.append(b)
                        else:
                            continue
                    solution = new_sol
                    improved_23 = True
                    break

                # Need to find 3 blocks covering all of uncovered_after
                # For each subset in uncovered_after, find candidate blocks
                # Use the most constrained subset to start
                if len(uncovered_after) > 30:
                    # Too many uncovered subsets, skip this pair
                    continue

                # Find blocks that cover the most uncovered subsets
                # For each block, count how many of uncovered_after it covers
                # This is expensive, so sample
                cand_blocks_set = set()
                for s in list(uncovered_after)[:10]:  # Sample up to 10
                    cand_blocks_set.update(subset_to_blocks[s])

                # For each candidate block, compute how many uncovered_after it covers
                best_triplet = None
                best_covered = 0
                uncovered_list = list(uncovered_after)

                # Greedy selection of 3 blocks
                remaining_uncovered = set(uncovered_list)
                selected_blocks = []

                for step in range(3):
                    if not remaining_uncovered:
                        break
                    best_b = None
                    best_count = 0
                    for b in list(cand_blocks_set)[:100]:  # Limit candidates
                        subs_b = set(block_to_subsets_np[b].tolist())
                        count = len(remaining_uncovered & subs_b)
                        if count > best_count:
                            best_count = count
                            best_b = b
                    if best_b is None or best_count == 0:
                        break
                    selected_blocks.append(best_b)
                    subs_b = set(block_to_subsets_np[best_b].tolist())
                    remaining_uncovered -= subs_b
                    if not remaining_uncovered:
                        break

                if not remaining_uncovered and len(selected_blocks) <= 3:
                    # Check that selected_blocks are not in solution (except b1, b2)
                    sol_set = set(solution)
                    sol_set.discard(b1)
                    sol_set.discard(b2)
                    if not any(b in sol_set for b in selected_blocks):
                        # Perform the exchange
                        # Remove b1 and b2
                        for s in subs1:
                            coverage[s] -= 1
                        for s in subs2:
                            coverage[s] -= 1
                        # Add selected_blocks
                        new_sol = [b for idx, b in enumerate(solution) if idx not in (i, j)]
                        for nb in selected_blocks:
                            subs_nb = block_to_subsets_np[nb]
                            coverage[subs_nb] += 1
                            new_sol.append(nb)
                        solution = new_sol
                        improved_23 = True
                        break

            if not improved_23:
                break

        # Final redundancy removal
        changed = True
        while changed and time_left() > 5.0:
            changed = False
            new_solution = []
            for b in solution:
                subs = block_to_subsets_np[b]
                if np.min(coverage[subs]) > 1:
                    coverage[subs] -= 1
                    changed = True
                else:
                    new_solution.append(b)
            solution = new_solution

        return solution

    # Run multiple strategies
    best_solution = None
    best_len = float('inf')

    # First run: standard greedy
    sol = greedy_standard()
    if sol is not None:
        sol = local_search(sol)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol.copy()

    # Second run: critical greedy
    if time_left() > 30.0:
        sol = greedy_critical()
        if sol is not None:
            sol = local_search(sol)
            if sol is not None and len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol.copy()

    # Additional runs with random restarts
    max_restarts = 10
    for run in range(max_restarts):
        if time_left() < 25.0:
            break

        # Randomize the greedy by shuffling block order consideration
        # We'll use a randomized greedy
        coverage = np.zeros(num_subsets, dtype=np.int32)
        solution = []
        uncovered_mask = np.ones(num_subsets, dtype=bool)
        uncovered_count = num_subsets

        while uncovered_count > 0 and time_left() > 5.0:
            scores = np.zeros(num_blocks, dtype=np.int32)
            for b_idx in range(num_blocks):
                subs = block_to_subsets_np[b_idx]
                scores[b_idx] = np.sum(uncovered_mask[subs])

            max_score = np.max(scores)
            if max_score == 0:
                break

            # Top-k randomization
            top_k = min(5, num_blocks)
            # Get top scores with some noise
            noisy_scores = scores + np.random.random(num_blocks) * max_score * 0.1
            top_indices = np.argsort(noisy_scores)[-top_k:]
            best_b = int(random.choice(top_indices))

            solution.append(best_b)
            subs = block_to_subsets_np[best_b]
            newly_covered = uncovered_mask[subs]
            uncovered_count -= np.sum(newly_covered)
            uncovered_mask[subs] = False

        if len(solution) > 0:
            sol = local_search(solution)
            if sol is not None and len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol.copy()

    if best_solution is None:
        # Fallback
        sol = greedy_standard()
        if sol is None:
            return []
        return [list(all_blocks[b]) for b in sol]

    return [list(all_blocks[b]) for b in best_solution]

if __name__ == "__main__":
    blocks = construct(19, 7, 4)
    print(len(blocks))


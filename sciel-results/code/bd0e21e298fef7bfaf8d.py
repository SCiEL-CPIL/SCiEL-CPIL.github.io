import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s

    # ---------- Precomputation ----------
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # For each block, the list of t-subset indices it covers
    block_coverages = []
    # For each t-subset, the list of blocks that cover it
    t_sub_to_blocks = [[] for _ in range(N)]

    for bi, b in enumerate(all_blocks):
        covers = []
        for s in itertools.combinations(b, t):
            ti = t_idx[s]
            covers.append(ti)
            t_sub_to_blocks[ti].append(bi)
        block_coverages.append(np.array(covers, dtype=np.int32))

    # ---------- Helper: build a valid covering via greedy ----------
    def greedy_cover(uncovered_mask, used_blocks_set=None, max_blocks=None):
        if used_blocks_set is None:
            used_blocks_set = set()
        blocks = []
        mask = uncovered_mask.copy()
        while np.any(mask):
            if time.time() > deadline - 5:
                return blocks
            unc_idx = np.where(mask)[0]
            if len(unc_idx) == 0:
                break
            target = random.choice(unc_idx)
            candidates = t_sub_to_blocks[target]
            best_bi = -1
            best_gain = 0
            for bi in candidates:
                if bi in used_blocks_set:
                    continue
                gain = np.sum(mask[block_coverages[bi]])
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    if gain >= 50:
                        break
            if best_bi == -1:
                break
            blocks.append(best_bi)
            used_blocks_set.add(best_bi)
            mask[block_coverages[best_bi]] = False
            if max_blocks and len(blocks) >= max_blocks:
                break
        return blocks

    # ---------- Helper: remove redundant blocks ----------
    def remove_redundant(blocks_list):
        if not blocks_list:
            return []
        cover_count = np.zeros(N, dtype=np.int32)
        for bi in blocks_list:
            cover_count[block_coverages[bi]] += 1
        new_blocks = []
        for bi in blocks_list:
            cov = block_coverages[bi]
            can_remove = True
            for ti in cov:
                if cover_count[ti] <= 1:
                    can_remove = False
                    break
            if can_remove:
                cover_count[cov] -= 1
            else:
                new_blocks.append(bi)
        return new_blocks

    # ---------- SA Local Search ----------
    def sa_local_search(blocks_list, max_time=130.0):
        """
        Simulated annealing local search on a *set* of blocks.
        Moves:
          - Delete: remove a block if it is redundant (all its t-subsets are covered by >= 1 other block)
          - Swap: replace one block with another, accepting if it improves or with probability exp(-delta/T)
        We maintain cover_count to quickly assess validity.
        """
        current = list(blocks_list)
        if not current:
            return current

        # Build cover_count
        cover_count = np.zeros(N, dtype=np.int32)
        for bi in current:
            cover_count[block_coverages[bi]] += 1

        current_size = len(current)
        best_size = current_size
        best_blocks = list(current)

        # SA parameters
        T_start = 10.0
        T_end = 0.01
        alpha = 0.9995  # cooling rate
        T = T_start

        rng = random.Random(12345)

        while time.time() < max_time and T > T_end:
            T *= alpha

            # Decide move type: 30% delete, 70% swap
            if rng.random() < 0.3 and len(current) > 1:
                # Delete move
                i = rng.randint(0, len(current) - 1)
                bi = current[i]
                cov = block_coverages[bi]
                # Check if redundant
                redundant = True
                for ti in cov:
                    if cover_count[ti] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Perform delete
                    cover_count[cov] -= 1
                    current.pop(i)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_blocks = list(current)
                # else: no change, just cool
            else:
                # Swap move: pick a block to remove, find a candidate to add
                if len(current) == 0:
                    continue
                i = rng.randint(0, len(current) - 1)
                bi = current[i]
                cov = block_coverages[bi]

                # Find t-subsets that would become uncovered if we remove bi
                uncovered_by_removal = [ti for ti in cov if cover_count[ti] == 1]

                if not uncovered_by_removal:
                    # Redundant: just delete
                    cover_count[cov] -= 1
                    current.pop(i)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_blocks = list(current)
                    continue

                # Need a replacement block that covers all uncovered_by_removal
                # Intersect candidate sets
                candidates = set(t_sub_to_blocks[uncovered_by_removal[0]])
                for ti in uncovered_by_removal[1:]:
                    candidates &= set(t_sub_to_blocks[ti])

                current_set = set(current)
                candidates -= current_set
                candidates.discard(bi)

                if not candidates:
                    continue

                # Score candidates: we want to minimize "loss" in cover_count
                # A good replacement covers the uncovered_by_removal and adds coverage elsewhere
                # Delta = (number of t-subsets that go from 1 to 0) - (number that go from 0 to 1)
                # But since we're swapping, we remove bi and add new_bi.
                # Net effect on cover_count:
                #   -1 for each ti in cov (bi's coverage)
                #   +1 for each ti in new_cov (new block's coverage)
                # We need all ti in uncovered_by_removal to remain >= 1, which is guaranteed
                #   if new_bi covers them (which it does by construction).
                # The "cost" is how many other t-subsets lose their only cover.
                best_new_bi = -1
                best_delta = float('inf')

                # Limit candidate evaluation for speed
                candidate_list = list(candidates)
                if len(candidate_list) > 200:
                    candidate_list = rng.sample(candidate_list, 200)

                for new_bi in candidate_list:
                    new_cov = block_coverages[new_bi]
                    # Compute delta: how many t-subsets go from cover_count 1 to 0 (bad)
                    # and how many go from 0 to 1 (good, but we're maintaining validity so 0->1 is fine)
                    # Actually, we need to check: after removing bi, some t-subsets in cov have count 1.
                    # After adding new_bi, those in new_cov get +1.
                    # The net change for a t-subset ti:
                    #   if ti in cov and ti in new_cov: net 0
                    #   if ti in cov and ti not in new_cov: net -1
                    #   if ti not in cov and ti in new_cov: net +1
                    # We need: for all ti, new_count >= 1.
                    # Since uncovered_by_removal are covered by new_bi, they're fine.
                    # For ti in cov \ new_cov: new_count = cover_count[ti] - 1. Must be >= 1 => cover_count[ti] >= 2.
                    # So delta = number of ti in cov \ new_cov where cover_count[ti] == 1.
                    # If delta == 0, the swap is valid and improves (size same, but we might enable future deletes).
                    # We prefer smaller delta.
                    delta = 0
                    for ti in cov:
                        if cover_count[ti] == 1 and ti not in set(new_cov):
                            delta += 1
                            if delta > best_delta:
                                break
                    if delta < best_delta:
                        best_delta = delta
                        best_new_bi = new_bi
                        if delta == 0:
                            break

                if best_new_bi == -1:
                    continue

                if best_delta == 0:
                    # Valid swap: perform it
                    cover_count[cov] -= 1
                    cover_count[block_coverages[best_new_bi]] += 1
                    current[i] = best_new_bi
                    # After swap, try to delete redundant blocks
                    # (the swap might have made some block redundant)
                    # We'll let the delete move handle this in the next iteration
                else:
                    # Worsening swap: accept with probability exp(-best_delta / T)
                    if rng.random() < np.exp(-best_delta / max(T, 0.01)):
                        cover_count[cov] -= 1
                        cover_count[block_coverages[best_new_bi]] += 1
                        current[i] = best_new_bi
                        # After an accepted worsening move, size is same but cover_count changed
                        # We don't update best_size here since size didn't change

            # Periodically try to remove redundant blocks
            if time.time() % 1 < 0.1:  # roughly every 10% of iterations
                # Quick redundancy check
                new_current = []
                for bi in current:
                    cov = block_coverages[bi]
                    can_remove = True
                    for ti in cov:
                        if cover_count[ti] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        cover_count[cov] -= 1
                    else:
                        new_current.append(bi)
                if len(new_current) < len(current):
                    current = new_current
                    current_size = len(current)
                    if current_size < best_size:
                        best_size = current_size
                        best_blocks = list(current)

        # Final redundancy removal
        best_blocks = remove_redundant(best_blocks)
        return best_blocks

    # ---------- Main ----------
    best_blocks = None
    best_size = float('inf')

    # Phase 1: Generate initial solutions with greedy (multiple seeds)
    for seed in range(5):
        if time.time() > deadline - 30:
            break
        random.seed(seed)
        uncovered = np.ones(N, dtype=bool)
        blocks = greedy_cover(uncovered, used_blocks_set=set())
        if blocks:
            # Verify
            test_cover = np.zeros(N, dtype=bool)
            for bi in blocks:
                test_cover[block_coverages[bi]] = True
            if np.all(test_cover):
                blocks = remove_redundant(blocks)
                if len(blocks) < best_size:
                    best_size = len(blocks)
                    best_blocks = list(blocks)

    if best_blocks is None:
        random.seed(42)
        uncovered = np.ones(N, dtype=bool)
        best_blocks = greedy_cover(uncovered)
        best_size = len(best_blocks)

    # Phase 2: SA local search
    if time.time() < deadline - 10:
        improved = sa_local_search(best_blocks, max_time=deadline - 10)
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(improved)

    # Phase 3: A few more SA passes with different random seeds
    for trial in range(3):
        if time.time() > deadline - 5:
            break
        # Slightly perturb the current best and run SA
        perturbed = list(best_blocks)
        # Add a random block
        if perturbed:
            random_bi = random.randint(0, num_blocks - 1)
            if random_bi not in set(perturbed):
                perturbed.append(random_bi)
        improved = sa_local_search(perturbed, max_time=deadline - 5 - trial * 2)
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(improved)

    # Convert back to point lists
    final_blocks = [list(all_blocks[bi]) for bi in best_blocks]

    return final_blocks

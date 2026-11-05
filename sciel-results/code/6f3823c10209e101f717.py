import itertools
import time
import random
from collections import defaultdict
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)

    best_blocks = None
    best_len = float('inf')

    # Multiple strategies with different random seeds
    trial = 0
    while time.time() < deadline:
        trial += 1
        seed = trial * 7919 + 42

        # Strategy 1: Greedy with ratio-based scoring
        blocks, valid = _greedy_cover_ratio(v, k, t, all_tsubsets, deadline, seed)
        if valid:
            blocks = _remove_redundant(blocks, v, k, t, deadline)
            blocks = _local_search(blocks, v, k, t, all_tsubsets, deadline, seed)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
                if best_len <= 76:
                    break

        # Strategy 2: Greedy with absolute gain
        if time.time() < deadline:
            blocks2, valid2 = _greedy_cover_abs(v, k, t, all_tsubsets, deadline, seed + 1)
            if valid2:
                blocks2 = _remove_redundant(blocks2, v, k, t, deadline)
                blocks2 = _local_search(blocks2, v, k, t, all_tsubsets, deadline, seed + 1)
                if len(blocks2) < best_len:
                    best_len = len(blocks2)
                    best_blocks = blocks2
                    if best_len <= 76:
                        break

        # Strategy 3: Greedy with rarity-weighted scoring
        if time.time() < deadline:
            blocks3, valid3 = _greedy_cover_rarity(v, k, t, all_tsubsets, deadline, seed + 2)
            if valid3:
                blocks3 = _remove_redundant(blocks3, v, k, t, deadline)
                blocks3 = _local_search(blocks3, v, k, t, all_tsubsets, deadline, seed + 2)
                if len(blocks3) < best_len:
                    best_len = len(blocks3)
                    best_blocks = blocks3
                    if best_len <= 76:
                        break

        # WalkSAT-style Max-SAT search to minimize block count
        if trial >= 3 and best_blocks is not None and time.time() < deadline - 15:
            ws_blocks = _walksat_minimize(best_blocks, v, k, t, all_tsubsets, deadline)
            if len(ws_blocks) < best_len:
                best_len = len(ws_blocks)
                best_blocks = ws_blocks
                if best_len <= 76:
                    break

        # If we've run several trials, do a final polish on the best
        if trial >= 6 and best_blocks is not None and time.time() < deadline - 10:
            polished = _deep_local_search(best_blocks, v, k, t, all_tsubsets, deadline)
            if len(polished) < best_len:
                best_len = len(polished)
                best_blocks = polished

    if best_blocks is None:
        # Fallback: deterministic greedy
        blocks, valid = _greedy_cover_abs(v, k, t, all_tsubsets, deadline, -1)
        if valid:
            best_blocks = _remove_redundant(blocks, v, k, t, deadline)
        else:
            best_blocks = []

    return [list(b) for b in best_blocks]


def _greedy_cover_ratio(v, k, t, all_tsubsets, deadline, seed_val):
    """
    Greedy covering using ratio-based scoring: gain / (cost of block).
    Since all blocks have the same size k, this reduces to gain, but we
    use a modified scoring that penalizes overlap with existing blocks.
    """
    rng = random.Random(seed_val)
    uncovered = set(all_tsubsets)
    blocks = []
    points = list(range(v))

    while uncovered:
        if time.time() > deadline:
            return None, False

        # Pick a random uncovered t-subset as seed
        seed = tuple(rng.choice(list(uncovered)))
        block = list(seed)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline:
                return None, False

            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            best_x = -1
            best_score = -1
            candidates = []

            for x in points:
                if x in block_set:
                    continue

                gain = 0
                for p in t_minus_1_subsets:
                    s = tuple(sorted(p + (x,)))
                    if s in uncovered:
                        gain += 1

                if gain == 0:
                    continue

                # Ratio-based: we want high gain relative to potential overlap
                # Use gain as primary, with small random perturbation for diversity
                score = gain + rng.random() * 0.1
                candidates.append((score, x))

            if not candidates:
                remaining = [x for x in points if x not in block_set]
                if not remaining:
                    return None, False
                chosen_x = rng.choice(remaining)
            else:
                candidates.sort(key=lambda c: -c[0])
                # Pick from top candidates with some randomness
                top_n = min(5, len(candidates))
                chosen_x = candidates[rng.randint(0, top_n - 1)][1]

            block.append(chosen_x)
            block_set.add(chosen_x)

        block.sort()
        blocks.append(tuple(block))

        for s in itertools.combinations(block, t):
            uncovered.discard(s)

    return blocks, True


def _greedy_cover_abs(v, k, t, all_tsubsets, deadline, seed_val):
    """
    Greedy covering using absolute gain scoring.
    """
    rng = random.Random(seed_val if seed_val >= 0 else time.time())
    uncovered = set(all_tsubsets)
    blocks = []
    points = list(range(v))

    while uncovered:
        if time.time() > deadline:
            return None, False

        seed = tuple(rng.choice(list(uncovered)))
        block = list(seed)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline:
                return None, False

            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            best_x = -1
            best_score = -1
            tiebreakers = []

            for x in points:
                if x in block_set:
                    continue

                gain = 0
                for p in t_minus_1_subsets:
                    s = tuple(sorted(p + (x,)))
                    if s in uncovered:
                        gain += 1

                if gain > best_score:
                    best_score = gain
                    best_x = x
                    tiebreakers = [x]
                elif gain == best_score and gain > 0:
                    tiebreakers.append(x)

            if best_x == -1:
                remaining = [x for x in points if x not in block_set]
                if not remaining:
                    return None, False
                chosen_x = rng.choice(remaining)
            else:
                if len(tiebreakers) > 1:
                    chosen_x = rng.choice(tiebreakers[:min(10, len(tiebreakers))])
                else:
                    chosen_x = best_x

            block.append(chosen_x)
            block_set.add(chosen_x)

        block.sort()
        blocks.append(tuple(block))

        for s in itertools.combinations(block, t):
            uncovered.discard(s)

    return blocks, True


def _greedy_cover_rarity(v, k, t, all_tsubsets, deadline, seed_val):
    """
    Greedy covering with rarity-weighted scoring.
    T-subsets that can be covered by fewer potential blocks are more valuable.
    """
    rng = random.Random(seed_val)
    uncovered = set(all_tsubsets)
    blocks = []
    points = list(range(v))

    # Precompute rarity: for each t-subset, how many k-blocks can cover it
    # A t-subset {a,b,c} can be in C(v-t, k-t) blocks
    # But we use a simpler heuristic: weight by 1/sqrt(coverage_count)
    # Actually, for uniform random blocks, all t-subsets have the same number
    # of potential covering blocks, so rarity is uniform. Instead, we weight
    # by how "hard" it is to cover, approximated by random perturbation.

    while uncovered:
        if time.time() > deadline:
            return None, False

        seed = tuple(rng.choice(list(uncovered)))
        block = list(seed)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline:
                return None, False

            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            best_x = -1
            best_score = -1
            candidates = []

            for x in points:
                if x in block_set:
                    continue

                gain = 0
                weighted_gain = 0.0
                for p in t_minus_1_subsets:
                    s = tuple(sorted(p + (x,)))
                    if s in uncovered:
                        gain += 1
                        # Weight by inverse of how many blocks could cover this
                        # (heuristic: use 1.0 for all, add small noise)
                        weighted_gain += 1.0 + rng.random() * 0.05

                if gain == 0:
                    continue

                score = weighted_gain
                candidates.append((score, x, gain))

            if not candidates:
                remaining = [x for x in points if x not in block_set]
                if not remaining:
                    return None, False
                chosen_x = rng.choice(remaining)
            else:
                candidates.sort(key=lambda c: -c[0])
                top_n = min(5, len(candidates))
                chosen_x = candidates[rng.randint(0, top_n - 1)][1]

            block.append(chosen_x)
            block_set.add(chosen_x)

        block.sort()
        blocks.append(tuple(block))

        for s in itertools.combinations(block, t):
            uncovered.discard(s)

    return blocks, True


def _remove_redundant(blocks, v, k, t, deadline):
    """
    Remove redundant blocks. A block is redundant if removing it
    doesn't leave any t-subset uncovered.
    """
    if not blocks:
        return blocks

    cover_count = defaultdict(int)
    for block in blocks:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1

    improved = list(blocks)

    for i in range(len(improved) - 1, -1, -1):
        if time.time() > deadline:
            break

        block = improved[i]
        redundant = True
        for s in itertools.combinations(block, t):
            if cover_count[s] <= 1:
                redundant = False
                break

        if redundant:
            for s in itertools.combinations(block, t):
                cover_count[s] -= 1
            improved.pop(i)

    return improved


def _local_search(blocks, v, k, t, all_tsubsets, deadline, seed_val):
    """
    Local search: try to replace blocks to reduce total count.
    Uses a combination of block replacement and redundancy removal.
    """
    if not blocks:
        return blocks

    rng = random.Random(seed_val)
    improved = list(blocks)

    # Build cover count
    cover_count = defaultdict(int)
    for block in improved:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1

    n_blocks = len(improved)
    max_iterations = 200

    for iteration in range(max_iterations):
        if time.time() > deadline:
            break

        # Try to find a block that can be replaced to make another block redundant
        any_change = False

        for i in range(n_blocks):
            if time.time() > deadline:
                break

            old_block = improved[i]
            old_tsubs = list(itertools.combinations(old_block, t))
            old_tsub_set = set(old_tsubs)

            # Find which t-subsets are covered only by this block
            unique_subs = [s for s in old_tsubs if cover_count[s] == 1]

            if not unique_subs:
                # This block is fully redundant, remove it
                for s in old_tsubs:
                    cover_count[s] -= 1
                    if cover_count[s] == 0:
                        del cover_count[s]
                improved.pop(i)
                n_blocks -= 1
                any_change = True
                break

            # Try to generate a replacement block that covers all unique_subs
            # and potentially makes other blocks redundant
            best_new_block = None
            best_score = -1

            for _ in range(30):
                if time.time() > deadline:
                    break

                # Generate candidate by modifying old_block
                new_block = list(old_block)
                remove_count = rng.randint(1, min(2, k))
                to_remove = rng.sample(new_block, remove_count)
                for x in to_remove:
                    new_block.remove(x)
                while len(new_block) < k:
                    x = rng.randint(0, v - 1)
                    if x not in new_block:
                        new_block.append(x)
                cand = tuple(sorted(new_block))

                if cand == old_block:
                    continue

                cand_tsubs = set(itertools.combinations(cand, t))

                # Check if cand covers all unique_subs
                if not all(s in cand_tsubs for s in unique_subs):
                    continue

                # Score: how many new t-subsets does cand cover?
                new_subs = cand_tsubs - old_tsub_set
                gain = len(new_subs)

                # Bonus for creating redundancy
                redundancy_bonus = 0
                for s in new_subs:
                    if cover_count.get(s, 0) >= 1:
                        redundancy_bonus += 1

                score = gain + redundancy_bonus * 0.5

                if score > best_score:
                    best_score = score
                    best_new_block = cand

            if best_new_block is not None and best_score > 0:
                # Replace
                for s in old_tsubs:
                    cover_count[s] -= 1
                    if cover_count[s] == 0:
                        del cover_count[s]
                for s in itertools.combinations(best_new_block, t):
                    cover_count[s] += 1
                improved[i] = best_new_block
                n_blocks -= 1
                any_change = True
                break

        if not any_change:
            break

    # Final redundancy removal
    improved = _remove_redundant(improved, v, k, t, deadline)

    return improved


def _deep_local_search(blocks, v, k, t, all_tsubsets, deadline):
    """
    More intensive local search for the best solution found.
    """
    if not blocks:
        return blocks

    improved = list(blocks)
    best = improved
    best_len = len(improved)

    for trial in range(10):
        if time.time() > deadline:
            break

        seed = trial * 104729 + 7
        candidate = _local_search(improved, v, k, t, all_tsubsets, deadline, seed)
        candidate = _remove_redundant(candidate, v, k, t, deadline)

        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)
            improved = candidate

    return best


def _walksat_minimize(blocks, v, k, t, all_tsubsets, deadline):
    """
    WalkSAT-style stochastic local search to minimize the number of selected blocks.
    
    Model: Each block in the initial solution is a Boolean variable (selected or not).
    Each t-subset coverage is a clause (at least one block containing it must be selected).
    Objective: minimize number of selected blocks while keeping all clauses satisfied.
    
    We use a penalty-based approach: score = num_selected + lambda * num_uncovered.
    Moves: flip a block (add or remove). Accept if score improves or with probability p
    (to escape local minima).
    """
    if not blocks:
        return blocks

    rng = random.Random(int(time.time() * 1000) % 100000)
    
    # Build mapping: for each t-subset, which blocks cover it
    # blocks is a list of tuples
    n_blocks = len(blocks)
    t_subsets = list(all_tsubsets)
    n_tsubs = len(t_subsets)
    
    # Precompute: for each block, the set of t-subset indices it covers
    # And for each t-subset, the list of block indices that cover it
    tsub_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    block_covers = []  # block_covers[i] = set of t-subset indices
    tsub_coverers = [[] for _ in range(n_tsubs)]  # tsub_coverers[j] = list of block indices
    
    for i, block in enumerate(blocks):
        covered = set()
        for s in itertools.combinations(block, t):
            idx = tsub_to_idx.get(s)
            if idx is not None:
                covered.add(idx)
                tsub_coverers[idx].append(i)
        block_covers.append(covered)
    
    # Initial state: all blocks selected
    selected = [True] * n_blocks
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    for i in range(n_blocks):
        for idx in block_covers[i]:
            cover_count[idx] += 1
    
    num_uncovered = np.sum(cover_count == 0)
    num_selected = n_blocks
    best_score = num_selected + 10 * num_uncovered
    best_selected = selected[:]
    best_num_selected = num_selected
    
    lambda_penalty = 10  # weight for uncovered constraints
    
    # WalkSAT parameters
    p_flip = 0.3  # probability of making a non-improving move
    max_steps = 5000
    patience = 200  # steps without improvement before changing strategy
    
    current_score = num_selected + lambda_penalty * num_uncovered
    steps_since_improve = 0
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Choose a block to flip
        # With probability (1-p), pick a block that improves score
        # With probability p, pick randomly
        if rng.random() < p_flip:
            i = rng.randint(0, n_blocks - 1)
        else:
            # Try to find an improving flip
            # If selected: removing it reduces num_selected by 1, but may increase num_uncovered
            # If not selected: adding it increases num_selected by 1, but may decrease num_uncovered
            best_i = -1
            best_delta = 0  # we want negative delta (improvement)
            
            # Sample a few candidates for speed
            n_samples = min(20, n_blocks)
            sampled = rng.sample(range(n_blocks), n_samples)
            
            for i in sampled:
                if selected[i]:
                    # Removing block i
                    delta_selected = -1
                    # Count how many t-subsets become uncovered
                    newly_uncovered = 0
                    for idx in block_covers[i]:
                        if cover_count[idx] == 1:
                            newly_uncovered += 1
                    delta_uncovered = newly_uncovered
                    delta = delta_selected + lambda_penalty * delta_uncovered
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                else:
                    # Adding block i
                    delta_selected = 1
                    # Count how many t-subsets become covered
                    newly_covered = 0
                    for idx in block_covers[i]:
                        if cover_count[idx] == 0:
                            newly_covered += 1
                    delta_uncovered = -newly_covered
                    delta = delta_selected + lambda_penalty * delta_uncovered
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
            
            if best_i == -1:
                i = rng.randint(0, n_blocks - 1)
            else:
                i = best_i
        
        # Perform the flip
        if selected[i]:
            # Remove block i
            selected[i] = False
            num_selected -= 1
            old_uncovered = num_uncovered
            for idx in block_covers[i]:
                cover_count[idx] -= 1
                if cover_count[idx] == 0:
                    num_uncovered += 1
            new_uncovered = num_uncovered
        else:
            # Add block i
            selected[i] = True
            num_selected += 1
            old_uncovered = num_uncovered
            for idx in block_covers[i]:
                if cover_count[idx] == 0:
                    num_uncovered -= 1
                cover_count[idx] += 1
            new_uncovered = num_uncovered
        
        current_score = num_selected + lambda_penalty * num_uncovered
        
        if current_score < best_score:
            best_score = current_score
            best_selected = selected[:]
            best_num_selected = num_selected
            steps_since_improve = 0
        else:
            steps_since_improve += 1
        
        # Occasionally try a more aggressive move: remove a random selected block
        # and see if we can recover
        if step % 100 == 50 and num_uncovered == 0 and num_selected > best_num_selected:
            # Try removing a random block and see if we can fix coverage
            sel_indices = [i for i in range(n_blocks) if selected[i]]
            if sel_indices:
                i = rng.choice(sel_indices)
                # Temporarily remove
                selected[i] = False
                num_selected -= 1
                for idx in block_covers[i]:
                    cover_count[idx] -= 1
                    if cover_count[idx] == 0:
                        num_uncovered += 1
                
                # Try to add blocks to fix coverage
                for _ in range(5):
                    if num_uncovered == 0:
                        break
                    # Find a block that covers the most uncovered t-subsets
                    best_add = -1
                    best_gain = 0
                    for j in range(n_blocks):
                        if selected[j]:
                            continue
                        gain = 0
                        for idx in block_covers[j]:
                            if cover_count[idx] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_add = j
                    if best_add == -1 or best_gain == 0:
                        break
                    selected[best_add] = True
                    num_selected += 1
                    for idx in block_covers[best_add]:
                        if cover_count[idx] == 0:
                            num_uncovered -= 1
                        cover_count[idx] += 1
                
                if num_uncovered > 0:
                    # Revert: re-add the removed block
                    selected[i] = True
                    num_selected += 1
                    for idx in block_covers[i]:
                        if cover_count[idx] == 0:
                            num_uncovered -= 1
                        cover_count[idx] += 1
                else:
                    current_score = num_selected + lambda_penalty * num_uncovered
                    if current_score < best_score:
                        best_score = current_score
                        best_selected = selected[:]
                        best_num_selected = num_selected
                        steps_since_improve = 0
    
    # Reconstruct blocks from best_selected
    result = []
    for i in range(n_blocks):
        if best_selected[i]:
            result.append(blocks[i])
    
    # Final redundancy removal
    result = _remove_redundant(result, v, k, t, deadline)
    
    return result

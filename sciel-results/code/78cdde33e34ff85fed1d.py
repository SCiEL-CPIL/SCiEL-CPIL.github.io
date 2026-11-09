import itertools
import random
import time
import sys
import numpy as np

_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]

    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)

    block_triple_indices = [None] * n_blocks
    triple_to_blocks = [[] for _ in range(n_triples)]

    for i, block in enumerate(all_blocks):
        indices = []
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            indices.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices[i] = indices

    _CACHE[key] = (all_blocks, block_triple_indices, triple_to_blocks, n_triples)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, n_triples, seed):
    rng = random.Random(seed)
    uncovered_set = set(range(n_triples))
    blocks_used = []
    used_blocks_set = set()

    triple_constraints = [len(triple_to_blocks[i]) for i in range(n_triples)]

    while uncovered_set:
        # Pick the uncovered triple with the minimum number of covering blocks
        min_count = float('inf')
        candidates_for_tri = []
        for tri in uncovered_set:
            count = triple_constraints[tri]
            if count < min_count:
                min_count = count
                candidates_for_tri = [tri]
            elif count == min_count:
                candidates_for_tri.append(tri)

        if not candidates_for_tri:
            break

        target_tri = candidates_for_tri[rng.randint(0, len(candidates_for_tri) - 1)]
        candidate_blocks = triple_to_blocks[target_tri]
        if not candidate_blocks:
            break

        best_block = None
        max_gain = -1
        for b_idx in candidate_blocks:
            if b_idx in used_blocks_set:
                continue
            gain = 0
            for t_idx in block_triple_indices[b_idx]:
                if t_idx in uncovered_set:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = b_idx
            elif gain == max_gain and best_block is not None:
                if rng.random() < 0.5:
                    best_block = b_idx

        if best_block is None:
            break

        blocks_used.append(best_block)
        used_blocks_set.add(best_block)
        for t_idx in block_triple_indices[best_block]:
            if t_idx in uncovered_set:
                uncovered_set.remove(t_idx)

    return blocks_used

def _local_search(v, k, t, initial_blocks, block_triple_indices, triple_to_blocks, n_triples, deadline):
    current_blocks = list(initial_blocks)

    triple_counts = [0] * n_triples
    for b in current_blocks:
        for idx in block_triple_indices[b]:
            triple_counts[idx] += 1

    num_uncovered = sum(1 for c in triple_counts if c == 0)
    if num_uncovered != 0:
        return current_blocks

    best_blocks = list(current_blocks)
    best_len = len(current_blocks)

    rng = random.Random(42)
    temperature = 3.0
    cooling_rate = 0.995
    min_temperature = 0.05

    def _cleanup_redundant():
        nonlocal best_blocks, best_len
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks) - 1, -1, -1):
                b_out = current_blocks[i]
                redundant = True
                for idx in block_triple_indices[b_out]:
                    if triple_counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    del current_blocks[i]
                    for idx in block_triple_indices[b_out]:
                        triple_counts[idx] -= 1
                    changed = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                    break

    while time.time() < deadline:
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)

        # Phase 1: Delete all redundant blocks
        _cleanup_redundant()

        if not current_blocks:
            break

        # Phase 2: Try moves
        move_type = rng.random()

        if move_type < 0.6:
            # 1-exchange: remove one block, add one block
            pos = rng.randint(0, len(current_blocks) - 1)
            b_out = current_blocks[pos]

            uncovered_triples = []
            for idx in block_triple_indices[b_out]:
                if triple_counts[idx] == 1:
                    uncovered_triples.append(idx)

            if not uncovered_triples:
                del current_blocks[pos]
                for idx in block_triple_indices[b_out]:
                    triple_counts[idx] -= 1
                continue

            min_triple = min(uncovered_triples, key=lambda x: len(triple_to_blocks[x]))
            candidates = set(triple_to_blocks[min_triple])
            for tri in uncovered_triples:
                if tri == min_triple:
                    continue
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break

            current_set = set(current_blocks)
            candidates.difference_update(current_set)

            if candidates:
                candidate_list = list(candidates)
                rng.shuffle(candidate_list)
                max_tries = min(5, len(candidate_list))
                best_candidate = None
                best_gain = -1
                for b_in in candidate_list[:max_tries]:
                    gain = 0
                    for idx in block_triple_indices[b_in]:
                        if triple_counts[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = b_in

                if best_candidate is not None:
                    # Fast penalty: count of triples that become uniquely covered
                    # after the swap minus those that were uniquely covered before.
                    # A triple is uniquely covered if count == 1.
                    # After removing b_out and adding b_in:
                    # - triples in b_out but not b_in: count decreases by 1
                    # - triples in b_in but not b_out: count increases by 1
                    # - triples in both: count unchanged
                    # Penalty = (new unique count) - (old unique count)
                    # We only need to consider affected triples.
                    b_out_set = set(block_triple_indices[b_out])
                    b_in_set = set(block_triple_indices[best_candidate])
                    affected = b_out_set | b_in_set
                    old_unique = 0
                    new_unique = 0
                    for idx in affected:
                        c = triple_counts[idx]
                        if c == 1:
                            old_unique += 1
                        if idx in b_out_set:
                            c -= 1
                        if idx in b_in_set:
                            c += 1
                        if c == 1:
                            new_unique += 1
                    penalty = new_unique - old_unique
                    if penalty <= 0 or rng.random() < np.exp(-penalty / temperature):
                        del current_blocks[pos]
                        for idx in block_triple_indices[b_out]:
                            triple_counts[idx] -= 1
                        current_blocks.append(best_candidate)
                        for idx in block_triple_indices[best_candidate]:
                            triple_counts[idx] += 1

            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature

        elif move_type < 0.85:
            # 2-exchange: remove two blocks, add one block
            if len(current_blocks) < 2:
                continue
            pos1 = rng.randint(0, len(current_blocks) - 1)
            pos2 = rng.randint(0, len(current_blocks) - 1)
            if pos1 == pos2:
                continue

            b_out1 = current_blocks[pos1]
            b_out2 = current_blocks[pos2]

            temp_counts = list(triple_counts)
            for idx in block_triple_indices[b_out1]:
                temp_counts[idx] -= 1
            for idx in block_triple_indices[b_out2]:
                temp_counts[idx] -= 1

            uncovered = [i for i in range(n_triples) if temp_counts[i] == 0]
            if not uncovered:
                if pos1 > pos2:
                    pos1, pos2 = pos2, pos1
                    b_out1, b_out2 = b_out2, b_out1
                del current_blocks[pos1]
                for idx in block_triple_indices[b_out1]:
                    triple_counts[idx] -= 1
                del current_blocks[pos2 - 1]
                for idx in block_triple_indices[b_out2]:
                    triple_counts[idx] -= 1
                continue

            min_triple = min(uncovered, key=lambda x: len(triple_to_blocks[x]))
            candidates = set(triple_to_blocks[min_triple])
            for tri in uncovered:
                if tri == min_triple:
                    continue
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break

            current_set = set(current_blocks)
            candidates.difference_update(current_set)

            if candidates:
                candidate_list = list(candidates)
                rng.shuffle(candidate_list)
                max_tries = min(5, len(candidate_list))
                best_candidate = None
                best_gain = -1
                for b_in in candidate_list[:max_tries]:
                    gain = 0
                    for idx in block_triple_indices[b_in]:
                        if temp_counts[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = b_in

                if best_candidate is not None:
                    if pos1 > pos2:
                        pos1, pos2 = pos2, pos1
                        b_out1, b_out2 = b_out2, b_out1
                    del current_blocks[pos1]
                    for idx in block_triple_indices[b_out1]:
                        triple_counts[idx] -= 1
                    del current_blocks[pos2 - 1]
                    for idx in block_triple_indices[b_out2]:
                        triple_counts[idx] -= 1
                    current_blocks.append(best_candidate)
                    for idx in block_triple_indices[best_candidate]:
                        triple_counts[idx] += 1

            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature

        else:
            # 3-exchange: remove three blocks, add one block
            if len(current_blocks) < 3:
                continue
            pos1 = rng.randint(0, len(current_blocks) - 1)
            pos2 = rng.randint(0, len(current_blocks) - 1)
            pos3 = rng.randint(0, len(current_blocks) - 1)
            if pos1 == pos2 or pos2 == pos3 or pos1 == pos3:
                continue

            b_out1 = current_blocks[pos1]
            b_out2 = current_blocks[pos2]
            b_out3 = current_blocks[pos3]

            temp_counts = list(triple_counts)
            for idx in block_triple_indices[b_out1]:
                temp_counts[idx] -= 1
            for idx in block_triple_indices[b_out2]:
                temp_counts[idx] -= 1
            for idx in block_triple_indices[b_out3]:
                temp_counts[idx] -= 1

            uncovered = [i for i in range(n_triples) if temp_counts[i] == 0]
            if not uncovered:
                # All three are redundant; remove one at a time
                _cleanup_redundant()
                continue

            min_triple = min(uncovered, key=lambda x: len(triple_to_blocks[x]))
            candidates = set(triple_to_blocks[min_triple])
            for tri in uncovered:
                if tri == min_triple:
                    continue
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break

            current_set = set(current_blocks)
            candidates.difference_update(current_set)

            if candidates:
                candidate_list = list(candidates)
                rng.shuffle(candidate_list)
                max_tries = min(5, len(candidate_list))
                best_candidate = None
                best_gain = -1
                for b_in in candidate_list[:max_tries]:
                    gain = 0
                    for idx in block_triple_indices[b_in]:
                        if temp_counts[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = b_in

                if best_candidate is not None:
                    # Remove the three blocks, add the one
                    positions = sorted([pos1, pos2, pos3], reverse=True)
                    blocks_to_remove = [b_out1, b_out2, b_out3]
                    # Map positions to blocks correctly
                    pos_block_map = {pos1: b_out1, pos2: b_out2, pos3: b_out3}
                    for p in positions:
                        b = pos_block_map[p]
                        idx = current_blocks.index(b)
                        del current_blocks[idx]
                        for t_idx in block_triple_indices[b]:
                            triple_counts[t_idx] -= 1
                    current_blocks.append(best_candidate)
                    for idx in block_triple_indices[best_candidate]:
                        triple_counts[idx] += 1

            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature

    return best_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    all_blocks, block_triple_indices, triple_to_blocks, n_triples = _precompute(v, k, t)

    best_blocks_global = None
    best_len_global = float('inf')

    seed = 0
    while time.time() < deadline - 10:
        remaining = deadline - time.time()
        if remaining < 5:
            break

        ls_deadline = time.time() + min(15, remaining - 5)

        greedy_blocks = _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, n_triples, seed)
        seed += 1

        if not greedy_blocks:
            continue

        ls_blocks = _local_search(
            v, k, t,
            greedy_blocks,
            block_triple_indices,
            triple_to_blocks,
            n_triples,
            ls_deadline
        )

        if len(ls_blocks) < best_len_global:
            best_len_global = len(ls_blocks)
            best_blocks_global = ls_blocks

    if best_blocks_global is None:
        greedy_blocks = _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, n_triples, 999)
        best_blocks_global = greedy_blocks

    result = [list(all_blocks[idx]) for idx in best_blocks_global]
    return result

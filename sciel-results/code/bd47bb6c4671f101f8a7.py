import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}

    # Precompute block triple indices for all possible blocks
    # For v=24, k=7: C(24,7) = 346104 blocks, each with C(7,3)=35 triples
    # We'll use lazy caching

    block_triple_cache = {}

    def get_block_triples(block):
        key = tuple(sorted(block))
        if key in block_triple_cache:
            return block_triple_cache[key]
        result = [triple_to_idx[triple] for triple in itertools.combinations(key, t)]
        block_triple_cache[key] = result
        return result

    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)

        coverage_count = np.zeros(num_triples, dtype=np.int32)
        uncovered = set(range(num_triples))
        blocks = []

        while uncovered:
            if time.time() > deadline:
                break

            # Pick an uncovered triple
            target_triple_idx = random.choice(list(uncovered))
            target_triple = all_triples[target_triple_idx]

            remaining_points = [p for p in points if p not in target_triple]
            best_block = None
            best_gain = -1

            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_triple + combo))
                b_triples = get_block_triples(block)
                gain = sum(1 for idx in b_triples if idx in uncovered)

                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    if best_gain == 35:
                        break

            if best_block is None:
                break

            blocks.append(list(best_block))

            b_triples = get_block_triples(best_block)
            for idx in b_triples:
                coverage_count[idx] += 1
                if coverage_count[idx] == 1:
                    uncovered.discard(idx)

        return blocks

    def local_search(blocks):
        block_tuples = [tuple(sorted(b)) for b in blocks]
        coverage_count = np.zeros(num_triples, dtype=np.int32)
        for b in block_tuples:
            for t_idx in get_block_triples(b):
                coverage_count[t_idx] += 1

        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Deletion pass
            for i in range(len(block_tuples)):
                if time.time() > deadline:
                    break
                b = block_tuples[i]
                b_triples = get_block_triples(b)
                if all(coverage_count[t_idx] >= 2 for t_idx in b_triples):
                    for t_idx in b_triples:
                        coverage_count[t_idx] -= 1
                    del block_tuples[i]
                    improved = True
                    break

            if time.time() > deadline:
                break
            if improved:
                continue

            # Exchange pass
            for i in range(len(block_tuples)):
                if time.time() > deadline:
                    break
                b = block_tuples[i]
                b_triples = get_block_triples(b)
                b_set = set(b_triples)
                unique_triples = [t_idx for t_idx in b_triples if coverage_count[t_idx] == 1]

                if not unique_triples:
                    continue

                mandatory_points = set()
                for t_idx in unique_triples:
                    mandatory_points.update(all_triples[t_idx])

                if len(mandatory_points) > k:
                    continue

                available_points = [p for p in points if p not in mandatory_points]
                num_extra_needed = k - len(mandatory_points)

                if num_extra_needed > len(available_points):
                    continue

                best_new_block = None
                best_gain = 0

                candidates = list(itertools.combinations(available_points, num_extra_needed))
                if len(candidates) > 5000:
                    random.shuffle(candidates)
                    candidates = candidates[:5000]

                for combo in candidates:
                    new_block = tuple(sorted(list(mandatory_points) + list(combo)))
                    if new_block == b:
                        continue

                    new_triples = get_block_triples(new_block)
                    new_set = set(new_triples)

                    if not all(t_idx in new_set for t_idx in unique_triples):
                        continue

                    # Count how many currently-covered-only-by-this-block triples we'd lose
                    lost = sum(1 for t_idx in b_set - new_set if coverage_count[t_idx] == 1)
                    # Count how many new triples we'd cover
                    gained = sum(1 for t_idx in new_set - b_set if coverage_count[t_idx] == 0)

                    net_gain = gained - lost
                    if net_gain > best_gain:
                        best_gain = net_gain
                        best_new_block = new_block

                if best_new_block is not None:
                    for t_idx in b_triples:
                        coverage_count[t_idx] -= 1
                    for t_idx in get_block_triples(best_new_block):
                        coverage_count[t_idx] += 1
                    block_tuples[i] = best_new_block
                    improved = True
                    break

        return [list(b) for b in block_tuples]

    best_blocks = None
    best_count = float('inf')

    # Multiple restarts
    for i in range(50):
        if time.time() > deadline - 5:
            break

        blocks = greedy_construct(random_seed=i)
        blocks = local_search(blocks)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

        if best_count <= 70:
            break

    # Final polish on best solution
    if best_blocks is not None and time.time() < deadline - 10:
        improved_blocks = local_search(best_blocks)
        if len(improved_blocks) < best_count:
            best_count = len(improved_blocks)
            best_blocks = improved_blocks

    if best_blocks is None:
        return []

    return best_blocks

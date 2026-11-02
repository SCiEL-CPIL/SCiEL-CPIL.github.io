"""Improved greedy covering for C(19,8,5) with deadline guard and better heuristics."""
# EVOLVE-BLOCK-START
import itertools
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    all_t_set = set(all_t_subsets)
    uncovered = set(all_t_set)
    blocks = []

    if not uncovered:
        return blocks

    # For speed, maintain a mapping from point to count of uncovered t-subsets containing it
    # But we need dynamic updates, so we'll use a different approach:
    # Track for each point, how many uncovered t-subsets contain it (approximate via sampling or exact)

    # Better approach: use a priority-based greedy with better point selection
    # Key insight: when building a block, we want to maximize coverage of uncovered t-subsets

    # Precompute: for each pair of points, etc. - too expensive
    # Instead, use a randomized greedy with multiple restarts if time permits

    def greedy_cover(uncovered, blocks, time_limit):
        """Greedy covering algorithm with improved heurstic for point selection."""
        while uncovered and time.time() < time_limit:
            # Pick the uncovered t-subset with the most 'pressure' (heuristic: first one)
            # Better: pick a random uncovered subset to diversify
            if len(uncovered) > 1000:
                # Use a random sample to find a good starting point
                sample = list(uncovered)
                import random
                start_subset = sample[random.randint(0, len(sample)-1)]
            else:
                start_subset = min(uncovered)

            block = list(start_subset)
            block_set = set(block)

            # Build the block by adding points that cover the most uncovered t-subsets
            while len(block) < k and time.time() < time_limit:
                best_point = -1
                best_gain = -1

                # For each candidate point, compute how many uncovered t-subsets it would help cover
                # A t-subset S is covered by adding point x to current block B if:
                #   S ⊆ B ∪ {x} and x ∈ S
                # Equivalently: S \ B ⊆ {x} and x ∈ S, i.e., S \ B = {x} or S \ B = ∅ (but S not yet covered)
                # More precisely: S is covered by B∪{x} iff S ∩ (B∪{x}) = S, i.e., S ⊆ B∪{x}
                # Since S is a t-subset and |B| < k, S ⊆ B∪{x} means S \ B ⊆ {x}, so |S \ B| ≤ 1
                # and if |S \ B| = 1 then the missing element must be x.

                # So gain(x) = number of uncovered S such that S ⊆ B ∪ {x} and x ∉ B
                # = number of uncovered S where S \ B = {x} (since x ∉ B)
                # = number of uncovered S where S contains x and S \ {x} ⊆ B

                for x in range(v):
                    if x in block_set:
                        continue

                    # Count uncovered t-subsets S where x ∈ S and S \ {x} ⊆ B
                    gain = 0
                    # S \ {x} is a (t-1)-subset of B, and S = (S\{x}) ∪ {x}
                    for rest in itertools.combinations(block, t-1):
                        s = tuple(sorted(rest + (x,)))
                        if s in uncovered:
                            gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_point = x

                if best_point == -1 or best_gain <= 0:
                    # No point adds coverage; add a random point to complete the block
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        best_point = available[0]
                    else:
                        break

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            block_tuple = tuple(block)
            blocks.append(block)

            # Remove all t-subsets covered by this block
            for s in itertools.combinations(block, t):
                uncovered.discard(s)

            # Check deadline
            if time.time() > deadline:
                break

        return blocks, uncovered

    # Try greedy with a few restarts using different random seeds
    import random
    best_blocks = None
    best_uncovered_count = len(uncovered)

    # First attempt: standard greedy
    blocks, uncovered = greedy_cover(uncovered, [], deadline)
    if not uncovered:
        return blocks

    # If not complete, try to improve by checking if we can do better with different strategies
    # For now, return what we have if it's valid, otherwise we need a complete cover

    # Verify completeness
    if uncovered:
        # We need a complete cover. Try a more aggressive approach:
        # Continue with simpler greedy that always picks the most constrained uncovered subset
        remaining_uncovered = set(uncovered)
        while remaining_uncovered and time.time() < deadline:
            # Pick the uncovered subset with the fewest candidate points that could extend existing blocks
            # Simpler: just pick the first uncovered subset and build a block around it
            start_subset = min(remaining_uncovered)
            block = list(start_subset)
            block_set = set(block)

            while len(block) < k and time.time() < deadline:
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        s = tuple(sorted(rest + (x,)))
                        if s in remaining_uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x

                if best_point == -1:
                    available = [x for x in range(v) if x not in block_set]
                    best_point = available[0] if available else -1

                if best_point == -1:
                    break

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)
            for s in itertools.combinations(block, t):
                remaining_uncovered.discard(s)

        if remaining_uncovered:
            # Last resort: add blocks for each remaining uncovered subset
            for s in remaining_uncovered:
                if time.time() > deadline:
                    break
                block = list(s)
                block_set = set(block)
                while len(block) < k:
                    available = [x for x in range(v) if x not in block_set]
                    if not available:
                        break
                    block.append(available[0])
                    block_set.add(available[0])
                block.sort()
                blocks.append(block)

    # Remove duplicates
    seen = set()
    unique_blocks = []
    for b in blocks:
        bt = tuple(b)
        if bt not in seen:
            seen.add(bt)
            unique_blocks.append(b)

    return unique_blocks

# EVOLVE-BLOCK-END

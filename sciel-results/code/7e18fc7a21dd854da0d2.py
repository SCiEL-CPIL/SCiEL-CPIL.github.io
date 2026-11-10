"""Covering design C(19,7,4) via greedy + local search with simulated-annealing-style removal."""
import itertools
import random
import time

def construct(v, k, t):
    deadline = time.time() + 110  # safety margin before 120s kill
    rng = random.Random(42)

    # Precompute all t-subsets as frozensets for fast membership
    all_tsubsets = list(itertools.combinations(range(v), t))
    all_tset = set(map(frozenset, all_tsubsets))

    def greedy_construct():
        """Greedy: repeatedly pick the block that covers the most uncovered t-subsets."""
        uncovered = set(all_tset)
        blocks = []
        while uncovered:
            best_block = None
            best_gain = -1
            # Sample candidate blocks to speed up; use all if small
            # For v=19,k=7, C(19,7)=50388, so we can be smart
            # Strategy: build block greedily by adding point with max marginal gain
            # Start from the uncovered t-subset with highest "rarity"
            # Simpler: try several starting points
            best_block_covered = 0
            # Try multiple seeds for the block
            candidates = []
            for seed_subset in list(uncovered)[:50]:
                block = list(seed_subset)
                while len(block) < k:
                    best_x = -1
                    best_g = -1
                    for x in range(v):
                        if x in block:
                            continue
                        new_block = block + [x]
                        # Count how many uncovered t-subsets are covered by new_block that weren't by block
                        # A t-subset is newly covered if it contains x and the other t-1 elements are in block
                        g = 0
                        for rest in itertools.combinations(block, t - 1):
                            ts = frozenset(rest + (x,))
                            if ts in uncovered:
                                g += 1
                        if g > best_g:
                            best_g = g
                            best_x = x
                    if best_x == -1:
                        break
                    block.append(best_x)
                if len(block) < k:
                    continue
                block.sort()
                # Count coverage
                covered = 0
                for ts in itertools.combinations(block, t):
                    if frozenset(ts) in uncovered:
                        covered += 1
                if covered > best_block_covered:
                    best_block_covered = covered
                    best_block = block
            if best_block is None:
                # Fallback: random block
                best_block = sorted(rng.sample(range(v), k))
                best_block_covered = sum(1 for ts in itertools.combinations(best_block, t) if frozenset(ts) in uncovered)
            blocks.append(best_block)
            for ts in itertools.combinations(best_block, t):
                uncovered.discard(frozenset(ts))
        return blocks, len(blocks)

    # Initial greedy construction
    blocks, n_blocks = greedy_construct()
    best_blocks = list(blocks)
    best_n = n_blocks

    # Local search: try to remove redundant blocks
    def compute_block_unique(blocks_list):
        """For each block, count how many t-subsets it uniquely covers."""
        # Count how many blocks cover each t-subset
        from collections import defaultdict
        ts_count = defaultdict(int)
        for b in blocks_list:
            for ts in itertools.combinations(b, t):
                ts_count[frozenset(ts)] += 1
        # A block is redundant if all its t-subsets are covered by >= 2 blocks
        unique_counts = []
        for b in blocks_list:
            unique = 0
            for ts in itertools.combinations(b, t):
                if ts_count[frozenset(ts)] == 1:
                    unique += 1
            unique_counts.append(unique)
        return unique_counts

    def try_remove_redundant(blocks_list):
        """Iteratively remove blocks that have no unique t-subsets."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            unique_counts = compute_block_unique(blocks_list)
            for i in range(len(blocks_list)):
                if unique_counts[i] == 0:
                    blocks_list.pop(i)
                    changed = True
                    break
        return blocks_list

    def local_search(blocks_list):
        """Try to improve by removing redundant blocks and swapping."""
        current = list(blocks_list)
        current = try_remove_redundant(current)
        best = current
        best_n = len(current)
        
        start_time = time.time()
        while time.time() - start_time < 30 and time.time() < deadline:
            # Try random removal + repair
            if len(current) > 10:
                idx = rng.randint(0, len(current) - 1)
                removed = current[idx]
                current.pop(idx)
                
                # Check if still valid
                if not is_valid(current, v, k, t, all_tset):
                    # Try to repair by adding a block that covers the most newly-uncovered t-subsets
                    # Find uncovered t-subsets
                    covered = set()
                    for b in current:
                        for ts in itertools.combinations(b, t):
                            covered.add(frozenset(ts))
                    uncovered = all_tset - covered
                    if uncovered:
                        # Add greedy block for uncovered
                        best_block = None
                        best_gain = -1
                        for _ in range(20):
                            seed = rng.choice(list(uncovered))
                            block = list(seed)
                            while len(block) < k:
                                best_x = -1
                                best_g = -1
                                for x in range(v):
                                    if x in block:
                                        continue
                                    g = sum(1 for rest in itertools.combinations(block, t-1) 
                                            if frozenset(rest + (x,)) in uncovered)
                                    if g > best_g:
                                        best_g = g
                                        best_x = x
                                if best_x == -1:
                                    break
                                block.append(best_x)
                            if len(block) == k:
                                block.sort()
                                gain = sum(1 for ts in itertools.combinations(block, t) if frozenset(ts) in uncovered)
                                if gain > best_gain:
                                    best_gain = gain
                                    best_block = block
                        if best_block:
                            current.append(best_block)
                        else:
                            current.append(removed)
                            continue
                    # Now try to remove more
                    current = try_remove_redundant(current)
                    if len(current) < best_n:
                        best = current
                        best_n = len(current)
                else:
                    if len(current) < best_n:
                        best = current
                        best_n = len(current)
                    current = try_remove_redundant(current)
                    if len(current) < best_n:
                        best = current
                        best_n = len(current)
            else:
                break
        
        return best, best_n

    def is_valid(blocks_list, v, k, t, all_tset):
        covered = set()
        for b in blocks_list:
            for ts in itertools.combinations(b, t):
                covered.add(frozenset(ts))
        return covered == all_tset

    # Run local search
    improved_blocks, improved_n = local_search(best_blocks)
    if improved_n < best_n:
        best_blocks = improved_blocks
        best_n = improved_n

    # Final check and cleanup
    best_blocks = try_remove_redundant(best_blocks)
    
    # Ensure validity
    if not is_valid(best_blocks, v, k, t, all_tset):
        # Fallback to original greedy
        best_blocks = list(blocks)
    
    return [sorted(b) for b in best_blocks]

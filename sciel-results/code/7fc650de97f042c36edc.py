import itertools
import time
import random

_DEADLINE_S = 120.0
_MARGIN_S = 10.0


def construct(v, k, t):
    start_time = time.time()
    rng = random.Random(42)

    def remaining():
        return max(0.0, _DEADLINE_S - _MARGIN_S - (time.time() - start_time))

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    blocks = []

    def block_covers(block):
        return set(itertools.combinations(block, t))

    def gain_for_point(block, x, target_set):
        """
        Calculate gain of adding x to block, specifically targeting t-subsets in target_set.
        """
        cnt = 0
        # We need to check if (rest + x) is in target_set
        # rest is a combination of t-1 elements from block
        for rest in itertools.combinations(block, t - 1):
            ts = tuple(sorted(rest + (x,)))
            if ts in target_set:
                cnt += 1
        return cnt

    # --- Greedy construction ---
    while uncovered and remaining() > 1.0:
        # Pick a random uncovered t-subset to diversify slightly, or min for determinism?
        # Min is deterministic and often good for baseline. Let's stick to min for stability.
        target = min(uncovered)
        block = list(target)
        block.sort()
        
        # Build the block greedily
        while len(block) < k and remaining() > 0.5:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            best_x = None
            best_gain = -1
            for x in candidates:
                g = gain_for_point(block, x, uncovered)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None or best_gain == 0:
                # Pick arbitrary if no gain
                best_x = candidates[0]
            block.append(best_x)
        
        block.sort()
        block = list(set(block))
        if len(block) < k:
            used = set(block)
            rem = [x for x in range(v) if x not in used]
            rem.sort()
            block.extend(rem[:k - len(block)])
            block.sort()
            
        if len(set(block)) != k:
            continue
            
        covered = block_covers(block)
        if not covered:
            continue
        uncovered -= covered
        blocks.append(block)

    # --- Local Improvement Phase ---
    # We will iterate:
    # 1. Remove redundant blocks.
    # 2. Try to replace blocks with better ones (targeting unique coverage).
    
    if remaining() > 2.0:
        # Helper to check if a block is redundant
        def is_redundant(block_idx, current_blocks):
            b = current_blocks[block_idx]
            cov_b = block_covers(b)
            for ts in cov_b:
                # Check if ts is covered by any other block
                only_this = True
                for j in range(len(current_blocks)):
                    if i == j:
                        continue
                    if ts in block_covers(current_blocks[j]):
                        only_this = False
                        break
                if only_this:
                    return False
            return True

        # To speed up redundancy check, maintain a coverage count map?
        # For v=19, k=7, t=4, C(19,4)=3876. Blocks ~150.
        # Checking redundancy is O(N * C(k,t) * N) which is 150 * 35 * 150 ~ 787k ops.
        # This is fast enough in Python if optimized.
        
        improved = True
        while improved and remaining() > 2.0:
            improved = False
            
            # Step 1: Remove redundant blocks
            new_blocks = []
            for i in range(len(blocks)):
                if is_redundant(i, blocks):
                    improved = True
                else:
                    new_blocks.append(blocks[i])
            blocks = new_blocks
            
            # Step 2: Try to improve individual blocks
            # For each block, find its unique coverage.
            # If unique coverage is non-empty, try to construct a new block that covers it better.
            n = len(blocks)
            for i in range(n):
                if remaining() < 2.0:
                    break
                b = blocks[i]
                cov_b = block_covers(b)
                
                # Find unique t-subsets covered by b
                unique_ts = set()
                for ts in cov_b:
                    only_this = True
                    for j in range(n):
                        if i == j:
                            continue
                        if ts in block_covers(blocks[j]):
                            only_this = False
                            break
                    if only_this:
                        unique_ts.add(ts)
                
                if not unique_ts:
                    continue # Should have been removed in Step 1, but safety check
                
                # Try to build a new block that covers unique_ts
                # Strategy: Start with a random ts from unique_ts, greedily add points to cover more unique_ts
                # If the new block is different from b and covers all unique_ts, replace.
                
                # To avoid infinite loops, we only replace if the new block is "better".
                # Better can mean: same size, but we just want to change the structure.
                # Actually, if we replace b with b_new that covers unique_ts, the count stays same.
                # But it might make other blocks redundant.
                
                # Let's try to construct a block covering unique_ts.
                # We need a block B_new such that unique_ts subset of C(B_new, t).
                
                # Heuristic:
                # 1. Pick a ts from unique_ts.
                # 2. Start B_new with ts.
                # 3. Add points to B_new to maximize coverage of remaining unique_ts.
                
                if len(unique_ts) == 0:
                    continue
                    
                target_ts = next(iter(unique_ts))
                new_block = list(target_ts)
                new_block.sort()
                
                while len(new_block) < k:
                    candidates = [x for x in range(v) if x not in new_block]
                    if not candidates:
                        break
                    best_x = None
                    best_gain = -1
                    for x in candidates:
                        g = gain_for_point(new_block, x, unique_ts)
                        if g > best_gain:
                            best_gain = g
                            best_x = x
                    if best_x is None or best_gain == 0:
                        # If we can't cover more unique_ts, pick random to fill
                        best_x = rng.choice(candidates)
                    new_block.append(best_x)
                
                new_block.sort()
                new_block = list(set(new_block))
                if len(new_block) < k:
                    used = set(new_block)
                    rem = [x for x in range(v) if x not in used]
                    rng.shuffle(rem)
                    new_block.extend(rem[:k - len(new_block)])
                    new_block.sort()
                
                # Verify new_block covers all unique_ts
                cov_new = block_covers(new_block)
                if unique_ts.issubset(cov_new):
                    # If new_block is different from old block, replace it
                    if new_block != b:
                        blocks[i] = new_block
                        improved = True

    return blocks

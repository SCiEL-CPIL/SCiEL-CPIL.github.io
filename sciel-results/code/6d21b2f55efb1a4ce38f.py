import itertools
import time
import random

_DEADLINE_S = 120.0
_MARGIN_S = 5.0


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
        cnt = 0
        for rest in itertools.combinations(block, t - 1):
            ts = tuple(sorted(rest + (x,)))
            if ts in target_set:
                cnt += 1
        return cnt

    # --- Greedy construction ---
    while uncovered and remaining() > 1.0:
        target = min(uncovered)
        block = list(target)
        block.sort()
        
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

    # --- Local Improvement Phase using Coverage Counts ---
    if remaining() > 2.0:
        # Helper to update coverage counts
        def update_counts(block, sign, counts):
            for ts in block_covers(block):
                counts[ts] += sign
                if counts[ts] < 0:
                    counts[ts] = 0 # Safety, shouldn't happen if logic is correct

        improved = True
        while improved and remaining() > 2.0:
            improved = False
            
            # Calculate current coverage counts
            cover_count = {}
            for b in blocks:
                for ts in block_covers(b):
                    cover_count[ts] = cover_count.get(ts, 0) + 1

            # Step 1: Remove redundant blocks
            new_blocks = []
            for b in blocks:
                redundant = True
                for ts in block_covers(b):
                    if cover_count.get(ts, 0) == 1:
                        redundant = False
                        break
                if not redundant:
                    new_blocks.append(b)
                else:
                    improved = True
                    
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                continue # Restart loop to re-evaluate with new set

            # Step 2: Try to replace blocks with better ones
            # A block is "essential" if it covers at least one t-subset with count 1.
            # If we replace it, we must cover those unique subsets.
            n = len(blocks)
            for i in range(n):
                if remaining() < 2.0:
                    break
                b = blocks[i]
                cov_b = block_covers(b)
                
                unique_ts = set()
                for ts in cov_b:
                    if cover_count.get(ts, 0) == 1:
                        unique_ts.add(ts)
                
                if not unique_ts:
                    continue

                # Try to build a new block that covers unique_ts
                # Strategy: Start with a random ts from unique_ts, greedily add points
                # to cover more unique_ts.
                
                target_ts = next(iter(unique_ts))
                new_block = list(target_ts)
                new_block.sort()
                
                while len(new_block) < k and remaining() > 0.5:
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
                if unique_ts.issubset(cov_new) and new_block != b:
                    # Potential swap. Check if this swap allows removal of other blocks?
                    # Simple heuristic: Just swap if it looks valid. 
                    # The next iteration of the while loop will check for new redundancies.
                    blocks[i] = new_block
                    improved = True
                    break # Restart inner loop to re-evaluate counts

    return blocks

import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_subsets = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_count = float('inf')
    
    def remove_redundant(blocks):
        """Remove blocks that do not cover any unique t-subset."""
        blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            # Build a count of how many times each t-subset is covered
            # Or simply, for each block, check if all its t-subsets are covered by others.
            # Efficient way:
            # 1. Count occurrences of each t-subset in all blocks.
            # 2. A block is redundant if for every t-subset S in it, count[S] > 1.
            
            for i in range(len(blocks)):
                block = blocks[i]
                is_redundant = True
                for sub in itertools.combinations(block, t):
                    # Check if this sub is covered by any other block
                    # This is O(N_blocks * C(k,t))
                    # We can precompute counts.
                    pass
            
            # Better: Precompute counts once per iteration of the while loop
            # But the set of blocks changes.
            
            # Let's do:
            # 1. Count coverage of each t-subset by all blocks.
            counts = {}
            for block in blocks:
                for sub in itertools.combinations(block, t):
                    counts[sub] = counts.get(sub, 0) + 1
            
            # 2. Identify redundant blocks
            redundant_indices = []
            for i, block in enumerate(blocks):
                redundant = True
                for sub in itertools.combinations(block, t):
                    if counts[sub] == 1:
                        # This sub is ONLY covered by this block
                        redundant = False
                        break
                if redundant:
                    redundant_indices.append(i)
            
            # 3. Remove them
            if redundant_indices:
                changed = True
                # Remove from end to start to keep indices valid
                for idx in sorted(redundant_indices, reverse=True):
                    blocks.pop(idx)
                    
        return blocks

    while time.time() < deadline:
        uncovered_set = set(all_subsets)
        uncovered_list = list(uncovered_set)
        blocks = []
        
        while uncovered_list:
            if time.time() > deadline:
                break
            
            # Pick a random target
            idx = random.randint(0, len(uncovered_list) - 1)
            target_subset = uncovered_list[idx]
            
            # If target is stale (covered by previous blocks in this run), skip
            if target_subset not in uncovered_set:
                # Remove from list
                uncovered_list.pop(idx)
                continue
                
            block = list(target_subset)
            
            # Build block
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                if not candidates:
                    break
                
                gains = []
                for x in candidates:
                    count = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        if new_sub in uncovered_set:
                            count += 1
                    gains.append(count)
                
                max_gain = max(gains)
                if max_gain == 0:
                    x = random.choice(candidates)
                else:
                    best_candidates = [x for x, g in zip(candidates, gains) if g == max_gain]
                    x = random.choice(best_candidates)
                
                block.append(x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                uncovered_set.discard(sub)
            
            # Filter list occasionally or every step?
            # Every step is safe.
            uncovered_list = [sub for sub in uncovered_list if sub in uncovered_set]
        
        if not uncovered_set:
            # Valid covering found
            # Apply redundancy removal
            blocks = remove_redundant(blocks)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
            
            # If we have a very good result, we might want to break early?
            # No, keep searching.

    if best_blocks is None:
        # Fallback
        uncovered_set = set(all_subsets)
        blocks = []
        while uncovered_set:
            target_subset = min(uncovered_set)
            block = list(target_subset)
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                gains = []
                for x in candidates:
                    count = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        if new_sub in uncovered_set:
                            count += 1
                    gains.append(count)
                max_gain = max(gains)
                x = candidates[gains.index(max_gain)]
                block.append(x)
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered_set.discard(sub)
        blocks = remove_redundant(blocks)
        best_blocks = blocks

    return best_blocks

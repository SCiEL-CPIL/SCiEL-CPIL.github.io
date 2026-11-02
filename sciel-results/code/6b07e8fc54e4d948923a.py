"""Optimized randomized local search for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for safety

    # Precompute all t-subsets and map them to IDs for fast indexing
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_id = {fs: i for i, fs in enumerate(all_tsubs)}
    total_tsubs = len(all_tsubs)
    
    # Helper to get t-subset IDs for a block
    def get_block_tsubs(block):
        return [tsub_to_id[frozenset(c)] for c in itertools.combinations(block, t)]

    def greedy_cover():
        """Construct an initial covering using a randomized greedy heuristic."""
        best_blocks = None
        best_count = float('inf')
        
        # Run a few iterations with different random seeds to find a good start
        for _ in range(10):
            cur_uncovered = [1] * total_tsubs
            cur_blocks = []
            remaining_uncovered = set(range(total_tsubs))
            
            while remaining_uncovered:
                # Pick a random uncovered t-subset to seed the new block
                seed_id = rng.choice(list(remaining_uncovered))
                seed_tsub = all_tsubs[seed_id]
                
                # Start block with this t-subset
                block = list(seed_tsub)
                
                # Extend block to size k
                # We want to add points that cover the most *currently uncovered* t-subsets.
                candidates = [x for x in range(v) if x not in block]
                
                while len(block) < k:
                    best_x = None
                    best_gain = -1
                    
                    # To speed up, we don't need to check all candidates if we find a high gain
                    # But for k=7, v=24, candidates are ~17. C(17,1) is small.
                    rng.shuffle(candidates)
                    
                    for x in candidates:
                        gain = 0
                        # Check coverage of t-subsets formed by (t-1) from block + x
                        for rest in itertools.combinations(block, t-1):
                            tsub = frozenset(list(rest) + [x])
                            tsub_id = tsub_to_id[tsub]
                            if cur_uncovered[tsub_id] > 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            
                    if best_x is None:
                        best_x = rng.choice(candidates)
                        
                    block.append(best_x)
                    candidates.remove(best_x)
                    
                block.sort()
                cur_blocks.append(block)
                
                # Update uncovered status
                block_tsubs = get_block_tsubs(block)
                for tid in block_tsubs:
                    if cur_uncovered[tid] > 0:
                        cur_uncovered[tid] = 0
                        remaining_uncovered.discard(tid)
                        
            if len(cur_blocks) < best_count:
                best_count = len(cur_blocks)
                best_blocks = cur_blocks
                
        return best_blocks

    # Initial covering
    blocks = greedy_cover()
    if not blocks:
        return []
        
    # Initialize coverage counts
    cover_count = [0] * total_tsubs
    for b in blocks:
        for tid in get_block_tsubs(b):
            cover_count[tid] += 1
            
    num_uncovered = sum(1 for c in cover_count if c == 0)
    
    best_blocks = [b[:] for b in blocks]
    best_len = len(blocks)
    
    # Local Search
    while time.time() - start_time < time_limit:
        # 1. Try to remove a block
        # A block is removable if it doesn't cover any t-subset uniquely (cover_count == 1)
        removable_idx = -1
        for i, b in enumerate(blocks):
            safe = True
            for tid in get_block_tsubs(b):
                if cover_count[tid] == 1:
                    safe = False
                    break
            if safe:
                removable_idx = i
                break
                
        if removable_idx != -1:
            b_to_remove = blocks[removable_idx]
            # Update counts
            for tid in get_block_tsubs(b_to_remove):
                cover_count[tid] -= 1
            blocks.pop(removable_idx)
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = [b[:] for b in blocks]
            continue
            
        # 2. Try to swap a block
        # If no block is removable, we try to replace a block to create redundancy or cover new areas.
        # We pick a random block to replace.
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            new_block = tuple(sorted(rng.sample(range(v), k)))
            
            if new_block == tuple(old_block):
                continue
                
            # Check validity of swap:
            # Removing old_block might uncover t-subsets that are only covered by it.
            # Adding new_block must cover those, or they must be covered by others.
            
            old_tsubs = get_block_tsubs(old_block)
            new_tsubs = get_block_tsubs(new_block)
            new_tsub_set = set(new_tsubs)
            
            valid_swap = True
            for tid in old_tsubs:
                # If this t-subset is only covered by old_block (count 1)
                # AND it is not covered by new_block, then the swap is invalid.
                if cover_count[tid] == 1 and tid not in new_tsub_set:
                    valid_swap = False
                    break
            
            if valid_swap:
                # Perform swap
                # Remove old
                for tid in old_tsubs:
                    cover_count[tid] -= 1
                # Add new
                for tid in new_tsubs:
                    cover_count[tid] += 1
                    
                blocks[i] = list(new_block)
                continue

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END

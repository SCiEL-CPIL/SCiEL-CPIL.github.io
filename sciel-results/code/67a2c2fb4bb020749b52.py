import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave buffer
    
    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(range(v), t))
    tsubs_set = set(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Precompute masks for all blocks?
    # C(24, 7) = 346,104 blocks.
    # This is a lot to precompute if we do it naively.
    # Let's generate them on the fly or use a smarter structure.
    
    # Strategy:
    # 1. Greedy with "random uncovered t-subset" pivot.
    # 2. ILS with delete-replace local search.
    
    # To make local search fast, we need to quickly check if a block is redundant
    # and to find replacements.
    # We will maintain a count of how many blocks cover each t-subset.
    
    def get_block_mask(block):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            mask |= (1 << idx)
        return mask

    # Precompute all blocks and their masks?
    # 346k blocks * C(7,3)=35 tsubs * bit op.
    # 346k * 35 = 12M operations. Might take 5-10s.
    # Let's try it. It enables O(1) mask lookups.
    
    all_blocks = []
    block_masks = []
    block_to_idx = {}
    
    for block in itertools.combinations(range(v), k):
        b = list(block)
        b.sort()
        mask = get_block_mask(b)
        all_blocks.append(b)
        block_masks.append(mask)
        block_to_idx[tuple(b)] = len(all_blocks) - 1
        
    # Greedy Cover
    def greedy_cover(seed):
        r = random.Random(seed)
        uncovered_mask = (1 << num_tsubs) - 1
        blocks = []
        
        while uncovered_mask:
            # Pick a random uncovered t-subset
            # Find an index where bit is set
            # Using bit tricks to find a set bit
            # Or just pick a random index and check?
            # num_tsubs = 2024. 2024 bits.
            # We can use:
            # import random
            # idx = r.randrange(num_tsubs)
            # while not (uncovered_mask >> idx) & 1:
            #    idx = r.randrange(num_tsubs)
            # This is fast enough.
            
            idx = r.randrange(num_tsubs)
            while not (uncovered_mask >> idx) & 1:
                idx = r.randrange(num_tsubs)
                
            target_ts = all_tsubs[idx]
            
            # Find all blocks containing target_ts
            # We need a mapping: t-subset -> list of block indices containing it
            # Let's precompute this: tsub_to_blocks
            # For each t-subset, which blocks contain it?
            # C(21, 4) = 5985 blocks per t-subset.
            # 2024 * 5985 = 12M entries. Memory might be high but okay?
            # 12M integers ~ 48MB. Acceptable.
            
            # Let's precompute tsub_to_blocks
            # Actually, we can do this on the fly if we have block_to_idx?
            # No, we need reverse index.
            
            # Let's assume we have tsub_to_blocks precomputed outside?
            # Or compute inside?
            # Computing inside is O(num_blocks) which is 346k.
            # 76 steps * 346k = 26M checks. Fast enough.
            
            best_block_idx = -1
            best_gain = -1
            best_block = None
            
            # Iterate all blocks? No, too slow if we do it 76 times?
            # 26M is okay.
            # But we can filter: only blocks containing target_ts.
            # How to find them fast?
            # We can precompute `points_in_block`? No.
            # Let's just iterate all blocks and check if they contain target_ts.
            # 346k * 76 = 26M.
            # Check: `set(target_ts).issubset(set(block))`?
            # Or use bitmask for points?
            # Let's use point bitmask for blocks too.
            
            # Re-precompute with point masks
            # point_mask[block_idx] = bitmask of points in block
            
            # Let's re-do precomputation to include point masks
            # And build tsub_to_blocks list
            
            # Wait, I'm inside the function. I should precompute once before the loop.
            
            # Let's restructure.
            
            # For now, let's just use the "random completion" greedy which was in the original code
            # but improve the local search.
            # The original greedy was: pick random uncovered ts, complete block greedily.
            # That is O(k * v * t) per block. Very fast.
            # The bottleneck is local search.
            
            # Let's stick to the fast greedy, but use bitsets for coverage tracking.
            
            # I will rewrite the whole thing to use bitsets for speed in local search.
            
            break # Break to restart with better design
            
        return blocks

    # Let's implement the full solution with precomputation.
    
    # Precomputation
    tsub_to_blocks = defaultdict(list)
    point_masks = []
    
    for i, b in enumerate(all_blocks):
        pm = 0
        for p in b:
            pm |= (1 << p)
        point_masks.append(pm)
        
        for ts in itertools.combinations(b, t):
            tsub_to_blocks[ts].append(i)
            
    # Greedy
    def greedy_cover(seed):
        r = random.Random(seed)
        uncovered_mask = (1 << num_tsubs) - 1
        blocks = []
        
        while uncovered_mask:
            # Pick random uncovered t-subset
            idx = r.randrange(num_tsubs)
            while not (uncovered_mask >> idx) & 1:
                idx = r.randrange(num_tsubs)
                
            target_ts = all_tsubs[idx]
            
            # Candidates: blocks containing target_ts
            candidates = tsub_to_blocks[target_ts]
            
            best_idx = -1
            best_gain = -1
            best_block = None
            
            for b_idx in candidates:
                gain = bin(block_masks[b_idx] & uncovered_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = b_idx
                    best_block = all_blocks[b_idx]
                
            if best_idx != -1:
                blocks.append(best_block)
                uncovered_mask &= ~block_masks[best_idx]
            else:
                # Should not happen if covering is possible
                break
                
        return blocks

    def local_search(blocks, seed):
        r = random.Random(seed)
        blocks = [list(b) for b in blocks]
        
        # Map block tuple to index in current list
        # Maintain coverage counts
        cov_counts = defaultdict(int)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cov_counts[ts] += 1
                
        improved = True
        while improved:
            improved = False
            
            # 1. Remove redundant
            for i in range(len(blocks)):
                b = blocks[i]
                redundant = True
                for ts in itertools.combinations(b, t):
                    if cov_counts[ts] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for ts in itertools.combinations(b, t):
                        cov_counts[ts] -= 1
                    del blocks[i]
                    improved = True
                    break # Restart loop
            
            if improved:
                continue
                
            # 2. Try to replace a block
            # Pick a random block
            if blocks:
                i = r.randrange(len(blocks))
                b = blocks[i]
                
                # Find unique t-subsets of b
                unique_ts = [ts for ts in itertools.combinations(b, t) if cov_counts[ts] == 1]
                
                if not unique_ts:
                    # Should have been removed
                    continue
                    
                # Try to find a replacement block b' that covers all unique_ts
                # and is not b
                # Candidates: blocks that contain the union of points in unique_ts?
                # Let's just try random blocks? No.
                # Let's try blocks that contain at least one point from each unique_ts?
                
                # Heuristic: Pick a random unique_ts, and try blocks containing it.
                # Filter those that cover ALL unique_ts.
                
                # To speed up, we can check if a candidate block covers all unique_ts.
                # Check: all(ts in b' for ts in unique_ts)
                
                # How many candidates?
                # Pick one unique_ts, say ut.
                # Candidates = tsub_to_blocks[ut]
                # For each candidate, check if it covers all other unique_ts.
                
                ut = r.choice(unique_ts)
                candidates = tsub_to_blocks[ut]
                
                found_replacement = False
                for b_idx in candidates:
                    if b_idx == block_to_idx[tuple(b)]:
                        continue
                    b_prime = all_blocks[b_idx]
                    
                    # Check if b_prime covers all unique_ts
                    covers_all = True
                    for uts in unique_ts:
                        if uts not in b_prime: # This is slow. b_prime is list.
                             # Use set?
                             pass
                             
                    # Let's use bitmasks for this check too?
                    # Or just convert b_prime to set once?
                    # len(unique_ts) is small (usually 1-5).
                    # len(candidates) is ~6000.
                    # 6000 * 5 checks. 30k ops. Fast.
                    
                    b_prime_set = set(b_prime)
                    covers_all = True
                    for uts in unique_ts:
                        if not set(uts).issubset(b_prime_set):
                            covers_all = False
                            break
                            
                    if covers_all:
                        # Perform swap
                        # Remove b
                        for ts in itertools.combinations(b, t):
                            cov_counts[ts] -= 1
                        # Add b_prime
                        for ts in itertools.combinations(b_prime, t):
                            cov_counts[ts] += 1
                            
                        blocks[i] = b_prime
                        improved = True
                        found_replacement = True
                        break
                        
                # If no replacement found, try a perturbation?
                # For now, just stop if no swap found in this iteration?
                # To avoid infinite loop, if we tried many times and no improvement, break.
                # We can add a counter.
                
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run ILS
    num_iterations = 0
    while time.time() - start_time < time_limit * 0.8:
        num_iterations += 1
        seed = num_iterations
        blocks = greedy_cover(seed)
        blocks = local_search(blocks, seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        # Perturb best solution for next iteration?
        # Or just run fresh greedy?
        # Fresh greedy is simpler.
        
    # Final local search
    if best_blocks:
        best_blocks = local_search(best_blocks, 12345)
        
    # Verification
    cov = defaultdict(int)
    for b in best_blocks:
        for ts in itertools.combinations(b, t):
            cov[ts] += 1
            
    for ts in all_tsubs:
        if cov.get(ts, 0) == 0:
            # Fallback
            b = list(ts)
            r = random.Random(999)
            while len(b) < k:
                x = r.choice([p for p in all_points if p not in b])
                b.append(x)
            b.sort()
            best_blocks.append(b)
            
    return best_blocks

import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute triple_idx for fast access
    # triple_idx[x, y, z] = index of (x, y, z) sorted
    triple_idx = np.full((v, v, v), -1, dtype=np.int32)
    for i, ts in enumerate(all_tsubsets):
        a, b, c = ts
        triple_idx[a, b, c] = i
        triple_idx[a, c, b] = i
        triple_idx[b, a, c] = i
        triple_idx[b, c, a] = i
        triple_idx[c, a, b] = i
        triple_idx[c, b, a] = i

    def get_block_triples(block):
        """Get indices of t-subsets covered by a block."""
        return [tsubset_to_idx[ts] for ts in itertools.combinations(block, t)]

    def greedy_cover(seed, time_limit):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsubsets, dtype=bool)
        blocks = []
        
        while np.any(uncovered) and time.time() < time_limit:
            # Pick a random uncovered t-subset
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = rng.choice(uncovered_indices)
            start_combo = all_tsubsets[start_idx]
            block = list(start_combo)
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_set = set(block)
                
                # Calculate gain for each candidate x
                # Gain is number of uncovered triples formed by x and pairs in block
                # For t=3, gain(x) = sum(1 for p1,p2 in block if uncovered[idx(x,p1,p2)])
                
                # Optimization: Precompute pair indices for current block?
                # Block size is small (<=7), so itertools.combinations is fine.
                block_pairs = list(itertools.combinations(block, 2))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for p1, p2 in block_pairs:
                        idx = triple_idx[x, p1, p2]
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_gain > 0:
                        # Tie-breaking: random
                        if rng.random() < 0.5:
                            best_x = x
                
                if best_x < 0:
                    break
                
                block.append(best_x)
            
            if len(block) < k:
                continue
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    uncovered[idx] = False
        
        if not np.any(uncovered):
            return [list(b) for b in blocks]
        return None

    def local_search(blocks, time_limit):
        """
        Local search using Critical Triple Replacement.
        1. Remove redundant blocks (blocks with no critical triples).
        2. If stuck, try to replace a block with a new one that covers its critical triples.
        """
        if not blocks:
            return []
        
        blocks = [list(b) for b in blocks]
        block_triples = [get_block_triples(b) for b in blocks]
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        
        for t_list in block_triples:
            for idx in t_list:
                cover_count[idx] += 1
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            # 1. Remove redundant blocks
            for i in range(len(blocks)):
                if time.time() > time_limit:
                    return [list(b) for b in blocks]
                
                redundant = True
                for idx in block_triples[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for idx in block_triples[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_triples.pop(i)
                    changed = True
                    break # Restart loop as indices changed
            
            if not changed:
                # 2. Try to improve/swap
                # Pick a random block to potentially replace
                if not blocks:
                    break
                    
                i = random.randint(0, len(blocks) - 1)
                B_i = blocks[i]
                # Find critical triples of B_i
                critical_triples = [ts for ts, idx in zip(itertools.combinations(B_i, t), block_triples[i]) if cover_count[idx] == 1]
                
                if not critical_triples:
                    # Should have been caught by redundancy check, but just in case
                    continue
                
                # We need a new block B_new that covers all critical_triples
                # And is not already in the cover
                # And ideally covers other uncovered or low-count triples
                
                # Generate candidate blocks containing all critical triples?
                # If |critical_triples| > 1, finding a block containing all is hard.
                # If |critical_triples| == 1, say {a,b,c}, we need a block containing a,b,c.
                
                if len(critical_triples) == 1:
                    ct = critical_triples[0]
                    # Generate random blocks containing ct
                    # k=7, t=3. Block must contain a,b,c.
                    # Remaining 4 points from v-3.
                    # C(21, 4) = 5985 candidates.
                    
                    # We can try a few random ones
                    found_swap = False
                    for _ in range(20): # Try 20 random swaps
                        if time.time() > time_limit:
                            break
                        rest = [x for x in range(v) if x not in ct]
                        random.shuffle(rest)
                        new_block = list(ct) + rest[:k-t]
                        new_block.sort()
                        
                        if new_block in blocks:
                            continue
                        
                        # Check if new block covers all critical triples (it does by construction)
                        # Check if the swap is valid:
                        # The new block must cover all triples that were uniquely covered by B_i.
                        # It does.
                        # But does it break coverage for other triples?
                        # No, because we are only removing B_i. Other blocks remain.
                        # The only risk is if B_i covered some triples that NO OTHER block covers.
                        # Those are exactly the critical_triples.
                        # Since new_block covers them, the cover remains valid.
                        
                        # Is it an improvement?
                        # It keeps the same number of blocks.
                        # But it might allow future removals.
                        # Let's accept the swap to explore the space.
                        
                        # Remove B_i
                        for idx in block_triples[i]:
                            cover_count[idx] -= 1
                        
                        # Add new_block
                        new_triples = get_block_triples(new_block)
                        for idx in new_triples:
                            cover_count[idx] += 1
                        
                        blocks[i] = new_block
                        block_triples[i] = new_triples
                        found_swap = True
                        changed = True
                        break
                        
                    if found_swap:
                        break # Restart redundancy check

        return blocks

    best_blocks = None
    best_count = float('inf')
    
    seed = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        attempt_time = min(remaining - 5, 15)
        
        blocks = greedy_cover(seed, time.time() + attempt_time)
        
        if blocks is not None:
            red_time = min(remaining - 5, 20)
            blocks = local_search(blocks, time.time() + red_time)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        seed += 1
    
    if best_blocks is None:
        # Fallback: just greedy
        blocks = greedy_cover(0, deadline - 1)
        if blocks:
            return blocks
        return []

    return best_blocks

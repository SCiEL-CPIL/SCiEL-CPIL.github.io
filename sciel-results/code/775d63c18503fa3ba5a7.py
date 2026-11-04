import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute indices for all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Precompute for each point x and each (t-1)-subset S, the bitmask contribution?
    # Actually, we just need to be able to compute mask(B) quickly.
    # mask(B) is the bitmask of all t-subsets contained in B.
    # For a block B of size k, there are C(k,t) such subsets.
    # We can precompute the bitmask for every possible k-subset? No, too many.
    # But we can compute it on the fly.
    
    def get_block_mask(block):
        mask = 0
        for tsub in itertools.combinations(block, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        return mask

    def popcount(x):
        return bin(x).count('1')

    # Greedy Construction
    def greedy_construct(rng, deadline):
        uncovered = (1 << num_tsubs) - 1
        blocks = []
        cover_counts = np.zeros(num_tsubs, dtype=np.int32)
        
        # Helper to update cover counts and uncovered mask
        def add_block(block):
            mask = get_block_mask(block)
            # Update cover counts
            for tsub in itertools.combinations(block, t):
                idx = tsub_to_idx[tsub]
                cover_counts[idx] += 1
            return mask

        while uncovered:
            if time.time() > deadline:
                break
            
            # Select seed
            # Find an uncovered t-subset
            # To speed up, we can just pick a random bit from uncovered?
            # No, we need the actual subset.
            # We can maintain a list of uncovered subsets?
            # Or just pick a random index and check if it's set?
            # Better: pick a random uncovered subset from a maintained list?
            # Since we update 'uncovered' bitmask, finding a set bit is fast.
            # But we need the subset corresponding to the index.
            idx_to_tsub = {i: sub for sub, i in tsub_to_idx.items()}
            
            # Pick a random uncovered t-subset
            # Get a random set bit
            # This is O(1) if we use a list of uncovered indices, but updating that list is O(num_tsubs).
            # Bitmask is better.
            # To pick a random set bit:
            #   rand_val = rng.getrandbits(num_tsubs)
            #   candidate = uncovered & rand_val
            #   if candidate: idx = (candidate & -candidate).bit_length() - 1
            #   else: retry
            # This might loop many times if uncovered is sparse.
            # Alternatively, maintain a list of uncovered indices and sample from it.
            # Updating the list when a block is added:
            #   For each tsub in block, if it was uncovered, remove from list.
            #   This is efficient if we use a set of uncovered indices.
            
            uncovered_set = set(all_tsubs) # This is slow to init and update.
            
            # Let's stick to bitmask for 'uncovered' status, but use a list for selection.
            # Actually, the previous approach of `rng.choice(list(uncovered))` was slow.
            # Let's use a list `uncovered_list` and a set `uncovered_set` for O(1) removal.
            
            # Re-initialize for each restart? No, this is inside the function.
            # Let's restructure to use lists for speed.
            
            # This function is getting complex. Let's simplify.
            # The bottleneck is the gain calculation.
            # Let's just run a standard greedy with random restarts and prune at the end.
            
            # For the seed, just pick a random t-subset from the full list, 
            # but ensure we prioritize uncovered ones.
            # If we just pick random, we might pick covered ones.
            # We can check if it's covered.
            
            # Let's try a different strategy:
            # 1. Generate a large set of random blocks.
            # 2. Prune.
            # 3. Fill gaps with greedy.
            
            # This is often more effective.
            
            pass # Placeholder

    # Let's implement the "Random Blocks + Prune + Greedy Fill" strategy.
    
    def solve():
        rng = random.Random(42)
        
        # 1. Generate random blocks
        num_random_blocks = 200
        blocks = []
        for _ in range(num_random_blocks):
            b = rng.sample(range(v), k)
            b.sort()
            blocks.append(b)
        
        # 2. Prune
        # Compute cover counts
        cover_counts = np.zeros(num_tsubs, dtype=np.int32)
        block_masks = []
        for b in blocks:
            mask = get_block_mask(b)
            block_masks.append(mask)
            for tsub in itertools.combinations(b, t):
                idx = tsub_to_idx[tsub]
                cover_counts[idx] += 1
        
        # Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks
                # Check if block i is redundant
                redundant = True
                for tsub in itertools.combinations(blocks[i], t):
                    idx = tsub_to_idx[tsub]
                    if cover_counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for tsub in itertools.combinations(blocks[i], t):
                        idx = tsub_to_idx[tsub]
                        cover_counts[idx] -= 1
                    blocks.pop(i)
                    block_masks.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        
        # 3. Greedy Fill
        # Find uncovered t-subsets
        uncovered_idx = [i for i in range(num_tsubs) if cover_counts[i] == 0]
        
        while uncovered_idx:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset
            idx = rng.choice(uncovered_idx)
            seed_tsub = all_tsubs[idx]
            
            # Build a block around seed_tsub
            block = list(seed_tsub)
            block_set = set(block)
            
            # Greedily add points
            # To speed up gain calc, we can just pick random points?
            # No, greedy is better.
            # Use the bitmask gain method?
            # For simplicity and speed, let's just add random points from the remaining
            # but biased towards points that appear in few blocks?
            # Or just random.
            
            # Let's use a simple greedy:
            # Current block B.
            # For each x not in B, count how many uncovered t-subsets it covers.
            # Uncovered t-subsets are those with cover_count == 0.
            # A t-subset S is covered by B U {x} if S \ {x} subset B and x in S.
            # So S must be of the form {x} U T where T subset B, |T|=t-1.
            # We can precompute for each x and each B, the list of such S?
            # No, B changes.
            
            # Let's just pick random points. It's fast.
            remaining_points = [p for p in range(v) if p not in block_set]
            rng.shuffle(remaining_points)
            
            # Add points until size k
            for p in remaining_points:
                if len(block) >= k:
                    break
                block.append(p)
                block_set.add(p)
            
            block.sort()
            
            # Add block
            blocks.append(block)
            mask = get_block_mask(block)
            block_masks.append(mask)
            
            # Update cover counts
            new_uncovered_idx = []
            for tsub in itertools.combinations(block, t):
                idx_t = tsub_to_idx[tsub]
                cover_counts[idx_t] += 1
                if cover_counts[idx_t] == 1:
                    # It was uncovered, now covered
                    pass
            
            # Recompute uncovered list (expensive if done every step)
            # Only do it if necessary?
            # For now, recompute.
            uncovered_idx = [i for i in range(num_tsubs) if cover_counts[i] == 0]
            
            # Prune again?
            # Maybe every 10 blocks?
            if len(blocks) % 10 == 0:
                 # Prune
                 changed = True
                 while changed:
                     changed = False
                     for i in range(len(blocks)):
                         if time.time() > deadline:
                             return blocks
                         redundant = True
                         for tsub in itertools.combinations(blocks[i], t):
                             idx = tsub_to_idx[tsub]
                             if cover_counts[idx] == 1:
                                 redundant = False
                                 break
                         if redundant:
                             for tsub in itertools.combinations(blocks[i], t):
                                 idx = tsub_to_idx[tsub]
                                 cover_counts[idx] -= 1
                             blocks.pop(i)
                             block_masks.pop(i)
                             changed = True
                             break
                             
        return blocks

    # Run multiple restarts
    best_blocks = None
    best_len = float('inf')
    
    for i in range(10):
        if time.time() > deadline:
            break
        rng = random.Random(100 + i)
        # Modify rng seed inside solve?
        # Let's pass rng to solve
        # Refactor solve to take rng
        
        blocks_i = solve_with_rng(rng)
        if len(blocks_i) < best_len:
            best_blocks = blocks_i
            best_len = len(blocks_i)
            
    if best_blocks is None:
        # Fallback to simple greedy
        best_blocks = simple_greedy()
        
    return best_blocks

def solve_with_rng(rng):
    # ... (implementation of the logic above)
    pass

def simple_greedy():
    # ...
    pass

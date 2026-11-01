import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Finds a covering design C(v, k, t) using randomized greedy initialization
    and local search (block deletion and point swaps).
    """
    # Precompute all t-subsets
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    # Deadline guard: 110 seconds to be safe
    deadline = time.time() + 110.0
    best_blocks = None
    best_count = float('inf')
    
    # Helper: Calculate coverage array from blocks
    def get_coverage(blocks):
        cov = np.zeros(n_subs, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cov[sub_to_idx[sub]] += 1
        return cov

    # Helper: Check if a block is redundant given current coverage
    # Assumes block is currently included in coverage
    def is_redundant(block, cov):
        for sub in itertools.combinations(block, t):
            if cov[sub_to_idx[sub]] == 1:
                return False
        return True

    # Helper: Randomized Greedy Construction
    def randomized_greedy():
        uncovered = set(all_subs)
        blocks = []
        while uncovered:
            # Pick a random uncovered subset to start the block
            start_sub = random.choice(list(uncovered))
            block = list(start_sub)
            
            while len(block) < k:
                best_gain = -1
                candidates = []
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    # Count how many uncovered t-subsets are covered by adding x
                    # A t-subset is covered if it contains x and t-1 elements from block
                    # We need to check if the subset is in 'uncovered'
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        if sub in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        candidates = [x]
                    elif gain == best_gain:
                        candidates.append(x)
                
                if best_gain == 0:
                    # No point adds value? Should not happen if uncovered is not empty
                    # and we are picking from uncovered, but just in case
                    break
                
                # Randomly pick from best candidates
                block.append(random.choice(candidates))
            
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        
        return blocks

    # Helper: Local Search
    def local_search(blocks, max_time):
        cov = get_coverage(blocks)
        improved = True
        steps = 0
        
        while improved and time.time() < max_time:
            improved = False
            steps += 1
            if steps > 10000: # Safety break
                break
            
            # Pass 1: Delete redundant blocks
            to_remove = []
            for i, block in enumerate(blocks):
                if is_redundant(block, cov):
                    to_remove.append(i)
            
            if to_remove:
                # Remove in reverse order to keep indices valid
                for i in sorted(to_remove, reverse=True):
                    block = blocks[i]
                    for sub in itertools.combinations(block, t):
                        cov[sub_to_idx[sub]] -= 1
                    blocks.pop(i)
                improved = True
                continue # Restart passes as coverage changed significantly
            
            # Pass 2: Point Swaps
            # Try a few random swaps to escape local optima
            for _ in range(50):
                if time.time() >= max_time:
                    break
                if len(blocks) == 0:
                    break
                
                b_idx = random.randint(0, len(blocks) - 1)
                block = blocks[b_idx]
                
                if len(block) < 2:
                    continue
                
                # Pick a point to remove
                p_idx = random.randint(0, len(block) - 1)
                p = block[p_idx]
                
                # Pick a point to add
                available = [x for x in range(v) if x not in block]
                if not available:
                    continue
                q = random.choice(available)
                
                # Check validity of swap p -> q
                # 1. Subsets containing p in old block must be covered by others
                # 2. Subsets containing q in new block must be covered by new block (trivial) 
                #    BUT subsets containing q in new block that were NOT in old block 
                #    are now covered by new block.
                #    The critical check is: 
                #    Any subset in OldBlock \ NewBlock must have cov > 1.
                #    Any subset in NewBlock \ OldBlock is covered by NewBlock.
                
                old_subs = set(itertools.combinations(block, t))
                new_block = block.copy()
                new_block[p_idx] = q
                new_block.sort()
                new_subs = set(itertools.combinations(new_block, t))
                
                removed_subs = old_subs - new_subs
                added_subs = new_subs - old_subs
                
                valid = True
                for sub in removed_subs:
                    if cov[sub_to_idx[sub]] <= 1:
                        valid = False
                        break
                
                if valid:
                    # Apply swap
                    for sub in removed_subs:
                        cov[sub_to_idx[sub]] -= 1
                    for sub in added_subs:
                        cov[sub_to_idx[sub]] += 1
                    
                    blocks[b_idx] = new_block
                    improved = True
                    break # Restart passes
            
            # Pass 3: Re-greedy a random block
            # This allows us to replace a block with a potentially better one
            if len(blocks) > 0 and time.time() < max_time:
                b_idx = random.randint(0, len(blocks) - 1)
                old_block = blocks[b_idx]
                
                # Temporarily remove
                for sub in itertools.combinations(old_block, t):
                    cov[sub_to_idx[sub]] -= 1
                
                # Find uncovered subsets
                uncovered_now = set()
                for sub in all_subs:
                    if cov[sub_to_idx[sub]] == 0:
                        uncovered_now.add(sub)
                
                if not uncovered_now:
                    # Old block was redundant, just leave it removed
                    blocks.pop(b_idx)
                    improved = True
                    continue
                
                # Build new block to cover uncovered_now
                # Start with a random uncovered sub
                start_sub = random.choice(list(uncovered_now))
                new_block = list(start_sub)
                
                while len(new_block) < k and uncovered_now:
                    best_gain = -1
                    candidates = []
                    for x in range(v):
                        if x in new_block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(new_block, t - 1):
                            sub = tuple(sorted(rest + (x,)))
                            if sub in uncovered_now:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            candidates = [x]
                        elif gain == best_gain:
                            candidates.append(x)
                    
                    if best_gain == 0:
                        break
                    
                    new_block.append(random.choice(candidates))
                
                # Check if new block covers all currently uncovered subsets
                # If not, the swap is invalid (we can't have uncovered subsets)
                # We only replace if it fixes the coverage
                valid = True
                for sub in uncovered_now:
                    if not all(pt in new_block for pt in sub):
                        valid = False
                        break
                
                if valid:
                    # Apply new block
                    for sub in itertools.combinations(new_block, t):
                        cov[sub_to_idx[sub]] += 1
                    blocks[b_idx] = new_block
                    improved = True
                else:
                    # Revert
                    for sub in itertools.combinations(old_block, t):
                        cov[sub_to_idx[sub]] += 1
                    # blocks[b_idx] is still old_block, no change needed

        return blocks

    # Main Loop
    while time.time() < deadline:
        # Generate a candidate
        candidate = randomized_greedy()
        
        # Refine
        refined = local_search(candidate, deadline)
        
        # Check if better
        if len(refined) < best_count:
            best_count = len(refined)
            best_blocks = refined
            
            # If we hit the theoretical lower bound or a very good number, we might stop early
            # Lower bound is 69. If we find 76, we are at best known.
            # If we find < 76, that's a new record!
            if best_count <= 76:
                # We can continue to try for even fewer, but 76 is the benchmark.
                # Let's keep searching for a bit longer if we have time.
                pass

    # If no blocks were found (should not happen), return empty or fallback
    if best_blocks is None:
        return []
        
    return best_blocks

"""Combined program: Greedy + Local Search for Covering Designs."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """Construct a covering design C(v,k,t) using randomized greedy + local search."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Precompute t-subsets and k-blocks
    tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(tsubs)
    tsub_idx = {ts: i for i, ts in enumerate(tsubs)}
    
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # Map block tuple to list of t-subset indices it covers
    # We store these as lists for easy iteration, but we can optimize access if needed.
    # For v=19, k=8, t=5:
    # n_tsubs = C(19,5) = 11628
    # n_blocks = C(19,8) = 75582
    # Each block covers C(8,5) = 56 t-subsets.
    # Total entries in block_tsubs = 75582 * 56 = 4,232,592.
    # This is manageable in memory.
    
    block_tsubs = {}
    for b in blocks:
        ts_idxs = [tsub_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsubs[b] = ts_idxs
        
    # Map t-subset tuple to list of blocks containing it
    # This is crucial for fast candidate retrieval in greedy and local search.
    tsub_blocks = {ts: [] for ts in tsubs}
    for b in blocks:
        for ts in itertools.combinations(b, t):
            tsub_blocks[ts].append(b)
            
    # Helper function to check if a set of t-subset indices is covered by a block
    # Using sets for fast intersection check
    # block_tsub_sets: map block -> set of t-subset indices
    block_tsub_sets = {b: set(idxs) for b, idxs in block_tsubs.items()}
    
    def _greedy_cover(rng):
        """Randomized greedy set cover."""
        uncovered = set(range(n_tsubs))
        selected_blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset
            target_idx = list(uncovered)[rng.randrange(len(uncovered))]
            target_tsub = tsubs[target_idx]
            
            # Get candidate blocks covering this t-subset
            candidates = tsub_blocks[target_tsub]
            
            best_block = None
            best_gain = -1
            
            for b in candidates:
                # Calculate gain: number of currently uncovered t-subsets covered by b
                # Using set intersection
                gain = len(block_tsub_sets[b] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            if best_block is None:
                break
                
            selected_blocks.append(best_block)
            uncovered -= block_tsub_sets[best_block]
            
        return selected_blocks

    def _local_search(blocks, rng, time_limit):
        """Local search: delete redundant blocks, swap to escape local minima."""
        start_time_ls = time.time()
        blocks = list(blocks)
        
        # Build initial coverage counts
        cov = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            for idx in block_tsubs[b]:
                cov[idx] += 1
        
        # To speed up local search, we can precompute a list of blocks for each t-subset
        # which we already have in tsub_blocks.
        
        while time.time() - start_time_ls < time_limit:
            improved = False
            
            # 1. Try to delete redundant blocks
            # A block is redundant if all t-subsets it covers are covered by at least one other block.
            # i.e., for all idx in block_tsubs[b], cov[idx] > 1.
            
            # We can iterate through blocks in random order.
            indices = list(range(len(blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() - start_time_ls > time_limit:
                    break
                b = blocks[i]
                idxs = block_tsubs[b]
                # Check if all are covered > 1 time
                if all(cov[idx] > 1 for idx in idxs):
                    # Remove block
                    for idx in idxs:
                        cov[idx] -= 1
                    blocks.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try to swap blocks
            # Pick a random block b.
            # If b is essential (covers some t-subset uniquely), we need to replace it.
            # Find a block b' that covers all the unique t-subsets of b.
            # To make this efficient, we can look for b' that shares many t-subsets with b 
            # but covers the unique ones.
            
            if not blocks:
                break
                
            found_swap = False
            num_tries = 20
            for _ in range(num_tries):
                if time.time() - start_time_ls > time_limit:
                    break
                    
                i = rng.randrange(len(blocks))
                b = blocks[i]
                idxs_b = block_tsubs[b]
                
                # Find critical t-subsets (covered only by b)
                critical_idxs = [idx for idx in idxs_b if cov[idx] == 1]
                
                if not critical_idxs:
                    # Redundant, handled in step 1, but just in case
                    for idx in idxs_b:
                        cov[idx] -= 1
                    blocks.pop(i)
                    improved = True
                    found_swap = True
                    break
                
                # Try to find a replacement
                # The replacement must cover all critical_idxs.
                # We can pick a random critical idx and look at blocks covering it.
                # Filter those that cover ALL critical_idxs.
                
                first_crit = critical_idxs[0]
                candidates = tsub_blocks[tsubs[first_crit]]
                
                # Shuffle candidates to diversify
                candidates = list(candidates)
                rng.shuffle(candidates)
                
                for b_prime in candidates:
                    if b_prime == b:
                        continue
                    
                    # Check if b_prime covers all critical_idxs
                    # Using set intersection
                    covers_all = all(idx in block_tsub_sets[b_prime] for idx in critical_idxs)
                    
                    if covers_all:
                        # Perform swap
                        # Remove b
                        for idx in idxs_b:
                            cov[idx] -= 1
                        # Add b_prime
                        for idx in block_tsubs[b_prime]:
                            cov[idx] += 1
                        blocks[i] = b_prime
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            if not found_swap:
                # No improvement found in this batch
                # We could try a more aggressive move or break.
                # Breaking is safer to avoid infinite loops if stuck.
                # But maybe try a "kick" move: remove a random block and re-greedy?
                # For now, break.
                break
                
        return blocks

    # Main loop: Multiple restarts
    best_blocks = None
    best_count = float('inf')
    
    # We can adjust the number of restarts based on time.
    # Each greedy takes some time, local search takes the rest.
    # Let's do a few quick greedy restarts, then local search on the best.
    
    # Time budget allocation:
    # 1. Greedy restarts: ~20s
    # 2. Local search: remaining time
    
    greedy_deadline = start_time + 20.0
    num_restarts = 0
    while time.time() < greedy_deadline:
        rng = random.Random(num_restarts * 101 + 7)
        blocks = _greedy_cover(rng)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        num_restarts += 1
        
    # Local search on the best solution found
    if best_blocks is not None:
        remaining_time = deadline - time.time() - 5.0 # 5s safety margin
        if remaining_time > 0:
            rng = random.Random(42)
            best_blocks = _local_search(best_blocks, rng, remaining_time)
            
    # Final validation and cleanup
    if best_blocks is None:
        # Fallback: simple greedy
        rng = random.Random(0)
        best_blocks = _greedy_cover(rng)
        
    # Ensure blocks are distinct and valid
    seen = set()
    final_blocks = []
    for b in best_blocks:
        key = tuple(sorted(b))
        if key not in seen and len(set(b)) == k and all(0 <= x < v for x in b):
            seen.add(key)
            final_blocks.append(list(b))
            
    return final_blocks
# EVOLVE-BLOCK-END

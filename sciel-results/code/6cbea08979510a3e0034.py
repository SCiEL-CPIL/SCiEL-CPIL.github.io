import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using fast greedy multi-start
    followed by local search refinement.
    """
    deadline = time.time() + 110.0  # Safety margin

    # Precompute t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute for each t-subset, the list of k-subsets that contain it
    tsub_to_ksubs = {}
    for ts in all_tsubs:
        ts_set = set(ts)
        remaining = [p for p in range(v) if p not in ts_set]
        ksubs = []
        for extra in itertools.combinations(remaining, k - t):
            block = tuple(sorted(ts + tuple(extra)))
            ksubs.append(block)
        tsub_to_ksubs[ts] = ksubs
    
    # Precompute for each k-subset, the bitset of t-subsets it covers
    # Using Python integers as bitsets for speed
    ksub_to_bitset = {}
    
    def get_ksub_bitset(block):
        block = tuple(block)
        if block not in ksub_to_bitset:
            bits = 0
            for comb in itertools.combinations(block, t):
                idx = tsub_to_idx[comb]
                bits |= (1 << idx)
            ksub_to_bitset[block] = bits
        return ksub_to_bitset[block]
    
    # Full coverage bitset
    full_coverage = (1 << n_tsubs) - 1
    
    def greedy_once(seed):
        rng = random.Random(seed)
        uncovered = full_coverage
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset
            # Find first set bit
            # Use bit_length to find position
            lowbit = uncovered & (-uncovered)
            start_idx = lowbit.bit_length() - 1
            S = all_tsubs[start_idx]
            
            # Get precomputed k-subsets containing S
            candidates = tsub_to_ksubs[S]
            
            # Evaluate gain for each candidate
            best_block = None
            best_gain = -1
            
            # Shuffle for tie-breaking
            candidates = list(candidates)
            rng.shuffle(candidates)
            
            for B in candidates:
                bits = get_ksub_bitset(B)
                gain = (uncovered & bits).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = B
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Update uncovered mask
            bits = get_ksub_bitset(best_block)
            uncovered &= ~bits
        
        return blocks

    def remove_redundant(blocks, max_iter=5):
        if not blocks:
            return blocks
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Count coverage of each t-subset using bitsets
            # For each t-subset, count how many blocks cover it
            cover_count = [0] * n_tsubs
            block_bitsets = []
            for b in blocks:
                bits = get_ksub_bitset(tuple(b))
                block_bitsets.append(bits)
                # Count bits set in each position
                # This is slow, so we'll use a different approach
                # For each t-subset index, check if it's in the bitset
                # Actually, let's compute cover_count differently
                # We'll iterate through set bits
                temp = bits
                while temp:
                    lowbit = temp & (-temp)
                    idx = lowbit.bit_length() - 1
                    cover_count[idx] += 1
                    temp ^= lowbit
            
            valid_blocks = []
            removed_any = False
            
            for i, b in enumerate(blocks):
                if time.time() > deadline:
                    return blocks
                
                bits = block_bitsets[i]
                # Check if all t-subsets in this block are covered by other blocks
                # i.e., cover_count[idx] > 1 for all idx in bits
                temp = bits
                redundant = True
                while temp:
                    lowbit = temp & (-temp)
                    idx = lowbit.bit_length() - 1
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                    temp ^= lowbit
                
                if redundant:
                    # Remove it
                    temp = bits
                    while temp:
                        lowbit = temp & (-temp)
                        idx = lowbit.bit_length() - 1
                        cover_count[idx] -= 1
                        temp ^= lowbit
                    removed_any = True
                else:
                    valid_blocks.append(b)
            
            if not removed_any:
                break
            blocks = valid_blocks
        
        return blocks

    def validate_covering(blocks):
        """Check if blocks form a valid covering."""
        if not blocks:
            return n_tsubs == 0
        covered = 0
        for b in blocks:
            covered |= get_ksub_bitset(tuple(b))
        return covered == full_coverage

    def local_search(blocks, rng, time_limit):
        """
        Local search: try to remove blocks by replacing them with better ones,
        or find redundant blocks to remove.
        """
        if not blocks:
            return blocks
        
        current = blocks[:]
        current_len = len(current)
        start_time = time.time()
        
        # Precompute bitsets
        current_bitsets = [get_ksub_bitset(tuple(b)) for b in current]
        
        # Try to improve by swapping blocks
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try removing each block and see if we can cover the gaps with fewer blocks
            for i in range(len(current)):
                if time.time() > time_limit:
                    break
                    
                # Remove block i
                without_i = current[:i] + current[i+1:]
                without_i_bitsets = current_bitsets[:i] + current_bitsets[i+1:]
                
                # Check what's uncovered
                covered = 0
                for bits in without_i_bitsets:
                    covered |= bits
                
                uncovered = full_coverage & ~covered
                if not uncovered:
                    # Block i is redundant!
                    current = without_i
                    current_bitsets = without_i_bitsets
                    improved = True
                    break
                
                # Try to cover uncovered with fewer than 1 block (impossible)
                # So we can't remove it unless it's redundant
                # Try replacing block i with a different block that covers more
                # Find t-subsets only covered by block i
                only_i = 0
                for j in range(len(current)):
                    if j != i:
                        only_i &= ~current_bitsets[j]
                only_i &= current_bitsets[i]
                
                if only_i:
                    # There are t-subsets only covered by block i
                    # Try to find a replacement block that covers these and more
                    # Pick a t-subset from only_i
                    lowbit = only_i & (-only_i)
                    idx = lowbit.bit_length() - 1
                    target_tsub = all_tsubs[idx]
                    
                    # Find candidate blocks covering target_tsub
                    candidates = tsub_to_ksubs[target_tsub]
                    candidates = list(candidates)
                    rng.shuffle(candidates)
                    
                    best_new_block = None
                    best_gain = -1
                    
                    for B in candidates:
                        if B == tuple(current[i]):
                            continue
                        bits = get_ksub_bitset(B)
                        # Gain: how many of the only_i t-subsets does this cover
                        gain = (only_i & bits).bit_count()
                        # Also consider overall coverage improvement
                        # But for now, just focus on covering only_i
                        if gain > best_gain:
                            best_gain = gain
                            best_new_block = B
                    
                    if best_new_block is not None and best_gain == only_i.bit_count():
                        # New block covers all the unique t-subsets
                        # Check if it adds coverage elsewhere
                        new_bits = get_ksub_bitset(best_new_block)
                        # This is valid if it covers at least what the old block covered uniquely
                        # Actually, we need to verify the covering is still valid
                        # Let's just try it
                        new_current = current[:]
                        new_current[i] = best_new_block
                        new_bitsets = current_bitsets[:]
                        new_bitsets[i] = new_bits
                        
                        # Validate
                        covered_check = 0
                        for bits in new_bitsets:
                            covered_check |= bits
                        if covered_check == full_coverage:
                            current = new_current
                            current_bitsets = new_bitsets
                            # Try to reduce again
                            current = remove_redundant(current, max_iter=2)
                            current_bitsets = [get_ksub_bitset(tuple(b)) for b in current]
                            improved = True
                            break
        
        return current

    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multi-start greedy
    num_starts = 0
    start_time = time.time()
    
    # Run greedy starts until we have enough time for local search
    ls_time_budget = 25.0  # Reserve 25s for local search
    greedy_deadline = deadline - ls_time_budget
    
    while time.time() < greedy_deadline:
        seed = num_starts
        blocks = greedy_once(seed)
        
        if not blocks:
            num_starts += 1
            continue
            
        # Reduce
        reduced_blocks = remove_redundant(blocks)
        
        if len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks[:]
            
        num_starts += 1
    
    if best_blocks is None:
        best_blocks = greedy_once(0)
        best_blocks = remove_redundant(best_blocks)
        best_len = len(best_blocks)
    
    # Phase 2: Local search refinement
    if time.time() < deadline - 5.0:
        rng = random.Random(12345)
        ls_deadline = deadline - 5.0
        
        # Run local search on the best solution
        improved = local_search(best_blocks, rng, ls_deadline)
        if len(improved) < best_len:
            best_len = len(improved)
            best_blocks = improved[:]
        
        # Try local search on a few other good solutions if time permits
        if time.time() < ls_deadline - 10.0:
            for _ in range(3):
                if time.time() > ls_deadline - 5.0:
                    break
                seed = num_starts
                num_starts += 1
                blocks = greedy_once(seed)
                if blocks:
                    reduced = remove_redundant(blocks)
                    if len(reduced) <= best_len + 5:  # Only try if close to best
                        improved = local_search(reduced, rng, ls_deadline)
                        if len(improved) < best_len:
                            best_len = len(improved)
                            best_blocks = improved[:]
    
    return best_blocks

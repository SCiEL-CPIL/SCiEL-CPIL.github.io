import itertools
import random
import time
import heapq
from collections import defaultdict

_DEADLINE = 145.0
_start_time = None

def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def construct(v, k, t):
    global _start_time
    _start_time = time.time()
    
    rng = random.Random(12345)
    
    # 1. Precomputation
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Block coverage as bitsets (Python ints)
    # Bit i is set if block covers t-subset i
    block_cover_bits = [0] * n_blocks
    # Also store list of t-subset indices for faster iteration if needed
    block_ts_indices = [[] for _ in range(n_blocks)]
    
    # Map t-subset index to list of blocks covering it
    ts_to_blocks = defaultdict(list)
    
    for b_idx, blk in enumerate(all_blocks):
        bits = 0
        ts_list = []
        for ts in itertools.combinations(blk, t):
            idx = ts_index[ts]
            bits |= (1 << idx)
            ts_list.append(idx)
            ts_to_blocks[idx].append(b_idx)
        block_cover_bits[b_idx] = bits
        block_ts_indices[b_idx] = ts_list

    def greedy_construct(seed):
        """Run greedy with random tie-breaking."""
        local_rng = random.Random(seed)
        uncovered_mask = (1 << n_ts) - 1 # All uncovered
        uncovered_count = n_ts
        
        in_set = [False] * n_blocks
        selected = []
        
        # Initial gains: number of uncovered t-subsets covered
        # Since all are uncovered initially, gain = C(k,t) for all.
        # We use a heap. To add randomness, we can perturb the initial key or pick randomly from top.
        # Let's use a max-heap with (-gain, random_tie, block_idx)
        
        # Precompute max possible gain
        max_gain = len(block_ts_indices[0])
        
        # Heap entries: (-gain, random_tie, block_idx)
        # We need to update gains.
        # Instead of a complex heap update, since N_blocks is 50k, 
        # and we do this ~150 times, a simple scan might be too slow?
        # 150 * 50k = 7.5M operations. Python can do ~10M simple ops/sec. 
        # So a full scan per step is ~0.75s per greedy run. 
        # 5 runs = 3.75s. This is acceptable.
        
        # Let's try the scan approach for simplicity and robustness, 
        # but we can optimize by only considering blocks that cover *some* uncovered t-subsets.
        
        while uncovered_count > 0 and _remaining() > 5.0:
            best_b = -1
            best_gain = 0
            best_tie = 0
            
            # Optimization: Iterate only through blocks that have potential gain > 0
            # A block has gain > 0 if it covers at least one uncovered t-subset.
            # We can maintain a list of "active" blocks?
            # Or just scan all.
            
            for b_idx in range(n_blocks):
                if in_set[b_idx]:
                    continue
                # Calculate gain: popcount of (block_cover_bits[b_idx] & uncovered_mask)
                # Python int bit_count is fast (C implementation)
                gain = (block_cover_bits[b_idx] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
                    best_tie = local_rng.random()
                elif gain == best_gain and gain > 0:
                    # Tie-breaking
                    tie = local_rng.random()
                    if tie > best_tie:
                        best_tie = tie
                        best_b = b_idx
                        
            if best_b == -1 or best_gain == 0:
                break
                
            in_set[best_b] = True
            selected.append(best_b)
            uncovered_mask &= ~block_cover_bits[best_b]
            uncovered_count -= best_gain
            
        # Convert to list of blocks
        return [list(all_blocks[b]) for b in selected]

    def local_search(blocks, seed, time_limit):
        """Improve covering via reduction and stochastic swaps."""
        local_rng = random.Random(seed)
        local_start = time.time()
        
        # Convert to indices for internal processing
        block_indices = []
        for blk in blocks:
            # Find index
            # This is slow if done naively. 
            # We should pass indices from greedy.
            pass
        
        # Let's refactor greedy to return indices.
        
        # Re-implement local search to work with indices.
        
        pass

    # Refactored plan:
    # 1. Greedy returns list of block indices.
    # 2. Local Search works on indices.
    
    def greedy_construct_indices(seed):
        local_rng = random.Random(seed)
        uncovered_mask = (1 << n_ts) - 1
        uncovered_count = n_ts
        in_set = [False] * n_blocks
        selected = []
        
        while uncovered_count > 0 and _remaining() > 5.0:
            best_b = -1
            best_gain = 0
            best_tie = 0
            
            for b_idx in range(n_blocks):
                if in_set[b_idx]:
                    continue
                gain = (block_cover_bits[b_idx] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
                    best_tie = local_rng.random()
                elif gain == best_gain and gain > 0:
                    tie = local_rng.random()
                    if tie > best_tie:
                        best_tie = tie
                        best_b = b_idx
                        
            if best_b == -1 or best_gain == 0:
                break
                
            in_set[best_b] = True
            selected.append(best_b)
            uncovered_mask &= ~block_cover_bits[best_b]
            uncovered_count -= best_gain
            
        return selected

    def local_search_indices(indices, seed, time_limit):
        local_rng = random.Random(seed)
        local_start = time.time()
        
        # State
        in_set = [False] * n_blocks
        for b in indices:
            in_set[b] = True
        
        current_size = len(indices)
        best_size = current_size
        best_indices = indices.copy()
        
        # Precompute coverage counts? 
        # For reduction, we need to know if a block is redundant.
        # Redundant if: For all ts in covers(B), there is another block B' in set covering ts.
        # Equivalent to: covers(B) is a subset of (Union of covers of other blocks in set).
        
        def get_union_others(exclude_b):
            union = 0
            for b in range(n_blocks):
                if in_set[b] and b != exclude_b:
                    union |= block_cover_bits[b]
            return union

        # This is O(N_blocks) per check. 
        # N_blocks = 50k. 
        # Reduction loop:
        #   Scan all in_set blocks (size ~150).
        #   For each, check redundancy.
        #   If we do this naively, it's 150 * 50k = 7.5M ops per reduction pass.
        #   Too slow if done frequently.
        
        # Optimization: Maintain `coverage_count` array.
        coverage_count = [0] * n_ts
        for b in indices:
            for ts_idx in block_ts_indices[b]:
                coverage_count[ts_idx] += 1
                
        def is_redundant(b_idx):
            for ts_idx in block_ts_indices[b_idx]:
                if coverage_count[ts_idx] == 1:
                    return False
            return True
            
        def remove_block(b_idx):
            for ts_idx in block_ts_indices[b_idx]:
                coverage_count[ts_idx] -= 1
            in_set[b_idx] = False
            
        def add_block(b_idx):
            for ts_idx in block_ts_indices[b_idx]:
                coverage_count[ts_idx] += 1
            in_set[b_idx] = True
            
        # Main Loop
        while _remaining() > time_limit:
            # Phase 1: Reduction
            changed = True
            while changed and _remaining() > time_limit + 1.0:
                changed = False
                # Iterate over a copy of current blocks to allow removal
                current_blocks = [b for b in range(n_blocks) if in_set[b]]
                # Randomize order to avoid bias
                local_rng.shuffle(current_blocks)
                
                for b_idx in current_blocks:
                    if not in_set[b_idx]:
                        continue
                    if is_redundant(b_idx):
                        remove_block(b_idx)
                        changed = True
                        current_size -= 1
                        if current_size < best_size:
                            best_size = current_size
                            best_indices = [b for b in range(n_blocks) if in_set[b]]
                        break # Restart scan after removal
        
            # Phase 2: Stochastic Swap
            # Pick a random block in set
            current_blocks = [b for b in range(n_blocks) if in_set[b]]
            if not current_blocks:
                break
                
            b_old = local_rng.choice(current_blocks)
            
            # Find unique t-subsets of b_old
            unique_ts = []
            for ts_idx in block_ts_indices[b_old]:
                if coverage_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
                    
            if not unique_ts:
                # Should have been removed in Phase 1
                continue
                
            # Find candidates for replacement
            # Intersection of ts_to_blocks for all ts in unique_ts
            if len(unique_ts) == 1:
                candidates = ts_to_blocks[unique_ts[0]]
            else:
                # Start with the smallest list
                min_idx = min(unique_ts, key=lambda x: len(ts_to_blocks[x]))
                candidates = set(ts_to_blocks[min_idx])
                for ts_idx in unique_ts:
                    if ts_idx == min_idx:
                        continue
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                        
            # Filter out blocks already in set
            candidates = [b for b in candidates if not in_set[b]]
            
            if candidates:
                # Pick a candidate
                # Heuristic: Pick one that covers the most "critical" t-subsets elsewhere?
                # Or just random.
                b_new = local_rng.choice(candidates)
                
                # Perform Swap
                remove_block(b_old)
                add_block(b_new)
                # Size remains same
                
                # Check if this swap enabled further reduction?
                # We will check in the next iteration's Phase 1.
                # To avoid cycling, we can track recent moves? 
                # Or just rely on the fact that if it doesn't lead to reduction, 
                # we might swap back. 
                # To mitigate, we can bias towards candidates that cover more 
                # t-subsets with coverage_count == 1 (making them potentially redundant later).
                
            else:
                # No single block can replace b_old.
                # Kick: Remove b_old, then greedily cover the now-uncovered t-subsets.
                # This increases size, so we only do it if we are stuck.
                # Let's skip kicks for now to keep it simple and fast.
                # If we are stuck, we might try a different b_old.
                pass
                
        # Return best found
        return best_indices

    # Run multiple greedy instances
    best_overall = None
    best_overall_size = float('inf')
    
    # Time budget allocation:
    # 20% for greedy, 80% for local search?
    # Or just run greedy until time allows, then local search on best.
    
    # Let's run 3 greedy variants.
    seeds = [1, 2, 3]
    candidates = []
    
    for seed in seeds:
        if _remaining() < 10.0:
            break
        idxs = greedy_construct_indices(seed)
        if idxs:
            candidates.append(idxs)
            
    if not candidates:
        # Fallback: simple greedy
        idxs = greedy_construct_indices(0)
        return [list(all_blocks[b]) for b in idxs]
        
    # Run local search on each candidate, with time sharing
    # Or pick the best initial one and run local search on it?
    # Better: Run local search on all, but time-slice.
    
    # Sort candidates by size
    candidates.sort(key=len)
    
    # Take the best 2 candidates for local search
    for idxs in candidates[:2]:
        if _remaining() < 5.0:
            break
        # Allocate remaining time / 2 for this local search
        time_budget = _remaining() / 2.0
        improved = local_search_indices(idxs, random.randint(0, 1000), time_budget)
        
        if len(improved) < best_overall_size:
            best_overall_size = len(improved)
            best_overall = improved
            
    if best_overall is None:
        best_overall = candidates[0]
        
    return [list(all_blocks[b]) for b in best_overall]

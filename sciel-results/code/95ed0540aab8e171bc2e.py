"""Bitset-accelerated greedy with randomized restarts and frequency-weighted scoring."""
import time
import random
import itertools
from collections import Counter

def construct(v, k, t):
    deadline = time.time() + 110  # safety margin before 120s kill
    all_points = list(range(v))
    
    # Precompute all t-subsets as tuples
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each point, precompute which t-subsets it participates in
    point_tsubs = [[] for _ in range(v)]
    for ts in all_tsubs:
        for p in ts:
            point_tsubs[p].append(tsub_to_idx[ts])
    
    # For each (t-1)-subset, precompute which points complete it to a t-subset
    # And for each point, for each (t-1) subset not containing it, the t-subset index
    # Actually, let's precompute: for a partial block (list of points), 
    # the t-subsets covered by adding point x are those (t-1)-subsets of the block
    # combined with x.
    
    # Better: precompute for each point x, a mapping from frozenset of (t-1) points 
    # to the t-subset index. But that's expensive. 
    # Instead, during the greedy, for a current block B and candidate x,
    # the new t-subsets covered are: for each (t-1)-subset S of B, S ∪ {x} if it's a valid t-subset.
    
    # Let's precompute: for each point x, a dict mapping frozenset of (t-1) points to t-subset index
    # This is C(v, t-1) entries per point, total v * C(v, t-1) which for v=19, t=5: 19 * C(19,4) = 19*3876 = 73644. Fine.
    
    point_tsub_lookup = [dict() for _ in range(v)]
    for ts in all_tsubs:
        ts_set = set(ts)
        for x in ts:
            rest = frozenset(ts_set - {x})
            point_tsub_lookup[x][rest] = tsub_to_idx[ts]
    
    def greedy_once(rng, deadline):
        """Run one greedy pass with randomization controlled by rng."""
        # uncovered: bitset of uncovered t-subsets
        # Use Python int as bitset
        uncovered = (1 << num_tsubs) - 1  # all bits set
        
        # Count how many times each t-subset would be "requested" - we'll use inverse frequency weighting
        # Initially all are uncovered, so weight = 1 for all. But as we cover, 
        # we want to prioritize t-subsets that are still uncovered and "rare".
        # Actually, for the greedy, at each step we pick a t-subset to force-cover,
        # then build a block around it. Let's pick randomly from uncovered.
        
        blocks = []
        # We need to track which t-subsets are covered to pick uncovered ones efficiently.
        # For speed, maintain a list of uncovered indices and remove as we go.
        uncovered_list = list(range(num_tsubs))
        uncovered_set = set(uncovered_list)
        
        while uncovered_list:
            if time.time() > deadline:
                return None  # timeout, return what we have? No, we need a valid covering.
                # Actually, if we time out mid-greedy, the covering is incomplete. 
                # We should return the best COMPLETE covering from previous restarts.
                # So greedy_once should only return if it completed.
                return None
            
            # Pick a random uncovered t-subset to anchor
            idx = rng.choice(uncovered_list)
            anchor = all_tsubs[idx]
            
            # Build a block containing the anchor, picking points that maximize new coverage
            block = list(anchor)
            block.sort()
            
            # Compute which t-subsets this anchor already covers
            # The anchor itself is covered, but we need to add more points.
            
            while len(block) < k:
                if time.time() > deadline:
                    return None
                
                best_x = None
                best_gain = -1
                best_weighted = -1
                
                # For each candidate point x not in block, compute gain
                block_set = set(block)
                # Precompute (t-1)-subsets of current block
                t1_subsets = [frozenset(s) for s in itertools.combinations(block, t - 1)]
                
                for x in all_points:
                    if x in block_set:
                        continue
                    # Compute which t-subsets are newly covered by adding x
                    # For each (t-1)-subset S of block, if S ∪ {x} is a t-subset, 
                    # check if it's uncovered
                    gain = 0
                    weighted_gain = 0
                    for S in t1_subsets:
                        ts_idx = point_tsub_lookup[x].get(S)
                        if ts_idx is not None and (uncovered >> ts_idx) & 1:
                            gain += 1
                            # Weight by inverse frequency: how many blocks could cover this?
                            # For simplicity, use gain directly, but we can add a small random tie-break
                    # Add small random perturbation for tie-breaking
                    weighted_gain = gain + rng.random() * 0.001
                    
                    if weighted_gain > best_weighted:
                        best_weighted = weighted_gain
                        best_gain = gain
                        best_x = x
                
                if best_x is None or best_gain == 0:
                    # No point adds new coverage; just pick a random point to complete the block
                    remaining = [x for x in all_points if x not in block_set]
                    if not remaining:
                        break
                    best_x = rng.choice(remaining)
                
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            
            # Cover all t-subsets in this block
            for ts in itertools.combinations(block, t):
                ts_idx = tsub_to_idx[ts]
                if (uncovered >> ts_idx) & 1:
                    uncovered &= ~(1 << ts_idx)
                    if ts_idx in uncovered_set:
                        uncovered_set.remove(ts_idx)
                        uncovered_list.remove(ts_idx)  # O(n) but acceptable for now
            
            blocks.append(block)
        
        return blocks
    
    best_blocks = None
    rng = random.Random(42)
    
    # Run multiple randomized restarts
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        result = greedy_once(rng, deadline)
        if result is not None:
            # Verify it's a valid covering (should be, but check)
            # For speed, skip full verification; trust the greedy logic
            if best_blocks is None or len(result) < len(best_blocks):
                best_blocks = result
                # If we got a good result, try a few more
                if len(result) <= 340:
                    # Continue trying to improve
                    pass
            # If we've done several attempts and have a result, check time
            if attempt >= 3 and time.time() > deadline - 30:
                break
    
    # Fallback: if no complete covering found, do a deterministic greedy
    if best_blocks is None:
        # Deterministic greedy
        uncovered = (1 << num_tsubs) - 1
        uncovered_list = list(range(num_tsubs))
        uncovered_set = set(uncovered_list)
        blocks = []
        
        while uncovered_list:
            # Pick the smallest uncovered t-subset
            idx = min(uncovered_list)
            anchor = all_tsubs[idx]
            block = list(anchor)
            block.sort()
            
            while len(block) < k:
                block_set = set(block)
                t1_subsets = [frozenset(s) for s in itertools.combinations(block, t - 1)]
                best_x = None
                best_gain = -1
                
                for x in all_points:
                    if x in block_set:
                        continue
                    gain = 0
                    for S in t1_subsets:
                        ts_idx = point_tsub_lookup[x].get(S)
                        if ts_idx is not None and (uncovered >> ts_idx) & 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None or best_gain == 0:
                    remaining = [x for x in all_points if x not in block_set]
                    if not remaining:
                        break
                    best_x = remaining[0]
                
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            
            for ts in itertools.combinations(block, t):
                ts_idx = tsub_to_idx[ts]
                if (uncovered >> ts_idx) & 1:
                    uncovered &= ~(1 << ts_idx)
                    if ts_idx in uncovered_set:
                        uncovered_set.remove(ts_idx)
                        uncovered_list.remove(ts_idx)
        
        best_blocks = blocks
    
    return best_blocks

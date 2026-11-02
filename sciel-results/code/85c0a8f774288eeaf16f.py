import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (evaluator kills at 150s)

    def calc_score(U):
        if not U or 0 not in U:
            return 0.0
        U = sorted(U)
        if len(U) < 2:
            return 0.0
        
        max_u = U[-1]
        if max_u == 0:
            return 0.0
            
        sum_set = set()
        diff_set = set()
        
        n = len(U)
        for i in range(n):
            ui = U[i]
            for j in range(n):
                sum_set.add(ui + U[j])
                diff_set.add(ui - U[j])
        
        size_sum = len(sum_set)
        size_diff = len(diff_set)
        
        if size_sum == 0:
            return 0.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
            
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        return 1.0 + math.log(ratio) / denom

    best_score = 0.0
    best_set = [0, 1, 3]
    best_score = calc_score(best_set)

    # Strategy: Hierarchical "Staircase" Construction
    # We build U as a union of blocks B_0, B_1, ..., B_{k-1} such that:
    # B_0 = {0}
    # B_i is shifted by offset S_i
    # The gaps between blocks are superincreasing to prevent cross-block sum collisions.
    # Specifically, if max(B_i) < min(B_{i+1}) - max(B_i), then sums involving different blocks
    # are distinct from sums within a single block (and from sums of other pairs).
    #
    # If we ensure strict separation:
    # Let M_i = max element of block i (relative to block start).
    # Let S_i be the start of block i.
    # Condition: S_{i+1} > S_i + M_i + M_{i+1} roughly.
    # Actually, for sumsets to be disjoint, we need the ranges of sums to not overlap.
    # Range of sums within block i: [2*S_i, 2*S_i + 2*M_i] (approx, assuming 0 in block or similar structure)
    # This is complex. A simpler sufficient condition for |U+U| to be the sum of |B_i+B_i| is that
    # the sumsets B_i+B_i are disjoint.
    # If B_i is contained in [S_i, S_i + M_i], then B_i+B_i is in [2S_i, 2S_i + 2M_i].
    # If 2S_{i+1} > 2S_i + 2M_i + 2M_{i+1} (gap between ranges), they are disjoint.
    # A stronger, simpler condition: S_{i+1} > S_i + M_i + M_{i+1} + 1.
    # Then 2S_{i+1} > 2S_i + 2M_i + 2M_{i+1} + 2 > max(B_i+B_i).
    #
    # Similarly for differences:
    # B_i - B_j. If i != j, the ranges are [S_i - M_j, S_i + M_i] and [S_j - M_i, S_j + M_j].
    # If the blocks are well-separated, cross-difference sets are also disjoint from internal difference sets
    # and from each other, provided the separation is large enough.
    #
    # If we assume full disjointness of all sum/diff components:
    # |U+U| = sum_i |B_i+B_i|
    # |U-U| = sum_i |B_i-B_i| + sum_{i!=j} |B_i-B_j|
    # Note |B_i-B_j| = |B_i+B_j| (by mapping x -> -x).
    # So |U-U| = sum_i |B_i-B_i| + sum_{i!=j} |B_i+B_j|
    #
    # Let's optimize the blocks.
    # To maximize the ratio |U-U|/|U+U|, we want blocks with high internal ratio |B-B|/|B+B|.
    # A single element block {0} has |B+B|=1, |B-B|=1. Ratio 1.
    # A block {0, 1} has |B+B|=2 ({0,1,2}), |B-B|=3 ({-1,0,1}). Ratio 1.5.
    # A block {0, 1, 3} has |B+B|=5 ({0,1,2,3,4,6} -> wait. 0,1,3. Sums: 0,1,3, 1,2,4, 3,4,6. Set: {0,1,2,3,4,6}. Size 6.
    # Diffs: 0,1,3, -1,0,-3, -3,-4,-6. Set: {-6,-4,-3,-1,0,1,3,4,6}. Size 9.
    # Ratio 9/6 = 1.5.
    #
    # Let's try to find good small blocks first, then chain them.
    
    # Pre-compute best small sets
    # We will use a greedy approach to build the staircase.
    # We start with B_0 = {0}.
    # Then we add B_1, B_2, etc.
    # To keep the search space manageable, we will restrict the internal structure of blocks to be small (size <= 4 or 5).
    # And we will optimize the "shape" of the blocks.
    
    # Candidate block shapes (relative to 0)
    block_shapes = [
        [0],
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6], # Golomb ruler
        [0, 1, 2, 5],
        [0, 1, 3, 5],
        [0, 2, 3, 5, 8],
    ]
    
    # Pre-compute sum/diff sizes for these shapes
    # Note: The shape is defined relative to 0. When shifted by S, the sums are shifted by 2S, diffs by S.
    # The sizes remain the same.
    
    def get_block_stats(shape):
        s_set = set()
        d_set = set()
        for a in shape:
            for b in shape:
                s_set.add(a + b)
                d_set.add(a - b)
        return len(s_set), len(d_set)
    
    shape_stats = {}
    for sh in block_shapes:
        shape_stats[sh] = get_block_stats(sh)
        
    # Greedy Staircase Construction
    # We maintain a list of blocks used so far.
    # State: (current_blocks, current_score_estimate)
    # Actually, we can just simulate the construction.
    
    # Let's try a few fixed sequences of block shapes and see which is best.
    # Since the separation gaps grow exponentially, the max(U) will be dominated by the last block's position.
    # Let S_0 = 0.
    # S_{i+1} = S_i + M_i + M_{i+1} + 1.
    # Max(U) = S_{k-1} + M_{k-1}.
    
    # We will search over sequences of block shapes.
    # Limit length to 4 or 5 blocks to keep Max(U) reasonable and computation fast.
    
    best_sequence = None
    best_seq_score = 0.0
    
    # Recursive search for sequences
    def search_sequence(current_blocks, current_S, current_max_u, depth):
        nonlocal best_sequence, best_seq_score, best_score, best_set
        
        if time.time() > deadline:
            return
            
        if depth > 0:
            # Calculate score for this sequence
            # U is union of shifted blocks
            # We need to verify the disjointness assumption holds or just calculate directly.
            # Given the small number of blocks, direct calculation is fast and safe.
            
            U = []
            S = 0
            for sh in current_blocks:
                U.extend([x + S for x in sh])
                if sh:
                    S += sh[-1] + (current_blocks[current_blocks.index(sh)+1][-1] if current_blocks.index(sh)+1 < len(current_blocks) else 0) + 1
                    # Wait, the shift depends on the NEXT block's max.
                    # Let's re-calculate U properly.
            
            # Re-calculate U properly
            U = []
            S = 0
            for i, sh in enumerate(current_blocks):
                U.extend([x + S for x in sh])
                if i < len(current_blocks) - 1:
                    next_sh = current_blocks[i+1]
                    # Gap to ensure disjointness
                    # Max sum of current block: 2*S + 2*max(sh)
                    # Min sum of next block: 2*(S + gap)
                    # We need 2*(S+gap) > 2*S + 2*max(sh) + 2*max(next_sh)
                    # gap > max(sh) + max(next_sh)
                    # Let's use gap = max(sh) + max(next_sh) + 1
                    S += max(sh) + max(next_sh) + 1
            
            if not U:
                return
                
            U = list(set(U))
            if 0 not in U:
                return
                
            s = calc_score(U)
            if s > best_seq_score:
                best_seq_score = s
                best_sequence = U[:]
                
            # Update global best
            if s > best_score:
                best_score = s
                best_set = U[:]
        
        # Try adding more blocks
        if depth >= 6: # Max depth
            return
            
        if time.time() > deadline:
            return

        for sh in block_shapes:
            # Pruning: if the new block is too large, skip
            # Heuristic: keep max(U) under some limit, e.g., 10000
            # Estimate new max U
            if current_blocks:
                last_sh = current_blocks[-1]
                est_new_max = current_max_u + max(last_sh) + max(sh) + 1 + max(sh)
                if est_new_max > 5000:
                    continue
            else:
                est_new_max = max(sh)
                
            new_blocks = current_blocks + [sh]
            # Calculate new S and Max
            # We need to recalculate S for the whole chain to be sure, but for pruning we can estimate.
            # Let's just recurse.
            search_sequence(new_blocks, 0, 0, depth + 1)

    # Start search
    search_sequence([], 0, 0, 0)
    
    # Also try some random local improvements on the best found so far
    if time.time() < deadline - 10:
        random.seed(42)
        # Local search on best_set
        current_set = best_set[:]
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding a random element
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                if not current_set:
                    break
                max_u = max(current_set)
                # Try adding a number in [1, max_u*2]
                new_val = random.randint(1, max_u * 2 + 10)
                if new_val not in current_set:
                    new_set = current_set + [new_val]
                    new_set.sort()
                    s = calc_score(new_set)
                    if s > current_score:
                        current_score = s
                        current_set = new_set
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_set = current_set[:]
            # Try removing an element (if size > 2)
            if len(current_set) > 2:
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    idx = random.randint(1, len(current_set) - 1) # Don't remove 0
                    new_set = current_set[:idx] + current_set[idx+1:]
                    s = calc_score(new_set)
                    if s > current_score:
                        current_score = s
                        current_set = new_set
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_set = current_set[:]
            # Try swapping
            if len(current_set) > 2:
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    idx = random.randint(1, len(current_set) - 1)
                    old_val = current_set[idx]
                    max_u = max(current_set)
                    new_val = random.randint(1, max_u + 10)
                    if new_val != old_val and new_val not in current_set:
                        new_set = current_set[:]
                        new_set[idx] = new_val
                        new_set.sort()
                        s = calc_score(new_set)
                        if s > current_score:
                            current_score = s
                            current_set = new_set
                            improved = True
                            if s > best_score:
                                best_score = s
                                best_set = current_set[:]

    return best_set
# EVOLVE-BLOCK-END

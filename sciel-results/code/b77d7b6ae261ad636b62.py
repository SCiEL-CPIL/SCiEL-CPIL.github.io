import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # ============================================================
    # Strategy: Deterministic construction of S(5,6,12) via
    # the known algebraic structure, with backtracking fallback.
    #
    # S(5,6,12) has 132 blocks on 12 points.
    # We can construct it using the "short" and "long" blocks
    # from the Witt design construction.
    #
    # The key insight: S(5,6,12) can be constructed from the
    # Steiner system S(4,5,11) on 11 points by adding a 12th point.
    # For each 5-block B of S(4,5,11), we create two 6-blocks:
    #   {∞} ∪ B  and  complement of B in the 11 points
    # This gives 66 + 66 = 132 blocks.
    #
    # S(4,5,11) itself can be constructed from the 11-point
    # projective line or via explicit backtracking with symmetry breaking.
    # ============================================================
    
    # --- Construct S(4,5,11) on points 0..10 ---
    # S(4,5,11) has 66 blocks, each of size 5, covering all C(11,4)=330 4-subsets exactly once.
    
    points_11 = list(range(11))
    all_4subsets_11 = list(combinations(points_11, 4))
    total_4subsets = len(all_4subsets_11)  # 330
    
    # All 5-subsets of 11 points
    all_5subsets_11 = list(combinations(points_11, 5))
    
    # Map each 4-subset to the 5-subsets containing it
    four_to_fives = {s: [] for s in all_4subsets_11}
    five_to_fours = {}
    for f in all_5subsets_11:
        fours = list(combinations(f, 4))
        five_to_fours[f] = set(fours)
        for s in fours:
            four_to_fives[s].append(f)
    
    # Precompute as sets for fast intersection
    five_to_fours_set = {f: set(combinations(f, 4)) for f in all_5subsets_11}
    
    # Backtracking solver for S(4,5,11)
    # Uses Algorithm X style with bitmasks for speed
    
    # Assign an index to each 4-subset
    four_idx = {s: i for i, s in enumerate(all_4subsets_11)}
    
    # For each 5-subset, store the bitmask of 4-subsets it covers
    five_mask = {}
    for f in all_5subsets_11:
        mask = 0
        for s in five_to_fours_set[f]:
            mask |= (1 << four_idx[s])
        five_mask[f] = mask
    
    # For each 4-subset index, list of 5-subsets containing it
    four_to_fives_by_idx = [[] for _ in range(total_4subsets)]
    for f in all_5subsets_11:
        for s in five_to_fours_set[f]:
            four_to_fives_by_idx[four_idx[s]].append(f)
    
    def solve_s4511():
        """Solve S(4,5,11) using backtracking with bitmask."""
        covered_mask = 0
        chosen = []
        
        def backtrack(covered_mask, chosen):
            if time.time() > deadline:
                return None
            
            if covered_mask == (1 << total_4subsets) - 1:
                return chosen[:]
            
            if len(chosen) >= 66:
                return chosen[:] if covered_mask == (1 << total_4subsets) - 1 else None
            
            # Find uncovered 4-subset with fewest candidates
            best_col = None
            best_cands = None
            min_count = float('inf')
            
            # Iterate through uncovered 4-subsets
            for col in range(total_4subsets):
                if (covered_mask >> col) & 1:
                    continue
                cands = four_to_fives_by_idx[col]
                # Filter: candidate 5-set must not share any 4-subset with already chosen
                # i.e., five_mask[f] & covered_mask == 0
                valid = []
                for f in cands:
                    if five_mask[f] & covered_mask == 0:
                        valid.append(f)
                if len(valid) < min_count:
                    min_count = len(valid)
                    best_col = col
                    best_cands = valid
                    if min_count == 0:
                        return None
                    if min_count == 1:
                        break
            
            if best_col is None:
                return None
            
            # Sort candidates by how many new 4-subsets they cover
            best_cands.sort(key=lambda f: bin(five_mask[f] & ~covered_mask).count('1'), reverse=True)
            
            for f in best_cands:
                new_mask = covered_mask | five_mask[f]
                result = backtrack(new_mask, chosen + [f])
                if result is not None:
                    return result
            return None
        
        # Symmetry breaking: fix first block
        # By symmetry, we can fix the first block to be (0,1,2,3,4)
        first_block = (0, 1, 2, 3, 4)
        if first_block not in five_to_fours_set:
            # Should not happen
            first_block = all_5subsets_11[0]
        
        init_mask = five_mask[first_block]
        result = backtrack(init_mask, [first_block])
        
        if result is None:
            # Try without symmetry breaking
            result = backtrack(0, [])
        
        return result
    
    s4511_blocks = solve_s4511()
    
    if s4511_blocks is not None and len(s4511_blocks) == 66:
        # Extend to S(5,6,12)
        # Point 11 is the new infinity point
        # For each 5-block B of S(4,5,11):
        #   Block 1: B ∪ {11}  (size 6)
        #   Block 2: complement of B in {0..10}  (size 6)
        blocks_12 = []
        for b in s4511_blocks:
            b_set = set(b)
            # Block containing 11
            blocks_12.append(tuple(sorted(b_set | {11})))
            # Block not containing 11 (complement in 0..10)
            comp = tuple(sorted(set(range(11)) - b_set))
            blocks_12.append(comp)
        
        # Verify: should have 132 distinct blocks
        if len(set(blocks_12)) == 132:
            # Quick verification: no 5-subset appears in two blocks
            all_5sets_12 = list(combinations(range(12), 5))
            five_count = {s: 0 for s in all_5sets_12}
            for b in blocks_12:
                for s in combinations(b, 5):
                    five_count[s] += 1
                    if five_count[s] > 1:
                        break
            # If all good, return
            if all(v <= 1 for v in five_count.values()):
                return [list(b) for b in blocks_12]
    
    # --- Fallback: Direct backtracking for S(5,6,12) ---
    # This is the original approach but optimized
    
    points = range(12)
    all_5sets = list(combinations(points, 5))
    all_6sets = list(combinations(points, 6))
    
    # Map 5-subset to list of 6-subsets containing it
    blocks_containing_5set = {s: [] for s in all_5sets}
    block_5sets = {}
    
    for b in all_6sets:
        subsets_5 = list(combinations(b, 5))
        block_5sets[b] = set(subsets_5)
        for s in subsets_5:
            blocks_containing_5set[s].append(b)
    
    total_5sets = len(all_5sets)  # 792
    
    # Use bitmask approach for speed
    five_idx = {s: i for i, s in enumerate(all_5sets)}
    six_mask = {}
    for b in all_6sets:
        mask = 0
        for s in block_5sets[b]:
            mask |= (1 << five_idx[s])
        six_mask[b] = mask
    
    five_to_sixes = [[] for _ in range(total_5sets)]
    for b in all_6sets:
        for s in block_5sets[b]:
            five_to_sixes[five_idx[s]].append(b)
    
    def solve_s5612():
        """Solve S(5,6,12) using backtracking with bitmask."""
        covered_mask = 0
        chosen = []
        
        def backtrack(covered_mask, chosen):
            if time.time() > deadline:
                return None
            
            if covered_mask == (1 << total_5sets) - 1:
                return chosen[:]
            
            if len(chosen) >= 132:
                return chosen[:] if covered_mask == (1 << total_5sets) - 1 else None
            
            # Find uncovered 5-subset with fewest candidates
            best_col = None
            best_cands = None
            min_count = float('inf')
            
            for col in range(total_5sets):
                if (covered_mask >> col) & 1:
                    continue
                cands = five_to_sixes[col]
                valid = []
                for b in cands:
                    if six_mask[b] & covered_mask == 0:
                        valid.append(b)
                if len(valid) < min_count:
                    min_count = len(valid)
                    best_col = col
                    best_cands = valid
                    if min_count == 0:
                        return None
                    if min_count == 1:
                        break
            
            if best_col is None:
                return None
            
            # Sort by coverage
            best_cands.sort(key=lambda b: bin(six_mask[b] & ~covered_mask).count('1'), reverse=True)
            
            for b in best_cands:
                new_mask = covered_mask | six_mask[b]
                result = backtrack(new_mask, chosen + [b])
                if result is not None:
                    return result
            return None
        
        # Symmetry breaking: fix first block
        first_block = (0, 1, 2, 3, 4, 5)
        if first_block in six_mask:
            init_mask = six_mask[first_block]
            result = backtrack(init_mask, [first_block])
            if result is not None:
                return result
        
        # Try without symmetry breaking
        return backtrack(0, [])
    
    result = solve_s5612()
    
    if result is not None:
        return [list(b) for b in result]
    
    # --- Final fallback: Greedy ---
    covered = set()
    blocks = []
    shuffled_blocks = all_6sets[:]
    random.shuffle(shuffled_blocks)
    for b in shuffled_blocks:
        if block_5sets[b] & covered == set():
            blocks.append(b)
            covered.update(block_5sets[b])
    
    return [list(b) for b in blocks]

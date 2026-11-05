import sys
import time

def construct():
    # The Witt design S(5,6,12) has 132 blocks.
    # We construct it using the known orbit structure under M12.
    # A standard way to generate S(5,6,12) is via the Steiner system S(3,4,8) (Witt design W8)
    # or by using a known set of generators for the Mathieu group M12.
    
    # However, the most robust and "from scratch" way without external libraries is to
    # use a known explicit list of blocks or a deterministic algorithm that builds it.
    
    # One common construction uses the "octad" representation or the "dodecad" representation.
    # Here, we use a simple backtracking search that is extremely fast for this specific
    # highly symmetric problem, or we can just hardcode the 132 blocks if we can generate them.
    
    # Let's try to generate them using a known base block and a set of permutations.
    # The stabilizer of a point in M12 is M11. The stabilizer of two points is M10, etc.
    # The number of blocks is 132.
    # The stabilizer of a block (dodecad) has size 155520 / 132 = 1178.
    
    # Actually, a simpler approach:
    # We can use the fact that S(5,6,12) can be constructed from S(4,5,11) or similar.
    # But let's just write a fast solver that finds the 132 blocks.
    # The previous solver was slow. We can improve the heuristic or use a known construction.
    
    # A known construction:
    # 1. Start with the set of 12 points {0, ..., 11}.
    # 2. Use the fact that the blocks are the 132 "dodecads".
    # 3. We can generate them by taking the orbit of a base block under a set of generators for M12.
    
    # Generators for M12:
    # g1 = (0 1 2 3 4 5 6 7 8 9 10 11)  # 12-cycle
    # g2 = (0 1 2 3 4 5) (6 7 8 9 10 11) # 6-cycle
    # This might not generate M12.
    
    # Let's use a verified set of generators for M12 in S12.
    # From literature:
    # a = (0 1 2 3 4 5 6 7 8 9 10 11)
    # b = (0 1 2 3 4 5) (6 7 8 9 10 11)
    # c = (0 6) (1 7) (2 8) (3 9) (4 10) (5 11)
    # These might not be sufficient.
    
    # Alternative: Use the "binary Golay code" construction.
    # The 132 blocks of S(5,6,12) correspond to the 132 words of weight 6 in the binary Golay code.
    # We can generate the binary Golay code using a generator matrix.
    
    # Generator matrix for the [24,12,8] binary Golay code is well known.
    # But we need the [12,6,4] shortened code? No, the dodecads are weight 6 in the [24,12,8] code?
    # Actually, the Steiner system S(5,6,12) is related to the Leech lattice or the Golay code.
    # The 132 blocks of S(5,6,12) are the supports of the weight 6 words in the dual of the [12,6,4] code?
    
    # Let's just use a fast backtracking search with a good heuristic.
    # The key to speed is to fix the first few blocks to break symmetry.
    
    # By symmetry, we can assume:
    # Block 1 contains {0, 1, 2, 3, 4, 5}
    # Block 2 contains {0, 1, 2, 3, 6, 7}
    # Block 3 contains {0, 1, 2, 8, 9, 10}
    # Block 4 contains {0, 1, 11, ...}
    
    # Let's try this approach.
    
    from itertools import combinations
    
    n = 12
    points = range(n)
    
    # Precompute all 5-subsets and their IDs
    all_5_subsets = list(combinations(points, 5))
    subset_to_id = {s: i for i, s in enumerate(all_5_subsets)}
    num_5_subsets = len(all_5_subsets) # 792
    
    # Precompute row masks for all 6-subsets
    all_6_subsets = list(combinations(points, 6))
    num_6_subsets = len(all_6_subsets) # 924
    
    row_mask = [0] * num_6_subsets
    col_rows = [[] for _ in range(num_5_subsets)]
    
    for r, block in enumerate(all_6_subsets):
        mask = 0
        for i in range(6):
            sub = block[:i] + block[i+1:]
            c = subset_to_id[sub]
            mask |= (1 << c)
            col_rows[c].append(r)
        row_mask[r] = mask
        
    initial_available = (1 << num_5_subsets) - 1
    
    # Fixed blocks to break symmetry
    # Block 0: {0,1,2,3,4,5}
    # Block 1: {0,1,2,3,6,7}
    # Block 2: {0,1,2,8,9,10}
    # Block 3: {0,1,11, ...} - we need to find the rest.
    
    # Let's just search with these initial blocks fixed.
    # This reduces the search space significantly.
    
    def apply_block(available_mask, block_id):
        return available_mask & ~row_mask[block_id]
    
    # Find the IDs of the fixed blocks
    block0 = (0, 1, 2, 3, 4, 5)
    block1 = (0, 1, 2, 3, 6, 7)
    block2 = (0, 1, 2, 8, 9, 10)
    
    id0 = all_6_subsets.index(block0)
    id1 = all_6_subsets.index(block1)
    id2 = all_6_subsets.index(block2)
    
    # Check if these blocks are compatible
    # They share {0,1,2}, so they are compatible.
    
    available = initial_available
    selected = []
    
    for bid in [id0, id1, id2]:
        if (row_mask[bid] & ~available) != 0:
            return [] # Should not happen
        available = apply_block(available, bid)
        selected.append(bid)
        
    # Now search for the remaining 129 blocks
    # We can use a recursive solver with a strong heuristic.
    
    best_solution = []
    best_len = 0
    deadline = time.time() + 110.0
    
    def solve(available_mask, selected_rows):
        nonlocal best_solution, best_len, deadline
        
        if time.time() > deadline:
            return False
            
        if not available_mask:
            if len(selected_rows) > best_len:
                best_solution = selected_rows.copy()
                best_len = len(selected_rows)
            return True
            
        # Heuristic: Find the 5-subset with the fewest valid blocks
        min_count = float('inf')
        best_col = -1
        best_rows = []
        
        # To speed up, we can iterate only over available columns
        # But 792 is small, so iterating all is fine.
        
        for c in range(num_5_subsets):
            if not (available_mask & (1 << c)):
                continue
                
            valid_rows = []
            for r in col_rows[c]:
                if (row_mask[r] & ~available_mask) == 0:
                    valid_rows.append(r)
                    
            count = len(valid_rows)
            if count == 0:
                return False
                
            if count < min_count:
                min_count = count
                best_col = c
                best_rows = valid_rows
                
                if count == 1:
                    break
        
        # Try each row in best_rows
        # To break symmetry further, we can order the rows
        for r in best_rows:
            new_available = apply_block(available_mask, r)
            if solve(new_available, selected_rows + [r]):
                return True
                
        return False
        
    solve(available, selected)
    
    if best_solution:
        return [list(all_6_subsets[r]) for r in best_solution]
    else:
        return [list(all_6_subsets[r]) for r in selected]

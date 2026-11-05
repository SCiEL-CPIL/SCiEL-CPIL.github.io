import sys
from itertools import combinations

def construct():
    # S(5,6,12) construction via S(4,5,11) on points 0..10, with point 11 as infinity.
    # S(4,5,11) has 66 blocks. We solve it via backtracking on 11 points.
    
    n11 = 11
    points = list(range(n11))
    
    # All 4-subsets of 11 points: C(11,4) = 330
    all_4_subsets = list(combinations(points, 4))
    subset4_to_id = {s: i for i, s in enumerate(all_4_subsets)}
    num_4 = len(all_4_subsets)
    
    # All 5-subsets of 11 points: C(11,5) = 462
    all_5_subsets = list(combinations(points, 5))
    num_5 = len(all_5_subsets)
    
    # For each 5-subset, list of 4-subset IDs it contains
    row_cols = [None] * num_5
    for r, block in enumerate(all_5_subsets):
        cols = []
        for i in range(5):
            sub = block[:i] + block[i+1:]
            cols.append(subset4_to_id[sub])
        row_cols[r] = cols
    
    # column_rows[c] = set of 5-subset indices that contain 4-subset c
    column_rows = [set() for _ in range(num_4)]
    for r in range(num_5):
        for c in row_cols[r]:
            column_rows[c].add(r)
    
    solution = []
    
    def solve(available_cols, selected_rows, depth):
        if not available_cols:
            solution.extend(selected_rows)
            return True
        
        # Find column with fewest valid rows
        min_count = float('inf')
        best_col = -1
        best_rows = []
        
        for c in available_cols:
            valid_rows = []
            for r in column_rows[c]:
                if all(cc in available_cols for cc in row_cols[r]):
                    valid_rows.append(r)
            if len(valid_rows) == 0:
                return False
            if len(valid_rows) < min_count:
                min_count = len(valid_rows)
                best_col = c
                best_rows = valid_rows
                if min_count == 1:
                    break
        
        for r in best_rows:
            new_available = available_cols - set(row_cols[r])
            if solve(new_available, selected_rows + [r], depth + 1):
                return True
        
        return False
    
    initial_available = set(range(num_4))
    if solve(initial_available, [], 0):
        s4511_blocks = [list(all_5_subsets[r]) for r in solution]
    else:
        # Fallback: shouldn't happen, but return empty
        return []
    
    # Now construct S(5,6,12) blocks:
    # 66 blocks: each S(4,5,11) block extended with point 11
    blocks = [b + [11] for b in s4511_blocks]
    
    # 11 blocks without point 11:
    # These are the 11 "parallel" classes. In S(5,6,12), the blocks not containing infinity
    # form a parallel class structure. Actually, the 11 blocks not containing 11 are:
    # For each point i in 0..10, there's a block of the other 10 points minus 4?
    # No. Let me think again.
    #
    # S(5,6,12) has 132 blocks. 66 contain infinity (11), 66 don't.
    # The 66 that don't contain infinity: each is a 6-subset of {0..10}.
    #
    # Actually, the standard construction: S(5,6,12) is the unique Steiner system.
    # The blocks not containing infinity can be derived from the S(4,5,11) structure.
    #
    # Alternative approach: The 11 blocks not containing 11 are the "short" blocks.
    # In the Witt design, there are 11 blocks of size 6 that don't contain a given point,
    # and these 11 blocks partition the other 11 points into... no, 11 blocks of size 6
    # on 11 points would cover 66 point-occurrences, but 11 points * 11 blocks = 121?
    # No. Let me just solve S(5,6,12) directly but use the S(4,5,11) solution to guide it.
    #
    # Actually, a cleaner approach: just hardcode the 132 blocks or solve S(5,6,12) 
    # directly with a fast solver. Since we already have S(4,5,11), let's use the 
    # known relationship.
    #
    # The 66 blocks not containing 11 in S(5,6,12) are in bijection with the 66 blocks 
    # of S(4,5,11) via the complement? No.
    #
    # Let me just solve S(5,6,12) directly. It's 12 points, 924 6-subsets, 792 5-subsets.
    # The backtracking should be fast with good heuristics.
    
    # Revert to direct S(5,6,12) solving
    n = 12
    points12 = list(range(n))
    
    all_5_subsets12 = list(combinations(points12, 5))
    subset5_to_id = {s: i for i, s in enumerate(all_5_subsets12)}
    num_5_12 = len(all_5_subsets12)
    
    all_6_subsets = list(combinations(points12, 6))
    num_6 = len(all_6_subsets)
    
    row_cols12 = [None] * num_6
    for r, block in enumerate(all_6_subsets):
        cols = []
        for i in range(6):
            sub = block[:i] + block[i+1:]
            cols.append(subset5_to_id[sub])
        row_cols12[r] = cols
    
    column_rows12 = [set() for _ in range(num_5_12)]
    for r in range(num_6):
        for c in row_cols12[r]:
            column_rows12[c].add(r)
    
    solution12 = []
    
    def solve12(available_cols, selected_rows):
        if not available_cols:
            solution12.extend(selected_rows)
            return True
        
        min_count = float('inf')
        best_col = -1
        best_rows = []
        
        for c in available_cols:
            valid_rows = []
            for r in column_rows12[c]:
                if all(cc in available_cols for cc in row_cols12[r]):
                    valid_rows.append(r)
            if len(valid_rows) == 0:
                return False
            if len(valid_rows) < min_count:
                min_count = len(valid_rows)
                best_col = c
                best_rows = valid_rows
                if min_count == 1:
                    break
        
        for r in best_rows:
            new_available = available_cols - set(row_cols12[r])
            if solve12(new_available, selected_rows + [r]):
                return True
        
        return False
    
    initial_available12 = set(range(num_5_12))
    if solve12(initial_available12, []):
        blocks = [list(all_6_subsets[r]) for r in solution12]
        return blocks
    else:
        return []

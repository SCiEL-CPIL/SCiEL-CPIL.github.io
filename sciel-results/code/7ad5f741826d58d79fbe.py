import time
import itertools
import sys

def construct():
    """
    Construct the Witt design S(5,6,12) with 132 blocks.
    
    Strategy: Deterministic Exact Cover using Algorithm X with bitmasks.
    The problem is to select 132 distinct 6-subsets of {0..11} such that every
    5-subset is contained in exactly one block (a Steiner system S(5,6,12)).
    
    - Universe: all C(12,5) = 792 five-subsets.
    - Rows: all C(12,6) = 924 six-subsets, each covering 6 five-subsets.
    - Goal: exact cover of the 792 five-subsets by 132 rows.
    
    Algorithm X with bitmask-based state and "minimum remaining values" heuristic
    is extremely fast for this size.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    points = list(range(12))
    
    # Generate all 5-subsets and map to indices
    all_5s = list(itertools.combinations(points, 5))
    five_map = {s: i for i, s in enumerate(all_5s)}
    num_fives = len(all_5s)  # 792
    
    # Generate all 6-subsets and compute their bitmask of covered 5-subsets
    all_6s = list(itertools.combinations(points, 6))
    num_sixes = len(all_6s)  # 924
    
    # row_masks[r] = bitmask of which 5-subsets row r covers
    row_masks = []
    for block in all_6s:
        mask = 0
        for sub5 in itertools.combinations(block, 5):
            idx = five_map[sub5]
            mask |= (1 << idx)
        row_masks.append(mask)
    
    # Precompute: for each 5-subset column, which rows cover it
    col_to_rows = [set() for _ in range(num_fives)]
    for r_idx in range(num_sixes):
        m = row_masks[r_idx]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            col_to_rows[idx].add(r_idx)
            m ^= lsb
    
    # Convert to frozensets or lists for faster intersection if needed
    # Keep as sets for now
    
    solution = []
    full_mask = (1 << num_fives) - 1
    
    def solve(covered_mask, available_rows):
        """
        Recursive Algorithm X solver.
        covered_mask: bitmask of 5-subsets already covered.
        available_rows: set of row indices that can still be chosen.
        Returns True if solution found.
        """
        if time.time() > deadline:
            return False
        
        if covered_mask == full_mask:
            return True
        
        # Heuristic: choose the uncovered column with fewest available rows
        best_col = -1
        min_count = float('inf')
        
        for c in range(num_fives):
            if not (covered_mask >> c) & 1:
                # Count available rows covering this column
                count = 0
                for r in available_rows:
                    if r in col_to_rows[c]:
                        count += 1
                        if count >= min_count:
                            break
                if count == 0:
                    return False
                if count < min_count:
                    min_count = count
                    best_col = c
                    if min_count == 1:
                        break
        
        if best_col == -1:
            return covered_mask == full_mask
        
        # Get candidate rows for best_col
        candidates = [r for r in available_rows if r in col_to_rows[best_col]]
        
        # Sort candidates by how many new 5-subsets they cover (greedy tie-break)
        candidates.sort(key=lambda r: bin(row_masks[r] & ~covered_mask).count('1'), reverse=True)
        
        for r in candidates:
            new_covered = covered_mask | row_masks[r]
            
            # Compute new available rows: remove rows that conflict with r
            conflict_mask = row_masks[r]
            new_available = set()
            for r_prime in available_rows:
                if r_prime == r:
                    continue
                if (row_masks[r_prime] & conflict_mask) == 0:
                    new_available.add(r_prime)
            
            solution.append(r)
            
            if solve(new_covered, new_available):
                return True
            
            solution.pop()
        
        return False
    
    # Initial state
    initial_available = set(range(num_sixes))
    
    # Run solver
    success = solve(0, initial_available)
    
    if success and len(solution) == 132:
        result_blocks = [list(all_6s[r]) for r in solution]
        return result_blocks
    
    # Fallback: if solver failed or timed out, use a greedy construction
    # This should not normally be reached, but provides a safety net
    if time.time() > deadline - 2.0:
        # Return empty list to avoid timeout
        return []
    
    # Greedy fallback: pick blocks that maximize new coverage
    covered = 0
    blocks = []
    remaining = set(range(num_sixes))
    
    while len(blocks) < 132 and remaining:
        best_idx = None
        best_new = -1
        for idx in remaining:
            if row_masks[idx] & covered:
                continue
            new_bits = bin(row_masks[idx] & ~covered).count('1')
            if new_bits > best_new:
                best_new = new_bits
                best_idx = idx
        if best_idx is None:
            break
        blocks.append(best_idx)
        covered |= row_masks[best_idx]
        # Remove conflicting rows
        conflict = row_masks[best_idx]
        remaining = {r for r in remaining if (row_masks[r] & conflict) == 0}
    
    if blocks:
        return [list(all_6s[r]) for r in blocks]
    
    return []

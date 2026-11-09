import itertools
import time

def construct():
    """
    Construct the Witt design S(5,6,12) with 132 blocks.
    
    Strategy: Deterministic Exact Cover (Algorithm X with MRV heuristic) using bitmasks.
    The problem is to select 132 distinct 6-subsets of {0..11} such that no 5-subset
    is contained in more than one block. Since C(12,5)=792 and each block covers 6 
    distinct 5-subsets, 132 blocks cover exactly 792 5-subsets. Thus, we are looking 
    for an exact cover of the 792 five-subsets by 132 six-subsets.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    points = list(range(12))
    
    # 1. Precomputation
    # Generate all 5-subsets and map them to indices 0..791
    all_5s = list(itertools.combinations(points, 5))
    five_map = {s: i for i, s in enumerate(all_5s)}
    num_fives = len(all_5s)  # 792
    
    # Generate all 6-subsets (candidate blocks)
    all_6s = list(itertools.combinations(points, 6))
    num_sixes = len(all_6s)  # 924
    
    # Represent each 6-subset as a bitmask of the 792 five-subsets it covers
    # A 6-subset {a,b,c,d,e,f} covers the 5-subsets formed by removing one element.
    row_masks = []
    for block in all_6s:
        mask = 0
        for i in range(6):
            # Remove the i-th element to get a 5-subset
            sub5 = tuple(block[:i] + block[i+1:])
            idx = five_map[sub5]
            mask |= (1 << idx)
        row_masks.append(mask)
        
    # Precompute for each 5-subset index, which rows (6-subsets) cover it
    # This is needed for the MRV heuristic
    col_to_rows = [set() for _ in range(num_fives)]
    for r_idx, mask in enumerate(row_masks):
        m = mask
        while m:
            # Get index of lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            col_to_rows[idx].add(r_idx)
            m ^= lsb

    # Convert sets to lists for slightly faster iteration in some contexts, 
    # but we keep sets for intersection logic.
    # Actually, for MRV, we need to count available rows for each column.
    # We will maintain 'available_rows' as a set of row indices.

    solution = []
    
    def solve(covered_mask, available_rows):
        """
        Recursively find a set of rows that cover all bits in the universe.
        covered_mask: bits already covered.
        available_rows: set of row indices that can still be chosen.
        Returns True if solution found, and appends to 'solution'.
        """
        if time.time() > deadline:
            return False # Timeout safety
            
        # Check if all 5-subsets are covered
        # (1 << num_fives) - 1 is a mask with 792 bits set
        if covered_mask == (1 << num_fives) - 1:
            return True
            
        # Heuristic: Choose the column (5-subset) that is not yet covered and has
        # the fewest available rows that can cover it (MRV).
        best_col = -1
        min_count = float('inf')
        
        # Iterate through all 792 columns to find the one with min available rows
        # Optimization: We can skip covered columns
        for c in range(num_fives):
            if not (covered_mask & (1 << c)):
                # Count how many available rows cover this column
                # available_rows & col_to_rows[c]
                # Using set intersection
                count = len(available_rows & col_to_rows[c])
                if count == 0:
                    return False # Dead end: column c cannot be covered
                if count < min_count:
                    min_count = count
                    best_col = c
                    if min_count == 1:
                        break # Can't get better than 1 (forced choice)
                
        if best_col == -1:
            return False # No uncovered columns? Should have returned True earlier.
            
        # Choose a row from the candidates for best_col
        candidates = list(available_rows & col_to_rows[best_col])
        
        for r in candidates:
            # Try selecting row r
            new_covered = covered_mask | row_masks[r]
            
            # Update available rows: remove r and any row that conflicts with r
            # Two rows conflict if they share a 5-subset (i.e., their masks intersect)
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
    
    if success:
        # Reconstruct blocks
        result_blocks = [list(all_6s[r]) for r in solution]
        # Verify length
        if len(result_blocks) == 132:
            # Optional: Verify validity just in case
            seen_5 = set()
            valid = True
            for b in result_blocks:
                for s in itertools.combinations(b, 5):
                    if s in seen_5:
                        valid = False
                        break
                    seen_5.add(s)
                if not valid:
                    break
            if valid:
                return result_blocks
    
    # Fallback: If for some reason the solver fails or times out, 
    # return a greedy approximation to ensure a valid (though suboptimal) result.
    # This branch is highly unlikely for this specific problem instance.
    if time.time() > deadline - 5:
        return []
        
    # Simple greedy fallback
    all_blocks = all_6s[:]
    import random
    random.seed(42)
    random.shuffle(all_blocks)
    
    covered = set()
    blocks = []
    block_5subs_cache = {}
    
    for b in all_blocks:
        if b not in block_5subs_cache:
            block_5subs_cache[b] = list(itertools.combinations(b, 5))
        subs = block_5subs_cache[b]
        if any(s in covered for s in subs):
            continue
        blocks.append(list(b))
        covered.update(subs)
        
    return blocks

if __name__ == "__main__":
    blocks = construct()
    print(f"Found {len(blocks)} blocks")
    if len(blocks) == 132:
        seen = set()
        valid = True
        for b in blocks:
            for s in itertools.combinations(b, 5):
                if s in seen:
                    valid = False
                    break
                seen.add(s)
            if not valid:
                break
        print(f"Valid: {valid}, Total 5-subsets: {len(seen)}")
    else:
        print("Did not find optimal 132 blocks")

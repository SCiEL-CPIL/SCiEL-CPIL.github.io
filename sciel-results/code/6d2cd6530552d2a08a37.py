import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    DEADLINE = 110.0  # Leave 10s margin
    
    n_points = 12
    block_size = 6
    subset_size = 5
    
    # Generate all blocks (6-subsets)
    all_blocks = list(combinations(range(n_points), block_size))
    # Generate all constraints (5-subsets)
    all_constraints = list(combinations(range(n_points), subset_size))
    
    # Map constraint to index
    constraint_to_idx = {c: i for i, c in enumerate(all_constraints)}
    
    # Build the matrix: for each block, which constraints does it cover?
    # A block of size 6 contains C(6,5)=6 subsets of size 5.
    rows = []
    for b in all_blocks:
        covered_constraints = []
        for s in combinations(b, subset_size):
            covered_constraints.append(constraint_to_idx[s])
        rows.append(covered_constraints)
    
    num_rows = len(rows)
    num_cols = len(all_constraints)
    
    # DLX Implementation
    # Node structure: [left, right, up, down, col, row_idx, value]
    # We use arrays for speed.
    # Max nodes: num_rows * max_row_width + num_cols + 1
    max_width = max(len(r) for r in rows)
    max_nodes = num_rows * max_width + num_cols + 1
    
    # Initialize links
    # 0 is the header node
    # We'll create nodes dynamically or pre-allocate. Pre-allocating is faster.
    # Let's use a simple list of dicts or parallel lists.
    # Parallel lists are faster in Python.
    
    # Actually, for this small size, a simple recursive backtracking with bitsets or sets might be simpler and fast enough.
    # Let's try a simple backtracking with "Minimum Remaining Values" (MRV) heuristic.
    
    # State:
    # covered: set of 5-subsets already covered
    # blocks: list of chosen blocks
    
    # To make it fast, we represent 5-subsets as integers or use a boolean array.
    # 792 constraints. We can use a set of integers.
    
    # Precompute for each block (index) the list of constraint indices it covers.
    block_covered = []
    for r in rows:
        block_covered.append(r)
        
    # Sort blocks? Maybe by something? Not necessary if we use MRV on constraints.
    # But we pick a constraint, then pick a block covering it.
    
    # Let's use the DLX approach as it is robust.
    
    # DLX Node Representation:
    # We will use a class or simple lists.
    
    class DLXNode:
        __slots__ = ('L', 'R', 'U', 'D', 'C', 'row')
        def __init__(self, header=None):
            self.L = self
            self.R = self
            self.U = self
            self.D = self
            self.C = None
            self.row = -1
            if header:
                self.C = header
                header.R.L = self
                self.R = header.R
                header.R = self

    # Build DLX
    # Header node
    H = DLXNode()
    
    # Column headers
    col_headers = []
    for c in range(num_cols):
        ch = DLXNode(H)
        ch.C = ch
        col_headers.append(ch)
        
    # Row nodes
    # We need to link row nodes horizontally and vertically.
    # For each row, create nodes for each column in that row.
    
    # To speed up, we can just use a standard Python set-based backtracking if DLX overhead is too high in pure Python.
    # Let's estimate: 924 rows, 792 cols.
    # Backtracking with MRV:
    # 1. Find constraint with fewest uncovered blocks.
    # 2. Try each block covering it.
    
    # Let's implement the backtracking version first, it's less code and likely fast enough for n=12.
    
    covered = [False] * num_cols
    solution = []
    
    # Precompute: for each constraint, which blocks cover it?
    constraint_to_blocks = [[] for _ in range(num_cols)]
    for r_idx, r in enumerate(rows):
        for c_idx in r:
            constraint_to_blocks[c_idx].append(r_idx)
            
    # Heuristic: Choose the constraint with the minimum number of available blocks.
    # To avoid scanning all 792 constraints at each step, we can maintain counts, 
    # but with only 132 steps, scanning 792 * 132 ~ 100k operations is trivial.
    
    def solve(depth):
        if time.time() - start_time > DEADLINE:
            return False
            
        if depth == 132:
            return True
            
        # Find constraint with min available blocks
        min_count = float('inf')
        min_constraint = -1
        
        for c in range(num_cols):
            if not covered[c]:
                # Count available blocks
                count = 0
                for b_idx in constraint_to_blocks[c]:
                    # Check if block b_idx is still valid (all its constraints are uncovered)
                    # This check is expensive: O(6) per block.
                    # Total: 792 * (avg blocks per constraint) * 6.
                    # Avg blocks per constraint: 924 * 6 / 792 = 7.
                    # 792 * 7 * 6 ~ 33k ops per step.
                    # 132 steps * 33k ~ 4M ops. Very fast.
                    valid = True
                    for cc in block_covered[b_idx]:
                        if covered[cc]:
                            valid = False
                            break
                    if valid:
                        count += 1
                        if count < min_count:
                            min_count = count
                            min_constraint = c
                            if min_count == 1:
                                break # Can't do better than 1
                if min_count == 1:
                    break
                if count < min_count:
                    min_count = count
                    min_constraint = c
                    if min_count == 1:
                        break
                
        if min_constraint == -1:
            # No uncovered constraints? Should not happen if depth < 132
            return False
            
        # Try all blocks covering min_constraint
        for b_idx in constraint_to_blocks[min_constraint]:
            # Check validity
            valid = True
            for cc in block_covered[b_idx]:
                if covered[cc]:
                    valid = False
                    break
            if not valid:
                continue
                
            # Choose this block
            solution.append(b_idx)
            for cc in block_covered[b_idx]:
                covered[cc] = True
                
            if solve(depth + 1):
                return True
                
            # Undo
            solution.pop()
            for cc in block_covered[b_idx]:
                covered[cc] = False
                
        return False

    # Start solving
    if solve(0):
        # Convert indices to actual subsets
        return [list(all_blocks[i]) for i in solution]
    else:
        # Fallback: return whatever we have or empty? 
        # The task says "Maximize". If we fail, we return best effort.
        # But 132 is always found.
        return [list(all_blocks[i]) for i in solution]

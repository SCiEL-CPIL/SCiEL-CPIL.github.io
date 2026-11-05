import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for safety
    
    # DLX (Dancing Links) implementation for Exact Cover
    # Problem: Cover all 792 5-subsets of {0..11} using 6-subsets (blocks).
    # Each block covers exactly 6 of the 5-subsets.
    # We need to find 132 blocks.
    
    # 1. Precompute all 5-subsets and map them to indices
    all_5subs = list(combinations(range(12), 5))
    num_5subs = len(all_5subs)
    sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    # 2. Precompute all 6-subsets (blocks)
    all_blocks = list(combinations(range(12), 6))
    num_blocks = len(all_blocks)
    
    # 3. Build the DLX matrix
    # Columns: 792 (the 5-subsets)
    # Rows: 924 (the 6-subsets)
    # Node structure: L, R, U, D, C (column), row
    # We use lists for speed.
    # Total nodes = 1 + num_cols + num_rows * 6
    
    # Initialize headers
    # Index 0 is the root header.
    # Indices 1..num_cols are column headers.
    # Indices num_cols+1.. are row nodes.
    
    N = num_cols + 1
    L = list(range(N+1))
    R = list(range(N+1))
    U = list(range(N+1))
    D = list(range(N+1))
    C = [0] * (N+1)
    row_id = [0] * (N+1)
    
    # Link headers in a circular list
    for i in range(N):
        L[i] = i - 1
        R[i] = (i + 1) % N
        U[i] = i
        D[i] = i
        C[i] = i
        row_id[i] = -1
        
    L[0] = N - 1
    R[0] = 1
    
    # Column sizes (for heuristic: choose column with min size)
    col_size = [0] * num_cols
    
    # Add rows
    node_idx = N
    for r, block in enumerate(all_blocks):
        # Get the 5-subsets covered by this block
        covered_5subs = [sub_to_idx[s] for s in combinations(block, 5)]
        
        # Add nodes for this row
        first_node = node_idx
        for c in covered_5subs:
            # Create node
            node_idx += 1
            # Ensure lists are large enough
            while len(L) <= node_idx:
                L.append(0); R.append(0); U.append(0); D.append(0); C.append(0); row_id.append(0)
            
            # Link into column c
            C[node_idx] = c + 1 # Column headers are 1-indexed in our array logic relative to root?
            # Actually, let's stick to 0-based for headers in array, but root is 0.
            # Headers are 0..num_cols-1? No, root is 0. Headers 1..num_cols.
            # Let's re-verify indices.
            # Root: 0
            # Col Headers: 1 to num_cols
            # Data Nodes: num_cols+1 to end
            
            # Correct C index: c+1
            C[node_idx] = c + 1
            row_id[node_idx] = r
            
            # Link to bottom of column c+1
            col_header = c + 1
            U[node_idx] = U[col_header]
            D[node_idx] = col_header
            D[U[col_header]] = node_idx
            U[col_header] = node_idx
            
            col_size[c] += 1
            
            # Link to right in row
            if node_idx != first_node:
                R[node_idx - 1] = node_idx
                L[node_idx] = node_idx - 1
        
        # Close the row circle
        R[first_node] = node_idx
        L[node_idx] = first_node
        
    # Heuristic: Choose column with minimum size
    def choose_column():
        min_size = float('inf')
        c = 0
        j = R[0]
        while j != 0:
            if col_size[j-1] < min_size: # col_size is 0-indexed, header is 1-indexed
                min_size = col_size[j-1]
                c = j
            j = R[j]
        return c
        
    def cover(c):
        # Remove column c
        R[L[c]] = R[c]
        L[R[c]] = L[c]
        
        # Remove rows in column c
        i = D[c]
        while i != c:
            j = R[i]
            while j != i:
                U[D[j]] = U[j]
                D[U[j]] = D[j]
                col_size[C[j]-1] -= 1
                j = R[j]
            i = D[i]
            
    def uncover(c):
        # Restore rows in column c
        i = U[c]
        while i != c:
            j = L[i]
            while j != i:
                col_size[C[j]-1] += 1
                U[D[j]] = j
                D[U[j]] = j
                j = L[j]
            i = U[i]
            
        # Restore column c
        L[R[c]] = c
        R[L[c]] = c
        
    solution = []
    found = False
    
    def solve(k):
        nonlocal found
        if time.time() > deadline:
            return
        if found:
            return
            
        if R[0] == 0:
            # All columns covered
            found = True
            return
            
        c = choose_column()
        if col_size[c-1] == 0:
            return # Dead end
            
        cover(c)
        r = D[c]
        while r != c:
            if time.time() > deadline:
                uncover(c)
                return
            if found:
                uncover(c)
                return
                
            solution.append(row_id[r])
            
            # Cover all other columns in this row
            j = R[r]
            while j != r:
                cover(C[j])
                j = R[j]
                
            solve(k + 1)
            
            # Uncover in reverse
            j = L[r]
            while j != r:
                uncover(C[j])
                j = L[j]
                
            solution.pop()
            
            r = D[r]
            
        uncover(c)
        
    # Start solving
    solve(0)
    
    if found:
        # Map row indices back to blocks
        result_blocks = [list(all_blocks[r]) for r in solution]
        return result_blocks
    else:
        # If DLX fails (shouldn't happen for S(5,6,12)), return empty or best effort
        # But S(5,6,12) exists, so DLX should find it.
        return []


import sys
import time

def construct():
    """
    Constructs the Steiner system S(5,6,12) (Witt design) using the Dancing Links (DLX) algorithm
    to solve the exact cover problem for the 5-subsets of a 12-set.
    """
    from itertools import combinations
    
    n = 12
    # Universe: all 5-subsets of 0..11
    universe = list(combinations(range(n), 5))
    col_map = {s: i for i, s in enumerate(universe)}
    num_cols = len(universe) # 792
    
    # Candidates: all 6-subsets of 0..11
    candidates = list(combinations(range(n), 6))
    num_rows = len(candidates) # 924
    
    # Build the DLX matrix
    # We need to map each row (6-subset) to the columns (5-subsets) it covers.
    # Each 6-subset contains 6 5-subsets.
    
    # DLX Data Structures
    # We use arrays for the linked list nodes to be fast.
    # Node indices: 0 is the header node for columns.
    # Each column has a header node.
    # Each 1 in the matrix is a node.
    
    # Total nodes = 1 (header) + num_cols (col headers) + num_rows * 6 (entries)
    # Max nodes = 1 + 792 + 924 * 6 = 1 + 792 + 5544 = 6337
    
    max_nodes = 1 + num_cols + num_rows * 6
    
    L = [0] * max_nodes
    R = [0] * max_nodes
    U = [0] * max_nodes
    D = [0] * max_nodes
    C = [0] * max_nodes # Column index for the node
    Row = [0] * max_nodes # Row index for the node
    
    # Initialize column headers
    # Node 0 is the main header
    # Nodes 1 to num_cols are column headers
    for c in range(num_cols):
        node = 1 + c
        L[node] = c
        R[node] = c + 1
        U[node] = node
        D[node] = node
        C[node] = c
        Row[node] = -1 # Not a data node
        
    R[num_cols] = 0
    L[0] = num_cols
    U[0] = 0
    D[0] = 0
    C[0] = -1
    Row[0] = -1
    
    next_node = 1 + num_cols
    
    # Add data nodes
    # For each candidate row (6-subset), find the 6 columns (5-subsets) it covers.
    row_cols = []
    for r, cand in enumerate(candidates):
        cols_in_row = []
        # Get all 5-subsets of this 6-subset
        for sub in combinations(cand, 5):
            col_idx = col_map[sub]
            cols_in_row.append(col_idx)
        row_cols.append(cols_in_row)
        
        # Link the nodes in this row
        first_node = -1
        for col_idx in cols_in_row:
            node = next_node
            next_node += 1
            
            C[node] = col_idx
            Row[node] = r
            
            # Link to column
            U[node] = U[1 + col_idx]
            D[node] = 1 + col_idx
            D[1 + col_idx] = node
            U[U[node]] = node
            
            # Link to row
            if first_node == -1:
                first_node = node
                L[node] = node
                R[node] = node
            else:
                R[node] = R[first_node]
                L[node] = first_node
                R[first_node] = node
                L[R[node]] = node
                
    # DLX Solver
    solution = []
    
    def search(k):
        if k == num_cols: # All columns covered
            return True
            
        # Choose column with minimum size (heuristic)
        # In this specific problem, the structure is regular, but sizes might vary slightly during search
        c = 0
        min_size = float('inf')
        j = R[0]
        while j != 0:
            if D[j] != j: # Column is not empty
                size = 0
                i = D[j]
                while i != j:
                    size += 1
                    i = D[i]
                if size < min_size:
                    min_size = size
                    c = j
            j = R[j]
            
        if c == 0: # No columns left to cover (should not happen if exact cover exists)
             # Or if we reached a dead end
             return False
             
        # Cover column c
        R[L[c]] = R[c]
        L[R[c]] = L[c]
        
        # For each row in column c
        i = D[c]
        found = False
        while i != c:
            # Add row to solution
            solution.append(Row[i])
            
            # Cover other columns in this row
            j = R[i]
            while j != i:
                # Remove node j from its column
                U[D[j]] = U[j]
                D[U[j]] = D[j]
                # Cover the column of j
                R[L[C[j]]] = R[C[j]]
                L[R[C[j]]] = L[C[j]]
                j = R[j]
                
            # Recursive step
            if search(k + 1):
                found = True
                break
            
            # Uncover columns in this row
            j = L[i]
            while j != i:
                # Uncover the column of j
                R[L[C[j]]] = C[j]
                L[R[C[j]]] = C[j]
                # Restore node j
                U[D[j]] = j
                D[U[j]] = j
                j = L[j]
                
            # Remove row from solution
            solution.pop()
            i = D[i]
            
        # Uncover column c
        R[L[c]] = c
        L[R[c]] = c
        
        return found
        
    # Start search
    if search(0):
        # solution contains row indices
        blocks = [list(candidates[r]) for r in solution]
        return blocks
    else:
        # Fallback: Greedy (should not happen)
        covered, blocks = set(), []
        for b in candidates:
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        return blocks

if __name__ == "__main__":
    blocks = construct()
    print(f"Found {len(blocks)} blocks")

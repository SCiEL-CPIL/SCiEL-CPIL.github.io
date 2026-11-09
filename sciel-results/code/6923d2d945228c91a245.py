import random
import time

def construct():
    n = 10
    start_time = time.time()
    timeout = 55.0 # Leave some buffer
    
    # Generate first square (cyclic)
    squares = []
    sq0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares.append(sq0)
    
    # Try to find more squares
    # Max MOLS for n=10 is 4.
    # We will attempt to find up to 3 more.
    
    while len(squares) < 4:
        if time.time() - start_time > timeout:
            break
        
        # Try to find the next square orthogonal to all existing ones
        # Use randomized restarts
        found = False
        # Increase attempts if we have fewer squares
        max_attempts = 50 if len(squares) == 1 else 20
        
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout:
                break
            
            next_sq = find_next_square_cspsolver(n, squares, attempt)
            if next_sq is not None:
                squares.append(next_sq)
                found = True
                break
        
        if not found:
            break
            
    return squares

def find_next_square_cspsolver(n, existing_squares, seed):
    """
    Finds a Latin square orthogonal to existing_squares using CSP with MRV and Forward Checking.
    """
    random.seed(seed)
    
    grid = [[-1 for _ in range(n)] for _ in range(n)]
    
    # Domains: list of lists of valid values for each cell
    domains = [[list(range(n)) for _ in range(n)] for _ in range(n)]
    
    # Used pairs for orthogonality
    used_pairs = [set() for _ in existing_squares]
    
    # Pre-compute groups for each existing square
    # groups[k][v] = list of (i, j) such that existing_squares[k][i][j] == v
    groups = []
    for k in range(len(existing_squares)):
        sq_k = existing_squares[k]
        g = {}
        for i in range(n):
            for j in range(n):
                v = sq_k[i][j]
                if v not in g:
                    g[v] = []
                g[v].append((i, j))
        groups.append(g)
    
    def backtrack(assigned_count):
        if assigned_count == n * n:
            return True
        
        # MRV: Find cell with smallest domain
        min_size = n + 1
        min_cell = None
        
        for i in range(n):
            for j in range(n):
                if grid[i][j] == -1:
                    size = len(domains[i][j])
                    if size < min_size:
                        min_size = size
                        min_cell = (i, j)
                        if size == 0:
                            return False
                        if size == 1:
                            # Can't get smaller than 1 (unless 0)
                            # We continue to check for 0
                            pass
        
        if min_cell is None:
            return True # All assigned
        
        i, j = min_cell
        vals = domains[i][j][:]
        random.shuffle(vals)
        
        for val in vals:
            # Check orthogonality
            valid = True
            for k, sq_k in enumerate(existing_squares):
                v_k = sq_k[i][j]
                pair = (val, v_k)
                if pair in used_pairs[k]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Assign
            grid[i][j] = val
            
            # Record used pairs
            pairs_to_remove = []
            for k, sq_k in enumerate(existing_squares):
                v_k = sq_k[i][j]
                pair = (val, v_k)
                used_pairs[k].add(pair)
                pairs_to_remove.append((k, pair))
            
            # Propagate constraints
            changed_cells = []
            
            # 1. Row constraint
            for c in range(n):
                if c != j and val in domains[i][c]:
                    domains[i][c].remove(val)
                    changed_cells.append((i, c))
            
            # 2. Col constraint
            for r in range(n):
                if r != i and val in domains[r][j]:
                    domains[r][j].remove(val)
                    changed_cells.append((r, j))
            
            # 3. Orthogonality constraint
            for k, sq_k in enumerate(existing_squares):
                v_k = sq_k[i][j]
                cells_with_val = groups[k][v_k]
                for (r2, c2) in cells_with_val:
                    if (r2, c2) != (i, j):
                        if val in domains[r2][c2]:
                            domains[r2][c2].remove(val)
                            changed_cells.append((r2, c2))
            
            # Check for domain wipeout
            wipeout = False
            for (r, c) in changed_cells:
                if len(domains[r][c]) == 0:
                    wipeout = True
                    break
            
            if not wipeout:
                if backtrack(assigned_count + 1):
                    return True
            
            # Undo
            grid[i][j] = -1
            for k, pair in pairs_to_remove:
                used_pairs[k].remove(pair)
            
            # Restore domains
            for (r, c) in changed_cells:
                if val not in domains[r][c]:
                    domains[r][c].append(val)
                
        return False
    
    result = backtrack(0)
    if result:
        return grid
    return None

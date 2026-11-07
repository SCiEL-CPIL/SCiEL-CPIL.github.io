import time

def construct():
    n = 10
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # Construct two orthogonal Latin squares of order 10
    # L1(i,j) = (i + j) % 10
    # L2(i,j) = (i + 2*j) % 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]

    # Verification functions
    def is_latin(sq):
        for row in sq:
            if sorted(row) != list(range(n)):
                return False
        for col in zip(*sq):
            if sorted(col) != list(range(n)):
                return False
        return True

    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return len(pairs) == n * n

    # Check validity of initial pair
    if not is_latin(L1) or not is_latin(L2) or not are_orthogonal(L1, L2):
        return [L1]

    squares = [L1, L2]

    # Search for a third orthogonal Latin square
    # Using backtracking with MRV heuristic and forward checking
    
    # Precompute the groups for L1 and L2
    # group_l1[s] = list of (i, j) where L1[i][j] == s
    # group_l2[s] = list of (i, j) where L2[i][j] == s
    group_l1 = {s: [] for s in range(n)}
    group_l2 = {s: [] for s in range(n)}
    for i in range(n):
        for j in range(n):
            group_l1[L1[i][j]].append((i, j))
            group_l2[L2[i][j]].append((i, j))

    # State for backtracking
    # L3 is a 10x10 grid, initially -1
    L3 = [[-1 for _ in range(n)] for _ in range(n)]
    
    # Track used symbols in each constraint
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    l1_used = [set() for _ in range(n)]
    l2_used = [set() for _ in range(n)]

    def get_domains():
        # Calculate the domain size for each unassigned cell
        # This is used for MRV heuristic
        domains = {}
        for i in range(n):
            for j in range(n):
                if L3[i][j] != -1:
                    continue
                domain = []
                for s in range(n):
                    if s in row_used[i]: continue
                    if s in col_used[j]: continue
                    if s in l1_used[L1[i][j]]: continue
                    if s in l2_used[L2[i][j]]: continue
                    domain.append(s)
                domains[(i, j)] = domain
        return domains

    def backtrack():
        if time.time() > deadline:
            return False
            
        # Find the unassigned cell with the smallest domain (MRV)
        best_cell = None
        best_domain = None
        min_size = n + 1
        
        for i in range(n):
            for j in range(n):
                if L3[i][j] != -1:
                    continue
                # Check constraints
                domain = []
                for s in range(n):
                    if s in row_used[i]: continue
                    if s in col_used[j]: continue
                    if s in l1_used[L1[i][j]]: continue
                    if s in l2_used[L2[i][j]]: continue
                    domain.append(s)
                
                if len(domain) < min_size:
                    min_size = len(domain)
                    best_cell = (i, j)
                    best_domain = domain
                    
                    if min_size == 0:
                        return False
                    if min_size == 1:
                        break
            if min_size == 1:
                break
                
        if best_cell is None:
            # All cells assigned
            return True
            
        i, j = best_cell
        
        for s in best_domain:
            # Assign
            L3[i][j] = s
            row_used[i].add(s)
            col_used[j].add(s)
            l1_used[L1[i][j]].add(s)
            l2_used[L2[i][j]].add(s)
            
            if backtrack():
                return True
            
            # Undo
            L3[i][j] = -1
            row_used[i].remove(s)
            col_used[j].remove(s)
            l1_used[L1[i][j]].remove(s)
            l2_used[L2[i][j]].remove(s)
            
        return False

    # Run the search
    if backtrack():
        # Verify the third square
        if is_latin(L3) and are_orthogonal(L1, L3) and are_orthogonal(L2, L3):
            squares.append(L3)
        else:
            # Should not happen if backtrack is correct, but just in case
            pass

    # Convert to ensure all values are Python ints
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])

    return result

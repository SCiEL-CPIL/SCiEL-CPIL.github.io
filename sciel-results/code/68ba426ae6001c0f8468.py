"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 10}

def construct():
    n = P['n']
    deadline = time.time() + 80.0  # 80s safety margin
    
    # We know at most 2 MOLS of order 10 exist.
    # Use backtracking to find 2 MOLS.
    
    # Start with the cyclic square as base
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find a second MOLS orthogonal to base using backtracking
    # The second square must satisfy: for each cell (i,j), pair (base[i][j], second[i][j]) is unique
    
    # Use constraint propagation: for the second square, each row and column must be a permutation of 0..9
    # and the pairs with base must all be distinct
    
    def try_find_second(base_sq, max_time):
        """Find a Latin square orthogonal to base_sq using backtracking."""
        n = len(base_sq)
        # For each position (i,j), the value v must satisfy:
        # 1. v not in row i of second
        # 2. v not in col j of second  
        # 3. pair (base_sq[i][j], v) not already used
        
        # Precompute: for each base value b, how many times it appears (should be n times each)
        # The orthogonality constraint: all 100 pairs must be distinct
        
        second = [[-1]*n for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair_used = set()
        
        # Order cells by most constrained first (MRV heuristic)
        # Initially all have 10 options, so use row-major order
        cells = [(i, j) for i in range(n) for j in range(n)]
        
        # Try different orderings for better pruning
        random.shuffle(cells)
        
        def backtrack(idx):
            if time.time() > max_time:
                return False
            if idx == n*n:
                return True
            
            i, j = cells[idx]
            bval = base_sq[i][j]
            
            # Find available values
            available = set(range(n)) - row_used[i] - col_used[j]
            
            # Further filter by pair constraint
            for v in list(available):
                if (bval, v) in pair_used:
                    available.discard(v)
            
            if not available:
                return False
            
            # Try values in random order for diversity
            vals = list(available)
            random.shuffle(vals)
            
            for v in vals:
                second[i][j] = v
                row_used[i].add(v)
                col_used[j].add(v)
                pair_used.add((bval, v))
                
                if backtrack(idx + 1):
                    return True
                
                second[i][j] = -1
                row_used[i].discard(v)
                col_used[j].discard(v)
                pair_used.discard((bval, v))
            
            return False
        
        return backtrack(0), second
    
    # Try to find a second MOLS
    success, second = try_find_second(base, deadline)
    
    if success:
        # Verify orthogonality
        pairs = set()
        for i in range(n):
            for j in range(n):
                pairs.add((base[i][j], second[i][j]))
        if len(pairs) == n*n:
            # Convert to proper Python int lists
            result = []
            for sq in [base, second]:
                result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
            return result
    
    # Fallback: return just the base square
    return [[[int(base[i][j]) for j in range(n)] for i in range(n)]]
# EVOLVE-BLOCK-END

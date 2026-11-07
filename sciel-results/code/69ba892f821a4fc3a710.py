import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    n = 10
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # Construct two orthogonal Latin squares of order 10
    # Using linear forms over Z_10
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

    # For order 10, it is a well-established result that no three mutually
    # orthogonal Latin squares exist. We attempt a bounded search for a third
    # square using backtracking with MRV heuristic and forward checking, but
    # expect it to fail quickly. If it succeeds, we include it.

    def find_third_square(existing_squares):
        """Find a Latin square orthogonal to all existing squares using
        backtracking with MRV and forward checking."""
        
        # Precompute for each cell (i,j) the set of allowed values based on
        # orthogonality constraints with each existing square.
        # For orthogonality with square sq: at positions where sq has value v,
        # the new square must use each symbol exactly once.
        
        # Build constraint: for each existing square, for each value v,
        # the set of positions where sq[i][j] == v
        pos_by_val = []
        for sq in existing_squares:
            pbyv = defaultdict(set)
            for i in range(n):
                for j in range(n):
                    pbyv[sq[i][j]].add((i, j))
            pos_by_val.append(pbyv)
        
        # State: grid, row_avail, col_avail, ortho_used
        grid = [[-1] * n for _ in range(n)]
        row_avail = [set(range(n)) for _ in range(n)]
        col_avail = [set(range(n)) for _ in range(n)]
        ortho_used = [set() for _ in existing_squares]
        
        # Count nodes visited
        nodes = [0]
        
        def get_candidates(i, j):
            """Get candidate values for cell (i,j)."""
            cands = row_avail[i] & col_avail[j]
            for k, sq in enumerate(existing_squares):
                ex_val = sq[i][j]
                cands = cands - {v for (ev, v) in ortho_used[k] if ev == ex_val}
            return cands
        
        def backtrack():
            if time.time() - start_time > deadline - 5:
                return False
            
            nodes[0] += 1
            
            # Find unassigned cell with fewest candidates (MRV)
            best_cell = None
            best_cands = None
            best_count = n + 1
            
            for i in range(n):
                for j in range(n):
                    if grid[i][j] == -1:
                        cands = get_candidates(i, j)
                        cnt = len(cands)
                        if cnt < best_count:
                            best_count = cnt
                            best_cell = (i, j)
                            best_cands = cands
                            if cnt == 0:
                                return False  # Dead end
                            if cnt == 1:
                                break
                if best_count == 1:
                    break
            
            if best_cell is None:
                return True  # All cells assigned
            
            i, j = best_cell
            
            # Forward checking: try each candidate
            for val in sorted(best_cands):
                # Place
                grid[i][j] = val
                row_avail[i].discard(val)
                col_avail[j].discard(val)
                for k, sq in enumerate(existing_squares):
                    ex_val = sq[i][j]
                    ortho_used[k].add((ex_val, val))
                
                # Check for immediate dead ends (forward checking)
                valid = True
                for ii in range(n):
                    for jj in range(n):
                        if grid[ii][jj] == -1:
                            if len(get_candidates(ii, jj)) == 0:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid and backtrack():
                    return True
                
                # Unplace
                grid[i][j] = -1
                row_avail[i].add(val)
                col_avail[j].add(val)
                for k, sq in enumerate(existing_squares):
                    ex_val = sq[i][j]
                    ortho_used[k].discard((ex_val, val))
            
            return False
        
        if backtrack():
            return grid
        return None

    # Try to find a third square with a tight time budget
    if time.time() - start_time < deadline - 10:
        third = find_third_square(squares)
        if third is not None:
            if is_latin(third):
                valid = True
                for sq in squares:
                    if not are_orthogonal(third, sq):
                        valid = False
                        break
                if valid:
                    squares.append(third)

    # Convert to ensure all values are Python ints
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])

    return result
# EVOLVE-BLOCK-END

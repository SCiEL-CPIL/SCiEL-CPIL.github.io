import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 10
    symbols = list(range(n))
    
    # 1. Generate the first Latin Square (Cyclic)
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [L1]
    
    # Check if we have time to search for a second square
    if time.time() > deadline - 5.0:
        return squares

    # 2. Search for a second Latin Square L2 orthogonal to L1
    # We use backtracking with MRV (Minimum Remaining Values) heuristic.
    
    # Pre-compute the forbidden values for each cell based on L1
    # For orthogonality, if L1[r][c] = v1, and we place v2 in L2[r][c],
    # the pair (v1, v2) must not have appeared before.
    # Since L1 is fixed, we track which v2 values are "used" for each v1 value
    # across the whole grid? No, orthogonality is global: every pair (a,b) appears exactly once.
    # This is equivalent to: for each value a in L1, the values in L2 at positions where L1=a must be a permutation of 0..9.
    # And for each value b in L2, the values in L1 at positions where L2=b must be a permutation of 0..9.
    # Actually, the standard definition is that the set of ordered pairs {(L1[r][c], L2[r][c])} is all distinct.
    
    # Let's define the constraints more simply for the backtracker:
    # 1. L2 must be a Latin Square.
    #    - Each row is a permutation of 0..9.
    #    - Each col is a permutation of 0..9.
    # 2. Orthogonality with L1.
    #    - For any two cells (r1,c1) and (r2,c2), if L1[r1][c1] == L1[r2][c2] and L2[r1][c1] == L2[r2][c2], then (r1,c1) == (r2,c2).
    #    - Equivalently, for each symbol s in 0..9, let Pos_s = {(r,c) | L1[r][c] == s}.
    #      The values L2[r][c] for (r,c) in Pos_s must be all distinct.
    #    - This means that if we look at the subgrid of L2 formed by the positions where L1 is s, it must contain each symbol 0..9 exactly once.
    
    # Let's implement a backtracker that fills L2 cell by cell.
    
    L2 = [[None for _ in range(n)] for _ in range(n)]
    
    # Candidate sets
    row_cands = [set(symbols) for _ in range(n)]
    col_cands = [set(symbols) for _ in range(n)]
    
    # For orthogonality:
    # For each symbol s in L1, we need to track which symbols in L2 are already used in the cells where L1 is s.
    # ortho_cands[s] = set of symbols in L2 that are still available for cells where L1 == s.
    ortho_cands = [set(symbols) for _ in range(n)]
    
    # Initialize ortho_cands? No, they start as full sets.
    # But we can pre-calculate the positions for each L1 symbol to help MRV?
    # Or just check dynamically.
    
    # To speed up MRV, we can pre-calculate the list of cells for each L1 symbol.
    l1_pos = {s: [] for s in range(n)}
    for r in range(n):
        for c in range(n):
            l1_pos[L1[r][c]].append((r, c))
            
    # We fill cells in a specific order. MRV suggests picking the cell with the fewest candidates.
    # Candidates for (r,c) = row_cands[r] & col_cands[c] & ortho_cands[L1[r][c]]
    
    # To make MRV efficient, we can maintain a priority queue or just scan for the min each time.
    # Since n=10, scanning 100 cells is cheap.
    
    def get_candidates(r, c):
        return row_cands[r] & col_cands[c] & ortho_cands[L1[r][c]]
        
    def is_consistent():
        # Check if any candidate set is empty
        for r in range(n):
            if not row_cands[r]:
                return False
            if not col_cands[r]: # Wait, col_cands is indexed by c
                pass
        for c in range(n):
            if not col_cands[c]:
                return False
        for s in range(n):
            if not ortho_cands[s]:
                return False
        return True
        
    # We need a way to undo assignments.
    # We will use a stack of changes.
    
    def backtrack():
        if time.time() > deadline:
            return False
            
        # Find the empty cell with the minimum number of candidates (MRV)
        best_r, best_c = -1, -1
        min_cands = n + 1
        best_cand_set = None
        
        for r in range(n):
            for c in range(n):
                if L2[r][c] is None:
                    cands = get_candidates(r, c)
                    if len(cands) < min_cands:
                        min_cands = len(cands)
                        best_r, best_c = r, c
                        best_cand_set = cands
                        if min_cands == 1:
                            break
            if min_cands == 1:
                break
                
        if best_r == -1:
            # No empty cells left, we found a solution!
            return True
            
        if min_cands == 0:
            return False
            
        # Try each candidate in the best cell
        # Randomize the order to avoid deterministic failures or speed up finding *a* solution?
        # For MOLS, any solution is valid. Randomization might help escape local minima if we were doing local search,
        # but for backtracking, it just changes the search tree.
        # Let's iterate in sorted order for determinism, or random for variety.
        # Given the small size, sorted is fine.
        
        for v in sorted(best_cand_set):
            # Assign v to L2[best_r][best_c]
            L2[best_r][best_c] = v
            
            # Remove v from constraints
            row_cands[best_r].remove(v)
            col_cands[best_c].remove(v)
            ortho_cands[L1[best_r][best_c]].remove(v)
            
            # Forward checking: Ensure no other cell has 0 candidates
            consistent = True
            for r in range(n):
                if not row_cands[r]:
                    consistent = False
                    break
            if consistent:
                for c in range(n):
                    if not col_cands[c]:
                        consistent = False
                        break
            if consistent:
                for s in range(n):
                    if not ortho_cands[s]:
                        consistent = False
                        break
                        
            if consistent:
                if backtrack():
                    return True
                    
            # Undo assignment
            L2[best_r][best_c] = None
            row_cands[best_r].add(v)
            col_cands[best_c].add(v)
            ortho_cands[L1[best_r][best_c]].add(v)
            
        return False

    # Run the backtracker
    if backtrack():
        # Verify the solution just in case
        # Check Latin property
        valid = True
        for r in range(n):
            if sorted(L2[r]) != symbols:
                valid = False
                break
        if valid:
            for c in range(n):
                col = [L2[r][c] for r in range(n)]
                if sorted(col) != symbols:
                    valid = False
                    break
        if valid:
            # Check Orthogonality
            seen = set()
            for r in range(n):
                for c in range(n):
                    p = (L1[r][c], L2[r][c])
                    if p in seen:
                        valid = False
                        break
                    seen.add(p)
                if not valid:
                    break
            if valid:
                squares.append(L2)
        
    # Return the result
    # Ensure types are correct (Python ints and lists)
    # L1 and L2 are already lists of lists of ints.
    
    return squares

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    # Deadline guard: 60s limit, keep 5s buffer
    deadline = time.time() + 55.0
    n = 10
    
    # Base square: addition table mod 10
    # This is a valid Latin square
    A0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [A0]
    
    def find_second_square():
        """
        Attempts to find a Latin square A1 orthogonal to A0.
        Uses backtracking with MRV heuristic and constraint propagation.
        Returns None if not found within time limit.
        """
        grid = [[-1] * n for _ in range(n)]
        row_masks = [0] * n
        col_masks = [0] * n
        pair_used = [[False] * n for _ in range(n)]
        
        # Normalization: Fix first row of A1 to be 0..9
        for j in range(n):
            grid[0][j] = j
            row_masks[0] |= (1 << j)
            col_masks[j] |= (1 << j)
            pair_used[j][j] = True
        
        # Track remaining cells
        remaining = [(i, j) for i in range(1, n) for j in range(n)]
        
        def backtrack(remaining_cells):
            if time.time() > deadline:
                return False
            
            if not remaining_cells:
                return True
            
            # MRV heuristic: pick cell with fewest valid values
            best_idx = -1
            best_count = n + 1
            best_candidates = None
            
            for idx, (i, j) in enumerate(remaining_cells):
                count = 0
                candidates = []
                for s in range(n):
                    bit = 1 << s
                    if not (row_masks[i] & bit) and not (col_masks[j] & bit) and not pair_used[A0[i][j]][s]:
                        count += 1
                        candidates.append(s)
                if count < best_count:
                    best_count = count
                    best_idx = idx
                    best_candidates = candidates
                    if count == 1:
                        break
            
            if best_count == 0:
                return False
            
            i, j = remaining_cells[best_idx]
            new_remaining = remaining_cells[:best_idx] + remaining_cells[best_idx+1:]
            
            # Try candidates in order
            for s in best_candidates:
                bit = 1 << s
                grid[i][j] = s
                row_masks[i] |= bit
                col_masks[j] |= bit
                pair_used[A0[i][j]][s] = True
                
                if backtrack(new_remaining):
                    return True
                
                # Backtrack
                grid[i][j] = -1
                row_masks[i] &= ~bit
                col_masks[j] &= ~bit
                pair_used[A0[i][j]][s] = False
            
            return False
        
        if backtrack(remaining):
            return grid
        return None

    # Try to find the second square
    A1 = find_second_square()
    
    if A1 is not None:
        # Verify strictly to avoid invalid submission
        ok = True
        for i in range(n):
            if len(set(A1[i])) != n:
                ok = False
                break
        if ok:
            for j in range(n):
                col = [A1[i][j] for i in range(n)]
                if len(set(col)) != n:
                    ok = False
                    break
        if ok:
            seen_pairs = set()
            for i in range(n):
                for j in range(n):
                    p = (A0[i][j], A1[i][j])
                    if p in seen_pairs:
                        ok = False
                        break
                    seen_pairs.add(p)
                if not ok:
                    break
        if ok:
            squares.append(A1)
            
    return squares

# EVOLVE-BLOCK-END

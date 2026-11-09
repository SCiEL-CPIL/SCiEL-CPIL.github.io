import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 10
    
    # Initial square
    sq1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [sq1]
    
    # Try to find up to 8 more squares
    for _ in range(8):
        if time.time() > deadline:
            break
        next_sq = find_next_square(squares, n, deadline)
        if next_sq is None:
            break
        squares.append(next_sq)
        
    return squares

def find_next_square(existing_squares, n, deadline):
    grid = [[-1 for _ in range(n)] for _ in range(n)]
    row_masks = [0] * n
    col_masks = [0] * n
    forbidden_pairs = [set() for _ in existing_squares]
    
    def backtrack():
        if time.time() > deadline:
            return False
        
        # Find cell with minimum remaining values (MRV)
        min_count = n + 1
        best_r, best_c = -1, -1
        best_candidates = []
        
        for r in range(n):
            for c in range(n):
                if grid[r][c] != -1:
                    continue
                
                candidates = []
                for v in range(n):
                    bit = 1 << v
                    if row_masks[r] & bit or col_masks[c] & bit:
                        continue
                    valid = True
                    for k, sq in enumerate(existing_squares):
                        if (sq[r][c], v) in forbidden_pairs[k]:
                            valid = False
                            break
                    if valid:
                        candidates.append(v)
                
                if len(candidates) < min_count:
                    min_count = len(candidates)
                    best_r, best_c = r, c
                    best_candidates = candidates
                    if min_count == 0:
                        return False
                    if min_count == 1:
                        break # Can't do better than 1
            
            if min_count == 1:
                break
        
        if best_r == -1:
            return True # All cells filled
        
        # Try candidates in best cell
        for v in best_candidates:
            bit = 1 << v
            grid[best_r][best_c] = v
            row_masks[best_r] |= bit
            col_masks[best_c] |= bit
            for k, sq in enumerate(existing_squares):
                forbidden_pairs[k].add((sq[best_r][best_c], v))
            
            if backtrack():
                return True
            
            # Undo
            grid[best_r][best_c] = -1
            row_masks[best_r] &= ~bit
            col_masks[best_c] &= ~bit
            for k, sq in enumerate(existing_squares):
                forbidden_pairs[k].remove((sq[best_r][best_c], v))
                
        return False

    if backtrack():
        return grid
    return None
# EVOLVE-BLOCK-END

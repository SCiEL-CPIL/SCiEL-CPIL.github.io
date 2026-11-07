import sys

def construct():
    n = 10
    # L0: Cyclic Latin Square
    L0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    squares = [L0]
    
    def find_orthogonal_square(existing_squares):
        # current_sq is a list of lists of None or int
        current_sq = [[None for _ in range(n)] for _ in range(n)]
        
        # Pre-compute constraints for existing squares
        # For each existing square, we need to track which pairs (existing_val, new_val) are used
        # We will update these as we fill current_sq
        
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        # pair_used[k] is a set of (existing_val, new_val) for existing_squares[k]
        pair_used = [set() for _ in range(len(existing_squares))]
        
        def backtrack(idx):
            if idx == n * n:
                return True
            
            i = idx // n
            j = idx % n
            
            # Try all values 0..n-1
            # Optimization: Order values by least constraining? Or just 0..n-1
            for v in range(n):
                # Check row
                if v in row_used[i]:
                    continue
                # Check col
                if v in col_used[j]:
                    continue
                
                # Check orthogonality with existing squares
                valid = True
                for k, esq in enumerate(existing_squares):
                    ev = esq[i][j]
                    if (ev, v) in pair_used[k]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Place v
                current_sq[i][j] = v
                row_used[i].add(v)
                col_used[j].add(v)
                for k, esq in enumerate(existing_squares):
                    pair_used[k].add((esq[i][j], v))
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                current_sq[i][j] = None
                row_used[i].remove(v)
                col_used[j].remove(v)
                for k, esq in enumerate(existing_squares):
                    pair_used[k].remove((esq[i][j], v))
            
            return False
        
        # Simple index order
        if backtrack(0):
            return [row[:] for row in current_sq]
        return None
    
    # Find second square
    sq2 = find_orthogonal_square(squares)
    if sq2:
        squares.append(sq2)
        
    # Try to find third square
    sq3 = find_orthogonal_square(squares)
    if sq3:
        squares.append(sq3)
        
    # Return the squares
    # Ensure types are correct
    result = []
    for sq in squares:
        result.append([list(map(int, row)) for row in sq])
        
    return result

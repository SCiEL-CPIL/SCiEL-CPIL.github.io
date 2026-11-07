import time
import random

def construct():
    n = 10
    deadline = time.time() + 50.0
    
    # S_0
    S0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [S0]
    
    def is_valid_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
            col = [sq[r][i] for r in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True

    def is_orthogonal_to_all(candidate, existing_squares):
        for esq in existing_squares:
            pairs = set()
            for i in range(n):
                for j in range(n):
                    p = (candidate[i][j], esq[i][j])
                    if p in pairs:
                        return False
                    pairs.add(p)
        return True

    def find_next_square(existing_squares):
        current_sq = [[0]*n for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair_used_sets = [set() for _ in range(len(existing_squares))]
        
        # Fix first row to 0..n-1
        for j in range(n):
            s = j
            current_sq[0][j] = s
            row_used[0].add(s)
            col_used[j].add(s)
            for k, esq in enumerate(existing_squares):
                p = (s, esq[0][j])
                pair_used_sets[k].add(p)
                
        def backtrack(row, col):
            if row == n:
                return True
            
            next_row = row
            next_col = col + 1
            if next_col == n:
                next_row = row + 1
                next_col = 0
                
            used_sym = set(row_used[row]) | set(col_used[col])
            valid_syms = []
            for s in range(n):
                if s in used_sym:
                    continue
                ok = True
                for k, esq in enumerate(existing_squares):
                    p = (s, esq[row][col])
                    if p in pair_used_sets[k]:
                        ok = False
                        break
                if ok:
                    valid_syms.append(s)
            
            if not valid_syms:
                return False
            
            random.shuffle(valid_syms)
            
            for s in valid_syms:
                current_sq[row][col] = s
                row_used[row].add(s)
                col_used[col].add(s)
                
                added_pairs = []
                for k, esq in enumerate(existing_squares):
                    p = (s, esq[row][col])
                    pair_used_sets[k].add(p)
                    added_pairs.append((k, p))
                    
                if backtrack(next_row, next_col):
                    return True
                    
                for k, p in added_pairs:
                    pair_used_sets[k].remove(p)
                row_used[row].remove(s)
                col_used[col].remove(s)
            
            return False
        
        return backtrack(1, 0)

    while len(squares) < 9:
        if time.time() > deadline:
            break
        
        found = False
        # Try a few times with different random states
        for _ in range(20):
            if time.time() > deadline:
                break
            if find_next_square(squares):
                if is_valid_latin(current_sq) and is_orthogonal_to_all(current_sq, squares):
                    squares.append(current_sq)
                    found = True
                    break
        if not found:
            break

    result = []
    for sq in squares:
        new_sq = []
        for row in sq:
            new_row = []
            for val in row:
                if isinstance(val, bool):
                    new_row.append(int(val))
                elif not isinstance(val, int):
                    new_row.append(int(val))
                else:
                    new_row.append(val)
            new_sq.append(new_row)
        result.append(new_sq)
        
    return result

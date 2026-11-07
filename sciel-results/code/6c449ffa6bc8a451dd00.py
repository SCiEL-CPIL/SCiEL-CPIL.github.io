def construct():
    n = 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [L1]
    
    def find_orthogonal(existing, n):
        new_sq = [[None] * n for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair_used = set()
        prefixes = [[tuple(sq[i][j] for sq in existing) for j in range(n)] for i in range(n)]
        
        import sys
        sys.setrecursionlimit(10000)
        
        def backtrack():
            best_i, best_j, best_options = -1, -1, None
            min_count = n + 1
            for i in range(n):
                for j in range(n):
                    if new_sq[i][j] is not None:
                        continue
                    options = []
                    for sym in range(n):
                        if sym not in row_used[i] and sym not in col_used[j]:
                            if prefixes[i][j] + (sym,) not in pair_used:
                                options.append(sym)
                    if len(options) < min_count:
                        min_count = len(options)
                        best_i, best_j, best_options = i, j, options
                        if min_count == 0:
                            return False
                        if min_count == 1:
                            break
                if min_count == 1:
                    break
            
            if best_i == -1:
                return True
            
            i, j = best_i, best_j
            for sym in best_options:
                new_sq[i][j] = sym
                row_used[i].add(sym)
                col_used[j].add(sym)
                pair_used.add(prefixes[i][j] + (sym,))
                
                if backtrack():
                    return True
                
                new_sq[i][j] = None
                row_used[i].remove(sym)
                col_used[j].remove(sym)
                pair_used.remove(prefixes[i][j] + (sym,))
            
            return False
        
        if backtrack():
            return new_sq
        return None
    
    L2 = find_orthogonal(squares, n)
    if L2 is not None:
        squares.append(L2)
    
    L3 = find_orthogonal(squares, n)
    if L3 is not None:
        squares.append(L3)
    
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    return result

import time
import random

DEADLINE = 50.0 # Safety margin

def _is_latin(sq):
    n = len(sq)
    for i in range(n):
        if len(set(sq[i])) != n:
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if len(set(col)) != n:
            return False
    return True

def _is_orthogonal(a, b):
    n = len(a)
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (a[i][j], b[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def _find_second_square(base, n, deadline):
    """
    Find a Latin square orthogonal to base.
    Uses backtracking.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # We will fill sq row by row.
    # sq[i][j]
    # Constraints:
    # 1. sq is Latin
    # 2. (base[i][j], sq[i][j]) are unique
    
    sq = [[-1 for _ in range(n)] for _ in range(n)]
    used_in_row = [set() for _ in range(n)]
    used_in_col = [set() for _ in range(n)]
    used_pairs = set()
    
    # Symmetry breaking: fix first row?
    # Actually, we can fix sq[0][0] = 0.
    # But we need to ensure it's a valid permutation.
    
    def backtrack(i, j):
        if time.time() > deadline:
            return False
        if i == n:
            return True
        if j == n:
            return backtrack(i + 1, 0)
        
        # Try all symbols 0..n-1
        # Heuristic: try symbols that are less constrained?
        # For now, random order.
        symbols = list(range(n))
        rng.shuffle(symbols)
        
        for sym in symbols:
            if sym in used_in_row[i]:
                continue
            if sym in used_in_col[j]:
                continue
            pair = (base[i][j], sym)
            if pair in used_pairs:
                continue
            
            # Place
            sq[i][j] = sym
            used_in_row[i].add(sym)
            used_in_col[j].add(sym)
            used_pairs.add(pair)
            
            if backtrack(i, j + 1):
                return True
            
            # Undo
            sq[i][j] = -1
            used_in_row[i].remove(sym)
            used_in_col[j].remove(sym)
            used_pairs.remove(pair)
            
        return False

    # Add symmetry breaking: fix sq[0][0] = 0
    # This reduces search space by factor of n
    sq[0][0] = 0
    used_in_row[0].add(0)
    used_in_col[0].add(0)
    used_pairs.add((base[0][0], 0))
    
    # Backtrack from (0, 1)
    if backtrack(0, 1):
        return sq
    else:
        return None

def construct():
    start_time = time.time()
    n = 10
    deadline = start_time + DEADLINE
    
    # S1: Cyclic
    s1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find S2
    s2 = _find_second_square(s1, n, deadline)
    
    if s2 is not None:
        # Verify
        if _is_latin(s2) and _is_orthogonal(s1, s2):
            # Convert to list of lists of ints
            res = [s1, s2]
            # Ensure types
            final_res = []
            for sq in res:
                new_sq = []
                for row in sq:
                    new_row = [int(v) for v in row]
                    new_sq.append(new_row)
                final_res.append(new_sq)
            return final_res
    
    # Fallback: return 1 square
    return [s1]

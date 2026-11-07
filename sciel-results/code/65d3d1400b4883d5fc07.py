import time

def _is_latin(sq):
    n = len(sq)
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _orthogonal(sq1, sq2):
    n = len(sq1)
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (sq1[i][j], sq2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def _find_partner(L1, deadline):
    n = 10
    L2 = [[None for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    used_pairs = set()
    
    # Heuristic: fill cell with min options?
    # For simplicity, row-major first. If too slow, might timeout.
    # But n=10 is small.
    
    def bt(i, j):
        if time.time() > deadline:
            return False
        if i == n:
            return True
        if j == n:
            return bt(i+1, 0)
            
        # Available values
        avail = [v for v in range(n) if v not in row_used[i] and v not in col_used[j] and (L1[i][j], v) not in used_pairs]
        
        if not avail:
            return False
            
        for v in avail:
            L2[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            used_pairs.add((L1[i][j], v))
            
            if bt(i, j+1):
                return True
            
            L2[i][j] = None
            row_used[i].remove(v)
            col_used[j].remove(v)
            used_pairs.remove((L1[i][j], v))
            
        return False

    if bt(0, 0):
        return L2
    return None

def construct():
    deadline = time.time() + 50
    n = 10
    
    sq1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    partner = _find_partner(sq1, deadline)
    
    if partner:
        # Verify
        if _is_latin(partner) and _orthogonal(sq1, partner):
            return [sq1, partner]
            
    return [sq1]

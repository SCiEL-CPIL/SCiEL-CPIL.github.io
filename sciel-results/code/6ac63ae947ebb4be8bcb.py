import time

def _can_match(m, L):
    hi = m + L
    # For each i, get sorted list of multiples in (m, hi]
    cands = {}
    for i in range(1, 13):
        start = (m // i + 1) * i
        if start > hi:
            return False
        cands[i] = list(range(start, hi + 1, i))
    
    # Assign in order of fewest candidates first (most constrained)
    order = sorted(range(1, 13), key=lambda i: len(cands[i]))
    
    used = set()
    
    def dfs(idx):
        if idx == 12:
            return True
        i = order[idx]
        for x in cands[i]:
            if x not in used:
                used.add(x)
                if dfs(idx + 1):
                    return True
                used.remove(x)
        return False
    
    return dfs(0)

def _f_12(m):
    for L in range(1, 145):
        if _can_match(m, L):
            return L
    return 144

def construct():
    deadline = time.time() + 85
    best_m = 12
    best_L = _f_12(12)
    
    period = 27720
    
    for m in range(0, period):
        if time.time() > deadline:
            break
        L = _f_12(m)
        if L > best_L:
            best_L = L
            best_m = m
    
    return best_m

import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = 128
    aps = [[] for _ in range(n)]
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            idx = [a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1]
            for i in idx:
                others = [j for j in idx if j != i]
                aps[i].append(others)
                
    def backtrack(k, pos, colors):
        if time.time() > deadline:
            return False, None
            
        if pos == n:
            return True, colors[:]
            
        # Symmetry breaking
        if pos == 0:
            candidates = [0]
        elif pos == 1 and k > 1:
            candidates = [1]
        else:
            candidates = list(range(k))
            
        for c in candidates:
            valid = True
            for others in aps[pos]:
                count_c = sum(1 for j in others if colors[j] == c)
                if count_c >= 2:
                    valid = False
                    break
            if not valid:
                continue
                
            colors[pos] = c
            
            for others in aps[pos]:
                if all(j < pos for j in others):
                    vals = [colors[j] for j in others] + [c]
                    if len(set(vals)) == 2:
                        counts = Counter(vals)
                        if all(v == 2 for v in counts.values()):
                            valid = False
                            break
                            
            if not valid:
                continue
                
            success, res = backtrack(k, pos + 1, colors)
            if success:
                return True, res
                
        return False, None

    for k in [3, 4, 5, 6]:
        colors = [None] * n
        success, res = backtrack(k, 0, colors)
        if success:
            return res
            
    # Fallback
    colors = [0] * n
    for i in range(n):
        for c in range(128):
            ok = True
            for others in aps[i]:
                if all(j < i for j in others):
                    vals = [colors[j] for j in others] + [c]
                    if len(set(vals)) < 3:
                        ok = False
                        break
            if ok:
                colors[i] = c
                break
    return colors

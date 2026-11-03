import math

def construct():
    def score(U):
        U = sorted(set(U))
        if 0 not in U:
            return -1
        max_u = max(U)
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        ratio = len(diffs) / len(sums)
        if ratio <= 1:
            return 1.0
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    
    best_U = [0, 1, 3, 9]
    best_score = score(best_U)
    
    # Try small N with Greedy Golomb Ruler
    for N in range(3, 10):
        U = [0]
        diffs = set()
        for i in range(1, N):
            candidate = 1
            while True:
                # Check if candidate forms unique diffs with existing U
                valid = True
                new_diffs = []
                for u in U:
                    d = abs(candidate - u)
                    if d in diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    diffs.update(new_diffs)
                    U.append(candidate)
                    break
                candidate += 1
        
        s = score(U)
        if s > best_score:
            best_score = s
            best_U = U
            
    return best_U

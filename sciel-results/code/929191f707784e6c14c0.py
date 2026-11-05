import time

def construct():
    deadline = time.time() + 85
    
    N = 180
    
    # Precompute conflict pairs for each x
    conflict_pairs = {x: [] for x in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            numerator = a * b
            denominator = b - a
            if denominator > 0 and numerator % denominator == 0:
                c = numerator // denominator
                if c > b and c <= N:
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
    
    # Greedy: add from N down to 1
    included = set()
    
    for x in range(N, 0, -1):
        can_add = True
        for (y, z) in conflict_pairs[x]:
            if y in included and z in included:
                can_add = False
                break
        if can_add:
            included.add(x)
    
    # Local search
    excluded = set(range(1, N + 1)) - included
    
    def is_valid_with(x, base_set):
        for (y, z) in conflict_pairs[x]:
            if y in base_set and z in base_set:
                return False
        return True
    
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        
        # Try to add excluded elements directly
        for x in list(excluded):
            if time.time() > deadline - 3:
                break
            if is_valid_with(x, included):
                included.add(x)
                excluded.remove(x)
                improved = True
                continue
            # Try 1-swap
            blockers = set()
            for (y, z) in conflict_pairs[x]:
                if y in included:
                    blockers.add(y)
                if z in included:
                    blockers.add(z)
            
            for rem in list(blockers):
                if time.time() > deadline - 3:
                    break
                new_base = included - {rem}
                if is_valid_with(x, new_base):
                    included = new_base | {x}
                    excluded = set(range(1, N + 1)) - included
                    improved = True
                    break
            if improved:
                break
        
        # Try 2-swap kicks
        for x in list(excluded):
            if time.time() > deadline - 3:
                break
            if is_valid_with(x, included):
                included.add(x)
                excluded.remove(x)
                improved = True
                continue
            
            blockers = set()
            for (y, z) in conflict_pairs[x]:
                if y in included:
                    blockers.add(y)
                if z in included:
                    blockers.add(z)
            
            if len(blockers) > 25:
                continue
            
            blist = list(blockers)
            for i in range(len(blist)):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, len(blist)):
                    new_base = included - {blist[i], blist[j]}
                    if is_valid_with(x, new_base):
                        included = new_base | {x}
                        excluded = set(range(1, N + 1)) - included
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return list(included)

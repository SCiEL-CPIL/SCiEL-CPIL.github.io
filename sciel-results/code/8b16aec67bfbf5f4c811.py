"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    reachable = [False] * 201
    reachable[0] = True
    selected = []
    remaining = set(range(1, 101))
    
    while remaining:
        candidates = []
        for x in remaining:
            would_create_200 = False
            for s in range(200 - x + 1):
                if reachable[s] and s + x == 200:
                    would_create_200 = True
                    break
            if would_create_200:
                continue
            new_count = 0
            for s in range(200 - x + 1):
                if reachable[s] and not reachable[s + x]:
                    new_count += 1
            candidates.append((new_count, -x, x))  # min new_count, then max x
        
        if not candidates:
            break
        
        candidates.sort()
        best_x = candidates[0][2]
        selected.append(best_x)
        for s in range(200 - best_x, -1, -1):
            if reachable[s]:
                reachable[s + best_x] = True
        remaining.remove(best_x)
    
    return selected if selected else [1]
# EVOLVE-BLOCK-END

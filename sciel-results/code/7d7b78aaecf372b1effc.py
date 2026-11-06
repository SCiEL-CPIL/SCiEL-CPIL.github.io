"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Find a maximum subset of [1, 256] such that for every distinct pair a, b in the subset,
    (a * b) % (a + b) != 0.
    """
    n = 256
    # Build conflict adjacency lists
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Greedy independent set with minimum degree ordering
    def greedy_is():
        remaining = set(range(n))
        result = []
        while remaining:
            min_deg = n + 1
            min_v = None
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            if min_v is None:
                break
            result.append(min_v)
            remaining.discard(min_v)
            remaining -= conflicts[min_v]
        return result
    
    current = greedy_is()
    current_set = set(current)
    
    # Local search with 1-add and 1-remove-2-add moves
    for _ in range(500):
        improved = False
        
        # Try to add a single vertex
        for v in range(n):
            if v not in current_set and not (conflicts[v] & current_set):
                current_set.add(v)
                current.append(v)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-out, 2-in
        for v_out in list(current_set):
            temp = current_set - {v_out}
            # Find candidates that don't conflict with temp
            cands = []
            for v in range(n):
                if v not in temp and not (conflicts[v] & temp):
                    cands.append(v)
            # Try pairs
            for i in range(len(cands)):
                for j in range(i + 1, len(cands)):
                    if cands[i] == cands[j]:
                        continue
                    if cands[i] in temp or cands[j] in temp:
                        continue
                    if not (conflicts[cands[i]] & {cands[j]}):
                        current_set = temp | {cands[i], cands[j]}
                        current = list(current_set)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return [v + 1 for v in sorted(current)]

# EVOLVE-BLOCK-END

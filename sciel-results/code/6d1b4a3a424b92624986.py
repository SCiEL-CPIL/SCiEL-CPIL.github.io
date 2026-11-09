import time
from itertools import combinations

def _can_cover(K_set, N=40):
    """Check if chord set K_set covers all cycle lengths 3..N-1."""
    all_lengths = set(range(3, N))
    covered = set()
    for k in K_set:
        covered.add(k + 1)
        covered.add(N + 1 - k)
    # Double chords
    K_list = sorted(K_set)
    for i in range(len(K_list)):
        for j in range(i + 1, len(K_list)):
            diff = K_list[j] - K_list[i]
            if diff >= 1:
                covered.add(diff + 2)
    return all_lengths.issubset(covered)

def _find_15_chord_solution():
    """Search for a 15-chord solution by trying removals and swaps from the 16-chord base."""
    N = 40
    base_K = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 24]
    base_K_set = set(base_K)
    
    # First, verify the base works
    if not _can_cover(base_K_set, N):
        return None
    
    # Try single removals
    for remove in base_K:
        candidate = base_K_set - {remove}
        if _can_cover(candidate, N):
            return sorted(candidate)
    
    # Try 1-for-1 swaps
    for remove in base_K:
        for add in range(2, N - 1):
            if add in base_K_set:
                continue
            candidate = (base_K_set - {remove}) | {add}
            if _can_cover(candidate, N):
                return sorted(candidate)
    
    # Try 2-for-2 swaps
    for remove1, remove2 in combinations(base_K, 2):
        for add1 in range(2, N - 1):
            if add1 in base_K_set:
                continue
            for add2 in range(add1 + 1, N - 1):
                if add2 in base_K_set:
                    continue
                candidate = (base_K_set - {remove1, remove2}) | {add1, add2}
                if _can_cover(candidate, N):
                    return sorted(candidate)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    N = 40
    edges_set = set()
    
    # Base: Hamiltonian cycle on vertices 0..39 (the "rim")
    for i in range(N):
        edges_set.add((min(i, (i + 1) % N), max(i, (i + 1) % N)))
    
    # Try to find a 15-chord solution
    K_15 = _find_15_chord_solution()
    
    if K_15 is not None and len(K_15) == 15:
        K = K_15
    else:
        # Fall back to the known 16-chord solution
        K = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 24]
    
    K_set = set(K)
    
    for k in K:
        edges_set.add((0, k))
    
    # Build cycles
    cycles = {}
    cycles[N] = list(range(N))
    
    for L in range(3, N):
        found = False
        
        # 1. Try single chord (forward)
        k1 = L - 1
        if k1 in K_set:
            cycles[L] = list(range(k1 + 1))
            found = True
        
        # 2. Try single chord (backward)
        if not found:
            k2 = N + 1 - L
            if k2 in K_set:
                cycles[L] = [0] + list(range(N - 1, k2 - 1, -1))
                found = True
        
        # 3. Try double chord
        if not found:
            diff = L - 2
            for u in K:
                v = u + diff
                if v in K_set and v != u:
                    cycles[L] = [0] + list(range(u, v + 1))
                    found = True
                    break
        
        if not found:
            raise ValueError(f"Cycle length {L} not covered by chord set {K}")
    
    # Validate
    valid = True
    for k in range(3, N + 1):
        if k not in cycles:
            valid = False
            break
        cyc = cycles[k]
        if len(cyc) != k:
            valid = False
            break
        if len(set(cyc)) != k:
            valid = False
            break
        for i in range(k):
            a = cyc[i]
            b = cyc[(i + 1) % k]
            if (min(a, b), max(a, b)) not in edges_set:
                valid = False
                break
        if not valid:
            break
    
    if valid:
        out = [cycles[k] for k in range(3, N + 1)]
        return {'edges': [list(e) for e in edges_set], 'cycles': out}
    else:
        # Fallback: fan graph (worse score, but valid)
        fan_edges = set()
        for i in range(1, N):
            fan_edges.add((0, i))
        for i in range(1, N - 1):
            fan_edges.add((i, i + 1))
        cycles_fb = []
        for k in range(3, N + 1):
            cycles_fb.append(list(range(k)))
        return {'edges': [list(e) for e in fan_edges], 'cycles': cycles_fb}

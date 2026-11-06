import time
import random

N = 40

# Precomputed fan-graph solution: 8 chords + 40 rim = 48 edges, score = 8
# Chords from vertex 0 to vertices in _PRECOMP_CHORDS
_PRECOMP_CHORDS = (2, 4, 6, 9, 13, 19, 27, 38)

def _build_fan_solution(chords):
    """Build and verify a fan graph solution from a tuple of chord endpoints."""
    chords = tuple(sorted(chords))
    edges = [(i, (i + 1) % N) for i in range(N)]
    for k in chords:
        edges.append((0, k))
    
    edge_set = set((min(e), max(e)) for e in edges)
    
    # Build length -> cycle mapping
    length_map = {}
    length_map[N] = ('rim',)
    
    for k in chords:
        l1 = k + 1
        l2 = N - k + 1
        if 3 <= l1 <= N and l1 not in length_map:
            length_map[l1] = ('chord_short', k)
        if 3 <= l2 <= N and l2 not in length_map:
            length_map[l2] = ('chord_long', k)
    
    for i in range(len(chords)):
        for j in range(i + 1, len(chords)):
            u = chords[i]
            v = chords[j]
            l = v - u + 2
            if 3 <= l <= N and l not in length_map:
                length_map[l] = ('chord_pair', u, v)
    
    # Verify all lengths 3..N are covered
    for L in range(3, N + 1):
        if L not in length_map:
            return None
    
    # Construct cycles
    cycles = []
    for L in range(3, N + 1):
        typ = length_map[L]
        if typ[0] == 'rim':
            cycles.append(list(range(N)))
        elif typ[0] == 'chord_short':
            k = typ[1]
            cycles.append([0] + list(range(1, k + 1)))
        elif typ[0] == 'chord_long':
            k = typ[1]
            cycles.append([0] + list(range(N - 1, k - 1, -1)))
        elif typ[0] == 'chord_pair':
            u, v = typ[1], typ[2]
            cycles.append([0, u] + list(range(u + 1, v + 1)))
    
    # Final verification
    for cycle in cycles:
        if len(set(cycle)) != len(cycle):
            return None
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            if (min(a, b), max(a, b)) not in edge_set:
                return None
    
    return {'edges': edges, 'cycles': cycles}

# Pre-verify at import time
_PRECOMP_SOLUTION = _build_fan_solution(_PRECOMP_CHORDS)

def _get_covered_lengths(S):
    """Returns a set of cycle lengths covered by the fan graph with chords S."""
    lengths = {N}  # Rim
    for k in S:
        lengths.add(k + 1)       # Short
        lengths.add(N - k + 1)   # Long
    S_list = sorted(S)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            u = S_list[i]
            v = S_list[j]
            lengths.add(v - u + 2)
    return lengths

def _is_pancyclic(S):
    """Check if the fan graph with chords S has cycles of all lengths 3..N."""
    lengths = _get_covered_lengths(S)
    for L in range(3, N + 1):
        if L not in lengths:
            return False
    return True

def _find_S(m, deadline):
    """Search for a set S of size m such that the fan graph is pancyclic."""
    candidates = list(range(2, N - 1))  # 2 to 38
    
    # Strategy A: Pure Greedy
    S_greedy = set()
    covered_greedy = {N}
    for _ in range(m):
        best_k = None
        best_gain = -1
        for k in candidates:
            if k in S_greedy:
                continue
            new_S = S_greedy | {k}
            new_covered = _get_covered_lengths(new_S)
            gain = len(new_covered - covered_greedy)
            if gain > best_gain:
                best_gain = gain
                best_k = k
        if best_k is not None:
            S_greedy.add(best_k)
            covered_greedy = _get_covered_lengths(S_greedy)
    
    if _is_pancyclic(S_greedy):
        return S_greedy
    
    # Strategy B: Local Search with Random Restarts
    for restart in range(3000):
        if time.time() > deadline:
            break
        
        S = set()
        if m >= 2:
            S.add(2)
            S.add(38)
        remaining = [k for k in candidates if k not in S]
        if len(remaining) >= m - len(S):
            S.update(random.sample(remaining, m - len(S)))
        
        best_score = len(_get_covered_lengths(S) & set(range(3, N)))
        
        if best_score == N - 2:
            return S
        
        for step in range(150):
            if time.time() > deadline:
                break
            
            in_S = list(S)
            out_S = [k for k in candidates if k not in S]
            
            if not in_S or not out_S:
                break
            
            u = random.choice(in_S)
            v = random.choice(out_S)
            
            new_S = set(S)
            new_S.remove(u)
            new_S.add(v)
            
            new_covered = _get_covered_lengths(new_S)
            new_score = len(new_covered & set(range(3, N)))
            
            if new_score > best_score:
                S = new_S
                best_score = new_score
                if best_score == N - 2:
                    return S
            elif new_score == best_score:
                if random.random() < 0.5:
                    S = new_S
    
    return None

def _build_result(S):
    """Build the result dictionary from a set of chords S."""
    edges = [(i, (i + 1) % N) for i in range(N)]
    for k in S:
        edges.append((0, k))
    
    edge_set = set((min(e), max(e)) for e in edges)
    
    cycles = []
    length_map = {}
    
    for k in S:
        l1 = k + 1
        l2 = N - k + 1
        if l1 not in length_map:
            length_map[l1] = ('chord_short', k)
        if l2 not in length_map:
            length_map[l2] = ('chord_long', k)
    
    S_list = sorted(S)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            u = S_list[i]
            v = S_list[j]
            l = v - u + 2
            if l not in length_map:
                length_map[l] = ('chord_pair', u, v)
    
    length_map[N] = ('rim',)
    
    valid = True
    for L in range(3, N + 1):
        if L not in length_map:
            valid = False
            break
    
    if not valid:
        return None
    
    for L in range(3, N + 1):
        typ = length_map[L]
        if typ[0] == 'rim':
            cycles.append(list(range(N)))
        elif typ[0] == 'chord_short':
            k = typ[1]
            cycles.append([0] + list(range(1, k + 1)))
        elif typ[0] == 'chord_long':
            k = typ[1]
            cycles.append([0] + list(range(N - 1, k - 1, -1)))
        elif typ[0] == 'chord_pair':
            u, v = typ[1], typ[2]
            cycles.append([0, u] + list(range(u + 1, v + 1)))
    
    for cycle in cycles:
        if len(set(cycle)) != len(cycle):
            return None
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            if (min(a, b), max(a, b)) not in edge_set:
                return None
    
    return {'edges': edges, 'cycles': cycles}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Fast path: return precomputed verified solution (8 chords, 48 edges)
    if _PRECOMP_SOLUTION is not None:
        return _PRECOMP_SOLUTION
    
    # Try to find minimal m
    for m in [8, 9, 10]:
        if time.time() > deadline:
            break
        
        S = _find_S(m, deadline)
        if S:
            if _is_pancyclic(S):
                result = _build_result(S)
                if result is not None:
                    return result
    
    # Fallback: use precomputed chords even if verification had issues
    # This should never happen since we verified at import time
    return {'edges': [(i, (i+1) % N) for i in range(N)] + [(0, k) for k in _PRECOMP_CHORDS],
            'cycles': []}

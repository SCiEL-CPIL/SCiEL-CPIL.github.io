"""Combined sparse pancyclic graph construction with chord-pair exploitation."""
import time
import random
import itertools

N = 40
START_TIME = time.time()
DEADLINE = 85  # seconds, keep margin before 90s kill

def elapsed():
    return time.time() - START_TIME

def check_validity(edges, cycles):
    """Check if all cycles are valid and all required edges exist."""
    edge_set = set()
    for e in edges:
        a, b = e
        if a == b:
            return False
        edge_set.add((min(a, b), max(a, b)))
    
    for k, cycle in enumerate(cycles, start=3):
        if len(cycle) != k:
            return False
        if len(set(cycle)) != k:
            return False
        for v in cycle:
            if v < 0 or v >= N:
                return False
        for i in range(k):
            u = cycle[i]
            v = cycle[(i + 1) % k]
            if (min(u, v), max(u, v)) not in edge_set:
                return False
    return True

def get_covered_lengths(S):
    """Given a set of chord distances S (each chord is (0,k)), compute all cycle lengths covered."""
    S = set(S)
    covered = set()
    covered.add(N)  # rim cycle
    for k in S:
        covered.add(k + 1)
        covered.add(N - k + 1)
    for u, v in itertools.combinations(S, 2):
        covered.add(abs(u - v) + 2)
    return covered

def is_pancyclic(S):
    """Check if chord set S covers all lengths 3..N."""
    covered = get_covered_lengths(S)
    for L in range(3, N + 1):
        if L not in covered:
            return False
    return True

def greedy_find_S(max_chords):
    """Greedy set-cover to find a small chord set covering all cycle lengths."""
    candidates = list(range(2, N))  # chord distances 2..39, but we use min(d, N-d)
    # Actually chord (0,k) for k in 1..N-1. The "distance" is min(k, N-k).
    # Let's just use k directly as the chord endpoint (0, k) for k in 1..N-1.
    # But (0,k) and (0,N-k) are symmetric. Let's use k in 2..N-2 to avoid trivial rim-adjacent.
    # Actually k=1 gives chord (0,1) which is already a rim edge. So k in 2..N-2.
    
    candidates = list(range(2, N - 1))
    
    # Try multiple starting seeds
    seeds = [
        {2, N - 2},  # covers lengths 3 and N-1
        {2, 3, N-3, N-2},
        {2, 3, 4, N-4, N-3, N-2},
        set(random.sample(candidates, min(6, max_chords))),
    ]
    
    best_S = None
    for seed in seeds:
        if not seed.issubset(set(candidates)):
            continue
        if len(seed) > max_chords:
            continue
        S = set(seed)
        
        while len(S) < max_chords:
            current_covered = get_covered_lengths(S)
            best_k = None
            best_gain = -1
            
            for k in candidates:
                if k in S:
                    continue
                new_S = S | {k}
                new_covered = get_covered_lengths(new_S)
                gain = len(new_covered - current_covered)
                if gain > best_gain:
                    best_gain = gain
                    best_k = k
                if is_pancyclic(new_S):
                    S = new_S
                    break
            
            if is_pancyclic(S):
                break
            
            if best_k is None:
                break
            S.add(best_k)
        
        if is_pancyclic(S):
            if best_S is None or len(S) < len(best_S):
                best_S = set(S)
            if len(S) <= 8:
                break
    
    # Local search: try to reduce S
    if best_S is not None:
        S = set(best_S)
        improved = True
        while improved and len(S) > 3:
            improved = False
            for k in list(S):
                trial = S - {k}
                if is_pancyclic(trial):
                    S = trial
                    improved = True
                    break
        
        # Try swaps
        for k_in in list(S):
            for k_out in candidates:
                if k_out in S:
                    continue
                trial = (S - {k_in}) | {k_out}
                if len(trial) < len(S) and is_pancyclic(trial):
                    S = trial
                    break
            else:
                continue
            break
        
        if len(S) < len(best_S):
            best_S = set(S)
    
    return best_S

def random_search_S(max_chords, budget_sec):
    """Random restart local search to find small pancyclic chord set."""
    candidates = list(range(2, N - 1))
    best_S = None
    deadline = time.time() + budget_sec
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        S = set(random.sample(candidates, min(max_chords, len(candidates))))
        
        # Local improvement
        for _ in range(50):
            if is_pancyclic(S):
                if best_S is None or len(S) < len(best_S):
                    best_S = set(S)
                break
            
            # Try removing
            removed = False
            for k in list(S):
                trial = S - {k}
                if is_pancyclic(trial):
                    S = trial
                    removed = True
                    break
            
            if removed:
                continue
            
            # Try swapping
            in_s = list(S)
            out_s = [x for x in candidates if x not in S]
            if not in_s or not out_s:
                break
            
            u = random.choice(in_s)
            v = random.choice(out_s)
            S.remove(u)
            S.add(v)
        
        if best_S is not None and len(best_S) <= 8:
            break
    
    return best_S

def build_solution(S):
    """Build edges and cycles from chord set S (chords are (0,k) for k in S)."""
    S = sorted(S)
    
    # Build edge set
    edge_set = set()
    # Rim
    for i in range(N):
        a, b = i, (i + 1) % N
        edge_set.add((min(a, b), max(a, b)))
    # Chords
    for k in S:
        a, b = 0, k
        edge_set.add((min(a, b), max(a, b)))
    
    edges = list(edge_set)
    
    # Build length-to-cycle map
    length_map = {}
    
    # Rim: length N
    length_map[N] = ('rim',)
    
    # Single chord cycles
    for k in S:
        l1 = k + 1
        l2 = N - k + 1
        if l1 not in length_map:
            length_map[l1] = ('chord_short', k)
        if l2 not in length_map:
            length_map[l2] = ('chord_long', k)
    
    # Pair chord cycles
    for u, v in itertools.combinations(S, 2):
        l = abs(u - v) + 2
        if l not in length_map:
            length_map[l] = ('chord_pair', min(u, v), max(u, v))
    
    # Check all lengths covered
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
            # Cycle: 0, 1, 2, ..., k (k+1 vertices)
            cycles.append(list(range(0, k + 1)))
        elif typ[0] == 'chord_long':
            k = typ[1]
            # Cycle: 0, N-1, N-2, ..., k (N-k+1 vertices)
            cycles.append([0] + list(range(N - 1, k - 1, -1)))
        elif typ[0] == 'chord_pair':
            u, v = typ[1], typ[2]
            # Cycle: 0, u, u+1, ..., v (|u-v|+2 vertices)
            # Edges: (0,u) chord, (u,u+1)..(v-1,v) rim, (v,0) chord
            cycles.append([0, u] + list(range(u + 1, v + 1)))
    
    return {'edges': edges, 'cycles': cycles}

def try_prune_edges(edges, cycles):
    """Try to remove edges one by one while maintaining validity."""
    edge_set = set()
    for e in edges:
        a, b = e
        edge_set.add((min(a, b), max(a, b)))
    
    current_edges = list(edge_set)
    changed = True
    while changed:
        changed = False
        for i in range(len(current_edges)):
            if elapsed() > DEADLINE - 5:
                break
            trial_edges = [e for j, e in enumerate(current_edges) if j != i]
            trial_set = set((min(a, b), max(a, b)) for a, b in trial_edges)
            valid = True
            for k, cycle in enumerate(cycles, start=3):
                for idx in range(k):
                    u = cycle[idx]
                    v = cycle[(idx + 1) % k]
                    if (min(u, v), max(u, v)) not in trial_set:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                current_edges = trial_edges
                changed = True
                break
        if elapsed() > DEADLINE - 5:
            break
    
    return [(a, b) for a, b in current_edges]

# EVOLVE-BLOCK-START
def construct():
    if elapsed() > DEADLINE - 5:
        # Fallback: fan graph
        edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
        cycles = [list(range(k)) for k in range(3, N + 1)]
        return {'edges': edges, 'cycles': cycles}
    
    # Strategy: find minimal chord set S such that rim + chords (0,k) for k in S is pancyclic.
    # Try greedy first, then random search for smaller sets.
    
    best_S = None
    
    # Greedy search
    S_greedy = greedy_find_S(12)
    if S_greedy is not None:
        best_S = S_greedy
    
    # Random search for potentially smaller sets
    remaining_time = DEADLINE - elapsed() - 5
    if remaining_time > 2:
        S_random = random_search_S(12, remaining_time)
        if S_random is not None:
            if best_S is None or len(S_random) < len(best_S):
                best_S = S_random
    
    if best_S is None:
        # Fallback to deterministic construction with 19 chords
        S_fallback = set(range(2, 21))
        best_S = S_fallback
    
    # Build solution
    result = build_solution(best_S)
    if result is None:
        # Fallback: fan graph
        edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
        cycles = [list(range(k)) for k in range(3, N + 1)]
        return {'edges': edges, 'cycles': cycles}
    
    edges = result['edges']
    cycles = result['cycles']
    
    # Verify
    if not check_validity(edges, cycles):
        edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
        cycles = [list(range(k)) for k in range(3, N + 1)]
        return {'edges': edges, 'cycles': cycles}
    
    # Prune edges
    if elapsed() < DEADLINE - 10:
        edges = try_prune_edges(edges, cycles)
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END

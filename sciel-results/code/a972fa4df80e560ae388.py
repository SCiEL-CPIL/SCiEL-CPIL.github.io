import time
import random
import itertools

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for safety

    def get_lengths_fan(S):
        """Get all cycle lengths achievable with fan chords S (chords from 0 to k for k in S)"""
        lengths = {N}  # rim
        for k in S:
            lengths.add(k + 1)       # short cycle
            lengths.add(N - k + 1)    # long cycle
        for u, v in itertools.combinations(S, 2):
            lengths.add(abs(u - v) + 2)  # chord pair cycle
        return lengths

    def is_pancyclic_fan(S):
        lengths = get_lengths_fan(S)
        for L in range(3, N + 1):
            if L not in lengths:
                return False
        return True

    def build_result_fan(S):
        """Build result from fan chord set S"""
        edges = []
        # Rim
        for i in range(N):
            edges.append((i, (i + 1) % N))
        # Chords from 0 to k
        for k in sorted(S):
            edges.append((0, k))
        
        # Generate cycles
        cycles = []
        length_map = {}
        
        for k in S:
            l1 = k + 1
            l2 = N - k + 1
            if l1 not in length_map:
                length_map[l1] = ('chord_short', k)
            if l2 not in length_map:
                length_map[l2] = ('chord_long', k)
                
        for u, v in itertools.combinations(sorted(S), 2):
            l = abs(u - v) + 2
            if l not in length_map:
                length_map[l] = ('chord_pair', min(u, v), max(u, v))
                
        # Rim
        length_map[N] = ('rim',)
        
        # Construct cycles 3..N
        for L in range(3, N + 1):
            if L not in length_map:
                return None
            
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
                
        return {'edges': edges, 'cycles': cycles}

    def reduce_chords(S, max_time=5.0):
        """Try to remove redundant chords from a feasible set S."""
        S = set(S)
        t0 = time.time()
        changed = True
        while changed and (time.time() - t0) < max_time:
            changed = False
            for k in sorted(S):
                candidate = S - {k}
                if is_pancyclic_fan(candidate):
                    S = candidate
                    changed = True
        return S

    def swap_improve(S, max_time=3.0):
        """Try to replace chords to potentially enable further reduction."""
        S = set(S)
        candidates = list(range(2, N - 1))
        t0 = time.time()
        rng = random.Random(7)
        
        for _ in range(200):
            if time.time() - t0 > max_time:
                break
            if not is_pancyclic_fan(S):
                break
            
            # Try removing one chord and adding another
            for k in list(S):
                if time.time() - t0 > max_time:
                    break
                S2 = S - {k}
                if not is_pancyclic_fan(S2):
                    continue
                # Try to find a replacement that allows further reduction
                for v in candidates:
                    if v in S2:
                        continue
                    S3 = S2 | {v}
                    if is_pancyclic_fan(S3) and len(S3) < len(S):
                        S = S3
                        break
        return reduce_chords(S, max_time=1.0)

    def greedy_find_S(m):
        candidates = list(range(2, N - 1))
        
        seeds = [
            [2, 38],
            [2, 3, 37, 38],
            [2, 3, 4, 36, 37, 38],
            [3, 5, 7, 33, 35, 37],
            [2, 4, 6, 34, 36, 38],
            [2, 38],
            [2],
            [38],
        ]
        
        for seed in seeds:
            if not set(seed).issubset(set(candidates)):
                continue
            S = set(seed)
            if len(S) > m:
                continue
            
            while len(S) < m:
                current_lengths = get_lengths_fan(S)
                missing = [L for L in range(3, N + 1) if L not in current_lengths]
                
                if not missing:
                    break
                
                best_k = None
                best_score = -1
                
                for k in candidates:
                    if k in S:
                        continue
                    new_S = S | {k}
                    new_lengths = get_lengths_fan(new_S)
                    gain = len(new_lengths - current_lengths)
                    
                    if gain > best_score:
                        best_score = gain
                        best_k = k
                
                if best_k is None:
                    break
                S.add(best_k)
            
            if is_pancyclic_fan(S):
                return S
        
        # Random restarts
        rng = random.Random(42)
        for restart in range(5000):
            if time.time() > deadline:
                return None
            
            S = set(rng.sample(candidates, m))
            
            for step in range(400):
                if time.time() > deadline:
                    return None
                if is_pancyclic_fan(S):
                    return S
                
                current_lengths = get_lengths_fan(S)
                missing = [L for L in range(3, N + 1) if L not in current_lengths]
                
                if not missing:
                    return S
                
                in_s = list(S)
                out_s = [x for x in candidates if x not in S]
                if not in_s or not out_s:
                    break
                
                u = rng.choice(in_s)
                v = rng.choice(out_s)
                S.remove(u)
                S.add(v)
        
        return None

    def find_S(m):
        S = greedy_find_S(m)
        if S:
            return S
        
        # Fallback: pure random search
        candidates = list(range(2, N - 1))
        rng = random.Random(123)
        attempts = 0
        
        while time.time() < deadline and attempts < 100000:
            attempts += 1
            S = set(rng.sample(candidates, m))
            if is_pancyclic_fan(S):
                return S
        
        return None

    # Search for minimal m, starting from small values
    best_S = None
    
    for m in [8, 9, 10, 11, 12, 13, 14]:
        if time.time() > deadline - 10:
            break
        
        S = find_S(m)
        if S:
            # Try to reduce further
            S_reduced = reduce_chords(S, max_time=2.0)
            if len(S_reduced) < len(S):
                S = S_reduced
            
            # Try swap improvement
            S_swapped = swap_improve(S, max_time=2.0)
            if len(S_swapped) < len(S):
                S = S_swapped
            
            if best_S is None or len(S) < len(best_S):
                best_S = S
                if len(S) <= 8:
                    break  # Can't do better than 8
    
    if best_S is None:
        # Fallback: larger search
        for m in [15, 16, 17, 18, 19]:
            if time.time() > deadline - 5:
                break
            S = find_S(m)
            if S:
                S_reduced = reduce_chords(S, max_time=1.0)
                if best_S is None or len(S_reduced) < len(best_S):
                    best_S = S_reduced
                break

    if best_S is not None:
        result = build_result_fan(best_S)
        if result:
            return result

    # Final fallback: Fan Graph
    edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
    cycles = []
    for L in range(3, N + 1):
        cycles.append(list(range(L)))
        
    return {'edges': edges, 'cycles': cycles}

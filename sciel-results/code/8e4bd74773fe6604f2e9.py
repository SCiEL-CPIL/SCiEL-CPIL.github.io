import time
import itertools
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    def get_lengths(S):
        lengths = {N}
        for k in S:
            lengths.add(k + 1)
            lengths.add(N - k + 1)
        for u, v in itertools.combinations(S, 2):
            lengths.add(abs(u - v) + 2)
        return lengths

    def is_pancyclic(S):
        lengths = get_lengths(S)
        for L in range(3, N + 1):
            if L not in lengths:
                return False
        return True

    def greedy_find_S(m, rng=None):
        target_lengths = set(range(3, N))
        candidates = list(range(2, N - 1))
        if rng:
            rng.shuffle(candidates)
        S = set()
        
        for _ in range(m):
            best_k = None
            best_gain = -1
            
            current_lengths = get_lengths(S)
            current_covered = len(current_lengths.intersection(target_lengths))
            
            for k in candidates:
                if k in S:
                    continue
                new_lengths = set(current_lengths)
                new_lengths.add(k + 1)
                new_lengths.add(N - k + 1)
                for existing in S:
                    new_lengths.add(abs(k - existing) + 2)
                new_covered = len(new_lengths.intersection(target_lengths))
                gain = new_covered - current_covered
                
                if gain > best_gain:
                    best_gain = gain
                    best_k = k
            
            if best_k is None:
                return None
            
            S.add(best_k)
            if len(get_lengths(S).intersection(target_lengths)) == len(target_lengths):
                break
                
        if is_pancyclic(S):
            return S
        
        # Local search: try to swap chords to improve coverage
        for _ in range(200):
            if time.time() > deadline:
                break
            current_covered = len(get_lengths(S).intersection(target_lengths))
            if current_covered == len(target_lengths):
                return S
            
            improved = False
            for k_out in list(S):
                S.remove(k_out)
                base_lengths = get_lengths(S)
                base_covered = len(base_lengths.intersection(target_lengths))
                
                best_k_in = None
                best_gain = -1
                
                for k_in in candidates:
                    if k_in in S:
                        continue
                    new_lengths = set(base_lengths)
                    new_lengths.add(k_in + 1)
                    new_lengths.add(N - k_in + 1)
                    for existing in S:
                        new_lengths.add(abs(k_in - existing) + 2)
                    new_covered = len(new_lengths.intersection(target_lengths))
                    gain = new_covered - base_covered
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_k_in = k_in
                
                if best_k_in is not None and best_gain > 0:
                    S.add(best_k_in)
                    improved = True
                    break
                else:
                    S.add(k_out)
            
            if not improved:
                break

        if is_pancyclic(S):
            return S
        return None

    def build_result(S):
        edges = []
        for i in range(N):
            edges.append((i, (i + 1) % N))
        for k in S:
            edges.append((0, k))
        
        cycles = []
        length_map = {}
        
        for k in S:
            l1 = k + 1
            l2 = N - k + 1
            if l1 not in length_map:
                length_map[l1] = ('chord_short', k)
            if l2 not in length_map:
                length_map[l2] = ('chord_long', k)
                
        for u, v in itertools.combinations(S, 2):
            l = abs(u - v) + 2
            if l not in length_map:
                length_map[l] = ('chord_pair', min(u, v), max(u, v))
                
        length_map[N] = ('rim',)
        
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

    # Try to find minimal m with randomized restarts
    rng = random.Random(42)
    for m in [8, 9, 10, 11]:
        if time.time() > deadline:
            break
        # Try deterministic first
        S = greedy_find_S(m)
        if S:
            result = build_result(S)
            if result:
                return result
        # Try randomized restarts
        attempts = 100 if m == 8 else 50
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            S = greedy_find_S(m, rng)
            if S:
                result = build_result(S)
                if result:
                    return result

    # Fallback: Fan Graph
    edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
    cycles = []
    for L in range(3, N + 1):
        cycles.append(list(range(L)))
        
    return {'edges': edges, 'cycles': cycles}

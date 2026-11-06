import time
import itertools
import random

N = 40

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s limit
    
    # Precompute coverage info for each chord endpoint k (2..38)
    # Each k covers lengths: k+1, N-k+1
    # Pairs (u,v) cover length: |u-v|+2
    
    candidates = list(range(2, N - 1))  # 2 to 38
    num_candidates = len(candidates)
    
    # For each candidate index, precompute the lengths it covers individually
    # and for each pair, precompute the length covered
    single_cover = [set() for _ in range(num_candidates)]
    for i, k in enumerate(candidates):
        l1 = k + 1
        l2 = N - k + 1
        if 3 <= l1 <= N:
            single_cover[i].add(l1)
        if 3 <= l2 <= N:
            single_cover[i].add(l2)
    
    # Precompute pair coverage: pair_cover[i][j] = length covered by pair (candidates[i], candidates[j])
    pair_cover = [[0] * num_candidates for _ in range(num_candidates)]
    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            l = abs(candidates[i] - candidates[j]) + 2
            pair_cover[i][j] = l
            pair_cover[j][i] = l
    
    def check_S_indices(idx_set):
        """Check if set of indices gives pancyclic coverage 3..N. Returns True if covered."""
        covered = {N}  # rim
        for i in idx_set:
            covered.update(single_cover[i])
        idx_list = sorted(idx_set)
        for a, b in itertools.combinations(idx_list, 2):
            l = pair_cover[a][b]
            if 3 <= l <= N:
                covered.add(l)
        for L in range(3, N + 1):
            if L not in covered:
                return False
        return True
    
    def build_solution(S):
        """Build edges and cycles from chord set S (list of actual vertex indices)."""
        S = sorted(S)
        edges = []
        # Rim edges
        for i in range(N):
            edges.append((i, (i + 1) % N))
        # Chord edges from 0
        for k in S:
            edges.append((0, k))
        
        # Map each length 3..N to a cycle construction
        length_map = {}
        length_map[N] = ('rim',)
        
        for k in S:
            l1 = k + 1
            l2 = N - k + 1
            if l1 not in length_map and 3 <= l1 <= N:
                length_map[l1] = ('chord_short', k)
            if l2 not in length_map and 3 <= l2 <= N:
                length_map[l2] = ('chord_long', k)
        
        for u, v in itertools.combinations(S, 2):
            l = abs(u - v) + 2
            if l not in length_map and 3 <= l <= N:
                length_map[l] = ('chord_pair', u, v)
        
        # Verify all lengths 3..N are covered
        for L in range(3, N + 1):
            if L not in length_map:
                return None
        
        # Generate cycles
        cycles = []
        for L in range(3, N + 1):
            entry = length_map[L]
            if entry[0] == 'rim':
                cycles.append(list(range(N)))
            elif entry[0] == 'chord_short':
                k = entry[1]
                cycles.append(list(range(0, k + 1)))
            elif entry[0] == 'chord_long':
                k = entry[1]
                cycles.append([0] + list(range(N - 1, k - 1, -1)))
            elif entry[0] == 'chord_pair':
                u, v = entry[1], entry[2]
                cycles.append([0, u] + list(range(u + 1, v + 1)))
        
        return {'edges': edges, 'cycles': cycles}
    
    random.seed(42)
    
    best_solution = None
    best_m = 999
    
    for m in [7, 8, 9, 10]:
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 300000 if m <= 8 else 150000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            if attempt % 3 == 0:
                # Pure random
                idx_set = set(random.sample(range(num_candidates), m))
            elif attempt % 3 == 1:
                # Structured: include indices for 2 and 38
                idx_set = set()
                idx_set.add(0)  # candidate 2
                idx_set.add(num_candidates - 1)  # candidate 38
                remaining = [x for x in range(num_candidates) if x not in idx_set]
                need = m - len(idx_set)
                if need > 0:
                    idx_set.update(random.sample(remaining, need))
            else:
                # Greedy-ish: start with a random subset, try to complete
                idx_set = set(random.sample(range(num_candidates), max(2, m // 2)))
                remaining = [x for x in range(num_candidates) if x not in idx_set]
                need = m - len(idx_set)
                if need > 0:
                    idx_set.update(random.sample(remaining, need))
            
            if len(idx_set) != m:
                continue
            
            if check_S_indices(idx_set):
                S = [candidates[i] for i in idx_set]
                result = build_solution(S)
                if result:
                    best_solution = result
                    best_m = m
                    found = True
                    break
        
        if found:
            break
    
    if best_solution:
        return best_solution
    
    # Fallback: fan graph (always valid)
    edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
    cycles = []
    for L in range(3, N + 1):
        cycles.append(list(range(L)))
    
    return {'edges': edges, 'cycles': cycles}

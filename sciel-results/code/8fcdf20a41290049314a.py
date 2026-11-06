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
    
    # Use bitmask for coverage: bit L set means length L is covered
    # Lengths 3..40 -> bits 3..40
    FULL_MASK = 0
    for L in range(3, N + 1):
        FULL_MASK |= (1 << L)
    
    # single_cover_mask[i] = bitmask of lengths covered by chord i alone
    single_cover_mask = [0] * num_candidates
    for i, k in enumerate(candidates):
        l1 = k + 1
        l2 = N - k + 1
        if 3 <= l1 <= N:
            single_cover_mask[i] |= (1 << l1)
        if 3 <= l2 <= N:
            single_cover_mask[i] |= (1 << l2)
    
    # pair_cover[i][j] = length covered by pair (i,j), or 0 if out of range
    pair_cover = [[0] * num_candidates for _ in range(num_candidates)]
    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            l = abs(candidates[i] - candidates[j]) + 2
            if 3 <= l <= N:
                pair_cover[i][j] = l
                pair_cover[j][i] = l
    
    def check_coverage_fast(idx_set):
        """Fast coverage check using bitmask. Returns True if all lengths 3..N covered."""
        mask = 1 << N  # rim covers length N
        for i in idx_set:
            mask |= single_cover_mask[i]
        idx_list = sorted(idx_set)
        for a_idx in range(len(idx_list)):
            for b_idx in range(a_idx + 1, len(idx_list)):
                a, b = idx_list[a_idx], idx_list[b_idx]
                l = pair_cover[a][b]
                if l:
                    mask |= (1 << l)
        return (mask & FULL_MASK) == FULL_MASK
    
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
    
    # Try deterministic greedy first
    def greedy_search(m):
        """Greedy: start empty, add chord that covers most uncovered lengths."""
        if time.time() > deadline:
            return None
        
        covered = {N}
        S = []
        remaining = set(range(num_candidates))
        
        for _ in range(m):
            best_idx = None
            best_gain = -1
            for i in remaining:
                # Compute new lengths covered
                new_lengths = set()
                new_lengths.update([l for l in [candidates[i] + 1, N - candidates[i] + 1] if 3 <= l <= N])
                for j in S:
                    l = abs(candidates[i] - candidates[j]) + 2
                    if 3 <= l <= N:
                        new_lengths.add(l)
                gain = len(new_lengths - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx is None:
                return None
            S.append(best_idx)
            remaining.discard(best_idx)
            # Update covered
            k = candidates[best_idx]
            covered.add(k + 1) if 3 <= k + 1 <= N else None
            covered.add(N - k + 1) if 3 <= N - k + 1 <= N else None
            for j in S[:-1]:
                l = abs(candidates[best_idx] - candidates[j]) + 2
                if 3 <= l <= N:
                    covered.add(l)
        
        if check_coverage_fast(set(S)):
            S_actual = [candidates[i] for i in S]
            result = build_solution(S_actual)
            if result:
                return result
        return None
    
    for m in [6, 7, 8, 9, 10]:
        if time.time() > deadline:
            break
        
        # Try greedy first
        result = greedy_search(m)
        if result:
            best_solution = result
            best_m = m
            break
        
        # Random search
        found = False
        max_attempts = 500000 if m <= 7 else 300000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            if attempt % 4 == 0:
                # Pure random
                idx_set = set(random.sample(range(num_candidates), m))
            elif attempt % 4 == 1:
                # Include extreme chords (2 and 38) for better coverage
                idx_set = {0, num_candidates - 1}
                remaining = [x for x in range(num_candidates) if x not in idx_set]
                need = m - len(idx_set)
                if need > 0:
                    idx_set.update(random.sample(remaining, need))
            elif attempt % 4 == 2:
                # Include chords near middle for pair coverage
                mid = num_candidates // 2
                idx_set = {mid}
                remaining = [x for x in range(num_candidates) if x not in idx_set]
                need = m - len(idx_set)
                if need > 0:
                    idx_set.update(random.sample(remaining, need))
            else:
                # Structured: spread out chords
                step = num_candidates // m
                idx_set = set()
                for i in range(m):
                    idx_set.add(min(i * step, num_candidates - 1))
                # Add some randomness
                if len(idx_set) < m:
                    remaining = [x for x in range(num_candidates) if x not in idx_set]
                    need = m - len(idx_set)
                    if need > 0:
                        idx_set.update(random.sample(remaining, need))
            
            if len(idx_set) != m:
                continue
            
            if check_coverage_fast(idx_set):
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

import time
import itertools
import random

N = 40

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s limit
    
    # Candidates are vertices 2 to 38
    candidates = list(range(2, N - 1))
    num_candidates = len(candidates)
    
    # Bitmask for lengths 3..N
    # Bit L is set if length L is covered
    FULL_MASK = 0
    for L in range(3, N + 1):
        FULL_MASK |= (1 << L)
    
    # Precompute single chord coverage masks
    # Chord k covers lengths k+1 and N-k+1
    single_mask = [0] * num_candidates
    for i, k in enumerate(candidates):
        l1 = k + 1
        l2 = N - k + 1
        if 3 <= l1 <= N:
            single_mask[i] |= (1 << l1)
        if 3 <= l2 <= N:
            single_mask[i] |= (1 << l2)
            
    # Precompute pair coverage
    # Pair (u, v) with u < v covers length v - u + 2
    # We store the length, or 0 if invalid
    pair_length = [[0] * num_candidates for _ in range(num_candidates)]
    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            l = candidates[j] - candidates[i] + 2
            if 3 <= l <= N:
                pair_length[i][j] = l
                pair_length[j][i] = l

    def check_coverage(idx_set):
        """
        Check if the set of candidate indices covers all lengths 3..N.
        Uses bitmask for speed.
        """
        mask = 1 << N  # Rim covers N
        for i in idx_set:
            mask |= single_mask[i]
        
        # Sort to iterate pairs efficiently (though order doesn't matter for mask OR)
        sorted_idx = sorted(idx_set)
        for a in range(len(sorted_idx)):
            i = sorted_idx[a]
            for b in range(a + 1, len(sorted_idx)):
                j = sorted_idx[b]
                l = pair_length[i][j]
                if l:
                    mask |= (1 << l)
        
        return (mask & FULL_MASK) == FULL_MASK

    def build_solution(S):
        """
        Build the final graph structure from the set of chord endpoints S.
        S is a list of vertex indices (2..38).
        """
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
                # Cycle: 0, 1, ..., k
                cycles.append(list(range(0, k + 1)))
            elif entry[0] == 'chord_long':
                k = entry[1]
                # Cycle: 0, N-1, ..., k
                cycles.append([0] + list(range(N - 1, k - 1, -1)))
            elif entry[0] == 'chord_pair':
                u, v = entry[1], entry[2]
                # Cycle: 0, u, u+1, ..., v
                cycles.append([0, u] + list(range(u + 1, v + 1)))
        
        return {'edges': edges, 'cycles': cycles}

    random.seed(42)
    
    best_solution = None
    best_m = 999
    
    # Search for smallest m
    # m=7 is likely the minimum. If not found quickly, try 8.
    for m in range(7, 11):
        if time.time() > deadline:
            break
        
        found = False
        # Increase attempts for smaller m as they are rarer
        max_attempts = 500000 if m == 7 else 200000
        
        # Structural bias: 
        # Indices 0 (vertex 2) and 36 (vertex 38) are very good.
        # Vertex 2 covers 3, 39.
        # Vertex 38 covers 39, 3.
        # Including both is redundant for 3 and 39, but they help with other lengths?
        # k=2: 3, 39.
        # k=38: 39, 3.
        # Actually, k=2 and k=38 cover the same single lengths.
        # However, pairs involving them cover specific lengths.
        # Let's try forcing inclusion of 2 and 38 in some fraction of attempts.
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            idx_set = set()
            
            if attempt % 3 == 0:
                # Pure random
                idx_set = set(random.sample(range(num_candidates), m))
            elif attempt % 3 == 1:
                # Force include vertex 2 (idx 0) and 38 (idx 36)
                # This is a strong heuristic for covering extremes
                idx_set.add(0)
                idx_set.add(num_candidates - 1)
                remaining = [x for x in range(num_candidates) if x not in idx_set]
                need = m - len(idx_set)
                if need > 0:
                    idx_set.update(random.sample(remaining, need))
                else:
                    # If m < 2, this branch is invalid, but m>=7
                    pass
            else:
                # Force include vertex 2 (idx 0)
                idx_set.add(0)
                remaining = [x for x in range(num_candidates) if x not in idx_set]
                need = m - 1
                idx_set.update(random.sample(remaining, need))
                
            if len(idx_set) != m:
                continue
                
            if check_coverage(idx_set):
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
    
    # Fallback: Fan graph (always valid, but high edge count)
    # This should rarely be reached if m=7 or 8 is found.
    edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
    cycles = []
    for L in range(3, N + 1):
        cycles.append(list(range(L)))
    
    return {'edges': edges, 'cycles': cycles}

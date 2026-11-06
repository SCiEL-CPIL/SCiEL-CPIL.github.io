import time
import itertools
import random

N = 40

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s limit

    candidates = list(range(2, N - 1))  # 2 to 38
    num_candidates = len(candidates)

    # Precompute coverage masks using bitmasks
    # Bit i corresponds to length i (we care about 3..39)
    
    def make_mask(lengths):
        m = 0
        for l in lengths:
            if 3 <= l <= 39:
                m |= (1 << l)
        return m

    # single_cover[i] = bitmask of lengths covered by chord i alone
    single_cover = [0] * num_candidates
    for i, k in enumerate(candidates):
        l1 = k + 1
        l2 = N - k + 1
        single_cover[i] = make_mask([l1, l2])
    
    # pair_cover[i][j] = bitmask of length covered by pair (i, j)
    pair_cover = [[0] * num_candidates for _ in range(num_candidates)]
    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            l = abs(candidates[i] - candidates[j]) + 2
            if 3 <= l <= 39:
                pair_cover[i][j] = (1 << l)
                pair_cover[j][i] = (1 << l)
    
    # Target mask: all lengths 3..39
    target_mask = 0
    for l in range(3, 40):
        target_mask |= (1 << l)

    def get_mask(idx_set):
        """Compute coverage mask for a set of chord indices."""
        mask = 0
        for i in idx_set:
            mask |= single_cover[i]
        idx_list = sorted(idx_set)
        for a in range(len(idx_list)):
            for b in range(a + 1, len(idx_list)):
                mask |= pair_cover[idx_list[a]][idx_list[b]]
        return mask

    def check_coverage(idx_set):
        """Check if idx_set covers all required lengths."""
        return (get_mask(idx_set) & target_mask) == target_mask

    def build_solution(S):
        """
        Build edges and cycles from chord set S (list of actual vertex indices).
        S is sorted.
        """
        edges = []
        # Rim edges
        for i in range(N):
            edges.append((i, (i + 1) % N))
        # Chord edges from 0
        for k in S:
            edges.append((0, k))
        
        # Map each length 3..39 to a cycle construction
        length_map = {}
        length_map[40] = ('rim',)
        
        # Single chords
        for k in S:
            l1 = k + 1
            l2 = N - k + 1
            if l1 not in length_map and 3 <= l1 <= 39:
                length_map[l1] = ('chord_short', k)
            if l2 not in length_map and 3 <= l2 <= 39:
                length_map[l2] = ('chord_long', k)
        
        # Pairs
        for u, v in itertools.combinations(S, 2):
            l = abs(u - v) + 2
            if l not in length_map and 3 <= l <= 39:
                length_map[l] = ('chord_pair', u, v)
        
        # Verify all lengths 3..40 are covered
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

    # Strategy 1: Greedy search with multiple seed strategies
    def greedy_from_seed(seed_set, m_max):
        """Greedy addition from a seed set, trying to reach full coverage."""
        S = set(seed_set)
        if len(S) > m_max:
            return None
        
        current_mask = get_mask(S)
        
        while len(S) < m_max:
            best_chord = None
            best_gain = -1
            best_new_mask = 0
            
            for i in range(num_candidates):
                if i in S:
                    continue
                new_mask = current_mask | single_cover[i]
                for j in S:
                    new_mask |= pair_cover[i][j]
                
                gain = bin(new_mask & ~current_mask).count('1')
                
                if gain > best_gain:
                    best_gain = gain
                    best_chord = i
                    best_new_mask = new_mask
            
            if best_chord is None or best_gain == 0:
                break
            
            S.add(best_chord)
            current_mask = best_new_mask
            
            if (current_mask & target_mask) == target_mask:
                return S.copy()
        
        if (current_mask & target_mask) == target_mask:
            return S.copy()
        
        return None

    # Define structured seeds
    base_seeds = [
        {2, 38},
        {2, 3, 37, 38},
        {2, 3, 4, 36, 37, 38},
        {2, 5, 10, 15, 25, 30, 35, 38},
        {2, 4, 6, 8, 32, 34, 36, 38},
        {2, 3, 5, 7, 33, 35, 37, 38},
    ]

    # Try m from 7 upwards
    for m in range(7, 14):
        if time.time() > deadline:
            break
        
        found = False
        
        # Phase 1: Greedy from structured seeds
        for seed in base_seeds:
            if time.time() > deadline:
                break
            result_set = greedy_from_seed(seed, m)
            if result_set:
                S = sorted([candidates[i] for i in result_set])
                result = build_solution(S)
                if result:
                    best_solution = result
                    best_m = len(S)
                    found = True
                    break
        
        if found:
            break
        
        # Phase 2: Random sampling with bias
        max_random = 30000 if m <= 8 else 15000
        for attempt in range(max_random):
            if time.time() > deadline:
                break
            
            # Mix of pure random and biased random
            if attempt % 3 == 0:
                # Pure random
                idx_set = set(random.sample(range(num_candidates), m))
            elif attempt % 3 == 1:
                # Biased: include 2 and 38
                idx_set = {0, num_candidates - 1}  # indices for 2 and 38
                remaining = [i for i in range(num_candidates) if i not in idx_set]
                idx_set.update(random.sample(remaining, m - 2))
            else:
                # Biased: include small and large values
                small = random.sample(range(0, 5), min(2, m // 3))
                large = random.sample(range(num_candidates - 5, num_candidates), min(2, m // 3))
                idx_set = set(small + large)
                remaining = [i for i in range(num_candidates) if i not in idx_set]
                need = m - len(idx_set)
                if need > 0 and len(remaining) >= need:
                    idx_set.update(random.sample(remaining, need))
                elif len(idx_set) > m:
                    idx_set = set(list(idx_set)[:m])
            
            if len(idx_set) != m:
                continue
            
            if check_coverage(idx_set):
                S = sorted([candidates[i] for i in idx_set])
                result = build_solution(S)
                if result:
                    best_solution = result
                    best_m = m
                    found = True
                    break
        
        if found:
            break
        
        # Phase 3: Local search / hill climbing
        max_local = 10000 if m <= 8 else 5000
        for attempt in range(max_local):
            if time.time() > deadline:
                break
            
            # Start with a random set of size m
            idx_set = set(random.sample(range(num_candidates), m))
            current_mask = get_mask(idx_set)
            
            # Try to improve by swapping
            improved = True
            while improved and time.time() < deadline:
                improved = False
                current_mask = get_mask(idx_set)
                
                if (current_mask & target_mask) == target_mask:
                    S = sorted([candidates[i] for i in idx_set])
                    result = build_solution(S)
                    if result:
                        best_solution = result
                        best_m = m
                        found = True
                        break
                    break
                
                # Try all possible swaps
                idx_list = list(idx_set)
                for i in range(len(idx_list)):
                    if time.time() > deadline:
                        break
                    for j in range(num_candidates):
                        if j in idx_set:
                            continue
                        new_set = set(idx_set)
                        new_set.remove(idx_list[i])
                        new_set.add(j)
                        new_mask = get_mask(new_set)
                        
                        old_gain = bin(current_mask & target_mask).count('1')
                        new_gain = bin(new_mask & target_mask).count('1')
                        
                        if new_gain > old_gain:
                            idx_set = new_set
                            current_mask = new_mask
                            improved = True
                            break
                    if improved:
                        break
                
                if found:
                    break
            
            if found:
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

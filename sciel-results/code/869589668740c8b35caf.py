import itertools
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _is_dissociated(denominators):
    """
    Check if the set of reciprocals of 'denominators' has distinct subset sums.
    """
    k = len(denominators)
    if k == 0:
        return True
    L = _lcm_list(denominators)
    ints = [L // d for d in denominators]
    
    sums = set()
    for mask in range(1 << k):
        s = 0
        for i in range(k):
            if mask & (1 << i):
                s += ints[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _find_minimal_bad_sets(universe):
    """
    Find all minimal subsets of 'universe' that are NOT dissociated.
    """
    bad_sets = []
    # Check size 3
    for comb in itertools.combinations(universe, 3):
        if not _is_dissociated(list(comb)):
            bad_sets.append(set(comb))
            
    # Check size 4
    bad_size_3 = [s for s in bad_sets if len(s) == 3]
    for comb in itertools.combinations(universe, 4):
        is_superset_of_bad = False
        for b in bad_size_3:
            if b.issubset(set(comb)):
                is_superset_of_bad = True
                break
        if is_superset_of_bad:
            continue
        if not _is_dissociated(list(comb)):
            bad_sets.append(set(comb))
            
    # Check size 5
    bad_size_4 = [s for s in bad_sets if len(s) == 4]
    for comb in itertools.combinations(universe, 5):
        is_superset_of_bad = False
        for b in bad_size_3 + bad_size_4:
            if b.issubset(set(comb)):
                is_superset_of_bad = True
                break
        if is_superset_of_bad:
            continue
        if not _is_dissociated(list(comb)):
            bad_sets.append(set(comb))
            
    return bad_sets

def _build_conflict_graph(universe, bad_sets):
    """
    Build adjacency bitmasks for the conflict graph.
    """
    n = len(universe)
    num_to_idx = {num: i for i, num in enumerate(universe)}
    
    adj_masks = [0] * n
    
    for bset in bad_sets:
        indices = [num_to_idx[x] for x in bset]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                u = indices[i]
                v = indices[j]
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                
    return adj_masks

def _max_independent_set(adj_masks, n):
    """
    Solve Maximum Independent Set using Branch and Bound with bitsets.
    Optimized greedy coloring bound.
    """
    best_set = 0
    best_size = 0
    
    # Helper for popcount
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    def greedy_color_bound(candidates):
        """
        Returns an upper bound on the size of the MIS of the subgraph induced by 'candidates'.
        Uses greedy clique cover of the conflict graph.
        """
        bound = 0
        rem = candidates
        while rem:
            bound += 1
            # Select seed with highest degree in 'rem' to form a larger clique
            seed_idx = -1
            max_deg = -1
            r = rem
            while r:
                bit = r & -r
                idx = bit.bit_length() - 1
                deg = popcount(adj_masks[idx] & rem)
                if deg > max_deg:
                    max_deg = deg
                    seed_idx = idx
                r ^= bit
            
            if seed_idx == -1:
                break
                
            seed_bit = 1 << seed_idx
            clique = seed_bit
            rem ^= seed_bit
            
            # Add vertices to clique
            # A vertex v can be added if it is adjacent to ALL vertices in 'clique'.
            # adj_masks[v] & clique == clique
            r = rem
            while r:
                bit = r & -r
                idx = bit.bit_length() - 1
                if (adj_masks[idx] & clique) == clique:
                    clique |= bit
                    rem ^= bit # Remove from remaining candidates as it's covered
                r ^= bit
                
        return bound

    def backtrack(candidates, current_set, current_size):
        nonlocal best_set, best_size
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
            
        # Upper bound check
        ub = greedy_color_bound(candidates)
        if current_size + ub <= best_size:
            return
            
        # Branching heuristic: Pick vertex with highest degree in current subgraph
        best_v = -1
        max_deg = -1
        c = candidates
        while c:
            bit = c & -c
            idx = bit.bit_length() - 1
            deg = popcount(adj_masks[idx] & candidates)
            if deg > max_deg:
                max_deg = deg
                best_v = idx
            c ^= bit
            
        if best_v == -1:
            return
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        new_candidates_include = candidates & ~adj_masks[best_v] & ~v_bit
        backtrack(new_candidates_include, current_set | v_bit, current_size + 1)
        
        # Branch 2: Exclude best_v
        new_candidates_exclude = candidates & ~v_bit
        backtrack(new_candidates_exclude, current_set, current_size)

    backtrack((1 << n) - 1, 0, 0)
    
    result = []
    for i in range(n):
        if best_set & (1 << i):
            result.append(i)
    return result

def construct():
    universe = list(range(1, 23))
    bad_sets = _find_minimal_bad_sets(universe)
    
    if not bad_sets:
        return universe
        
    adj_masks = _build_conflict_graph(universe, bad_sets)
    
    indices = _max_independent_set(adj_masks, len(universe))
    
    result = [universe[i] for i in indices]
    
    if not result:
        return [1]
        
    # Verification
    if not _is_dissociated(result):
        # Fallback
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return result
# EVOLVE-BLOCK-END

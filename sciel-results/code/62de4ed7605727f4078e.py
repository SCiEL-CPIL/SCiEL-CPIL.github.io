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
            
    # Check size 5 (just in case, though rare for small integers)
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
    Two numbers are connected if they appear in the same minimal bad set.
    """
    n = len(universe)
    # Map number to index
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
    Uses greedy coloring of the complement graph for upper bounds.
    """
    best_set = 0
    best_size = 0
    
    def popcount(x):
        return bin(x).count('1')
        
    def greedy_color_bound(candidates):
        """
        Returns an upper bound on the size of the MIS of the subgraph induced by 'candidates'.
        This is done by finding a clique cover of the subgraph (which is an independent set cover of the complement).
        We greedily find cliques in the original graph.
        Each clique can contribute at most 1 vertex to the MIS.
        So the number of cliques in the cover is an upper bound on the MIS size.
        """
        bound = 0
        rem = candidates
        while rem:
            bound += 1
            # Find a maximal clique in the original graph within 'rem'
            # A clique in G is a set of vertices where every pair is connected.
            # We build it greedily.
            clique = 0
            # Pick the first available vertex as a seed
            # To be deterministic and potentially faster, pick the one with highest degree in 'rem'?
            # For simplicity, pick the lowest index bit.
            seed_bit = rem & -rem
            seed_idx = (seed_bit.bit_length() - 1)
            clique = seed_bit
            rem ^= seed_bit
            
            # Try to add other vertices to the clique
            # A vertex u can be added if it is adjacent to ALL vertices in 'clique'.
            # i.e., adj_masks[u] & clique == clique
            # Iterate through remaining vertices
            r = rem
            while r:
                bit = r & -r
                idx = bit.bit_length() - 1
                if (adj_masks[idx] & clique) == clique:
                    clique |= bit
                    r ^= bit
                else:
                    r ^= bit
            rem &= ~clique
            
        return bound

    def backtrack(candidates, current_set, current_size):
        nonlocal best_set, best_size
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
            
        # Upper bound check
        # The maximum additional vertices we can add is bounded by the clique cover size of the remaining graph.
        ub = greedy_color_bound(candidates)
        if current_size + ub <= best_size:
            return
            
        # Branching heuristic: Pick a vertex to branch on.
        # Common heuristic: Pick the vertex with the highest degree in the current subgraph.
        # Or simply the first one.
        # Let's pick the vertex with the highest degree in the induced subgraph.
        best_v = -1
        max_deg = -1
        c = candidates
        while c:
            bit = c & -c
            idx = bit.bit_length() - 1
            # Degree of idx in the subgraph induced by 'candidates'
            deg = popcount(adj_masks[idx] & candidates)
            if deg > max_deg:
                max_deg = deg
                best_v = idx
            c ^= bit
            
        if best_v == -1:
            return
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        # New candidates are those in 'candidates' that are NOT adjacent to best_v, and exclude best_v itself.
        new_candidates_include = candidates & ~adj_masks[best_v] & ~v_bit
        backtrack(new_candidates_include, current_set | v_bit, current_size + 1)
        
        # Branch 2: Exclude best_v
        new_candidates_exclude = candidates & ~v_bit
        backtrack(new_candidates_exclude, current_set, current_size)

    backtrack((1 << n) - 1, 0, 0)
    
    # Convert bitmask back to list of numbers
    result = []
    for i in range(n):
        if best_set & (1 << i):
            result.append(i) # Return indices? No, we need the numbers.
            # Wait, the function returns indices. The caller needs to map back.
            # Let's return the indices and map outside, or map inside.
            # Let's map inside for cleanliness.
    return result

def construct():
    universe = list(range(1, 23))
    bad_sets = _find_minimal_bad_sets(universe)
    
    if not bad_sets:
        return universe
        
    adj_masks = _build_conflict_graph(universe, bad_sets)
    
    # _max_independent_set returns indices
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

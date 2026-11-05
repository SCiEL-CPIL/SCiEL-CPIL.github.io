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
    n = len(universe)
    
    # We check sizes 3 up to 6. 
    # Size 2: 1/a = 1/b implies a=b, impossible for distinct. 
    # So minimal bad sets have size >= 3.
    
    for size in range(3, 7):
        for comb in itertools.combinations(universe, size):
            # Check if this combination is a superset of any already found bad set
            # If so, it's not minimal.
            is_superset = False
            for b in bad_sets:
                if b.issubset(set(comb)):
                    is_superset = True
                    break
            if is_superset:
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
    Solve Maximum Independent Set using Branch and Bound.
    Uses greedy coloring of the complement graph for upper bounds.
    """
    best_set = 0
    best_size = 0
    
    def popcount(x):
        return bin(x).count('1')

    def greedy_color_bound(candidates):
        """
        Returns an upper bound on the size of the MIS of the subgraph induced by 'candidates'.
        This is done by greedily coloring the COMPLEMENT graph of the candidates.
        A coloring of the complement graph corresponds to a clique cover of the original graph.
        The number of colors is an upper bound on the size of the Independent Set.
        """
        if not candidates:
            return 0
            
        # Get list of vertices in candidates
        verts = []
        c = candidates
        while c:
            bit = c & -c
            idx = bit.bit_length() - 1
            verts.append(idx)
            c ^= bit
            
        # Sort vertices by degree in the original graph (within candidates) descending.
        degrees = []
        for v in verts:
            deg = popcount(adj_masks[v] & candidates)
            degrees.append((deg, v))
        degrees.sort(reverse=True)
        
        uncolored = set(verts)
        num_colors = 0
        
        while uncolored:
            num_colors += 1
            # Find a maximal clique in the original graph within 'uncolored'
            # Start with the vertex with highest degree in original graph
            max_deg = -1
            start_v = None
            for v in uncolored:
                deg = popcount(adj_masks[v] & candidates)
                if deg > max_deg:
                    max_deg = deg
                    start_v = v
            
            if start_v is None:
                break
                
            clique = {start_v}
            uncolored.remove(start_v)
            
            clique_mask = 1 << start_v
            
            # Try to add other vertices to the clique
            to_remove = []
            for v in list(uncolored):
                if (adj_masks[v] & clique_mask) == clique_mask:
                    clique.add(v)
                    clique_mask |= (1 << v)
                    to_remove.append(v)
            
            for v in to_remove:
                uncolored.remove(v)
                
        return num_colors

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
            
        # Branching heuristic: Pick a vertex to branch on.
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

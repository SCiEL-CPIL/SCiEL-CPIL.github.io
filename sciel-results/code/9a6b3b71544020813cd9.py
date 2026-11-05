"""Few edges in 5-uniform hypergraphs without Property B (m(5))"""
import time
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0

def _elapsed():
    return time.time() - _START

def _check_time():
    return _elapsed() > _DEADLINE


def _is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph has a proper 2-coloring
    using backtracking with unit propagation.
    Returns True if a valid 2-coloring exists, False otherwise.
    """
    if not edges:
        return True

    # Build vertex-edge incidence for fast lookups
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)

    colors = [-1] * n
    # edge_state: [count_0, count_1] for each edge
    # We recompute unassigned on the fly to avoid complex state maintenance bugs,
    # but for n<=26 and E<=200, this is fast enough if optimized.
    # However, for speed in the inner loop, we'll maintain a list of unassigned per edge.
    # Let's stick to a robust incremental update for propagation.
    
    # For simplicity and robustness in this rewrite, we will use a slightly slower
    # but very clean backtracking that recomputes edge states or uses a simple stack.
    # Given the checker is the bottleneck, let's optimize the search order (MRV)
    # and propagation.

    def propagate():
        """
        Performs unit propagation.
        Returns False if a contradiction is found.
        """
        changed = True
        while changed:
            changed = False
            for i in range(len(edges)):
                e = edges[i]
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if c0 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if colors[v] == -1:
                        colors[v] = 1
                        changed = True
                        # Early exit not strictly necessary, loop handles it
                elif c1 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if colors[v] == -1:
                        colors[v] = 0
                        changed = True
                        
        return True

    def backtrack():
        # Find unassigned vertex with highest degree (MRV-like heuristic)
        best_v = -1
        best_deg = -1
        for v in range(n):
            if colors[v] == -1:
                deg = len(vertex_edges[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        
        if best_v == -1:
            # All assigned, check final consistency (should be caught by propagate)
            for i in range(len(edges)):
                e = edges[i]
                c0 = sum(1 for v in e if colors[v] == 0)
                if c0 == 5 or c0 == 0:
                    return False
            return True
        
        # Try color 0
        colors[best_v] = 0
        if propagate():
            if backtrack():
                return True
        colors[best_v] = -1
        
        # Try color 1
        colors[best_v] = 1
        if propagate():
            if backtrack():
                return True
        colors[best_v] = -1
        
        return False

    # Initial propagation
    if not propagate():
        return False
    
    return backtrack()


def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-coloring, minimizing edges.
    Uses Greedy Set Cover on the dual space of colorings.
    """
    # For n <= 12, 2^n is manageable (up to 4096).
    # We want to find a set of edges such that every 2-coloring has at least one monochromatic edge.
    # This is equivalent to covering the set of all 2-colorings with the "bad" sets of edges.
    
    # Let's try n=12 first.
    # If we can't find a small set, we might try n=10 or n=11, but n=12 is a good balance
    # between the number of colorings and the density of edges needed.
    # Actually, smaller n means fewer edges potentially? 
    # K_9^5 has 126 edges. 
    # For n=10, K_10^5 has 252 edges.
    # The "covering" approach finds a minimal subset.
    
    # Let's use n=12.
    n = 12
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    num_colorings = 1 << n
    
    # Precompute for each edge, the bitmask of colorings where it is monochromatic.
    # A coloring is monochromatic for edge e if all v in e have color 0 OR all v in e have color 1.
    # Colorings are represented as bitmasks of length n.
    # If we use integer 0..2^n-1, bit v is the color of vertex v.
    # "Monochromatic 0" means bits v are all 0.
    # "Monochromatic 1" means bits v are all 1.
    
    # Optimization: 
    # Instead of iterating all 2^n colorings for each edge (num_edges * 2^n ops),
    # we can iterate all colorings once and for each, determine which edges are monochromatic.
    # But we need `coverage[edge_index]` = bitmask of colorings.
    # It might be faster to iterate edges and compute the mask.
    # For a fixed edge e = {v0, v1, v2, v3, v4}:
    # The colorings where it is mono-0 are those where bits v0..v4 are 0.
    # The colorings where it is mono-1 are those where bits v0..v4 are 1.
    # We can construct these masks directly without iterating all 2^n.
    
    coverage = [0] * num_edges
    
    for idx, e in enumerate(all_edges):
        mask_mono_0 = 0
        mask_mono_1 = 0
        
        # We need to iterate over all combinations of the other n-5 vertices?
        # No, we can just iterate 0 to 2^n-1 if 2^n is small enough.
        # 2^12 = 4096. 1324 * 4096 ~ 5.4M operations. This is fast in Python? 
        # 5.4M simple bitwise ops might take ~0.5-1s. Acceptable.
        # But we can do it smarter.
        
        # Let's stick to the 2^n loop for clarity and moderate speed, 
        # or use a smarter generation.
        # Actually, iterating 4096 times for 1324 edges is 5.4M iterations.
        # In Python, a simple loop taking ~100ns per iteration -> 0.5s. 
        # This is fine.
        
        for c in range(num_colorings):
            is_mono_0 = True
            is_mono_1 = True
            for v in e:
                bit = (c >> v) & 1
                if bit != 0:
                    is_mono_0 = False
                if bit != 1:
                    is_mono_1 = False
                if not is_mono_0 and not is_mono_1:
                    break
            
            if is_mono_0 or is_mono_1:
                coverage[idx] |= (1 << c)
        
        if _check_time():
            break # Fallback if too slow, though unlikely

    # Greedy Set Cover
    uncovered = (1 << num_colorings) - 1
    selected_edges = []
    
    while uncovered:
        if _check_time():
            break
        best_idx = -1
        best_count = 0
        for i in range(num_edges):
            # If this edge is already selected, skip? 
            # In greedy set cover, we can pick it again but it won't help. 
            # We should pick from unselected.
            if i in selected_edges:
                continue
            cnt = bin(coverage[i] & uncovered).count('1')
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        
        if best_idx == -1:
            break # Should not happen if uncovered != 0 and all edges cover something
        
        selected_edges.append(best_idx)
        uncovered &= ~coverage[best_idx]
        
    # Convert indices to edge lists
    current_edges = [all_edges[i] for i in selected_edges]
    
    # Refinement: Try to remove edges one by one
    # We check if the remaining set still covers all colorings.
    # This is much faster than _is_2_colorable because we can use bitmasks.
    
    def is_covering(edge_indices):
        union_mask = 0
        for i in edge_indices:
            union_mask |= coverage[i]
        return (union_mask & ((1 << num_colorings) - 1)) == ((1 << num_colorings) - 1)
    
    # Greedy removal refinement
    # Repeat a few times to shake out local minima
    for _ in range(3):
        if _check_time():
            break
        improved = True
        while improved:
            improved = False
            # Shuffle for randomness in removal order? 
            # Or just iterate.
            for i in range(len(selected_edges)):
                if _check_time():
                    break
                # Try removing edge at index i
                candidate = selected_edges[:i] + selected_edges[i+1:]
                if is_covering(candidate):
                    selected_edges = candidate
                    improved = True
                    break # Restart scan
    
    current_edges = [all_edges[i] for i in selected_edges]
    
    # Final verification with the exact checker to be safe
    if _is_2_colorable(n, current_edges):
        # This should not happen. If it does, we have a bug or a timeout in construction.
        # Fallback to a known valid (but larger) construction?
        # K_9^5 is definitely non-2-colorable.
        n = 9
        current_edges = [list(e) for e in combinations(range(n), 5)]
        # We don't have time to minimize this, but it's a valid hypergraph.
        # However, the score will be bad.
        # Let's hope the bitmask check is correct.
        # The bitmask check `is_covering` is exact for the property "no 2-coloring exists".
        # Why? Because a 2-coloring exists iff there is a coloring c such that NO edge is monochromatic.
        # This is equivalent to: there is a c such that for all e, c is NOT in coverage[e].
        # This is equivalent to: c is NOT in (Union of coverage[e]).
        # So no 2-coloring exists iff Union of coverage[e] == All Onions.
        # So `is_covering` is exactly equivalent to `_is_2_colorable` returning False.
        
        # If this happens, it might be due to `selected_edges` being empty or invalid state.
        pass

    return {'n': n, 'edges': [list(e) for e in current_edges]}

# Alias to keep seed interface if any other part expects it, though task says construct() is entry.
# The seed had _is_2_colorable_sat. I defined _is_2_colorable.
# The task says: "Keep the function names and signatures of the seed program."
# Seed had:
# def _is_2_colorable_sat(n, edges):
# def construct():
# I defined _is_2_colorable. I should alias _is_2_colorable_sat to _is_2_colorable just in case.
_is_2_colorable_sat = _is_2_colorable

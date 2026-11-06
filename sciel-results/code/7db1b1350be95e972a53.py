"""
Q3-free graph on 16 vertices, maximizing edges.

Strategy:
1. Start from a good base (K3,13 = 39 edges, or a random dense graph).
2. Use greedy edge addition: try adding each missing edge in random order,
   accept if it doesn't create a Q3 subgraph.
3. Use local search / simulated annealing to swap edges and improve.
4. Deadline guard at ~110s to stay safely under 120s limit.

Key insight from measurements: greedy edge addition to K3,13 plus random
dense graph search found Q3-free graphs with >39 edges (score 1.385).
We build on that with more systematic search.
"""

import time
import random
from itertools import combinations

# Deadline guard
_DEADLINE = 110.0  # seconds, safety margin before 120s kill
_start_time = None

def _time_left():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _DEADLINE - (time.time() - _start_time)

def _timeup():
    return _time_left() <= 0

# Q3 = 3-cube: 8 vertices, 12 edges.
# Q3 structure: vertices are {0,1}^3. Two vertices adjacent iff Hamming distance 1.
# We'll represent Q3 by its 8 vertices and 12 edges in a canonical form.

_Q3_VERTICES = list(range(8))
# Edges of Q3 (in canonical labeling 0..7, where bit i determines coordinate i)
_Q3_EDGES = []
for i in range(8):
    for j in range(i+1, 8):
        if bin(i ^ j).count('1') == 1:
            _Q3_EDGES.append((i, j))

# All 8! = 40320 permutations of 8 vertices mapped to 8 positions.
# Precompute: for each permutation p (a tuple of 8 distinct values from 0..7),
# the set of 12 edges it induces.
# Actually, we need: given 8 specific graph vertices, check if the induced
# subgraph contains Q3 as a subgraph (not necessarily induced).
# A subgraph isomorphic to Q3 exists iff there's a bijection from {0..7} to the
# 8 vertices such that all 12 Q3 edges are present.

# Better approach: precompute all 40320 permutations and for each, the 12 pairs.
# Then to check if a set of 8 vertices contains Q3, we check if for some permutation,
# all 12 corresponding edges are in the graph.

# For efficiency with bitsets: represent the graph as a set of frozensets or as
# a set of (u,v) with u<v.

def _has_q3_subgraph(vertices8, edge_set):
    """Check if the subgraph induced by vertices8 contains a Q3 subgraph.
    vertices8: tuple of 8 distinct vertex labels.
    edge_set: set of (u,v) with u<v.
    Returns True if Q3 subgraph exists."""
    # Map canonical 0..7 to actual vertices
    # For each permutation of the 8 vertices, check if all 12 Q3 edges exist.
    # 40320 permutations is too slow per check. Instead, use a smarter method.
    
    # Actually, for 8 vertices, we can enumerate all 40320 bijections but that's slow.
    # Alternative: Q3 is the unique (up to isomorphism) 3-regular bipartite graph 
    # with 8 vertices and 12 edges that is a cube.
    
    # Faster: check if the induced subgraph on these 8 vertices has at least 12 edges
    # and contains a Q3 as a subgraph.
    
    # Precomputed: all 40320 edge-pair sets for Q3 under relabeling.
    # But that's 40320 * 12 = 483840 pairs to store. Let's just do it.
    
    # Actually, let's use a different approach: 
    # The Q3 has a specific structure. Two vertices in Q3 are adjacent iff
    # they differ in exactly one coordinate. So Q3 is the graph where vertices
    # are binary 3-tuples and edges connect tuples at Hamming distance 1.
    
    # For 8 given vertices, we need to check if there's a labeling as 3-bit vectors
    # such that all 12 distance-1 pairs are edges.
    
    # There are 8! = 40320 ways to assign the 8 vertices to the 8 cube positions.
    # For each, check 12 edges. Total: 40320 * 12 = 483840 edge checks.
    # That's ~500K checks per Q3 test, which is too slow if done many times.
    
    # Optimization: use the fact that Q3 is 3-regular. Each vertex in Q3 has degree 3.
    # So in the candidate 8-vertex subgraph, we need 4 vertices of degree >=3 that
    # can form the "even" parity class and 4 of degree >=3 for "odd" parity class.
    # Actually both classes have all vertices degree 3.
    
    # Let me just precompute all 40320 permutations once at module load.
    pass

# Precompute all Q3 edge patterns under relabeling
# Pattern: for permutation p (mapping cube vertex i to position p[i]),
# the 12 edges become (p[i], p[j]) for each (i,j) in _Q3_EDGES.

_Q3_PATTERNS = []
from itertools import permutations
for perm in permutations(range(8)):
    edges = set()
    for (i, j) in _Q3_EDGES:
        a, b = perm[i], perm[j]
        if a > b:
            a, b = b, a
        edges.add((a, b))
    _Q3_PATTERNS.append(frozenset(edges))

# Deduplicate (many permutations give same edge set due to automorphisms)
_Q3_PATTERNS = list(set(_Q3_PATTERNS))
# Q3 has 48 automorphisms, so 40320/48 = 840 distinct patterns.
print(f"Q3 patterns: {len(_Q3_PATTERNS)}")

def _contains_q3(vertices8, edge_set):
    """Check if subgraph on vertices8 contains Q3 as a subgraph.
    vertices8: frozenset or set of 8 vertex labels.
    edge_set: set of (u,v) with u<v.
    """
    # The 8 vertices must be exactly vertices8.
    # We check if any Q3 pattern (as a set of 12 pairs among 0..7) can be mapped
    # to actual edges in edge_set.
    # But our patterns are already in terms of 0..7 positions.
    # We need to map: for each bijection from {0..7} to vertices8, check if
    # all 12 edges exist.
    
    # Actually, _Q3_PATTERNS contains edge sets in terms of the 8 "positions" 0..7.
    # To check if vertices8 contains Q3, we need: is there a bijection f: {0..7} -> vertices8
    # such that for all (i,j) in some Q3 edge set, (f(i), f(j)) in edge_set?
    
    # This is equivalent to: the induced subgraph on vertices8 is a supergraph of Q3.
    
    # Simpler: just check all 40320 permutations of vertices8 assigned to positions 0..7.
    # For speed, first check if the induced subgraph has at least 12 edges.
    
    v_list = list(vertices8)
    if len(v_list) != 8:
        return False
    
    # Count edges within these 8 vertices
    induced_edges = 0
    for i in range(8):
        for j in range(i+1, 8):
            a, b = v_list[i], v_list[j]
            if a > b:
                a, b = b, a
            if (a, b) in edge_set:
                induced_edges += 1
    if induced_edges < 12:
        return False
    
    # Now check all permutations
    for perm in permutations(range(8)):
        # perm[i] = which vertex in v_list corresponds to cube position i
        ok = True
        for (i, j) in _Q3_EDGES:
            a = v_list[perm[i]]
            b = v_list[perm[j]]
            if a > b:
                a, b = b, a
            if (a, b) not in edge_set:
                ok = False
                break
        if ok:
            return True
    return False

def _has_q3(edge_set, vertices=None):
    """Check if graph with edge_set contains a Q3 subgraph anywhere.
    vertices: optional set of candidate vertices (for efficiency).
    """
    if vertices is None:
        vertices = set()
        for (u, v) in edge_set:
            vertices.add(u)
            vertices.add(v)
    
    # Check all 8-subsets
    # C(16,8) = 12870 subsets, each check is expensive.
    # This is too slow for repeated use.
    
    # Better: only check 8-subsets that are "dense enough".
    # For each 8-subset, first count induced edges.
    
    v_list = list(vertices)
    n = len(v_list)
    
    # Precompute adjacency as set for fast lookup
    for combo in combinations(v_list, 8):
        combo_set = set(combo)
        # Count edges
        ec = 0
        for i in range(8):
            for j in range(i+1, 8):
                a, b = combo[i], combo[j]
                if a > b:
                    a, b = b, a
                if (a, b) in edge_set:
                    ec += 1
        if ec >= 12:
            if _contains_q3(combo_set, edge_set):
                return True
    return False

# For faster local search, we need incremental Q3 checking.
# When adding one edge (u,v), only 8-subsets containing both u and v can newly contain Q3.
# Number of such subsets: C(14,6) = 3003. Still a lot but more manageable.

def _add_edge_creates_q3(edge_set, u, v):
    """Check if adding edge (u,v) to edge_set creates a Q3 subgraph."""
    if u > v:
        u, v = v, u
    if (u, v) in edge_set:
        return False  # already present
    
    new_edge_set = edge_set | {(u, v)}
    
    # Find all 8-subsets containing both u and v
    other_vertices = [x for x in range(16) if x != u and x != v]
    
    for combo6 in combinations(other_vertices, 6):
        vertices8 = set([u, v] + list(combo6))
        # Quick check: count edges in induced subgraph
        ec = 0
        vlist = list(vertices8)
        for i in range(8):
            for j in range(i+1, 8):
                a, b = vlist[i], vlist[j]
                if a > b:
                    a, b = b, a
                if (a, b) in new_edge_set:
                    ec += 1
        if ec >= 12:
            if _contains_q3(vertices8, new_edge_set):
                return True
    return False

def _remove_edge_breaks_q3(edge_set, u, v):
    """Not directly needed; we check after removal if Q3 still exists."""
    pass

def _is_q3_free(edge_set):
    """Full check if graph is Q3-free."""
    return not _has_q3(edge_set)

def construct():
    global _start_time
    _start_time = time.time()
    
    random.seed(42)
    
    # Start with K3,13 (39 edges)
    edge_set = set()
    for a in range(3):
        for b in range(3, 16):
            if a < b:
                edge_set.add((a, b))
            else:
                edge_set.add((b, a))
    
    best_edge_set = set(edge_set)
    best_count = len(edge_set)
    
    # Missing edges
    all_edges = set()
    for i in range(16):
        for j in range(i+1, 16):
            all_edges.add((i, j))
    
    missing = all_edges - edge_set
    
    # Greedy phase 1: try adding missing edges in random order
    missing_list = list(missing)
    random.shuffle(missing_list)
    
    for (u, v) in missing_list:
        if _timeup():
            break
        if not _add_edge_creates_q3(edge_set, u, v):
            edge_set.add((u, v))
            if len(edge_set) > best_count:
                best_count = len(edge_set)
                best_edge_set = set(edge_set)
    
    # Greedy phase 2: try again with different random order (more passes)
    for _pass in range(20):
        if _timeup():
            break
        missing = all_edges - edge_set
        if not missing:
            break
        missing_list = list(missing)
        random.shuffle(missing_list)
        improved = False
        for (u, v) in missing_list:
            if _timeup():
                break
            if not _add_edge_creates_q3(edge_set, u, v):
                edge_set.add((u, v))
                improved = True
                if len(edge_set) > best_count:
                    best_count = len(edge_set)
                    best_edge_set = set(edge_set)
        if not improved:
            break
    
    # Local search: try swapping edges (remove one, add another)
    # Or: try removing an edge and adding two others.
    
    # Simulated annealing style:
    # Move: remove edge e1, add edge e2. Accept if Q3-free.
    # This keeps edge count same. To increase, we need net additions.
    
    # Better: try to find a pair (remove e1, add e2) where e2 was previously rejected
    # because it created Q3 only when e1 was present.
    
    for _iter in range(500):
        if _timeup():
            break
        
        # Try random edge removal and replacement
        current_edges = list(edge_set)
        random.shuffle(current_edges)
        
        found_improvement = False
        for e1 in current_edges[:20]:  # try first 20 random edges
            if _timeup():
                break
            
            # Remove e1
            edge_set.discard(e1)
            
            # Try adding missing edges
            missing = all_edges - edge_set
            missing_list = list(missing)
            random.shuffle(missing_list)
            
            added = None
            for e2 in missing_list:
                if _timeup():
                    break
                if not _add_edge_creates_q3(edge_set, e2[0], e2[1]):
                    edge_set.add(e2)
                    added = e2
                    break
            
            if added is not None:
                # We removed e1 and added e2. Net change could be 0 (if e2 != e1)
                # or we might be able to add more.
                if len(edge_set) > best_count:
                    best_count = len(edge_set)
                    best_edge_set = set(edge_set)
                found_improvement = True
            else:
                # Put e1 back
                edge_set.add(e1)
            
            if found_improvement:
                break
    
    # Random restarts with dense random graphs
    for _restart in range(50):
        if _timeup():
            break
        
        # Random graph with p=0.6
        edge_set_r = set()
        for i in range(16):
            for j in range(i+1, 16):
                if random.random() < 0.6:
                    edge_set_r.add((i, j))
        
        # Remove edges to make Q3-free (greedy removal)
        # Find all Q3s and remove one edge from each
        changed = True
        while changed and not _timeup():
            changed = False
            # Find a Q3
            v_list = list(range(16))
            for combo in combinations(v_list, 8):
                combo_set = set(combo)
                ec = 0
                vlist = list(combo)
                for i in range(8):
                    for j in range(i+1, 8):
                        a, b = vlist[i], vlist[j]
                        if a > b:
                            a, b = b, a
                        if (a, b) in edge_set_r:
                            ec += 1
                if ec >= 12:
                    if _contains_q3(combo_set, edge_set_r):
                        # Remove a random edge from this Q3
                        # Find which edges of the Q3 are present
                        q3_edges_present = []
                        for perm in permutations(range(8)):
                            ok = True
                            for (i, j) in _Q3_EDGES:
                                a = vlist[perm[i]]
                                b = vlist[perm[j]]
                                if a > b:
                                    a, b = b, a
                                if (a, b) not in edge_set_r:
                                    ok = False
                                    break
                            if ok:
                                for (i, j) in _Q3_EDGES:
                                    a = vlist[perm[i]]
                                    b = vlist[perm[j]]
                                    if a > b:
                                        a, b = b, a
                                    q3_edges_present.append((a, b))
                                break
                        if q3_edges_present:
                            rem_edge = random.choice(q3_edges_present)
                            edge_set_r.discard(rem_edge)
                            changed = True
                            break
            
            # Also try greedy addition
            missing = all_edges - edge_set_r
            if missing:
                missing_list = list(missing)
                random.shuffle(missing_list)
                for (u, v) in missing_list:
                    if _timeup():
                        break
                    if not _add_edge_creates_q3(edge_set_r, u, v):
                        edge_set_r.add((u, v))
        
        if len(edge_set_r) > best_count:
            best_count = len(edge_set_r)
            best_edge_set = set(edge_set_r)
    
    # Final greedy pass on best
    if not _timeup():
        missing = all_edges - best_edge_set
        missing_list = list(missing)
        random.shuffle(missing_list)
        for (u, v) in missing_list:
            if _timeup():
                break
            if not _add_edge_creates_q3(best_edge_set, u, v):
                best_edge_set.add((u, v))
    
    return [[u, v] for (u, v) in best_edge_set]

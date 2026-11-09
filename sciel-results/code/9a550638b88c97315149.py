"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

_N = 16
_ALL_PAIRS = list(itertools.combinations(range(_N), 2))
_N_PAIRS = len(_ALL_PAIRS)  # 120
_PAIR_TO_IDX = {p: i for i, p in enumerate(_ALL_PAIRS)}
_IDX_TO_PAIR = list(_ALL_PAIRS)

# Precompute for each vertex, the list of pairs incident to it
_VERTEX_PAIRS = [[] for _ in range(_N)]
for i, (u, v) in enumerate(_ALL_PAIRS):
    _VERTEX_PAIRS[u].append(i)
    _VERTEX_PAIRS[v].append(i)

# Q3 has 8 vertices and 12 edges.
# A subgraph isomorphic to Q3: we need to check if the 12 edges of some
# embedding of Q3 are all present in our graph.
#
# Strategy: enumerate all 8-vertex subsets? That's C(16,8)=12870, and for each
# check if it contains a Q3. Too slow potentially.
#
# Better: use bitmask representation. Represent graph as a 120-bit integer
# (or use a set of edge indices). For Q3 detection, we can use the fact that
# Q3 is a 3-cube. A graph on 8 vertices contains Q3 as a subgraph iff it has
# at least 12 edges and contains a specific structure.
#
# Actually, the most efficient way: precompute all possible Q3 embeddings.
# A Q3 embedding is a bijection from {0,1,2,3,4,5,6,7} to 8 vertices of K16.
# The 12 edges of Q3 in the abstract are:
# (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
#
# Number of embeddings: P(16,8) * (aut(Q3)) but we can just enumerate all
# injective maps from 8 cube vertices to 16 graph vertices, check if all 12
# edges are present.
#
# That's 16*15*14*13*12*11*10*9 = 518,918,400 which is too many.
#
# Alternative: use the fact that Q3 is a bipartite graph with parts of size 4.
# For each 4+4 split of vertices, check if the bipartite subgraph contains Q3.
#
# Actually, let me think differently. The key insight from the knowledge:
# "Bitmask-based Q3 detection is significantly faster than matrix-based approach"
#
# Let me use a different approach: represent the graph as a set of edge bitmasks.
# For Q3 detection, I'll use the following:
# - Q3 has 8 vertices, 12 edges, each vertex degree 3.
# - To check if a set of 8 vertices induces a Q3 subgraph, we need to check
#   if there exists a labeling making all 12 edges present.
#
# More practical: iterate over all 8-subsets of vertices (C(16,8)=12870),
# and for each, check if the induced subgraph contains Q3. For a fixed 8-set,
# the induced subgraph has at most 28 edges. We need to check if it contains
# Q3 as a subgraph. Since Q3 has exactly 12 edges, and is 3-regular, we can
# check: does the 8-vertex graph have a 3-regular spanning subgraph that is Q3?
#
# Actually simpler: for an 8-vertex graph, check if it contains Q3. Since Q3
# is the only 3-regular graph on 8 vertices that is bipartite with parts of
# size 4... wait, there are multiple 3-regular graphs on 8 vertices.
#
# Let me just precompute all Q3 edge patterns on 8 labeled vertices and check
# if any permutation of our 8-vertex set matches.
#
# Q3 edges (using vertices 0-7):
# 0-1, 0-2, 0-4, 1-3, 1-5, 2-3, 2-6, 3-7, 4-5, 4-6, 5-7, 6-7
#
# For each 8-subset of {0,...,15}, and for each bijection from {0,...,7} to
# that subset, check if all 12 edges are in the graph. But that's 8! = 40320
# per subset, times 12870 subsets = 519M checks. Too slow.
#
# Better approach: use the bipartite structure. Q3 is bipartite with parts
# A={0,2,4,6} and B={1,3,5,7} (using even/odd in binary representation).
# The edges connect A to B only.
#
# For a given 8-vertex set S, we need to find a bipartition into A,B (|A|=|B|=4)
# such that all edges of Q3 are present. But there are C(8,4)=70 ways to split.
# For each split, we need to check if the bipartite graph between A and B
# contains Q3 as a subgraph. Q3 as a bipartite graph is a specific 4x4 pattern.
#
# Actually, let me use a completely different strategy:
# 
# Since we're doing local search (simulated annealing), we need fast Q3 detection
# when adding/removing a single edge. When we add edge (u,v), we only need to
# check if this creates a NEW Q3. A new Q3 must include the edge (u,v).
#
# So: given current graph G and new edge e=(u,v), check if there exists a Q3
# in G+{e} that uses edge e. This means: find 6 more vertices such that the
# 8 vertices form a Q3 containing e.
#
# In Q3, each edge is part of exactly 2 faces (squares). Actually in Q3, each
# edge belongs to exactly 2 4-cycles (faces). And each edge is in exactly
# C(12-1, 11-1)... let me think. In Q3, each edge is in 2 squares.
#
# More directly: in Q3, if we fix an edge (a,b), the remaining 6 vertices and
# 10 edges must complete the cube. The structure is rigid.
#
# Let me enumerate: in Q3 with vertices as 3-bit strings, edge (000,001).
# The cube is determined. The other 6 vertices are 010,011,100,101,110,111.
# The edges among these and to 000,001 are fixed.
#
# So for a fixed edge (u,v) in our graph, a Q3 containing it corresponds to
# choosing 6 more vertices and a specific structure. The number of Q3s
# containing a given edge in K16: we need to choose 6 more vertices from 14,
# and then check if the specific Q3 structure exists.
#
# Actually, the number of distinct Q3 subgraphs in K16 containing a specific edge:
# Choose 6 more vertices from remaining 14: C(14,6) = 3003. For each such set,
# check if it forms a Q3 with the fixed edge. But the structure is very specific.
#
# Let me just precompute all Q3s in K16. A Q3 is determined by its 8 vertices
# and the specific edge pattern. The number of Q3 subgraphs in K16 is:
# C(16,8) * (number of Q3 structures on 8 labeled vertices) / (aut(Q3))
#
# Number of non-isomorphic Q3 structures on 8 labeled vertices: the number of
# distinct edge sets that form Q3 on vertices {0,...,7}. Since Q3 is unique up
# to isomorphism, the number is 8!/|Aut(Q3)|. |Aut(Q3)| = 48 (hyperoctahedral group).
# So 40320/48 = 840 distinct edge patterns.
#
# Total Q3s in K16: C(16,8) * 840 = 12870 * 840 = 10,810,800. That's a lot
# to precompute but manageable if we're smart.
#
# Actually, for the local search, I'll use a different approach:
# When adding edge (u,v), check all 6-vertex subsets of the remaining 14 vertices.
# For each, check if {u,v} + subset forms a Q3. This is C(14,6)=3003 checks per
# edge addition. Each check is O(1) if we precompute.
#
# Let me precompute: for each edge e and each 6-set S disjoint from e's vertices,
# whether e ∪ S forms a Q3. Store as a list for each edge of the 6-sets that
# complete it to a Q3.
#
# Total storage: 120 edges * 3003 6-sets = 360,360 entries. Each entry is a
# bitmask of 6 vertices. This is feasible.
#
# Wait, but we also need to check that ALL 12 edges of the Q3 are present in
# the current graph. So for a candidate Q3 (determined by edge e and 6-set S),
# we need to verify that the other 11 edges are in the graph.
#
# Let me restructure:
# 1. Precompute all Q3s in K16 as frozensets of 12 edge indices.
# 2. For each Q3, store the set of its 12 edge indices.
# 3. During search, maintain a count for each Q3 of how many of its edges are present.
# 4. When adding edge e, increment count for all Q3s containing e. If any reaches 12, reject.
# 5. When removing edge e, decrement count for all Q3s containing e.
#
# Number of Q3s: ~10.8M. Each has 12 edges. Total edge-Q3 incidences: 10.8M * 12 = 129.6M.
# That's too much memory.
#
# Alternative: don't precompute all Q3s. Instead, for each edge, precompute the
# list of 6-sets that with it form a Q3. Then when adding edge e, for each such
# 6-set, check if the other 11 edges are present.
#
# For each edge, there are C(14,6)=3003 candidate 6-sets. But most won't form a Q3.
# The number of Q3s containing a specific edge in K16: 
# In K16, fix edge (0,1). Choose 6 more vertices from {2,...,15}. For each choice,
# there's at most one way to make it a Q3 (since Q3 structure is rigid given an edge).
# Actually, the number of Q3s in K16 containing a specific edge is C(14,6) * (fraction
# that actually form Q3). But since we're in K16 (complete graph), every 8-vertex set
# contains many Q3s. The number of Q3s on a specific 8-vertex set is 840 (as computed).
# Each Q3 has 12 edges, so each edge in the 8-set is in 840*12/28 = 360 Q3s.
# Wait, that's per 8-set. Total Q3s containing edge (0,1) = C(14,6) * 360? No...
#
# Let me just count: total Q3s in K16 = C(16,8) * 840 = 10,810,800.
# Each Q3 has 12 edges, total edge-Q3 incidences = 10,810,800 * 12 = 129,729,600.
# Per edge: 129,729,600 / 120 = 1,081,080 Q3s per edge. That's a lot.
#
# This approach is too memory-heavy. Let me use a different strategy.
#
# Strategy: Use simulated annealing with fast Q3 detection.
# For Q3 detection when adding edge (u,v):
#   - The new Q3 must use edge (u,v).
#   - In Q3, edge (u,v) has a specific local structure.
#   - Enumerate possible "neighbors" of u and v in the Q3.
#
# In Q3, each vertex has degree 3. So u has 2 other neighbors in the Q3 (besides v),
# and v has 2 other neighbors (besides u). 
#
# Let me enumerate all Q3 structures containing a specific edge. Fix edge (0,1) in
# Q3 with vertices {0,1,2,3,4,5,6,7}. The Q3 edges are:
# 0-1, 0-2, 0-4, 1-3, 1-5, 2-3, 2-6, 3-7, 4-5, 4-6, 5-7, 6-7
#
# Vertex 0 connects to: 1, 2, 4
# Vertex 1 connects to: 0, 3, 5
#
# So the other neighbors of 0 are {2,4} and of 1 are {3,5}.
# Then 2 connects to 0,3,6; 4 connects to 0,5,6; 3 connects to 1,2,7; 5 connects to 1,4,7;
# 6 connects to 2,4,7; 7 connects to 3,5,6.
#
# So given edge (u,v) in our graph, a Q3 containing it requires:
# - Two other neighbors a,b of u (a≠b, a,b≠v)
# - Two other neighbors c,d of v (c≠d, c,d≠u)
# - One more vertex e (not in {u,v,a,b,c,d})
# - One more vertex f (not in {u,v,a,b,c,d,e})
# - Edges: u-a, u-b, v-c, v-d, a-c, a-e, b-d, b-e, c-f, d-f, e-f, and the cube structure.
#
# This is getting complex. Let me just use a simpler but still fast method:
#
# For each pair of vertices (u,v), precompute the list of 6-tuples (a,b,c,d,e,f)
# such that {u,v,a,b,c,d,e,f} with specific edges forms a Q3. Then when adding
# edge (u,v), check if any such 6-tuple has all its required edges present.
#
# But precomputing this is also heavy. Let me just do the check on-the-fly
# with a reasonable bound.
#
# Actually, let me use the approach from the knowledge: "simulated annealing-style
# edge swaps can escape local optima". I'll use a bitmask for the graph and
# precompute Q3 patterns more cleverly.
#
# New plan:
# 1. Represent graph as a set of edge indices (0 to 119).
# 2. Precompute for each edge e, the list of "Q3 completions": for each 6-set S
#    disjoint from e, if e∪S can form a Q3, store the 11 other edges needed.
# 3. During search, when adding edge e, check all completions: if all 11 other
#    edges are present, reject.
#
# Step 2 precomputation: For each edge e (120 edges), for each 6-subset of the
# 14 remaining vertices (3003 subsets), check if the 8 vertices can form a Q3
# with e as one of the edges. If yes, store the 11 other edge indices.
#
# Total precomputation: 120 * 3003 = 360,360 checks. Each check: verify if the
# 8 vertices with a specific Q3 structure containing e works. This is O(1) per
# check if done right.
#
# For a fixed 8-vertex set and fixed edge e in it, the number of Q3 structures
# on those 8 vertices containing e is: 840 Q3s total on 8 vertices, each has 12
# edges, so each edge is in 840*12/28 = 360 Q3s. But we're fixing which edge e
# is, so for a specific 8-set and specific edge in it, there are 360 Q3s.
#
# Wait, that means for each (edge, 6-set) pair, there are up to 360 Q3 structures.
# That's too many to store.
#
# Let me simplify: I'll just check Q3 existence by trying all 840 Q3 patterns
# on the 8 vertices. For each pattern, check if all 12 edges are present.
#
# During local search, when adding edge (u,v):
#   For each 6-subset of remaining 14 vertices:
#     For each of the 360 Q3 patterns containing edge (u,v) on these 8 vertices:
#       Check if all 12 edges are present.
#
# That's 3003 * 360 = 1,081,080 edge checks per edge addition. Each check is
# O(1) (bit test in a bitmask). Total: ~1M operations per edge addition.
# With 120 edges, a full check is 120M operations. That might be too slow
# for many iterations.
#
# Let me use a different approach: maintain a "Q3 count" array. For each Q3 in
# K16, track how many of its 12 edges are in the current graph. When adding
# edge e, increment the count for all Q3s containing e. If any reaches 12, reject.
#
# But there are 10.8M Q3s, which is too many to store.
#
# OK let me just use a simpler heuristic approach with simulated annealing and
# a fast-but-approximate Q3 check, then verify at the end.
#
# Actually, let me use the following efficient method:
#
# Key insight: In Q3, every edge is in exactly 2 faces (4-cycles). So if I add
# edge (u,v), a new Q3 must contain a 4-cycle through (u,v). 
#
# For edge (u,v), find all common neighbors... no, in Q3, u and v are adjacent,
# so they share no common neighbors in the cube (since it's bipartite).
#
# Let me just implement a direct check: given a graph (as a set of edges) and
# a candidate edge (u,v), check if adding (u,v) creates a Q3.
#
# Method: 
# 1. Find all vertices adjacent to u in the current graph (excluding v): call this set Nu.
# 2. Find all vertices adjacent to v in the current graph (excluding u): call this set Nv.
# 3. In Q3, u has 2 other neighbors and v has 2 other neighbors.
# 4. For each pair (a,b) in Nu (a≠b) and each pair (c,d) in Nv (c≠d):
#    Check if {u,v,a,b,c,d} plus 2 more vertices can complete a Q3.
#
# This is still complex. Let me just use a timeout-based approach with a
# reasonably efficient check.
#
# I'll use the following: represent the graph as a 120-bit integer (using Python
# big integers or a list of 2 64-bit integers). For Q3 detection, I'll precompute
# all Q3 edge patterns as 120-bit masks, but only store them in a compressed form.
#
# Actually, let me just go with a practical approach:
#
# 1. Start with a good initial graph (e.g., K3,13 = 39 edges, or better).
# 2. Use simulated annealing: randomly add/remove/swap edges.
# 3. For Q3 check, use a reasonably fast method.
# 4. Keep the best graph found.
#
# For the Q3 check, I'll use: when adding edge (u,v), enumerate all 8-vertex
# subsets containing u and v, and check if the induced subgraph on those 8
# vertices (with the new edge) contains Q3.
#
# Number of 8-subsets containing u and v: C(14,6) = 3003.
# For each, check if it contains Q3: try all 840 patterns (or use a faster check).
#
# Total: 3003 * 840 = 2,522,520 pattern checks per edge addition. Each check is
# 12 edge lookups. So ~30M operations per edge addition. That's too slow.
#
# Let me optimize: for each 8-vertex subset, precompute the 840 Q3 patterns as
# bitmasks of the 28 possible edges within those 8 vertices. Then checking is
# just a bitmask AND operation.
#
# Precompute: for each 8-subset (12870 of them), store the 840 Q3 edge patterns
# as 28-bit integers. Total: 12870 * 840 = 10,810,800 entries of 28 bits.
# That's about 10.8M * 4 bytes = 43MB. Feasible.
#
# Then during search, for edge (u,v) and 8-subset S containing u,v:
#   Compute the 28-bit mask of edges present in S (from current graph + new edge).
#   Check if any of the 840 Q3 patterns is a subset of this mask.
#   This is: for each of 840 patterns, check (mask & pattern) == pattern.
#   Or: check if mask is a superset of any pattern.
#
# To speed this up, I can precompute for each 8-subset a list of the 840 patterns,
# and use a trie or other structure. But 840 checks per subset * 3003 subsets =
# 2.5M checks. Each check is O(1) with bit operations. Should be fast enough in
# C, but in Python it might be slow.
#
# Let me use numpy for the bit operations to vectorize.
#
# Actually, let me just implement it and see. The time limit is 150s.
#
# Let me use a simpler approach: start with a known good graph and do local
# search with a fast Q3 check that only checks relevant 8-subsets.

def _build_q3_patterns():
    """Precompute all Q3 edge patterns on 8 vertices (labeled 0-7)."""
    # Q3 edges: vertices are 3-bit numbers 0-7
    # Edge between i and j if they differ in exactly 1 bit
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            diff = i ^ j
            if diff & (diff - 1) == 0:  # power of 2, differ in 1 bit
                q3_edges.append((i, j))
    # q3_edges has 12 edges
    # Now generate all 8! / 48 = 840 distinct patterns by permuting vertex labels
    # Actually, let's just generate all permutations and deduplicate
    from itertools import permutations
    patterns = set()
    for perm in permutations(range(8)):
        # perm maps original vertex i to perm[i]
        edge_set = frozenset((min(perm[i], perm[j]), max(perm[i], perm[j])) for i, j in q3_edges)
        patterns.add(edge_set)
    return list(patterns)

# Precompute Q3 patterns
_Q3_PATTERNS = _build_q3_patterns()
assert len(_Q3_PATTERNS) == 840, f"Expected 840 patterns, got {len(_Q3_PATTERNS)}"

# For each 8-subset of {0,...,15}, precompute the mapping from pattern to edge indices
# and the list of Q3 patterns as tuples of edge indices
_ALL_8_SUBSETS = list(itertools.combinations(range(16), 8))

# Precompute: for each 8-subset, a list of 840 frozensets of edge indices
_8SUBSET_Q3_EDGES = {}
for subset in _ALL_8_SUBSETS:
    subset_set = set(subset)
    # Map local vertex (0-7) to global vertex
    local_to_global = {i: subset[i] for i in range(8)}
    edges_in_subset = []
    for i in range(8):
        for j in range(i+1, 8):
            u, v = local_to_global[i], local_to_global[j]
            idx = _PAIR_TO_IDX[(min(u,v), max(u,v))]
            edges_in_subset.append(idx)
    # For each Q3 pattern, get the corresponding global edge indices
    q3_edge_lists = []
    for pattern in _Q3_PATTERNS:
        global_edges = []
        for (i, j) in pattern:
            u, v = local_to_global[i], local_to_global[j]
            idx = _PAIR_TO_IDX[(min(u,v), max(u,v))]
            global_edges.append(idx)
        q3_edge_lists.append(frozenset(global_edges))
    _8SUBSET_Q3_EDGES[frozenset(subset)] = q3_edge_lists

def _has_q3(graph_edges, edge_set):
    """Check if graph (set of edge indices) contains Q3. 
    edge_set is the set of all edges in the graph."""
    # Check all 8-subsets
    for subset in _ALL_8_SUBSETS:
        subset_set = set(subset)
        # Get edges within this subset that are in the graph
        local_edges = set()
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                idx = _PAIR_TO_IDX[(min(u,v), max(u,v))]
                if idx in edge_set:
                    local_edges.add(idx)
        # Check if any Q3 pattern is a subset of local_edges
        q3_patterns = _8SUBSET_Q3_EDGES[frozenset(subset)]
        for pattern in q3_patterns:
            if pattern <= local_edges:
                return True
    return False

def _add_edge_creates_q3(edge_set, new_edge_idx):
    """Check if adding new_edge_idx to edge_set creates a Q3."""
    # Only need to check 8-subsets containing both vertices of new_edge_idx
    u, v = _IDX_TO_PAIR[new_edge_idx]
    for subset in _ALL_8_SUBSETS:
        if u not in subset or v not in subset:
            continue
        subset_set = set(subset)
        # Get edges within this subset that are in the graph (plus new edge)
        local_edges = set()
        for i in range(8):
            for j in range(i+1, 8):
                a, b = subset[i], subset[j]
                idx = _PAIR_TO_IDX[(min(a,b), max(a,b))]
                if idx == new_edge_idx or idx in edge_set:
                    local_edges.add(idx)
        # Check if any Q3 pattern is a subset of local_edges
        q3_patterns = _8SUBSET_Q3_EDGES[frozenset(subset)]
        for pattern in q3_patterns:
            if pattern <= local_edges:
                return True
    return False

def _remove_edge_safe(edge_set, edge_idx):
    """Remove edge from set. Always safe (removing can't create Q3)."""
    edge_set.discard(edge_idx)

def _init_graph():
    """Initialize with K3,13 (39 edges)."""
    edges = set()
    for a in range(3):
        for b in range(3, 16):
            idx = _PAIR_TO_IDX[(a, b)]
            edges.add(idx)
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    best_edges = _init_graph()
    best_count = len(best_edges)
    
    random.seed(42)
    
    # Simulated annealing
    current_edges = set(best_edges)
    current_count = best_count
    temp = 10.0
    cooling = 0.999
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Choose move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a random edge
            candidates = [i for i in range(_N_PAIRS) if i not in current_edges]
            if not candidates:
                continue
            edge_idx = random.choice(candidates)
            if _add_edge_creates_q3(current_edges, edge_idx):
                continue
            current_edges.add(edge_idx)
            new_count = len(current_edges)
            if new_count > best_count:
                best_count = new_count
                best_edges = set(current_edges)
        elif move_type < 0.7:
            # Remove a random edge
            if not current_edges:
                continue
            edge_idx = random.choice(list(current_edges))
            current_edges.discard(edge_idx)
        else:
            # Swap: remove one, add another
            if not current_edges:
                continue
            remove_idx = random.choice(list(current_edges))
            candidates = [i for i in range(_N_PAIRS) if i not in current_edges and i != remove_idx]
            if not candidates:
                continue
            add_idx = random.choice(candidates)
            # Try the swap
            current_edges.discard(remove_idx)
            if not _add_edge_creates_q3(current_edges, add_idx):
                current_edges.add(add_idx)
            else:
                current_edges.add(remove_idx)  # revert
                continue
        
        # Cooling
        temp *= cooling
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Convert to list of [u, v]
    result = []
    for idx in sorted(best_edges):
        u, v = _IDX_TO_PAIR[idx]
        result.append([u, v])
    
    return result

# EVOLVE-BLOCK-END

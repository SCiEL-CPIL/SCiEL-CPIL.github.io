"""Greedy dense Q3-free graph on 16 vertices: start from a known Q3-free seed and add edges greedily."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 16
DEADLINE = 145.0  # safety margin before 150s kill

# Q3: the 3-cube has 8 vertices and 12 edges.
# A subgraph isomorphic to Q3 exists iff there is a set of 8 vertices
# inducing at least 12 edges forming the cube structure.
# 
# Key insight: Q3 is bipartite with parts of size 4.
# The 8 vertices of Q3 can be labeled as binary triples.
# Two vertices are adjacent iff they differ in exactly one coordinate.
#
# For a 16-vertex graph, we need to check all subsets of 8 vertices
# that could host a Q3. There are C(16,8) = 12870 such subsets.
# For each, we check if the induced subgraph contains Q3 as a subgraph.
#
# Faster approach: use the fact that Q3 has exactly 12 edges among 8 vertices.
# A 8-vertex induced subgraph has Q3 as a subgraph iff it has >= 12 edges
# AND the edge set contains a Q3 edge pattern.
#
# Even faster: precompute all 12 edge patterns of Q3 (up to vertex relabeling).
# For each 8-subset, count edges; if < 12, skip. If >= 12, check if any
# permutation of the 8 vertices makes all 12 Q3-edges present.
#
# Best approach for speed: 
# 1. Precompute the 12 edges of Q3 as (i,j) pairs on {0..7}.
# 2. For a candidate 8-vertex subset S = {v0..v7}, for each bijection f: Q3_vertices -> S,
#    check if all 12 Q3 edges map to edges in G.
#    There are 8! = 40320 bijections, too many.
#
# Better: exploit symmetry. Q3 is vertex-transitive and edge-transitive.
# We can fix vertex 0 of Q3 to v0, and try all assignments of the 3 neighbors
# of Q3-vertex-0 to the neighbors of v0 in S.
#
# Most efficient: use bitmasks.
# Precompute Q3 adjacency as 8-bit masks: q3_adj[i] = bitmask of neighbors of i in Q3.
# For an 8-subset with vertices [v0..v7], build an 8-bit adjacency matrix.
# Check if Q3 is a subgraph: there exists a permutation p of 8 such that
# for all i, q3_adj[i] & ~(1<<i) is subset of induced_adj[p(i)] mapped back.
#
# Actually the cleanest fast check: 
# Q3 has 12 edges. In an 8-vertex graph, Q3 is a subgraph iff there's a
# labeling where all 12 Q3 edges are present. We can enumerate by fixing
# which 4 vertices form one bipartition part.
#
# Q3 bipartition: even parity {0,3,5,6} and odd parity {1,2,4,7} (using binary labels).
# Wait, let me be precise. Q3 vertices are {0,1,2,3,4,5,6,7} as binary triples.
# Vertex i has parity = popcount(i) % 2.
# Even parity: 0(000), 3(011), 5(101), 6(110) -> 4 vertices
# Odd parity: 1(001), 2(010), 4(100), 7(111) -> 4 vertices
# Edges connect even to odd only.
#
# So Q3 is bipartite with parts E={0,3,5,6} and O={1,2,4,7}.
# Each even vertex connects to 3 odd vertices, each odd to 3 even.
#
# For a given 8-subset S of our graph, Q3 is a subgraph iff:
# There exists a partition of S into E' and O' (each of size 4) such that:
#   - Every vertex in E' is adjacent in G to at least 3 vertices in O'
#   - Every vertex in O' is adjacent in G to at least 3 vertices in E'
#   - AND the specific Q3 edge pattern is satisfied (not just degrees >= 3)
#
# Actually the degree condition isn't sufficient. We need the exact edge set to contain Q3.
#
# Fastest practical check for 16 vertices:
# Precompute all C(16,4) = 1820 ways to choose the "even" part.
# For each such 4-set E, the "odd" part O is some 4-set from the remaining 12.
# But we need |O|=4 and E and O disjoint, so we need to choose O from remaining 12.
# That's C(16,4)*C(12,4) which is too large.
#
# Alternative: just iterate over all C(16,8)=12870 8-subsets, and for each,
# check if it contains Q3. For each 8-subset, there are C(8,4)=70 ways to split
# into two 4-sets. For each split (E,O), check if the bipartite graph between
# E and O contains Q3's bipartite structure (which is a 3-regular bipartite graph
# on 4+4 vertices, specifically the "cube bipartite" which is K4,4 minus a perfect matching).
#
# K4,4 minus a perfect matching has 12 edges. So the induced bipartite graph on
# (E,O) must contain K4,4 minus some perfect matching as a subgraph.
#
# Even simpler: for an 8-subset, just check if the induced subgraph has >= 12 edges
# AND for the specific Q3 pattern. Let me just use a direct check.
#
# Most pragmatic approach given time constraints:
# Precompute: for every 8-subset of {0..15}, and for every way to label it as Q3,
# record which pairs of vertices in the original 16-vertex graph must be edges.
# Then checking if a graph is Q3-free means: for no 8-subset S and no Q3-labeling of S
# are all 12 required edges present.
#
# Number of checks: C(16,8) * (number of distinct Q3 labelings up to automorphism)
# Q3 has 48 automorphisms (the hyperoctahedral group B3). So 8!/48 = 840 distinct labelings.
# Total: 12870 * 840 ≈ 10.8 million edge-set checks. Each check is 12 bitmask ANDs.
# This is feasible in ~1-2 seconds in Python with bitmasks.
#
# Let me precompute the 840 labelings as tuples of 12 edge-pairs (in the 0..7 space),
# then for each 8-subset, map to actual vertices and check.

def precompute_q3_labelings():
    """Generate all distinct Q3 labelings as sets of 12 edge-pairs on {0..7}."""
    # Q3 edges: (i,j) where popcount(i^j) == 1
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.append((i, j))
    # q3_edges has 12 edges
    assert len(q3_edges) == 12
    
    # Generate all 8! permutations, but use the automorphism group to reduce.
    # Q3 automorphisms: permutations of {0..7} that preserve the Q3 structure.
    # We'll just generate all 40320 and deduplicate.
    
    from itertools import permutations
    seen = set()
    labelings = []
    
    for perm in permutations(range(8)):
        # perm maps Q3 vertex i to position perm[i]
        # The edges in the labeled graph are: (perm[i], perm[j]) for each (i,j) in q3_edges
        edge_set = frozenset(frozenset((perm[i], perm[j])) for (i, j) in q3_edges)
        if edge_set not in seen:
            seen.add(edge_set)
            labelings.append(edge_set)
    
    print(f"  Distinct Q3 labelings: {len(labelings)}", file=__import__('sys').stderr)
    return labelings

def construct():
    t0 = time.time()
    
    def time_left():
        return (DEADLINE - 5.0) - (time.time() - t0) > 0
    
    # Precompute Q3 labelings
    labelings = precompute_q3_labelings()
    
    # Precompute all 8-subsets of {0..15}
    all_8_subsets = list(combinations(range(16), 8))
    print(f"  8-subsets: {len(all_8_subsets)}", file=__import__('sys').stderr)
    
    # For each 8-subset, precompute the list of (edge_pairs) that would create Q3
    # edge_pairs: for each labeling, the set of 12 pairs of indices into the 8-subset
    # Actually, let's precompute for each 8-subset S=(s0..s7), for each labeling L,
    # the actual vertex pairs that must all be edges.
    
    # For speed, represent graph as a list of 16 integers (bitmasks).
    # adj[i] has bit j set iff edge (i,j) exists.
    
    def is_q3_free(adj):
        """Check if the graph with adjacency bitmasks adj is Q3-free."""
        for S in all_8_subsets:
            # S is a tuple of 8 vertices
            s0,s1,s2,s3,s4,s5,s6,s7 = S
            # For each labeling, check if all 12 edges are present
            for edge_set in labelings:
                ok = True
                for (a, b) in edge_set:
                    # a and b are in {0..7}, map to actual vertices
                    va = S[a]
                    vb = S[b]
                    if not (adj[va] >> vb) & 1:
                        ok = False
                        break
                if ok:
                    return False
        return True
    
    # Start with a known Q3-free graph: the empty graph, then add edges greedily.
    # Better: start with a dense graph known to be Q3-free.
    # The complement of Q3 has 28-12=16 edges. That's sparse.
    # 
    # Let's start with a random medium-density graph and verify it's Q3-free,
    # then add edges greedily.
    # 
    # Actually, let's just start empty and add all 120 edges in a good order.
    
    all_edges = list(combinations(range(16), 2))
    
    # Order edges to prioritize dense structures that avoid Q3.
    # Strategy: add edges in a random order, trying each one.
    # Use multiple random restarts if time allows.
    
    best_adj = [0] * 16
    best_edge_count = 0
    
    def edge_count(adj):
        c = 0
        for i in range(16):
            c += bin(adj[i]).count('1')
        return c // 2
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def tries_exceed_time():
        return (time.time() - t0) > (DEADLINE - 10.0)
    
    # Try multiple random orderings
    random.seed(42)
    
    for trial in range(100):
        if not time_left():
            break
        
        adj = [0] * 16
        edges_in_graph = set()
        
        # Shuffle edge order
        ordered_edges = all_edges[:]
        random.shuffle(ordered_edges)
        
        for (u, v) in ordered_edges:
            if tries_exceed_time():
                break
            # Try adding edge (u,v)
            add_edge(adj, u, v)
            edges_in_graph.add((u, v) if u < v else (v, u))
            
            if not is_q3_free(adj):
                remove_edge(adj, u, v)
                edges_in_graph.discard((u, v) if u < v else (v, u))
        
        cnt = edge_count(adj)
        if cnt > best_edge_count:
            best_edge_count = cnt
            best_adj = adj[:]
        
        if not time_left():
            break
    
    # Local search: try to improve by swapping edges
    if time_left():
        adj = best_adj[:]
        
        # Build set of current edges
        current_edges = set()
        missing_edges = set()
        for (u, v) in all_edges:
            if (adj[u] >> v) & 1:
                current_edges.add((u, v))
            else:
                missing_edges.add((u, v))
        
        improved = True
        while improved and time_left():
            improved = False
            
            # Try adding a missing edge
            for (u, v) in list(missing_edges):
                if tries_exceed_time():
                    break
                add_edge(adj, u, v)
                if is_q3_free(adj):
                    missing_edges.discard((u, v))
                    current_edges.add((u, v))
                    improved = True
                else:
                    remove_edge(adj, u, v)
            
            # Try removing a current edge and adding a different one (swap)
            if not improved:
                for (u, v) in list(current_edges):
                    if tries_exceed_time():
                        break
                    remove_edge(adj, u, v)
                    for (a, b) in list(missing_edges):
                        add_edge(adj, a, b)
                        if is_q3_free(adj):
                            # Accept the swap
                            current_edges.discard((u, v))
                            current_edges.add((a, b))
                            missing_edges.discard((a, b))
                            missing_edges.add((u, v))
                            improved = True
                            break
                        else:
                            remove_edge(adj, a, b)
                    if improved:
                        break
                    else:
                        # Restore
                        add_edge(adj, u, v)
    
    # Convert back to edge list
    result_edges = []
    for i in range(16):
        for j in range(i+1, 16):
            if (best_adj[i] >> j) & 1:
                result_edges.append([i, j])
    
    print(f"  Best edge count: {len(result_edges)}", file=__import__('sys').stderr)
    return result_edges

# EVOLVE-BLOCK-END

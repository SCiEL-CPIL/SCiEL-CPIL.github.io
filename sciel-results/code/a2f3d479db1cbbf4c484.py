"""Q3-free graph on 16 vertices: maximize edges via randomized greedy with fast Q3 check."""
import time
import random
import itertools
import sys

# Deadline guard: evaluator kills at 150s, we target finishing by ~110s for safety.
_DEADLINE = 110.0

# Precompute all 8 vertices of Q3 as 3-bit tuples
_Q3_VERTS = list(itertools.product([0, 1], repeat=3))  # 8 vertices
# Edges of Q3: pairs differing in exactly one coordinate
_Q3_EDGES = []
for i in range(8):
    for j in range(i + 1, 8):
        a = _Q3_VERTS[i]
        b = _Q3_VERTS[j]
        diff = sum(1 for k in range(3) if a[k] != b[k])
        if diff == 1:
            _Q3_EDGES.append((i, j))
# 12 edges in Q3
assert len(_Q3_EDGES) == 12

# For each Q3 edge (i,j), the set of other Q3 vertices that form a "neighborhood pattern"
# We'll use a different approach: for a given graph, check if it contains Q3 as a subgraph.
# Since 16 vertices and we're building incrementally, we check after each edge addition.

# Faster: precompute all injective mappings from Q3 vertices to graph vertices is too expensive.
# Instead, for each edge (u,v) added, check if it completes a Q3.
# A Q3 has 12 edges. When we add edge (u,v), it could be the 12th edge of some Q3.
# For a fixed edge (u,v) in the graph, a Q3 containing it requires:
#   - u and v are adjacent in Q3 (differ in 1 bit)
#   - 6 more vertices that complete the cube
# 
# Alternative: since 16 vertices, we can enumerate all 16P8 = 16!/(16-8)! = 518,918,400 mappings.
# That's too many. But we can be smarter.

# Better approach: For each set of 8 vertices, check if the induced subgraph contains Q3.
# C(16,8) = 12870 subsets. For each, check if the induced subgraph on those 8 vertices
# contains Q3 as a subgraph (not necessarily induced). 
# For 8 vertices, Q3 is a specific 12-edge graph. We need to check if the induced subgraph
# on 8 vertices has all 12 edges of Q3 present (under some labeling).
# Since Q3 has 8 vertices, we need to check all 8! = 40320 permutations? No, that's per subset.
# 12870 * 40320 = 518,918,400 checks. Each check is 12 edge lookups. ~6 billion ops. Too slow in Python.

# Even better: use the fact that Q3 is bipartite (4+4). 
# Precompute all pairs of 4-sets that could be the two partitions of a Q3.
# For a Q3, the two partitions each have 4 vertices, and every vertex in one partition
# is connected to exactly 3 vertices in the other partition.

# Strategy: Randomized greedy with incremental Q3 detection.
# When adding edge (u,v), check if it creates a Q3.
# A Q3 containing edge (u,v): u and v must be in opposite partitions of the Q3's bipartition.
# The Q3 has 4 vertices in each partition. u is in partition A, v is in partition B (or vice versa).
# For u to be in a Q3 with v, there must exist 3 other vertices a1,a2,a3 in A and 3 other
# vertices b1,b2,b3 in B such that the induced structure is Q3.

# Actually, let's use a different incremental check:
# After adding edge (u,v), a Q3 is formed if there exist 6 other vertices such that
# the 8-vertex subgraph contains Q3. We can enumerate:
#   - Choose 3 more vertices for u's partition (from the remaining 14): C(14,3) = 364
#   - Choose 3 more vertices for v's partition (from the remaining 11): C(11,3) = 165
#   - Total per edge: 364 * 165 = 59,960 candidate 8-sets
#   - For each, check if the 12 Q3 edges are all present: 12 checks
#   - Total per edge addition: ~720,000 operations
#   - For ~100 edges to add: ~72 million operations. Doable in ~10-20s in Python.

# Let's precompute for each pair of vertices (u,v) the list of candidate (A_set, B_set) pairs
# where A contains u, B contains v, |A|=4, |B|=4, and A,B partition 8 distinct vertices.
# Actually, we need to also consider u in B and v in A.

# Precomputation: for each ordered pair (u,v), u!=v, list of (A, B) where:
#   A is a 4-set containing u, B is a 4-set containing v, A ∩ B = ∅, A ∪ B has 8 elements.
# Then for each such (A,B), check if the 12 edges of Q3 (with u in A, v in B) are all in the graph.

# Number of such (A,B) pairs per (u,v):
#   Choose 3 more for A from remaining 14: C(14,3) = 364
#   Choose 3 more for B from remaining 11 (after removing A): C(11,3) = 165
#   Total: 364 * 165 = 59,960 per ordered pair
#   For 16*15 = 240 ordered pairs: 240 * 59,960 = 14,390,400 precomputed entries.
#   Each entry is a tuple of two frozensets. Memory: ~14M * (2*4*8 bytes) ≈ 14M * 64 ≈ 900MB. Too much.

# Alternative: don't precompute, compute on the fly but cache per (u,v) the list of (A,B).
# Or better: for each (u,v), generate the candidates lazily and check.

# Let's use a simpler but effective approach:
# Randomized greedy: start with empty graph, add edges in random order, skip if it creates Q3.
# To check if adding (u,v) creates a Q3, we enumerate all 8-sets containing u and v,
# and check if the induced subgraph on that 8-set contains Q3.
# C(14,6) = 3003 eight-sets containing u and v. For each, check if it contains Q3.
# To check if an 8-vertex graph contains Q3: try all 8! = 40320 labelings? No.
# Better: Q3 is the unique 3-regular bipartite graph on 4+4 vertices.
# So for an 8-set S, check if the induced subgraph G[S] has a Q3 subgraph.
# G[S] has at most 28 edges. We need to find if there's a 4+4 bipartition where
# each vertex has degree exactly 3 within the subgraph, and the structure is Q3.
# Actually, Q3 is the unique 3-regular bipartite graph on 8 vertices with girth 6.
# But checking this for each 8-set is still expensive.

# Let's use a more practical approach: 
# Since we're doing randomized greedy, and the graph is dense, we can use a simpler check:
# After building a candidate graph, verify it's Q3-free by checking all C(16,8)=12870 subsets.
# For each 8-subset, check if the induced subgraph contains Q3.
# To check if an 8-vertex graph contains Q3: 
#   - Q3 is 3-regular bipartite (4,4). 
#   - Enumerate all ways to split 8 vertices into two 4-sets: C(8,4)/2 = 35 bipartitions.
#   - For each bipartition (A,B), check if the induced subgraph is exactly Q3 (or contains Q3).
#   - Since we want subgraph (not induced), we need: all 12 edges of Q3 present.
#   - For a fixed bipartition (A,B), the Q3 edges are determined by the labeling.
#   - But the labeling matters. For a fixed A,B, how many Q3 subgraphs can fit?
#   - Q3 has automorphism group of size 48. The number of distinct edge sets isomorphic to Q3
#     on a fixed 4+4 vertex set: 4! * 4! / 48 = 24*24/48 = 12. Wait, let me think again.
#   - Actually, the number of ways to label Q3 on 8 specific vertices with a fixed bipartition:
#     The automorphism group of Q3 has order 48. The number of distinct embeddings = 8! / 48 = 40320/48 = 840.
#     But with fixed bipartition, we fix which 4 are in A and which 4 in B, so: 4! * 4! / (48/2) ... 
#     This is getting complicated. Let's just try all 8! labelings? No, 40320 * 12870 = 518M. Too slow.

# Practical approach: Use a randomized greedy with a fast necessary condition, then verify at the end.
# Necessary condition for Q3: the graph must not contain a 4+4 bipartite subgraph where each vertex
# has degree at least 3 in the subgraph and the total edges >= 12 with the right structure.
# Actually, let's just do the full check at the end for the best candidate, and during greedy
# use a fast heuristic to avoid obvious Q3s.

# Even better: use the known structure. Q3-free means no 8 vertices induce (contain) a Q3.
# Let's just do randomized greedy with full Q3 check after each edge, using the 8-set enumeration.
# For each new edge (u,v), check all C(14,6)=3003 eight-sets containing u,v.
# For each 8-set, check if it contains Q3. To check if an 8-vertex graph H contains Q3:
#   - H has up to 28 edges. Q3 has 12 edges.
#   - We can check: does H contain a 3-regular bipartite subgraph on 4+4 vertices?
#   - Enumerate all C(8,4)=70 ways to choose 4 vertices for one side (but we need bipartition).
#   - For each 4-set A (and B = complement), check if there's a Q3 with A and B as partitions.
#   - For fixed A,B: the Q3 on A,B is determined by which vertices in A connect to which in B.
#     There are 4! = 24 possible "perfect matching" patterns... no, Q3 is not a matching.
#     In Q3, each vertex in A connects to 3 vertices in B. So for each a in A, choose 3 of 4 in B: C(4,3)=4 choices. 4^4 = 256 possible edge sets.
#     But not all 256 are Q3. Q3 is specific. The number of distinct Q3 edge sets on fixed A,B:
#     It's the number of 3-regular bipartite graphs on 4+4 that are isomorphic to Q3.
#     Q3 is the unique 3-regular bipartite graph on 8 vertices with girth 6. 
#     Actually, there are exactly 2 non-isomorphic 3-regular bipartite graphs on 4+4: Q3 and K4,4 minus a perfect matching... no, K4,4 minus perfect matching is 3-regular bipartite on 4+4, which is actually Q3! Let me verify.
#     K4,4 has 16 edges. Remove a perfect matching (4 edges) gives 12 edges, 3-regular. Is this Q3? 
#     Q3: vertices are 3-bit strings. Bipartition by parity. Each vertex connects to 3 others (flip one bit). 
#     The complement of the bipartite adjacency: each vertex in A is NOT connected to exactly 1 vertex in B (the one differing in all 3 bits). So the non-edges form a perfect matching. Yes! Q3 = K4,4 minus a perfect matching.
#     So for fixed A,B, Q3 corresponds to choosing a perfect matching M in K4,4 and taking all edges except M.
#     Number of perfect matchings in K4,4: 4! = 24. So there are 24 possible Q3 edge sets on fixed A,B.
#     So to check if H contains Q3: for each of the 70 4-sets A (B=complement), for each of 24 perfect matchings M, check if all 12 edges of (K4,4 \ M) are in H. 
#     Total per 8-set: 70 * 24 = 1680 checks, each 12 edge lookups = 20,160 ops.
#     For 3003 eight-sets per edge: 3003 * 20160 ≈ 60.5M ops per edge. For 100 edges: 6B ops. Too slow.

# We need a faster check. Let's use the fact that we're doing randomized greedy and the graph is dense.
# Alternative: use a SAT-like approach or just do fewer restarts with a smarter check.

# Let's use a different strategy: 
# 1. Precompute all C(16,8) = 12870 eight-vertex subsets.
# 2. For each subset, precompute the list of 24*70 = 1680 possible Q3 edge sets (as bitmasks of the 28 possible edges within the 8 vertices).
# 3. During greedy, maintain the graph as a set of edges. When adding edge (u,v), for each 8-set containing u,v,
#    check if any Q3 edge set is fully contained in the current graph restricted to that 8-set.
# 4. Use bitmasks for fast checking.

# Let's implement this:
# For each 8-set S, map its 8 vertices to indices 0..7. The 28 possible edges are indexed 0..27.
# A Q3 edge set is a 28-bit integer (or Python int) with 12 bits set.
# Precompute for each 8-set S: list of 1680 Q3 masks (each a 28-bit int).
# Also precompute for each 8-set S: a mapping from (vertex_in_S, vertex_in_S) to edge_index_in_28.

# This is heavy but doable. Let's estimate memory: 12870 * 1680 * 8 bytes ≈ 173 MB. Acceptable.

# Actually, let's simplify: since we only need to check if adding (u,v) creates a Q3,
# we only need 8-sets containing both u and v. There are C(14,6) = 3003 such sets.
# For each, we need to check if the current graph (plus new edge) contains a Q3 within that 8-set.
# We can maintain for each 8-set a count of how many Q3 edge sets are "almost complete" (11 of 12 edges present).
# When adding edge (u,v), for each 8-set containing u,v, check if the new edge completes any Q3.

# Let's implement a clean randomized greedy with this approach.

def _precompute_q3_masks():
    """Precompute for each 8-subset of {0..15}, the list of Q3 edge masks (28-bit ints)."""
    from itertools import combinations
    
    all_vertices = list(range(16))
    eight_sets = list(combinations(all_vertices, 8))
    
    # For each 8-set, compute the 1680 Q3 masks
    q3_masks_per_set = []
    
    for S in eight_sets:
        # Map vertices in S to local indices 0..7
        local_map = {v: i for i, v in enumerate(S)}
        # Edge index mapping: (i,j) with i<j to index 0..27
        edge_idx = {}
        idx = 0
        for i in range(8):
            for j in range(i+1, 8):
                edge_idx[(i,j)] = idx
                idx += 1
        
        # Generate all 1680 Q3 masks
        masks = []
        for A in combinations(range(8), 4):
            B = tuple(i for i in range(8) if i not in A)
            # For each perfect matching between A and B
            # A is a 4-tuple, B is a 4-tuple
            # Perfect matching: permutation of B
            for perm in itertools.permutations(B):
                # The Q3 edge set: all edges between A and B except the matching edges
                # Matching edges: (A[k], perm[k]) for k in 0..3
                matching_edges = set()
                for k in range(4):
                    a_idx = A[k]
                    b_idx = perm[k]
                    # Convert to local edge index
                    if a_idx < b_idx:
                        matching_edges.add(edge_idx[(a_idx, b_idx)])
                    else:
                        matching_edges.add(edge_idx[(b_idx, a_idx)])
                
                # All bipartite edges between A and B: 4*4 = 16 edges
                all_bip_edges = set()
                for a in A:
                    for b in B:
                        if a < b:
                            all_bip_edges.add(edge_idx[(a, b)])
                        else:
                            all_bip_edges.add(edge_idx[(b, a)])
                
                # Q3 edges = all_bip_edges - matching_edges
                q3_edge_set = all_bip_edges - matching_edges
                # Convert to 28-bit mask
                mask = 0
                for e in q3_edge_set:
                    mask |= (1 << e)
                masks.append(mask)
        
        q3_masks_per_set.append(masks)
    
    return eight_sets, q3_masks_per_set, local_maps, edge_idxs

# This precomputation is too heavy. Let's use a simpler approach.

def construct():
    start_time = time.time()
    
    # We'll use randomized greedy with a fast Q3 check.
    # For each candidate edge (u,v), check if adding it creates a Q3.
    # Fast check: enumerate all 8-sets containing u,v, and for each, check if the
    # induced subgraph contains Q3 using a precomputed method.
    
    # Given time constraints, let's use a simpler but effective approach:
    # 1. Try multiple random orderings of edges.
    # 2. For each ordering, greedily add edges, checking Q3-freeness.
    # 3. For Q3 check, use the 8-set enumeration with a fast bit-mask method.
    
    # Precompute: for each 8-set, the list of Q3 masks.
    # But this is expensive. Let's use a different fast check:
    # When adding edge (u,v), a Q3 is created iff there exist 6 other vertices such that
    # the 8-vertex subgraph contains Q3. We check this by:
    #   For each 6-subset T of the other 14 vertices:
    #     S = {u,v} ∪ T
    #     Check if G[S] ∪ {(u,v)} contains Q3.
    #   To check if an 8-vertex graph contains Q3: 
    #     Q3 = K4,4 minus perfect matching.
    #     For each 4-subset A of S (70 choices), B = S\A:
    #       For each perfect matching M between A and B (24 choices):
    #         Check if all 12 edges of (K4,4[A,B] \ M) are in G.
    #     Total: 70 * 24 = 1680 checks per 8-set, each 12 edge lookups.
    #     Per new edge: 3003 * 1680 * 12 ≈ 60M ops. Too slow for many edges.
    
    # Let's use a different approach: since we're maximizing edges, and the seed gives 39 edges,
    # let's try to find a better structure. Known results: ex(16, Q3) is at least 39.
    # Let's try a different construction: complement of a sparse Q3-containing graph.
    # Or use a known extremal construction.
    
    # Actually, let's just do a randomized greedy with a timeout, and use a faster Q3 check
    # by precomputing for each pair (u,v) the list of 8-sets and using a cached edge set.
    
    # Simplified approach: use a set for edges, and for Q3 check, use the fact that
    # we only need to check 8-sets containing the new edge. Use a timeout per check.
    
    best_edges = None
    best_count = 0
    
    # Try several random restarts
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        # Random permutation of all 120 edges
        all_edges = [(u, v) for u in range(16) for v in range(u+1, 16)]
        random.shuffle(all_edges)
        
        current_edges = set()
        edge_list = []
        
        for (u, v) in all_edges:
            if time.time() - start_time > _DEADLINE - 2:
                break
            
            # Check if adding (u,v) creates a Q3
            if _creates_q3(current_edges, u, v):
                continue
            
            current_edges.add((u, v))
            edge_list.append([u, v])
        
        if len(edge_list) > best_count:
            best_count = len(edge_list)
            best_edges = edge_list[:]
            if best_count >= 45:  # Good enough, stop early
                break
    
    if best_edges is None:
        # Fallback: the seed construction
        best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    
    return best_edges


def _creates_q3(current_edges, u, v):
    """Check if adding edge (u,v) to current_edges creates a Q3 subgraph."""
    # Enumerate all 6-subsets of vertices other than u,v
    other_vertices = [x for x in range(16) if x != u and x != v]
    
    for T in itertools.combinations(other_vertices, 6):
        S = frozenset([u, v] + list(T))
        # Check if the subgraph on S (including the new edge (u,v)) contains Q3
        if _subgraph_contains_q3(S, current_edges, (u, v)):
            return True
    
    return False


def _subgraph_contains_q3(S, current_edges, new_edge):
    """Check if the graph on vertex set S (with edges from current_edges plus new_edge) contains Q3."""
    # S has 8 vertices
    S_list = list(S)
    local_map = {v: i for i, v in enumerate(S_list)}
    
    # Build edge set for S: edges in current_edges where both endpoints in S, plus new_edge
    local_edges = set()
    for (a, b) in current_edges:
        if a in S and b in S:
            local_edges.add(frozenset([a, b]))
    local_edges.add(frozenset([new_edge[0], new_edge[1]]))
    
    # Check if this 8-vertex graph contains Q3
    # Q3 = K4,4 minus perfect matching
    # For each 4-subset A of S (70 choices), B = S\A
    for A in itertools.combinations(S_list, 4):
        A_set = frozenset(A)
        B_set = S - A_set
        # For each perfect matching between A and B (24 choices)
        A_list = list(A)
        B_list = list(B_set)
        for perm in itertools.permutations(B_list):
            # Check if all 12 edges of (K4,4[A,B] \ matching) are present
            ok = True
            for a in A_list:
                for b in B_list:
                    # This edge should be present unless it's in the matching
                    if frozenset([a, b]) in local_edges:
                        continue
                    # Check if this is a matching edge
                    is_matching = False
                    for k in range(4):
                        if A_list[k] == a and perm[k] == b:
                            is_matching = True
                            break
                    if not is_matching:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    
    return False

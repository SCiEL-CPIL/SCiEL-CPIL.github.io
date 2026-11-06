<<<<<<< SEARCH
import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    n = 16
    
    # Precompute all 8-vertex subsets
    subsets_8 = list(itertools.combinations(range(n), 8))
    
    # For fast Q3 detection: Q3 is the 3-dimensional hypercube.
    # Two vertices in Q3 are adjacent iff their labels (3-bit) differ in exactly 1 bit.
    # We need to check if an 8-vertex induced subgraph contains Q3 as a subgraph.
    # Key insight: Q3 is bipartite with parts of size 4 (even/odd parity vertices).
    # For a candidate 8-set, we can try all 4! * 4! = 576 ways to assign the 
    # 8 vertices to the 8 positions of Q3, but that's too many.
    # Better: Precompute all 8! / 48 = 3360 distinct Q3 labelings up to automorphism.
    # Actually, let's use a different approach: 
    # Q3 has 12 edges. For a given 8-vertex set, if it contains Q3, then there exists
    # a bijection from the 8 vertices to {0,...,7} such that edges in the graph correspond
    # to pairs differing in exactly 1 bit.
    
    # Precompute: for each of the 8 positions in Q3, its neighbors (4 each)
    # pos 0 (000): neighbors 1(001), 2(010), 4(100)
    # pos 1 (001): neighbors 0(000), 3(011), 5(101)
    # pos 2 (010): neighbors 0(000), 3(011), 6(110)
    # pos 3 (011): neighbors 1(001), 2(010), 7(111)
    # pos 4 (100): neighbors 0(000), 5(101), 6(110)
    # pos 5 (101): neighbors 1(001), 4(100), 7(111)
    # pos 6 (110): neighbors 2(010), 4(100), 7(111)
    # pos 7 (111): neighbors 3(011), 5(101), 6(110)
    
    # Instead of checking all 40320 permutations, use the bipartite structure.
    # Q3 is bipartite: even parity {0,3,5,6} and odd parity {1,2,4,7}
    # For each 8-vertex set, try all ways to split into two 4-sets (70 ways),
    # then check if there's a perfect matching in the bipartite graph between them
    # that corresponds to Q3's bipartite structure.
    
    # Precompute: for each partition of {0..7} into two 4-sets, the required adjacency pattern
    # Actually, simpler: precompute all valid Q3 edge sets as frozensets of pairs of positions
    # There are only 12 edges in Q3. For a labeling, the edge set is determined.
    
    # Let's precompute all 8!/(48) = 3360 distinct Q3 subgraphs on positions 0-7
    # But that's a lot. Let's use a smarter check.
    
    # Fast check: For an 8-vertex set S, build the adjacency matrix.
    # Then for each of the 70 ways to split S into A (4 verts) and B (4 verts),
    # check if the bipartite graph between A and B contains Q3's bipartite edges.
    # Q3's bipartite edges: even->odd connections.
    # Even positions: {0,3,5,6}, Odd positions: {1,2,4,7}
    # Edges: 0-1, 0-2, 0-4, 3-1, 3-2, 3-7, 5-1, 5-4, 5-7, 6-2, 6-4, 6-7
    
    # For a given split (A,B), we need to check if there exists a bijection 
    # f: A -> {0,3,5,6} and g: B -> {1,2,4,7} such that for every Q3 edge (e,o),
    # the edge (f^{-1}(e), g^{-1}(o)) exists in the graph.
    
    # This is equivalent to: the bipartite adjacency matrix between A and B 
    # (after appropriate relabeling) contains the Q3 bipartite pattern.
    
    # Precompute: the 12 edges of Q3 as (even_idx, odd_idx) pairs where 
    # even_idx in {0,1,2,3} corresponds to {0,3,5,6} and odd_idx in {0,1,2,3} corresponds to {1,2,4,7}
    even_positions = [0, 3, 5, 6]  # indices 0,1,2,3 in even set
    odd_positions = [1, 2, 4, 7]   # indices 0,1,2,3 in odd set
    q3_edges = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,3), (2,0), (2,2), (2,3), (3,1), (3,2), (3,3)]
    q3_edges_set = set(q3_edges)
    
    # Precompute all 70 splits of {0..7} into two 4-sets
    splits = list(itertools.combinations(range(8), 4))
    
    # For each split, precompute the permutations needed
    # For a split (even_set_indices, odd_set_indices), we need to check if there exist
    # permutations pe of {0,1,2,3} and po of {0,1,2,3} such that for all (e,o) in q3_edges,
    # adj[even_set[pe[e]]][odd_set[po[o]]] is True
    
    # Precompute for each split the list of (pe, po) pairs to check - but that's 24*24=576 per split
    # Total: 70 * 576 = 40320 checks per 8-vertex set. Still too many for 12870 sets.
    
    # Better: Use Hall's theorem or bitmask matching.
    # For a given split and permutation pe of the even side, the required edges from 
    # even vertex pe[i] are to odd vertices po[0], po[1], ... according to q3_edges.
    # This is getting complex. Let's just use a fast bitset-based check with early termination.
    
    # Alternative fast approach: 
    # 1. Count edges in the 8-vertex induced subgraph. If < 12, no Q3.
    # 2. If >= 12, do a more detailed check.
    # In practice, most 8-vertex subsets in a sparse graph will have < 12 edges.
    
    # Precompute permutation lists for speed
    perms_4 = list(itertools.permutations(range(4)))
    
    def has_q3(adj_matrix):
        """
        adj_matrix: 16x16 list of lists of bools
        Returns True if the graph contains Q3 as a subgraph.
        """
        for s in subsets_8:
            # Quick edge count with early termination
            ec = 0
            for i in range(8):
                vi = s[i]
                for j in range(i+1, 8):
                    if adj_matrix[vi][s[j]]:
                        ec += 1
                        if ec >= 12:
                            break
                if ec >= 12:
                    break
            if ec < 12:
                continue
            
            # Detailed check: try all 70 splits
            for split in splits:
                even_set = [s[i] for i in split]
                # Build odd_set efficiently
                even_set_set = set(split)
                odd_set = [s[i] for i in range(8) if i not in even_set_set]
                
                # Build 4x4 adjacency matrix between even_set and odd_set
                bipartite = [[adj_matrix[even_set[i]][odd_set[j]] for j in range(4)] for i in range(4)]
                
                # Try all 24 permutations of the odd side (pe = identity)
                for po in perms_4:
                    ok = True
                    for e in range(4):
                        for o in range(4):
                            if (e, o) in q3_edges_set:
                                if not bipartite[e][po[o]]:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        return True
                
                # Try all 24 permutations of the even side (po = identity)
                for pe in perms_4:
                    ok = True
                    for e in range(4):
                        for o in range(4):
                            if (e, o) in q3_edges_set:
                                if not bipartite[pe[e]][o]:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        return True
        return False
    
    # Initial graph: Try a known Q3-free construction
    # Start with complete graph minus some edges, or use a known dense Q3-free graph
    # For n=16, a good starting point might be the complete bipartite K_{8,8} which has 64 edges
    # but K_{8,8} contains Q3? Let's think... Q3 is bipartite with parts of 4, so K_{8,8} 
    # contains many Q3 subgraphs. So K_{8,8} is NOT Q3-free.
    
    # Better starting point: Turan-type or other constructions.
    # Let's start with a random graph and use local search.
    
    adj = [[False]*n for _ in range(n)]
    current_edges = set()
    
    # Start with a moderately dense random graph
    random.seed(42)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i][j] = True
                adj[j][i] = True
                current_edges.add((i, j))
    
    # Remove any Q3 subgraphs by removing edges
    changed = True
    while changed and time.time() < deadline - 30:
        changed = False
        if has_q3(adj):
            # Find a Q3 and remove an edge from it
            # For simplicity, just remove a random edge and recheck
            edge_list = list(current_edges)
            if edge_list:
                rem_u, rem_v = random.choice(edge_list)
                adj[rem_u][rem_v] = False
                adj[rem_v][rem_u] = False
                current_edges.remove((rem_u, rem_v))
                changed = True
    
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # Local search: try adding edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for restart in range(5):
        if time.time() > deadline - 15:
            break
        
        # Work from current best
        adj = [[False]*n for _ in range(n)]
        current_edges = set(best_edges)
        for u, v in best_edges:
            adj[u][v] = True
            adj[v][u] = True
        
        non_edges = [e for e in all_edges if e not in current_edges]
        random.shuffle(non_edges)
        
        # Greedy addition
        for u, v in non_edges:
            if time.time() > deadline - 5:
                break
            if (u, v) in current_edges:
                continue
            adj[u][v] = True
            adj[v][u] = True
            current_edges.add((u, v))
            
            if has_q3(adj):
                adj[u][v] = False
                adj[v][u] = False
                current_edges.remove((u, v))
            else:
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = set(current_edges)
        
        # 2-opt: remove one edge, add up to two
        if time.time() < deadline - 10:
            rem_u, rem_v = random.choice(list(current_edges))
            adj[rem_u][rem_v] = False
            adj[rem_v][rem_u] = False
            current_edges.remove((rem_u, rem_v))
            
            non_edges_2 = [e for e in all_edges if e not in current_edges]
            random.shuffle(non_edges_2)
            added = 0
            for u, v in non_edges_2:
                if added >= 2 or time.time() > deadline - 5:
                    break
                adj[u][v] = True
                adj[v][u] = True
                current_edges.add((u, v))
                if has_q3(adj):
                    adj[u][v] = False
                    adj[v][u] = False
                    current_edges.remove((u, v))
                else:
                    added += 1
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = set(current_edges)
        
        random.shuffle(non_edges)
    
    result = [[u, v] for u, v in best_edges]
    return result
=======
import itertools
import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    n = 16
    
    # All possible edges in K16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(all_edges)  # 120
    
    # Edge index mapping
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute Q3 structure
    # Q3 vertices: 0..7, edges between vertices differing in exactly 1 bit
    q3_vert_edges = []
    for v in range(8):
        for d in range(3):
            w = v ^ (1 << d)
            if w > v:
                q3_vert_edges.append((v, w))
    # q3_vert_edges has 12 edges
    
    # For each embedding of Q3 into 16 vertices, determine which pairs of K16-edges
    # are "conflicting" (i.e., both appear in the same Q3 subgraph)
    # Two edges e1, e2 in K16 conflict if there exists a Q3 embedding containing both.
    
    # Strategy: For each choice of 8 vertices (C(16,8) = 12870), and each automorphism
    # of Q3, we get a set of 12 K16-edges that form a Q3. Any two edges in such a set
    # conflict with each other.
    
    # To be efficient, we'll build the conflict graph incrementally.
    # conflict[i][j] = True if edge i and edge j cannot both be in a Q3-free graph.
    
    # Actually, a cleaner formulation: 
    # A set of edges is Q3-free iff it does NOT contain all 12 edges of any Q3 embedding.
    # So we want the maximum subset of edges that avoids containing any complete Q3.
    # This is equivalent to: find the maximum set of edges such that for every Q3 embedding,
    # at least one edge of that embedding is NOT in our set.
    
    # This is a hitting set / partial cover problem. Alternatively, model as MWIS on a
    # hypergraph where each hyperedge is a Q3 embedding (12 edges), and we want to select
    # the max number of edges while hitting every hyperedge (i.e., not selecting all 12).
    
    # Simpler: Use branch and bound directly on edge selection with Q3-free constraint.
    
    # Precompute all Q3 embeddings as sets of edge indices.
    # For each 8-vertex subset, try all 8! / 48 = 3360 distinct Q3 labelings.
    # That's 12870 * 3360 = ~43 million, too many.
    
    # Better: For each 8-vertex subset, check if it can host a Q3 by trying to match.
    # But for the conflict graph, we need all pairs of edges that appear together in some Q3.
    
    # Alternative: Build the conflict graph by, for each pair of edges (e1, e2),
    # checking if they can be extended to a full Q3. This is expensive but doable with pruning.
    
    # Let's use a different approach: Direct B&B on the edge set with a fast Q3-free check.
    
    # Fast Q3-free check using bitsets:
    # Represent the graph as a set of edge indices. For a given edge set, check Q3-free-ness
    # by iterating over all 8-vertex subsets and checking if the induced subgraph contains Q3.
    
    # To speed up, precompute for each 8-vertex subset the 12 "required" edge patterns
    # under each Q3 labeling, and use bit operations.
    
    # Let's precompute all distinct Q3 edge patterns on 8 labeled vertices.
    # Two Q3 embeddings are equivalent up to automorphism of Q3.
    # Q3 has 48 automorphisms (the wreath product S2 wr S3).
    # So there are 8! / 48 = 3360 distinct edge sets for Q3 on 8 labeled vertices.
    
    # Precompute these 3360 patterns.
    q3_patterns = set()
    for perm in itertools.permutations(range(8)):
        # perm maps position p in Q3 to vertex perm[p] in the 8-set
        edges = frozenset()
        for (a, b) in q3_vert_edges:
            u, v = perm[a], perm[b]
            if u > v:
                u, v = v, u
            edges.add((u, v))
        q3_patterns.add(edges)
    
    # q3_patterns now has 3360 frozensets, each containing 12 pairs (0-7).
    
    # For an 8-vertex subset S = (s0, s1, ..., s7), a Q3 embedding corresponds to
    # a pattern P in q3_patterns where the edges (si, sj) for (i,j) in P are all present.
    
    # Precompute for each 8-subset S, the list of patterns that could match.
    # Actually, all 3360 patterns could potentially match any 8-subset, so we just
    # check if any pattern's edges are all in the graph.
    
    subsets_8 = list(itertools.combinations(range(n), 8))
    
    # For speed, precompute for each 8-subset S the list of edge-index pairs for each pattern.
    # pattern_edge_indices[S_idx][pattern_idx] = list of 12 edge indices in K16
    
    # This is 12870 * 3360 * 12 = ~516 million entries, too much memory.
    
    # Instead, compute on the fly with caching.
    
    # Let's use a direct B&B approach with a fast incremental Q3 check.
    
    # Key optimization: When adding an edge, only check Q3s that include that edge.
    # Precompute for each edge, the list of 8-subsets that contain both endpoints.
    
    edge_subsets = [[] for _ in range(num_edges)]
    for si, S in enumerate(subsets_8):
        S_set = set(S)
        for i in range(8):
            for j in range(i+1, 8):
                if S[i] < S[j]:
                    e = (S[i], S[j])
                else:
                    e = (S[j], S[i])
                if e in edge_to_idx:
                    edge_subsets[edge_to_idx[e]].append(si)
    
    # For each edge, store which 8-subsets contain it.
    # When we add edge e, we only need to check Q3s within those 8-subsets.
    
    # For each 8-subset, precompute the 3360 patterns as lists of 12 edge indices.
    # Cache this.
    pattern_cache = {}
    
    def get_pattern_indices(si):
        """Return list of 3360 patterns, each as a tuple of 12 edge indices, for 8-subset si."""
        if si in pattern_cache:
            return pattern_cache[si]
        S = subsets_8[si]
        S_set = set(S)
        # Map vertex to position in S
        vert_to_pos = {S[i]: i for i in range(8)}
        
        patterns = []
        for pat in q3_patterns:
            idx_list = []
            valid = True
            for (u, v) in pat:
                su, sv = S[u], S[v]
                if su < sv:
                    e = (su, sv)
                else:
                    e = (sv, su)
                if e not in edge_to_idx:
                    valid = False
                    break
                idx_list.append(edge_to_idx[e])
            if valid:
                patterns.append(tuple(idx_list))
        
        pattern_cache[si] = patterns
        return patterns
    
    # Precompute patterns for all subsets (may take a while)
    for si in range(len(subsets_8)):
        get_pattern_indices(si)
    
    # Now do B&B: select max edges such that no Q3 is fully contained.
    # State: selected set of edges, represented as a boolean array or set.
    
    # Use a greedy + local search hybrid with the fast incremental check.
    
    def is_q3_free(edge_set):
        """Check if a set of edges (as a set of edge indices) is Q3-free."""
        edge_bool = [False] * num_edges
        for e in edge_set:
            edge_bool[e] = True
        
        for si in range(len(subsets_8)):
            patterns = get_pattern_indices(si)
            for pat in patterns:
                # Check if all 12 edges in this pattern are selected
                all_present = True
                for ei in pat:
                    if not edge_bool[ei]:
                        all_present = False
                        break
                if all_present:
                    return False
        return True
    
    # This is still slow. Let's use a smarter incremental approach.
    
    # Maintain for each 8-subset, the count of how many of its Q3 patterns are "complete"
    # (all 12 edges selected). A graph is Q3-free iff no pattern is complete.
    
    # When adding edge e, for each 8-subset containing e, check if any pattern becomes complete.
    
    # Precompute for each edge e and each 8-subset si containing e, the list of patterns
    # in si that include e.
    
    edge_pattern_in_subset = {}  # (e, si) -> list of pattern indices in si that include e
    
    for e in range(num_edges):
        for si in edge_subsets[e]:
            patterns = get_pattern_indices(si)
            # Find which patterns include edge e
            pats_with_e = []
            for pi, pat in enumerate(patterns):
                if e in pat:
                    pats_with_e.append(pi)
            if pats_with_e:
                edge_pattern_in_subset[(e, si)] = pats_with_e
    
    # Now do local search with fast incremental Q3 check.
    
    def try_add_edge(edge_bool, selected_count, e):
        """Try adding edge e. Return True if graph remains Q3-free, and update edge_bool."""
        # Check all 8-subsets containing e
        for si in edge_subsets[e]:
            patterns = get_pattern_indices(si)
            pats_with_e = edge_pattern_in_subset.get((e, si), [])
            for pi in pats_with_e:
                pat = patterns[pi]
                # Check if all other 11 edges in this pattern are already selected
                all_others = True
                for ei in pat:
                    if ei != e and not edge_bool[ei]:
                        all_others = False
                        break
                if all_others:
                    # Adding e would complete this Q3
                    return False
        # Safe to add
        edge_bool[e] = True
        return True
    
    def remove_edge(edge_bool, e):
        edge_bool[e] = False
    
    # Local search
    random.seed(42)
    
    best_count = 0
    best_edge_bool = [False] * num_edges
    
    for restart in range(20):
        if time.time() > deadline - 20:
            break
        
        # Start with a random dense graph
        edge_bool = [False] * num_edges
        selected = set()
        
        # Random initial assignment
        for e in range(num_edges):
            if random.random() < 0.6:
                edge_bool[e] = True
                selected.add(e)
        
        # Remove Q3s by removing random edges
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            # Check if Q3-free
            q3_found = False
            for si in range(len(subsets_8)):
                patterns = get_pattern_indices(si)
                for pat in patterns:
                    all_present = all(edge_bool[ei] for ei in pat)
                    if all_present:
                        q3_found = True
                        # Remove a random edge from this Q3
                        rem_e = random.choice(pat)
                        edge_bool[rem_e] = False
                        selected.discard(rem_e)
                        changed = True
                        break
                if q3_found:
                    break
            if not q3_found:
                break
        
        current_count = len(selected)
        
        if current_count > best_count:
            best_count = current_count
            best_edge_bool = edge_bool[:]
        
        # Local search: try adding edges
        non_edges = [e for e in range(num_edges) if not edge_bool[e]]
        random.shuffle(non_edges)
        
        for e in non_edges:
            if time.time() > deadline - 5:
                break
            if try_add_edge(edge_bool, current_count, e):
                selected.add(e)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_edge_bool = edge_bool[:]
        
        # Try 1-1 swaps: remove one edge, try to add two
        if time.time() < deadline - 5 and len(selected) > 0:
            for _ in range(50):  # Try 50 random swaps
                if time.time() > deadline - 5:
                    break
                rem_e = random.choice(list(selected))
                edge_bool[rem_e] = False
                selected.discard(rem_e)
                current_count -= 1
                
                non_edges_2 = [e for e in range(num_edges) if not edge_bool[e]]
                random.shuffle(non_edges_2)
                added = 0
                for e in non_edges_2:
                    if added >= 2 or time.time() > deadline - 5:
                        break
                    if try_add_edge(edge_bool, current_count, e):
                        selected.add(e)
                        current_count += 1
                        added += 1
                
                if current_count > best_count:
                    best_count = current_count
                    best_edge_bool = edge_bool[:]
    
    # Convert best_edge_bool to edge list
    result = []
    for e in range(num_edges):
        if best_edge_bool[e]:
            u, v = all_edges[e]
            result.append([u, v])
    
    return result
>>>>>>> REPLACE


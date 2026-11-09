import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    n = 16
    # Vertices 0..7 are Part A, 8..15 are Part B
    # We will start with a dense bipartite graph between A and B
    # and add edges within A and within B.
    
    # Q3 is bipartite with parts of size 4.
    # If G contains Q3, it must be within a bipartite subgraph.
    # The bipartite edges of G are the edges between A and B.
    # Edges within A and within B do not contribute to the bipartite structure 
    # of a potential Q3 if we consider the "bipartite part" of G to be the A-B edges.
    # Wait, if we add edges within A, can they be part of a Q3?
    # Q3 is bipartite. A subgraph isomorphic to Q3 must be bipartite.
    # If we have edges within A, say (u,v) with u,v in A, they cannot be in the Q3 
    # unless the Q3 is not induced? No, "subgraph isomorphic" means we pick a subset of edges.
    # The Q3 subgraph itself must be bipartite.
    # If we pick a Q3 subgraph from G, its edges must form a bipartite graph.
    # The vertices of the Q3 can be split into two sets X, Y such that all Q3 edges go between X and Y.
    # If G has edges within A, they are just extra edges not used in the Q3 subgraph, 
    # UNLESS the Q3 subgraph uses vertices from both A and B in a way that mixes them?
    # No. If the Q3 subgraph uses vertices $v_1, \dots, v_8$, and we color them black/white for the Q3,
    # all edges in the Q3 connect black to white.
    # If two vertices $u, v$ are in A, and we map them to the same color in Q3 (e.g., both black), 
    # then the edge (u,v) in G is not used in the Q3 subgraph (since Q3 has no edges between blacks).
    # If we map them to different colors (u black, v white), then the edge (u,v) in G WOULD be used 
    # if it exists in the Q3 subgraph. But Q3 has no edges between vertices in the same part of its bipartition.
    # So, if we map $u \in A$ to Black and $v \in A$ to White, and (u,v) is an edge in G, 
    # it could potentially be an edge in the Q3 subgraph.
    # This means intra-partition edges CAN be part of a Q3 if the bipartition of the Q3 
    # does not align with the partition A/B of our construction.
    
    # HOWEVER, a simpler and more robust strategy:
    # 1. Construct a bipartite graph $H$ between A and B that is $Q_3$-free.
    #    Since $Q_3$ is a subgraph of $K_{4,4}$, $H$ must not contain a subgraph isomorphic to $Q_3$.
    #    Actually, if $H$ is bipartite, any subgraph is bipartite.
    #    We need $H$ to not contain $Q_3$ as a subgraph.
    # 2. Add edges within A and within B.
    #    Let $G = H \cup E(A) \cup E(B)$.
    #    Does adding $E(A)$ and $E(B)$ create a $Q_3$?
    #    A $Q_3$ in $G$ must use some edges from $H$, $E(A)$, and $E(B)$.
    #    Since $Q_3$ is bipartite, the vertices of the $Q_3$ can be 2-colored.
    #    This 2-coloring might not respect the A/B partition.
    #    Example: A Q3 could have 3 vertices in A and 5 in B? No, Q3 is 3-regular.
    #    If we make A and B cliques, we add many edges.
    #    Is it safe to assume that if the bipartite core $H$ is $Q_3$-free, and we add intra-partition edges, 
    #    we might create a Q3 that "crosses" the partition?
    #    Yes. Consider a Q3 where vertices are $a_1, a_2, a_3$ in A and $b_1, \dots, b_5$ in B.
    #    This is getting complicated.
    
    # ALTERNATIVE STRATEGY: 
    # Use the known extremal graph structure.
    # $ex(n, Q_3)$ is related to Turan graphs or similar.
    # For $n=16$, a good candidate is a dense graph that avoids the specific 8-vertex pattern.
    
    # Let's stick to a verified approach:
    # 1. Start with a known Q3-free graph (e.g., from the current program's seed or a random one).
    # 2. Use a faster local search.
    
    # The current program's bottleneck is `contains_q3_in_subset_fast`.
    # It checks all 35 splits of 8 vertices.
    # We can optimize the check by precomputing "forbidden patterns" or using bitsets more aggressively.
    
    # Let's try a different structural approach:
    # Construct a graph based on a 4-partite structure?
    # Q3 is 3-regular.
    
    # Let's go back to the bipartite idea but make it safe.
    # If we ensure that the subgraph induced by any 8 vertices is NOT a Q3, we are safe.
    # The current program does this.
    
    # How to speed it up?
    # 1. Use NumPy? No, Python loops for bit operations are often faster than NumPy overhead for small sizes.
    # 2. Better heuristic for edge addition.
    
    # Let's try a "Greedy with Backtracking" or "Simulated Annealing" style, 
    # but with a MUCH faster validity check.
    
    # Key Insight: 
    # A graph contains Q3 iff there is a set of 8 vertices $S$ and a bijection to $V(Q_3)$ 
    # such that all edges of $Q_3$ are present in $G$.
    # This is equivalent to: There exist two sets $X, Y$ of size 4, disjoint, 
    # such that the bipartite graph between $X$ and $Y$ in $G$ is a supergraph of $Q_3$.
    # $Q_3$ has 12 edges. $K_{4,4}$ has 16 edges.
    # So $G[X,Y]$ must have at least 12 edges, AND the missing 4 edges must form a matching 
    # (specifically, the complement of $Q_3$ in $K_{4,4}$ is a perfect matching).
    # Wait, $Q_3$ is $K_{4,4}$ minus a perfect matching.
    # So $G$ contains $Q_3$ iff there exists $X, Y$ (size 4) such that 
    # the edges missing in $G$ between $X$ and $Y$ are a subset of a perfect matching.
    # (Because if the missing edges are a matching $M$, then $G[X,Y] = K_{4,4} - M'$, 
    # where $M' \supseteq M$. If $M'$ is larger than a matching, $G[X,Y]$ has fewer edges than $Q_3$ 
    # or is not isomorphic. Wait.
    # If $G[X,Y]$ contains $Q_3$, then there is a subgraph of $G[X,Y]$ isomorphic to $Q_3$.
    # Since $Q_3$ has 12 edges, $G[X,Y]$ must have at least 12 edges.
    # The missing edges in $K_{4,4}$ relative to $G[X,Y]$ must be such that 
    # removing them from $K_{4,4}$ leaves a graph containing $Q_3$.
    # Actually, it's simpler: $G[X,Y]$ contains $Q_3$ iff the set of missing edges 
    # $M_{miss} = E(K_{4,4}) \setminus E(G[X,Y])$ does NOT cover all edges of any perfect matching?
    # No. $Q_3 = K_{4,4} \setminus M_{perf}$.
    # So $G[X,Y]$ contains $Q_3$ iff there exists a perfect matching $M_{perf}$ such that 
    # $M_{perf} \cap M_{miss} = \emptyset$.
    # i.e., $M_{miss}$ is a subset of $K_{4,4} \setminus M_{perf}$.
    # This means $M_{miss}$ must NOT contain a full perfect matching.
    # If $M_{miss}$ contains a perfect matching, then $G[X,Y]$ is a subgraph of $K_{4,4} \setminus M_{miss}$, 
    # which has at most $16 - 4 = 12$ edges? No.
    # If $M_{miss}$ contains a perfect matching $M_p$, then $G[X,Y]$ has no edges in $M_p$.
    # Can $G[X,Y]$ still contain $Q_3$?
    # $Q_3$ requires 12 edges. If $G[X,Y]$ misses 4 edges that form a matching, it has 12 edges.
    # If those 4 missing edges form a matching, the remaining 12 edges form a graph. 
    # Is that graph $Q_3$?
    # $K_{4,4}$ minus a perfect matching is exactly $Q_3$ (isomorphic to it).
    # So if $M_{miss}$ is exactly a perfect matching, $G[X,Y] \cong Q_3$.
    # If $M_{miss}$ is a SUPERSET of a perfect matching, $G[X,Y]$ has < 12 edges, so no Q3.
    # If $M_{miss}$ is a SUBSET of a perfect matching (size < 4), $G[X,Y]$ has > 12 edges.
    # Does a graph with > 12 edges in a 4x4 bipartite setup contain Q3?
    # Yes, any supergraph of Q3 contains Q3.
    # So the condition is: $M_{miss}$ is NOT a superset of any perfect matching.
    # Equivalently: $M_{miss}$ does NOT contain a perfect matching.
    # A set of edges contains a perfect matching iff its size is >= 4 AND it's not "deficient".
    # By Hall's theorem, a bipartite graph has a perfect matching iff for all subsets S of left, 
    # |N(S)| >= |S|.
    # So $M_{miss}$ contains a perfect matching iff it is not "Hall-deficient".
    # So $G[X,Y]$ contains $Q_3$ iff $M_{miss}$ DOES NOT contain a perfect matching.
    # Wait, this logic is inverted?
    # Let's re-verify.
    # $G$ contains $Q_3$ iff $\exists X,Y$ such that $G[X,Y]$ contains $Q_3$.
    # $G[X,Y]$ contains $Q_3$ iff $E(G[X,Y]) \supseteq E(Q_3')$ for some $Q_3' \cong Q_3$.
    # $E(Q_3') = E(K_{4,4}) \setminus M_{perf}'$ for some perfect matching $M_{perf}'$.
    # So we need $E(K_{4,4}) \setminus M_{miss} \supseteq E(K_{4,4}) \setminus M_{perf}'$.
    # This implies $M_{miss} \subseteq M_{perf}'$.
    # So $G$ contains $Q_3$ iff there exists $X,Y$ and a perfect matching $M_{perf}'$ such that 
    # all edges missing in $G$ between $X,Y$ are in $M_{perf}'$.
    # In other words: The set of missing edges $M_{miss}$ is a SUBSET of some perfect matching.
    # This is the condition used in the current program! ("missing edges form a matching").
    
    # So the check is: For a given 8-set $S$ and split $X,Y$, 
    # let $M_{miss}$ be the missing edges.
    # Check if $M_{miss}$ is a subset of a perfect matching.
    # This is equivalent to: $M_{miss}$ is a matching (no shared vertices) AND size <= 4.
    # Wait, if $M_{miss}$ is a matching of size 2, is it a subset of a perfect matching? Yes.
    # If $M_{miss}$ is a matching of size 4, it IS a perfect matching.
    # If $M_{miss}$ has shared vertices, it's not a matching, so not a subset of a perfect matching.
    # So the condition is: **$M_{miss}$ is a matching.**
    
    # The current program checks this.
    
    # How to speed it up?
    # 1. Reduce the number of 8-sets to check.
    #    When adding edge (u,v), we only need to check 8-sets containing u,v.
    #    There are C(14,6) = 3003 such sets.
    #    For each set, we check 35 splits.
    #    Total checks: 3003 * 35 = 105,105.
    #    Each check is very fast (bit ops).
    #    In Python, 100k checks might take ~0.1-0.5s?
    #    If we do 1000 additions, that's 100-500s. Too slow.
    
    # We need to reduce the number of checks per addition.
    # Idea: Only check splits where u and v are in opposite parts (X and Y).
    # In a Q3, u and v must be adjacent. So they must be in opposite parts of the bipartition.
    # So for a given set S containing u,v, we only check splits where u in X, v in Y OR u in Y, v in X.
    # This reduces the 35 splits to roughly 35/2 = 17.5?
    # Actually, for a fixed set of 8, the number of splits is 35.
    # The number of splits where u and v are separated is:
    # Total splits: 35.
    # Splits where u,v are together: 
    #   Case 1: u,v in X. Then we need to choose 2 more for X from the other 6. C(6,2)=15.
    #   Case 2: u,v in Y. Similarly 15.
    #   Total together: 30.
    #   Total separated: 35 - 30 = 5? 
    #   Let's re-calculate.
    #   Total ways to choose 4 for X from 8: C(8,4) = 70.
    #   Unordered splits: 35.
    #   In the 70 ordered splits:
    #   u,v in X: Choose 2 more from 6 -> C(6,2) = 15.
    #   u,v in Y: Choose 2 more for X from 6 (so u,v not in X) -> C(6,4) = 15.
    #   u in X, v in Y: Choose 3 more for X from 6 -> C(6,3) = 20.
    #   u in Y, v in X: Choose 3 more for X from 6 (so v in X, u not) -> C(6,3) = 20.
    #   Total ordered: 15+15+20+20 = 70.
    #   Unordered splits where u,v separated: (20+20)/2 = 20.
    #   Unordered splits where u,v together: (15+15)/2 = 15.
    #   So we only need to check 20 splits instead of 35!
    #   That's a 43% reduction.
    
    # 2. Can we reduce the number of 8-sets?
    #    If we add (u,v), and we know the graph was Q3-free before, 
    #    any new Q3 must use (u,v).
    #    The Q3 must have u and v adjacent.
    #    Are there other constraints?
    #    The other 6 vertices must be chosen such that the induced subgraph on the 8 vertices 
    #    has the right properties.
    #    Hard to prune without checking.
    
    # 3. Can we check faster?
    #    Instead of iterating 3003 sets, can we iterate over potential Q3s?
    #    A Q3 containing (u,v) is defined by the 6 other vertices and the structure.
    #    This is essentially the same as iterating the sets.
    
    # Let's try the "20 splits" optimization and see if it's enough.
    # Also, we can precompute the list of splits for each pair (u,v) relative to the set?
    # No, the splits depend on the specific 8 vertices.
    
    # Alternative: Use a different graph model.
    # What if we start with a very dense graph and REMOVE edges?
    # Start with K16 (120 edges).
    # Find all Q3s (many).
    # Remove edges to break them.
    # This is a "Hitting Set" problem.
    # We want to remove a minimum number of edges to hit all Q3s.
    # This is NP-hard, but maybe greedy works well?
    # Greedy: Pick the edge that is in the most Q3s, remove it.
    # This requires counting Q3s per edge, which is expensive.
    
    # Let's stick to the addition strategy but optimize the check.
    
    # Implementation details:
    # 1. Precompute all 35 splits as (mask_a, mask_b, A_idx, B_idx).
    # 2. For a given 8-set and edge (u,v), determine which splits separate u and v.
    #    We can do this by checking the index of u and v in the 8-set.
    #    Let i_u be index of u in the set, i_v be index of v.
    #    A split separates them if (i_u in A and i_v in B) or (i_u in B and i_v in A).
    #    We can precompute for each split a bitmask of indices in A.
    #    Then for a given i_u, i_v, check if (mask_a has i_u) XOR (mask_a has i_v) is True.
    
    # Let's implement this.
    
    all_splits = []
    seen = set()
    for A in combinations(range(8), 4):
        B = tuple(i for i in range(8) if i not in A)
        key = frozenset([frozenset(A), frozenset(B)])
        if key not in seen:
            seen.add(key)
            mask_a = 0
            mask_b = 0
            for i in A:
                mask_a |= (1 << i)
            for i in B:
                mask_b |= (1 << i)
            all_splits.append((mask_a, mask_b, A, B))
            
    # Optimization: Store A and B as bitmasks of indices too?
    # A_idx is a tuple. We can precompute a list of booleans or just use the masks.
    # To check if i is in A: (mask_a >> i) & 1.
    
    def contains_q3_fast(v_list, edge_u, edge_v):
        """
        Checks if the 8-set v_list contains a Q3 that uses the edge (edge_u, edge_v).
        v_list: tuple of 8 vertices.
        edge_u, edge_v: the specific edge added.
        """
        # Find indices of edge_u and edge_v in v_list
        # v_list is a tuple, so we can use a dict or just linear search (size 8).
        idx_u = -1
        idx_v = -1
        for i in range(8):
            if v_list[i] == edge_u:
                idx_u = i
            elif v_list[i] == edge_v:
                idx_v = i
        
        if idx_u == -1 or idx_v == -1:
            return False # Should not happen
            
        # Build local adjacency masks
        local_adj = [0] * 8
        for i in range(8):
            vi = v_list[i]
            mask = 0
            for j in range(8):
                if i != j and adj[vi][v_list[j]]:
                    mask |= (1 << j)
            local_adj[i] = mask
            
        # Iterate splits
        for mask_a, mask_b, A_idx, B_idx in all_splits:
            # Check if split separates idx_u and idx_v
            # u in A, v in B OR u in B, v in A
            u_in_a = (mask_a >> idx_u) & 1
            v_in_a = (mask_a >> idx_v) & 1
            if u_in_a == v_in_a:
                continue # They are in the same part, so edge (u,v) cannot be in the Q3 bipartite structure
                
            # Now check if missing edges form a matching
            covered = 0
            is_matching = True
            missing_count = 0
            
            # We need to check edges between A and B.
            # We can iterate over the smaller set or just both.
            # A and B are fixed tuples.
            # We need to find missing edges between v_list[i] for i in A_idx and v_list[j] for j in B_idx.
            
            # Optimization: 
            # Instead of nested loops, we can compute the set of missing edges quickly.
            # For each i in A_idx:
            #   neighbors_in_B = local_adj[i] & mask_b
            #   missing_in_B = mask_b & ~neighbors_in_B
            #   Iterate bits in missing_in_B
            
            for i in A_idx:
                # Get neighbors of i in B
                # local_adj[i] has bits for all neighbors in v_list.
                # We only care about bits in B.
                neigh_b = local_adj[i] & mask_b
                # Missing bits in B
                miss_b = mask_b & ~neigh_b
                
                # Iterate through missing bits
                # If miss_b has more than 1 bit, we need to be careful.
                # If i is missing connection to multiple j in B, 
                # then (i, j1) and (i, j2) are both missing.
                # They share vertex i. So they don't form a matching.
                # Unless one of them is not part of the "matching" we are constructing?
                # No, ALL missing edges must form a matching.
                # So if i has 2 missing neighbors in B, it's not a matching.
                
                # Check if miss_b has more than 1 bit
                if miss_b & (miss_b - 1):
                    is_matching = False
                    break
                    
                if miss_b:
                    # Only one missing neighbor j
                    # Find j
                    # j is the index in v_list corresponding to the bit.
                    # We need to check if j is already covered.
                    # But wait, we also need to check if i is already covered.
                    # i is in A. j is in B.
                    # If i is covered by a previous missing edge, it's not a matching.
                    # If j is covered by a previous missing edge, it's not a matching.
                    
                    # Find j
                    # miss_b is a power of 2.
                    # j = log2(miss_b)
                    # We can use bit_length - 1
                    j = miss_b.bit_length() - 1
                    
                    if (covered & (1 << i)) or (covered & (1 << j)):
                        is_matching = False
                        break
                        
                    covered |= (1 << i)
                    covered |= (1 << j)
                    
            if is_matching:
                return True
                
        return False

    # Initialize graph
    adj = [[0] * n for _ in range(n)]
    edges = set()
    
    def add_edge(u, v):
        if u == v or adj[u][v]:
            return
        adj[u][v] = 1
        adj[v][u] = 1
        edges.add((min(u, v), max(u, v)))

    def remove_edge(u, v):
        if not adj[u][v]:
            return
        adj[u][v] = 0
        adj[v][u] = 0
        edges.discard((min(u, v), max(u, v)))
        
    # Start with a random dense graph? Or a known good one?
    # Let's start with a random graph with p=0.5 and repair it?
    # Or start with the seed from the previous program (K3,13 + K3)?
    # Let's start with a random graph and remove Q3s?
    # No, start with empty and add is safer for correctness if we verify.
    
    # Let's try starting with a random graph of density 0.4 and then adding edges.
    # Or better, start with the best known structure.
    
    # Let's initialize with a random graph and check validity, then add.
    # Actually, just start with empty and add greedily.
    
    random.seed(42)
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(all_pairs)
    
    best_edges = 0
    best_adj = [row[:] for row in adj]
    best_edge_set = set(edges)
    
    # Local Search
    # We will iterate through the shuffled pairs and try to add them.
    # If adding creates a Q3, we skip it (or try to remove another edge?).
    # Simple greedy: if valid, add.
    
    # To make it better, we can do multiple passes or random restarts.
    # But 120s is tight.
    
    # Let's use the "20 splits" optimization in the check.
    
    for u, v in all_pairs:
        if time.time() > deadline - 5:
            break
        if adj[u][v]:
            continue
            
        add_edge(u, v)
        
        # Check for Q3 containing (u, v)
        found_q3 = False
        other_vertices = [i for i in range(n) if i != u and i != v]
        
        for combo in combinations(other_vertices, 6):
            vertices = (u, v) + tuple(combo)
            if contains_q3_fast(vertices, u, v):
                found_q3 = True
                break
                
        if found_q3:
            remove_edge(u, v)
        else:
            if len(edges) > best_edges:
                best_edges = len(edges)
                best_adj = [row[:] for row in adj]
                best_edge_set = set(edges)
                
    # Second pass: Try to add edges that were skipped?
    # The order matters. A random order might miss some edges.
    # Let's do a few random restarts if time permits.
    
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            # Reset
            for row in adj:
                for col in range(n):
                    adj[row][col] = 0
            edges.clear()
            
            all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(all_pairs)
            
            for u, v in all_pairs:
                if time.time() > deadline - 5:
                    break
                if adj[u][v]:
                    continue
                    
                add_edge(u, v)
                
                found_q3 = False
                other_vertices = [i for i in range(n) if i != u and i != v]
                
                for combo in combinations(other_vertices, 6):
                    vertices = (u, v) + tuple(combo)
                    if contains_q3_fast(vertices, u, v):
                        found_q3 = True
                        break
                        
                if found_q3:
                    remove_edge(u, v)
                else:
                    if len(edges) > best_edges:
                        best_edges = len(edges)
                        best_adj = [row[:] for row in adj]
                        best_edge_set = set(edges)
                        
    return [list(e) for e in best_edge_set]

if __name__ == "__main__":
    res = construct()
    print(len(res))

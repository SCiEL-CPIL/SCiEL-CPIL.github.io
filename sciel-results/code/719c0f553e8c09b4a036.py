import time
import sys
from itertools import combinations
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _max_matching_size(adj, vertices):
    """
    Compute maximum matching size in the induced subgraph on 'vertices'.
    Used as a lower bound for set cover: each biclique can cover at most 1 edge 
    incident to a matched edge? No, a biclique can cover many edges.
    However, if we have a matching of size M in the remaining graph, 
    we need at least M bicliques? No, a biclique (L, R) covers all edges between L and R.
    If we pick a matching edge (u,v), a single biclique containing u and v can cover it.
    But can one biclique cover two disjoint matching edges (u,v) and (x,y)?
    Yes, if L={u,x} and R={v,y} and all 4 cross edges exist.
    So matching size is not a direct lower bound on the number of bicliques.
    
    Wait, let's rethink the lower bound.
    The problem is equivalent to covering edges with bicliques.
    A biclique is a complete bipartite subgraph.
    
    Let's use a simpler lower bound: 
    The size of the maximum matching in the graph is not a lower bound for biclique cover number directly in the way it is for edge cover.
    Actually, for vertex cover, matching is a lower bound.
    For edge cover (minimum number of edges to cover all vertices), it's different.
    For biclique cover:
    Consider the graph. If we have a matching of size k, can we cover all those k edges with 1 biclique?
    Yes, if they form a "grid" structure.
    However, if the matching edges are "independent" in a way that prevents them from being in the same biclique, we need more.
    
    Let's try a different bound.
    The clique number of the line graph? No.
    
    Let's stick to a heuristic bound or just use the greedy solution as the initial upper bound and branch on the most "constrained" edge.
    
    Actually, a good lower bound is: 
    Let G be the remaining graph.
    Let C be a biclique.
    If we pick an edge e=(u,v), any biclique covering e must contain u and v.
    
    Let's use the following:
    The minimum number of bicliques to cover a graph is at least the size of the maximum matching in the graph?
    No. Consider a cycle C4. Matching size 2. Biclique cover: 1 biclique (all 4 vertices, but C4 is not a biclique, it's a cycle). 
    Wait, C4 edges: (0,1), (1,2), (2,3), (3,0).
    Biclique 1: L={0,2}, R={1,3}. Edges: (0,1), (0,3), (2,1), (2,3).
    This covers all 4 edges of C4. So 1 biclique covers matching size 2.
    So matching size is NOT a lower bound.
    
    What is a lower bound?
    The size of a set of edges such that no two can be in the same biclique.
    Two edges (u,v) and (x,y) can be in the same biclique if {u,x} x {v,y} is a biclique.
    This requires edges (u,v), (u,y), (x,v), (x,y) to exist.
    So if we have a set of edges E' such that for any distinct e1, e2 in E', they cannot be part of the same biclique, then we need at least |E'| bicliques.
    
    How to find such a set?
    If we take a matching where no two edges can be extended to a biclique together, that works.
    Specifically, if we have a matching edge (u,v) and another (x,y), they can be in the same biclique if u-x, v-y, u-y, x-v edges exist (assuming u,x on left, v,y on right, or similar).
    Actually, for (u,v) and (x,y) to be in the same biclique, the subgraph induced by {u,v,x,y} must contain a biclique containing both.
    This happens if the edges form a K2,2 or are part of a larger one.
    If the edges are "independent" in the sense that there are no cross edges between their endpoints, they can't be in the same biclique.
    
    Let's compute a lower bound based on "independent edges".
    An edge e=(u,v) is "blocked" from edge f=(x,y) if the biclique containing e cannot contain f.
    
    Alternative strategy:
    Since n=30 is small, and the graph is dense, the number of maximal bicliques might be manageable.
    Let's generate all maximal bicliques.
    Then solve Set Cover exactly using Branch and Bound.
    
    Lower bound for Set Cover:
    Standard lower bound: ceil(remaining_edges / max_set_size).
    This is weak.
    Better: For each remaining edge, we need a set.
    If we can find a collection of edges such that each set in our candidate pool covers at most 1 of these edges, then the size of this collection is a lower bound.
    
    Let's find a maximal set of "conflicting" edges.
    Two edges conflict if no single biclique in our candidate pool covers both.
    If we find a maximal set of pairwise conflicting edges, that's a lower bound.
    
    How to check if two edges e1, e2 can be covered by the same biclique from the pool?
    Precompute for each pair of edges whether they are compatible.
    If not compatible, they conflict.
    Then find the maximum independent set in the conflict graph? No, maximum clique in the compatibility graph?
    We want a set of edges where NO two are compatible.
    This is a Maximum Independent Set in the graph where nodes are edges and edges represent compatibility.
    Or Maximum Clique in the graph where edges represent incompatibility.
    
    This might be expensive.
    Let's use a simpler bound: 
    Greedy matching of incompatible edges.
    Iterate through edges, if current edge is incompatible with all previously selected, add it.
    This gives a valid lower bound.
    
    Let's implement this.
    """
    pass

def _generate_maximal_bicliques(n, edges):
    """
    Generate all maximal bicliques.
    Using a recursive expansion.
    A biclique is (L, R).
    Start with a single edge (u, v). L={u}, R={v}.
    Expand: add x to L if x is connected to all in R.
    Add y to R if y is connected to all in L.
    Continue until no more can be added.
    To avoid duplicates, we need a canonical form or careful enumeration.
    
    Given n=30, this might explode.
    However, the graph is random-ish but dense.
    Let's try a heuristic:
    For each edge (u,v), expand to maximal biclique containing (u,v).
    Deduplicate.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    maximal_bicliques = set()
    
    def expand_biclique(L, R):
        L = frozenset(L)
        R = frozenset(R)
        # Check if this is maximal
        # Try to add vertices
        changed = True
        while changed:
            changed = False
            # Add to L
            for u in range(n):
                if u not in L and adj[u].issuperset(R):
                    L = L | {u}
                    changed = True
            # Add to R
            for v in range(n):
                if v not in R and adj[v].issuperset(L):
                    R = R | {v}
                    changed = True
        return L, R

    # Iterate over all edges to find maximal bicliques containing them
    # To reduce duplicates, we can restrict expansion order or just use a set of frozensets of edges
    biclique_edge_sets = set()
    
    for u, v in edges:
        L, R = expand_biclique({u}, {v})
        # Generate edge set
        e_set = frozenset((min(x,y), max(x,y)) for x in L for y in R if (x,y) in adj[x])
        if e_set:
            biclique_edge_sets.add(e_set)
            
    # Convert back to list of (L, R) for output, but for set cover we just need edge sets
    # We need to map edge sets back to L, R for the final answer
    # Since we might have same edge set from different L,R (isomorphic?), 
    # actually if edge set is same, L and R are determined up to swapping?
    # No, L and R are the bipartition.
    # If we have edge set E, we can recover L and R by taking the connected components of the bipartite graph?
    # No, a biclique is connected (if |L|,|R| > 1).
    # Actually, for a maximal biclique, L and R are unique (up to swap if the graph is complete bipartite and we allow swapping sides? 
    # The problem says "left_vertices, right_vertices". It doesn't say we can't swap, but usually we fix an ordering or just pick one.
    # The validation checks "cross pair must be a host edge".
    # If I return [L, R], it's valid. If I return [R, L], it's also valid.
    # So we just need one valid representation.
    
    # Let's store the bicliques as (frozenset(L), frozenset(R))
    bicliques = []
    seen_edge_sets = set()
    
    for u, v in edges:
        L, R = expand_biclique({u}, {v})
        e_set = frozenset((min(x,y), max(x,y)) for x in L for y in R if (x,y) in adj[x])
        if e_set not in seen_edge_sets:
            seen_edge_sets.add(e_set)
            # Ensure L and R are non-empty and disjoint
            if L and R and not (L & R):
                bicliques.append((L, R, e_set))
                
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    n = P['n']
    edges = P['edges']
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    
    # 1. Generate candidate bicliques
    bicliques = _generate_maximal_bicliques(n, edges)
    
    if not bicliques:
        return []
        
    # 2. Prepare for Set Cover
    # Universe: all edges
    # Sets: edge sets of bicliques
    # We want to select min number of sets to cover all edges.
    
    # Map edge to index for bitmasks or sets
    # Since |E| can be ~300-400, bitmasks are good.
    edge_to_idx = {e: i for i, e in enumerate(edge_set)}
    num_edges = len(edge_set)
    
    # Represent each biclique as a bitmask
    biclique_masks = []
    biclique_data = [] # (L, R)
    
    for L, R, e_set in bicliques:
        mask = 0
        for e in e_set:
            idx = edge_to_idx[e]
            mask |= (1 << idx)
        biclique_masks.append(mask)
        biclique_data.append((L, R))
        
    num_bicliques = len(biclique_masks)
    
    # Sort bicliques by size (larger first) to help pruning
    # Also remove subsets? If biclique A is subset of biclique B, A is never better than B.
    # Keep only maximal ones? We already generated maximal ones, but some might be subsets of others due to non-maximality in generation?
    # My generation expands to maximal, so they should be maximal.
    # But two different maximal bicliques might have same edge set? No, we deduped by edge set.
    
    # Remove bicliques that are subsets of others
    # This is O(N^2 * bitmask_ops)
    filtered_masks = []
    filtered_data = []
    for i in range(num_bicliques):
        is_subset = False
        for j in range(num_bicliques):
            if i == j: continue
            if (biclique_masks[i] & biclique_masks[j]) == biclique_masks[i]:
                if biclique_masks[i] != biclique_masks[j]:
                    is_subset = True
                    break
                # If equal, keep one (we deduped, so shouldn't happen)
        if not is_subset:
            filtered_masks.append(biclique_masks[i])
            filtered_data.append(biclique_data[i])
            
    num_bicliques = len(filtered_masks)
    
    # Sort by popcount descending
    order = sorted(range(num_bicliques), key=lambda i: bin(filtered_masks[i]).count('1'), reverse=True)
    sorted_masks = [filtered_masks[i] for i in order]
    sorted_data = [filtered_data[i] for i in order]
    
    # 3. Branch and Bound
    best_cover = []
    best_cost = num_bicliques + 1 # Upper bound
    
    # Initial greedy solution for upper bound
    remaining = (1 << num_edges) - 1
    greedy_sol = []
    temp_remaining = remaining
    while temp_remaining:
        best_idx = -1
        best_gain = 0
        for i in range(num_bicliques):
            if not (sorted_masks[i] & temp_remaining):
                continue
            gain = bin(sorted_masks[i] & temp_remaining).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        greedy_sol.append(best_idx)
        temp_remaining &= ~sorted_masks[best_idx]
        
    if temp_remaining == 0:
        best_cost = len(greedy_sol)
        best_cover = greedy_sol[:]
    else:
        # Fallback: individual edges
        best_cost = num_edges
        best_cover = []
        for i in range(num_edges):
            # Find any biclique containing edge i
            for j in range(num_bicliques):
                if (sorted_masks[j] >> i) & 1:
                    best_cover.append(j)
                    break
        if len(best_cover) > best_cost:
             best_cost = len(best_cover)

    # Lower Bound Function
    def get_lower_bound(remaining_mask, current_cost):
        if remaining_mask == 0:
            return current_cost
        # Simple bound: ceil(popcount(remaining) / max_set_size)
        # Max set size in remaining?
        # Or just use 1? That's weak.
        # Let's use: number of edges / max possible coverage per biclique.
        # Max possible coverage is max popcount of any biclique.
        max_cov = max(bin(m).count('1') for m in sorted_masks)
        if max_cov == 0:
            return float('inf')
        lb = (bin(remaining_mask).count('1') + max_cov - 1) // max_cov
        return current_cost + lb

    # Conflict-based lower bound
    # Precompute incompatibility? Too expensive.
    # Use a greedy independent set of edges in the remaining graph?
    # If we can find k edges in remaining_mask such that no single biclique covers two of them,
    # then we need at least k more bicliques.
    
    def greedy_conflict_lb(remaining_mask):
        # Get list of edges in remaining_mask
        edges_in_rem = [i for i in range(num_edges) if (remaining_mask >> i) & 1]
        if not edges_in_rem:
            return 0
        
        # Check compatibility between edges
        # Two edges e1, e2 are compatible if there exists a biclique covering both.
        # Precompute a matrix? 
        # Let's just check on the fly or precompute a "can_cover" matrix for edge pairs.
        # Since num_edges ~ 400, 400x400 matrix is 160k entries. Manageable.
        
        # Precompute compatibility matrix
        # compat[i][j] = True if some biclique covers both edge i and edge j
        # We can compute this by: for each biclique, for each pair of edges in it, mark compatible.
        # This is O(B * E^2) which might be slow if B is large.
        # Let's skip this for now and use the simple bound, or a weaker conflict bound.
        
        # Weaker bound: 
        # If we pick an edge, it must be covered.
        # Just return 1 if remaining != 0.
        return 1

    # To speed up B&B, we need a better LB.
    # Let's precompute for each edge, the list of bicliques covering it.
    edge_to_bicliques = defaultdict(list)
    for i, mask in enumerate(sorted_masks):
        for j in range(num_edges):
            if (mask >> j) & 1:
                edge_to_bicliques[j].append(i)
                
    # In B&B, when we pick a biclique, we update remaining.
    # We can also maintain a "forced" set? No.
    
    def branch_and_bound(remaining_mask, current_cost, current_selection):
        nonlocal best_cost, best_cover
        
        if time.time() > deadline:
            return
            
        if remaining_mask == 0:
            if current_cost < best_cost:
                best_cost = current_cost
                best_cover = current_selection[:]
            return
            
        # Lower bound
        lb = current_cost + get_lower_bound(remaining_mask, current_cost)
        if lb >= best_cost:
            return
            
        # Heuristic: Pick the edge with the fewest candidate bicliques
        # Among remaining edges
        min_candidates = float('inf')
        target_edge = -1
        for i in range(num_edges):
            if (remaining_mask >> i) & 1:
                # Count candidates
                # Candidates are bicliques that cover i and are not yet "excluded"?
                # In standard set cover, we don't exclude sets, we just pick them.
                # But we can prune if a set is already picked? No, we pick distinct sets.
                # So candidates are all bicliques covering i.
                # Wait, if we've already picked a biclique, we don't pick it again.
                # So we need to track which bicliques are available.
                # For simplicity in this implementation, we assume we pick from all, 
                # but we check if it's already in current_selection? 
                # No, set cover picks distinct sets.
                # So we should only consider bicliques not in current_selection.
                
                # Optimization: 
                # If we pick edge i, we must pick one of the bicliques covering i.
                # Let's pick the edge with the fewest *available* bicliques.
                
                count = 0
                for b_idx in edge_to_bicliques[i]:
                    if b_idx not in current_selection:
                        count += 1
                if count < min_candidates:
                    min_candidates = count
                    target_edge = i
                if min_candidates == 0:
                    return # Dead end
                
        if target_edge == -1:
            return
            
        # Branch on bicliques covering target_edge
        # Sort by size (larger first) to find good solutions earlier
        candidates = [b_idx for b_idx in edge_to_bicliques[target_edge] if b_idx not in current_selection]
        candidates.sort(key=lambda idx: bin(sorted_masks[idx]).count('1'), reverse=True)
        
        for b_idx in candidates:
            new_remaining = remaining_mask & ~sorted_masks[b_idx]
            current_selection.append(b_idx)
            branch_and_bound(new_remaining, current_cost + 1, current_selection)
            current_selection.pop()
            
            if time.time() > deadline:
                return

    branch_and_bound((1 << num_edges) - 1, 0, [])
    
    # Fallback if B&B didn't find anything better or timed out
    if best_cost > num_bicliques:
         # Use greedy
         if greedy_sol:
             best_cover = greedy_sol
             best_cost = len(greedy_sol)
         else:
             # Worst case: single edges
             best_cover = []
             for i in range(num_edges):
                 for j in range(num_bicliques):
                     if (sorted_masks[j] >> i) & 1:
                         best_cover.append(j)
                         break
             best_cost = len(best_cover)

    # Construct result
    result = []
    for idx in best_cover:
        L, R = sorted_data[idx]
        # Ensure types are Python int and list
        result.append([list(L), list(R)])
        
    # Validate
    # Check disjoint sides
    for L, R in result:
        if L and R:
            if set(L) & set(R):
                return [] # Invalid
        
    # Check partition
    covered_edges = set()
    for L, R in result:
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                if e not in edge_set:
                    return [] # Not a host edge
                if e in covered_edges:
                    return [] # Overlap
                covered_edges.add(e)
                
    if covered_edges != edge_set:
        return [] # Missing edges
        
    return result

# EVOLVE-BLOCK-END

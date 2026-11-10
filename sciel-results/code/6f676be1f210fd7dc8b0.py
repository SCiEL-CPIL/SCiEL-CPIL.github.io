import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # 1. Compute forbidden differences (nonzero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build adjacency lists
    # neighbors[u] is a list of v such that (u-v) % m is in forbidden
    # Note: The condition is symmetric because if d is a square, -d is also a square mod 205?
    # Let's verify: x^2 = d => (-x)^2 = d. So yes, if d is in F, m-d is in F.
    # Thus the graph is undirected.
    neighbors = [[] for _ in range(m)]
    for u in range(m):
        for v in range(m):
            if u != v and (u - v) % m in forbidden:
                neighbors[u].append(v)
    
    # Convert to sets for fast intersection if needed, but Tomita usually works with lists/arrays
    # For Tomita, we need efficient "neighbors in candidate set" operations.
    # Since N=205 is small, we can use bitmasks or just simple list filtering.
    # Bitmasks are very fast in Python if we use integers.
    
    # Let's use bitmasks for speed in the core loop.
    # adj_mask[u] is an integer where bit v is 1 if u and v are adjacent.
    adj_mask = [0] * m
    for u in range(m):
        mask = 0
        for v in neighbors[u]:
            mask |= (1 << v)
        adj_mask[u] = mask

    # Tomita Algorithm for Maximum Independent Set
    # We want an independent set in the original graph, which is a clique in the complement graph.
    # Or we can run Tomita directly on the independent set formulation.
    # Standard Tomita finds Max Clique. 
    # Max Independent Set in G == Max Clique in G_complement.
    
    # Let's define the complement adjacency masks.
    # comp_adj_mask[u] has bit v set if u and v are NOT adjacent in G (and u != v).
    comp_adj_mask = [0] * m
    for u in range(m):
        # All bits 0..204 set, then clear u and its neighbors in G
        full_mask = (1 << m) - 1
        # Clear u
        full_mask &= ~(1 << u)
        # Clear neighbors
        full_mask &= ~adj_mask[u]
        comp_adj_mask[u] = full_mask

    best_clique = []
    best_size = 0
    
    def expand(clique, candidates):
        """
        Recursively expand the clique.
        clique: list of nodes in current clique
        candidates: set of nodes that can be added to clique (adjacent to all in clique)
        """
        nonlocal best_clique, best_size
        
        # Upper bound: len(clique) + len(candidates)
        if len(clique) + len(candidates) <= best_size:
            return
        
        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
        
        # Greedy coloring of candidates to get upper bounds for branches
        # We color the subgraph induced by candidates in the complement graph?
        # No, for Max Clique, we color the candidates such that no two nodes with the same color are adjacent in the *original* graph?
        # Wait. Tomita's coloring is on the *candidate* subgraph.
        # If we are finding a clique in G, we color G[candidates].
        # The upper bound for a branch adding vertex v is len(clique) + color[v].
        
        # Let's implement a simple greedy coloring for the induced subgraph of candidates in G.
        # Actually, it's easier to work with the complement if we are doing Independent Set directly, 
        # but standard Tomita is for Cliques. Let's stick to Clique on G_complement.
        # So we are finding a Max Clique in H = G_complement.
        # Candidates are nodes in H that are adjacent to all nodes in current clique.
        
        # Greedy coloring of H[candidates]
        # We want to partition candidates into independent sets of H (which are cliques of G).
        # The color number is the upper bound on the clique size in H[candidates].
        
        remaining = list(candidates)
        # Sort by degree in H? Or just arbitrary?
        # Standard Tomita sorts by degree in the candidate subgraph to improve bounds.
        
        # For N=205, sorting is cheap.
        # Let's calculate degrees in H induced by candidates.
        # deg_H(v) = popcount(comp_adj_mask[v] & candidates_mask)
        
        # To make it fast, let's just use a simple ordering or degree ordering.
        # Given the small size, even unsorted might work, but sorted is better for pruning.
        
        # Let's convert candidates to a list and sort by degree in H descending.
        cand_list = list(candidates)
        
        # Calculate degrees in H for sorting
        def get_deg(node):
            return bin(comp_adj_mask[node] & (lambda x: sum([1<<i for i in x]) if False else 0))(0) # This is getting messy with bitmasks and sets.
            
        # Let's stick to simple sets for logic clarity, bitmask for intersection if needed.
        # Actually, for N=205, sets are fine.
        
        # Greedy Coloring on H[candidates]
        # We assign colors such that adjacent nodes in H have different colors.
        # color[i] is the color assigned to node i.
        colors = {}
        color_class = 0
        uncolored = set(candidates)
        
        # To optimize, we can process nodes in order of decreasing degree in H.
        # But for simplicity and correctness first, let's just iterate.
        # A better heuristic: Order by degree in H descending.
        
        # Let's compute degrees in H for sorting
        degs = {}
        for v in candidates:
            # Count neighbors in candidates within H
            degs[v] = len([u for u in candidates if u != v and (comp_adj_mask[v] >> u) & 1])
            
        # Sort candidates by degree descending
        sorted_cands = sorted(candidates, key=lambda x: degs[x], reverse=True)
        
        # Greedy coloring
        # We will assign the smallest possible color.
        # Actually, Tomita's specific coloring algorithm is a bit specific:
        # It colors nodes sequentially. The color of a node is the smallest integer not used by its already-colored neighbors in the candidate set.
        
        color_of_node = {}
        # We need to know which colors are used by neighbors in the *already colored* part.
        # Since we are coloring the whole candidate set, we can do it in one pass if we order them?
        # No, standard greedy coloring requires handling conflicts.
        
        # Let's use a simple greedy coloring:
        # Iterate through sorted_cands.
        # For each node v, find the smallest color c such that no neighbor of v in H (that has been colored) has color c.
        
        # Wait, the bound is: max color assigned is an upper bound on clique size.
        
        used_colors_by_node = {} # node -> color
        
        for v in sorted_cands:
            # Find neighbors of v in H that are in candidates
            # And check their colors
            forbidden_colors = set()
            for u in candidates:
                if u == v: continue
                # Check if u is adjacent to v in H
                if (comp_adj_mask[v] >> u) & 1:
                    if u in used_colors_by_node:
                        forbidden_colors.add(used_colors_by_node[u])
            
            # Assign smallest available color
            c = 1
            while c in forbidden_colors:
                c += 1
            used_colors_by_node[v] = c
            
        # Now, we branch on nodes in reverse order of their color (highest color first)
        # Why? Because if a node has a high color, it's likely part of a large clique.
        # Also, the upper bound for branch v is len(clique) + used_colors_by_node[v].
        
        # We iterate through nodes sorted by color descending.
        # But we must be careful: if we pick a node v, the new candidates are candidates intersected with neighbors of v in H.
        # And we must remove v and all nodes with color >= color[v]? 
        # No, the standard Tomita branch-and-bound works like this:
        # 1. Color the candidates.
        # 2. Iterate through candidates in order of decreasing color.
        # 3. For each candidate v:
        #    a. If len(clique) + color[v] <= best_size, break (prune rest).
        #    b. Add v to clique.
        #    c. New candidates = candidates ∩ neighbors_H(v).
        #    d. Recursively expand.
        #    e. Remove v from candidates. (Wait, does removing v affect the colors of others? 
        #       The coloring was done on the *original* candidate set. 
        #       As we remove nodes, the upper bounds might become loose, but they remain valid upper bounds.
        #       However, Tomita's algorithm typically re-colors or updates? 
        #       Actually, the standard implementation iterates through the colored nodes. 
        #       The key is that `color[v]` is an upper bound on the clique size in the subgraph induced by {v} ∪ {nodes in candidates with color >= color[v]}.
        #       So if we process nodes in decreasing color order, and remove them from the candidate pool, 
        #       the remaining nodes have lower colors.
        
        # Let's refine the loop:
        # We have `cand_set` (current candidates) and `coloring` (map node -> color).
        # We iterate through nodes in `cand_set` sorted by color descending.
        # Let `order` be list of nodes in cand_set sorted by color desc.
        
        # We maintain a local `cand_pool` which is a copy of candidates.
        # As we visit a node v, we:
        # 1. Check bound.
        # 2. Recurse with clique + [v] and candidates = cand_pool ∩ neighbors_H(v).
        # 3. Remove v from cand_pool.
        # 4. Update colors? No, the colors are static from the initial coloring of the *initial* candidate set for this level.
        #    This is a valid upper bound strategy.
        
        order = sorted(candidates, key=lambda x: used_colors_by_node[x], reverse=True)
        
        # We need to be careful: `candidates` passed to `expand` is a set.
        # We should not modify it directly if it's shared, but here it's local.
        
        # However, there is a subtle point: The coloring bound `color[v]` applies to the subgraph induced by v and all nodes in the *current* candidate set that have color >= color[v].
        # Since we are removing nodes one by one, and we process in decreasing color order, 
        # when we are at node v, all nodes with color > color[v] have already been processed (and either chosen or discarded).
        # The nodes remaining in the pool that have color >= color[v] are exactly those with color == color[v] (since > are gone).
        # So the bound is valid.
        
        # But wait, if we remove a node u with color 5, and then look at node v with color 4, 
        # the subgraph for v's bound should only consider nodes with color >= 4.
        # The nodes with color 5 are gone. So the bound color[v]=4 is an upper bound on the clique size in the remaining subgraph?
        # Yes, because the original coloring guaranteed that the max clique in the *entire* set is <= max color.
        # By removing nodes, the max clique can only decrease.
        
        # Implementation detail: 
        # We need `neighbors_H(v)` intersection with `current_pool`.
        
        current_pool = set(candidates)
        
        for v in order:
            if v not in current_pool:
                continue # Might have been removed? No, we only remove v itself. 
                         # But what if a previous branch removed it? 
                         # In this sequential loop, we only remove v after processing it.
                         # So v is always in current_pool at the start of its iteration.
            
            # Bound check
            if len(clique) + used_colors_by_node[v] <= best_size:
                # Since we are processing in decreasing color order, 
                # all subsequent nodes have color <= used_colors_by_node[v].
                # So they will also fail the bound check.
                break
            
            # Branch
            new_clique = clique + [v]
            
            # New candidates: nodes in current_pool that are adjacent to v in H
            # And we should probably exclude v itself (though intersection handles it if we don't include v in neighbors)
            # neighbors_H(v) does not include v.
            
            v_mask = comp_adj_mask[v]
            new_cands = set()
            for u in current_pool:
                if u == v:
                    continue
                if (v_mask >> u) & 1:
                    new_cands.add(u)
            
            # Recurse
            expand(new_clique, new_cands)
            
            # Remove v from pool for subsequent branches
            current_pool.discard(v)
            
            # Optimization: If we found a clique of size equal to the upper bound of the *initial* candidate set, 
            # we might be able to prune more aggressively, but the break condition above handles most of it.
            
    # Initial call
    # Candidates are all nodes 0..204
    initial_candidates = set(range(m))
    # We need to handle the case where m=0? No, m=205.
    
    # To ensure determinism and potentially better pruning, we can sort initial candidates?
    # The coloring inside `expand` will handle the ordering.
    
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Tomita is deterministic. No need for restarts.
    # However, if the graph is large, it might take time. 205 is small.
    # Let's run it.
    
    expand([], initial_candidates)
    
    # The result is `best_clique` which is a clique in H (complement of G).
    # This corresponds to an independent set in G.
    
    # Convert to numpy array
    if not best_clique:
        # Fallback if something went wrong (should not happen)
        return np.array([0], dtype=np.int64)
        
    return np.array(best_clique, dtype=np.int64)

# Test locally? No, just provide the function.
# The function name must be furstenberg_sarkozy_205.

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses Bitset-based Branch and Bound for Maximum Independent Set.
    """
    m = 205
    
    # 1. Precompute forbidden differences and adjacency lists
    forbidden = set(pow(x, 2, m) for x in range(1, m))
    forbidden.discard(0)
    
    # adj[i] is a set of neighbors j such that (i-j) % m is in forbidden
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            if (i - j) % m in forbidden:
                adj[i].add(j)
                adj[j].add(i) # Undirected graph

    # Convert adjacency to bitmasks for fast operations
    # conflict_mask[i] has bit j set if i and j are adjacent (conflict)
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        conflict_mask[i] = mask

    # Precompute degrees for vertex ordering (greedy coloring heuristic)
    # We want to order vertices by degree to make coloring tight (fewer colors)
    # High degree vertices are harder to color, so we process them first?
    # Actually, in greedy coloring, processing high-degree vertices first often leads to more colors?
    # No, usually we want to color vertices with high degree first to constrain their neighbors.
    # However, for the *bound* to be tight, we want the coloring to be as "efficient" as possible?
    # The bound is: Size of MIS <= Number of Colors in Greedy Coloring.
    # To get a *low* number of colors (tight bound), we should color vertices in an order that minimizes conflicts.
    # A common heuristic is to sort by degree descending (DSATUR-like logic but simpler).
    
    # Let's precompute a static order based on degree descending.
    # In the subgraph, degrees change, but static order is a good proxy and faster.
    order = sorted(range(m), key=lambda x: len(adj[x]), reverse=True)
    
    # To make this dynamic-ish and fast, we can just use the static order for the coloring step.
    # This is O(n^2) per call.

    def greedy_color_bound(candidates_mask):
        """
        Returns an upper bound on the size of the maximum independent set in the subgraph induced by candidates.
        Uses a greedy coloring of the conflict graph restricted to candidates.
        The number of colors used is an upper bound on the clique number of the complement,
        which is the independence number of the original graph.
        Wait, let's re-verify the logic.
        
        Problem: Max Independent Set (MIS) in G.
        Equivalent to: Max Clique in G_complement.
        
        Standard B&B for Max Clique uses a greedy coloring of the *candidate* graph (which is G_complement here? No).
        
        Let's stick to the standard "Tomita" algorithm for Max Clique.
        We are looking for Max Clique in H = G_complement.
        
        In H, vertices u, v are connected if they are NOT connected in G (i.e., (u-v) not in forbidden).
        
        The greedy coloring bound for Max Clique in H:
        We color the vertices of H. The number of colors is an upper bound on the clique size in H.
        
        So, we need to greedily color the graph H restricted to candidates.
        Edges in H: u~v if u!=v and u,v not in adj.
        
        Algorithm for coloring H:
        1. Order candidates.
        2. For each vertex v in order:
           Assign the smallest color c such that no neighbor of v in H has color c.
           Neighbor in H means: NOT neighbor in G.
           
        So for a vertex v, we check its neighbors in G.
        If a neighbor w in G has color c, does that prevent v from having color c in H?
        No. w is a neighbor in G, so w is NOT a neighbor in H.
        So w having color c does NOT prevent v from having color c in H.
        
        The vertices that PREVENT v from having color c in H are the vertices u such that u is a neighbor of v in H.
        u is a neighbor of v in H iff u is NOT a neighbor of v in G.
        
        So, to assign a color to v in H:
        We look at all other candidates u.
        If u is NOT in adj[v] (and u != v), then u is a neighbor in H.
        If any such u has color c, we cannot assign c to v.
        
        This requires checking all other candidates for each v. O(n^2).
        
        Optimization:
        We can maintain an array `color_of_vertex` for the current coloring step.
        
        Let's implement this carefully.
        """
        if candidates_mask == 0:
            return 0
            
        # Get list of candidate vertices
        # We use the static order 'order' but filtered to candidates
        # To speed up, we can just iterate 0..m-1 and check mask.
        
        # colors[i] stores the color assigned to vertex i, -1 if not assigned
        colors = [-1] * m
        num_colors = 0
        
        # We iterate through vertices in a specific order.
        # A good order for coloring to minimize colors is often "Largest First" or similar.
        # We'll use the static 'order' (degree descending in G).
        
        for v in order:
            if not (candidates_mask & (1 << v)):
                continue
                
            # Determine available colors for v in H
            # A color c is forbidden if there exists a neighbor u of v in H with color c.
            # Neighbor u in H means: u is candidate, u != v, and NOT (u in adj[v]).
            
            # We need to check colors of all candidates u such that u is NOT in adj[v].
            # This is O(n) per vertex. Total O(n^2).
            
            # Optimization:
            # Instead of checking all u, we can just check if the color c is present among the non-neighbors.
            
            # Let's just iterate through all candidates u != v.
            # If u is NOT in adj[v], it's a neighbor in H.
            # If colors[u] == c, then c is forbidden.
            
            # To make this faster, we can precompute "non-neighbor mask" for each v?
            # non_adj_mask[v] = candidates_mask & ~(1<<v) & ~conflict_mask[v]
            
            # Then for each color c, we check if any vertex in non_adj_mask[v] has color c.
            
            # Let's try a simpler O(n^2) implementation first, as n=205 is small.
            
            forbidden_colors = set()
            # Iterate through all other candidates
            temp_mask = candidates_mask & ~(1 << v)
            # We need to iterate through bits of temp_mask
            # But we only care about u where (u not in adj[v])
            
            # Let's precompute a list of candidates for this call?
            # No, just iterate 0..m-1.
            
            for u in range(m):
                if u == v:
                    continue
                if not (candidates_mask & (1 << u)):
                    continue
                
                # Check if u is a neighbor in H
                # u is neighbor in H if u is NOT in adj[v]
                if not (conflict_mask[v] & (1 << u)):
                    # u is a neighbor in H
                    if colors[u] != -1:
                        forbidden_colors.add(colors[u])
            
            # Find smallest color not in forbidden_colors
            c = 0
            while c in forbidden_colors:
                c += 1
            
            colors[v] = c
            if c >= num_colors:
                num_colors = c + 1
            
        return num_colors

    def branch_and_bound(candidates_mask, current_size, best_size_ref):
        """
        Recursive B&B for Max Independent Set.
        candidates_mask: bitmask of vertices still available.
        current_size: size of the independent set chosen so far.
        best_size_ref: list containing the best size found so far (to allow mutation in recursion).
        """
        if candidates_mask == 0:
            return
        
        # Update best if current is larger
        # Note: We don't know the final size yet, just the lower bound.
        # But we can update best_size_ref with current_size + something?
        # No, just track the best complete solution.
        
        # Upper bound check
        bound = greedy_color_bound(candidates_mask)
        if current_size + bound <= best_size_ref[0]:
            return
            
        # Select a vertex to branch on
        # Heuristic: Pick a vertex with high degree in the remaining graph?
        # Or just pick the first one in 'order'?
        # Let's pick a vertex from candidates.
        # To improve pruning, we should pick a vertex that splits the graph well.
        # High degree in G (low degree in H) might be good?
        
        # Let's just pick the first vertex in the static order that is in candidates.
        v = -1
        for u in order:
            if candidates_mask & (1 << u):
                v = u
                break
        
        if v == -1:
            return
            
        # Branch 1: Include v
        # New candidates: candidates without v and without its neighbors in G
        # Because if we include v, we cannot include any neighbor in G.
        new_candidates_include = candidates_mask & ~(1 << v) & ~conflict_mask[v]
        branch_and_bound(new_candidates_include, current_size + 1, best_size_ref)
        
        # Branch 2: Exclude v
        # New candidates: candidates without v
        new_candidates_exclude = candidates_mask & ~(1 << v)
        branch_and_bound(new_candidates_exclude, current_size, best_size_ref)

    # Main Setup
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Initial best size can be improved by a quick greedy heuristic
    # Greedy: Pick vertex with min degree? Or random?
    # Let's do a simple greedy to get a decent starting bound.
    best_size_ref = [0]
    curr_mask = 0
    temp_candidates = (1 << m) - 1
    
    # Greedy Initialization
    # We can use a simple heuristic: always pick the vertex with the smallest degree in the remaining graph.
    # This is O(n^3) but n=205, so 205^3 ~ 8M. Fast enough.
    
    # Actually, let's just run the B&B with an initial bound of 0.
    # The B&B will find a solution quickly.
    # To speed up the first prune, let's run a greedy MIS first.
    
    # Greedy MIS:
    # While candidates exist:
    #   Pick v with min degree in G restricted to candidates.
    #   Add v to solution.
    #   Remove v and N(v) from candidates.
    
    sol_mask = 0
    cand = (1 << m) - 1
    while cand:
        # Find min degree vertex in G restricted to cand
        min_deg = m
        min_v = -1
        for v in range(m):
            if cand & (1 << v):
                # Count neighbors in cand
                deg = bin(cand & conflict_mask[v]).count('1')
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
        
        if min_v == -1:
            break
            
        sol_mask |= (1 << min_v)
        # Remove min_v and its neighbors from cand
        cand &= ~(1 << min_v) & ~conflict_mask[min_v]
        
    best_size_ref[0] = bin(sol_mask).count('1')
    
    # Now run B&B
    # To avoid recursion depth issues or to control time, we might need to be careful.
    # But 205 is small.
    
    # We need to ensure we don't exceed time.
    # The recursive function doesn't check time.
    # We can add a global time check inside the recursion?
    # It's safer to wrap the call in a try/except or use a flag.
    
    time_exceeded = [False]
    
    def bounded_bnb(candidates_mask, current_size, best_size_ref):
        if time_exceeded[0]:
            return
        if time.time() > deadline:
            time_exceeded[0] = True
            return
            
        if candidates_mask == 0:
            return
            
        bound = greedy_color_bound(candidates_mask)
        if current_size + bound <= best_size_ref[0]:
            return
            
        v = -1
        for u in order:
            if candidates_mask & (1 << u):
                v = u
                break
        if v == -1:
            return
            
        # Include v
        new_candidates_include = candidates_mask & ~(1 << v) & ~conflict_mask[v]
        bounded_bnb(new_candidates_include, current_size + 1, best_size_ref)
        
        # Exclude v
        if time_exceeded[0]:
            return
        new_candidates_exclude = candidates_mask & ~(1 << v)
        bounded_bnb(new_candidates_exclude, current_size, best_size_ref)
        
        # Update best size?
        # We only update best_size_ref when we find a complete valid set?
        # No, in B&B for MIS, we accumulate the set.
        # Here, `current_size` is the size of the partial set.
        # We don't store the set itself, just the size.
        # The metric is the size.
        # So if we reach a leaf (candidates=0), the current_size is a valid size.
        # But we also prune if current_size + bound <= best.
        # So best_size_ref should be updated when we find a better solution.
        # However, `current_size` is just a count.
        # We need to know if the current partial solution + any completion is valid.
        # Yes, by definition of the recursion.
        
        # Wait, `best_size_ref` is only updated if we find a size larger than current best.
        # Where do we update it?
        # We should update it when we add a vertex? No, that's not a complete solution.
        # We update it when we finish a branch?
        # Actually, `current_size` is a lower bound on the final size of that branch.
        # But it's not a valid solution until we stop adding.
        # However, the greedy initialization gave us a valid solution of size `best_size_ref[0]`.
        # In the B&B, if we reach a state where we cannot add more vertices (cand=0), then `current_size` is a valid solution size.
        # But we also prune branches.
        # The standard Tomita algorithm updates the best clique size when it finds a clique of size > best.
        # Here, we are building an Independent Set.
        # We can update `best_size_ref[0]` with `current_size` at any point?
        # No, `current_size` is the size of the set we've picked so far.
        # This set IS a valid independent set.
        # So yes, we can update `best_size_ref[0] = max(best_size_ref[0], current_size)` at the start of the function?
        # That would be safe.
        
        if current_size > best_size_ref[0]:
            best_size_ref[0] = current_size

    # Run B&B
    # Start with all vertices
    initial_candidates = (1 << m) - 1
    bounded_bnb(initial_candidates, 0, best_size_ref)
    
    # If time exceeded, we still have best_size_ref[0]
    # But we lost the actual set!
    # We only tracked the size.
    # We need to reconstruct the set or track it.
    
    # Let's re-run with set tracking?
    # Or, since we found the size, we can just run the heuristic again and hope it finds that size?
    # No, that's risky.
    
    # Better approach: Track the best mask.
    
    best_mask_ref = [0]
    
    def bounded_bnb_with_mask(candidates_mask, current_mask, current_size, best_size_ref, best_mask_ref):
        if time_exceeded[0]:
            return
        if time.time() > deadline:
            time_exceeded[0] = True
            return
            
        if candidates_mask == 0:
            if current_size > best_size_ref[0]:
                best_size_ref[0] = current_size
                best_mask_ref[0] = current_mask
            return
            
        bound = greedy_color_bound(candidates_mask)
        if current_size + bound <= best_size_ref[0]:
            return
            
        # Update best if current is better?
        # current_mask is a valid independent set.
        if current_size > best_size_ref[0]:
            best_size_ref[0] = current_size
            best_mask_ref[0] = current_mask
            
        v = -1
        for u in order:
            if candidates_mask & (1 << u):
                v = u
                break
        if v == -1:
            return
            
        # Include v
        new_candidates_include = candidates_mask & ~(1 << v) & ~conflict_mask[v]
        new_mask_include = current_mask | (1 << v)
        bounded_bnb_with_mask(new_candidates_include, new_mask_include, current_size + 1, best_size_ref, best_mask_ref)
        
        # Exclude v
        if time_exceeded[0]:
            return
        new_candidates_exclude = candidates_mask & ~(1 << v)
        bounded_bnb_with_mask(new_candidates_exclude, current_mask, current_size, best_size_ref, best_mask_ref)

    # Reset time flag
    time_exceeded[0] = False
    best_size_ref[0] = bin(sol_mask).count('1')
    best_mask_ref[0] = sol_mask
    
    bounded_bnb_with_mask(initial_candidates, 0, 0, best_size_ref, best_mask_ref)
    
    best_mask = best_mask_ref[0]
    
    # Convert best_mask to array
    result = []
    for i in range(m):
        if best_mask & (1 << i):
            result.append(i)
            
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END

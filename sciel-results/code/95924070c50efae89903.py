import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    n = 256
    
    # 1. Build conflict graph: edge (a,b) if (a*b) % (a+b) == 0
    adj_mask = [0] * (n + 1)
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj_mask[a] |= (1 << (b - 1))
                adj_mask[b] |= (1 << (a - 1))
                
    # Complement graph for Max Clique: edge (a,b) if (a*b) % (a+b) != 0
    full_mask = (1 << n) - 1
    comp_adj_mask = [0] * (n + 1)
    for v in range(1, n + 1):
        comp_adj_mask[v] = full_mask ^ adj_mask[v] ^ (1 << (v - 1))
    
    # 2. Initial greedy solution
    best_size = 0
    best_set = []
    
    # Greedy: sort by conflict degree ascending, add if valid
    conflict_deg = [0] * (n + 1)
    for v in range(1, n + 1):
        conflict_deg[v] = bin(adj_mask[v]).count('1')
    
    init_set = []
    init_mask = 0
    for v in sorted(range(1, n + 1), key=lambda x: conflict_deg[x]):
        if adj_mask[v] & init_mask == 0:
            init_set.append(v)
            init_mask |= (1 << (v - 1))
    
    best_size = len(init_set)
    best_set = init_set[:]
    
    # Also try: all odds
    odds = [v for v in range(1, n + 1) if v % 2 == 1]
    valid = True
    for i in range(len(odds)):
        for j in range(i+1, len(odds)):
            a, b = odds[i], odds[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break
    if valid and len(odds) > best_size:
        best_size = len(odds)
        best_set = odds[:]
    
    # Local search: try adding excluded vertices
    in_set_mask = 0
    for v in best_set:
        in_set_mask |= (1 << (v - 1))
    for _ in range(200):
        improved = False
        for v in range(1, n + 1):
            if in_set_mask & (1 << (v - 1)):
                continue
            if adj_mask[v] & in_set_mask == 0:
                best_set.append(v)
                in_set_mask |= (1 << (v - 1))
                best_size += 1
                improved = True
        if not improved:
            break
        
    # 3. Tomita's Branch and Bound for Maximum Clique on complement graph
    # Use iterative approach with explicit stack to avoid recursion overhead
    
    # Helper: extract vertices from mask, sorted by degree in induced subgraph
    def get_vertices_by_degree(mask, adj_masks):
        verts = []
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length()
            verts.append(v)
            temp ^= lsb
        # Sort by degree in the induced subgraph (descending)
        # Degree of v in induced subgraph = popcount(adj_masks[v] & mask)
        verts.sort(key=lambda v: bin(adj_masks[v] & mask).count('1'), reverse=True)
        return verts
    
    # Tomita MAXCLQ with coloring bound
    # We'll use a recursive function but with optimizations
    
    def expand(P_list, clique_size, clique):
        nonlocal best_size, best_set
        
        if time.time() - start_time > timeout:
            return
            
        if not P_list:
            if clique_size > best_size:
                best_size = clique_size
                best_set = clique[:]
            return
            
        # Greedy Coloring with degree ordering
        # P_list should already be sorted by degree descending
        # Color in that order for tighter bounds
        
        color_of = [0] * (n + 1)
        colored_mask = 0
        max_color = 0
        v_colors = []
        
        for v in P_list:
            # Find colors of colored neighbors in complement graph
            neighbor_mask = comp_adj_mask[v] & colored_mask
            used_colors = set()
            temp = neighbor_mask
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length()
                if color_of[u] > 0:
                    used_colors.add(color_of[u])
                temp ^= lsb
                
            c = 1
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c > max_color:
                max_color = c
            v_colors.append((v, c))
            colored_mask |= (1 << (v - 1))
        
        # Sort by color ascending for branching
        v_colors.sort(key=lambda x: x[1])
        
        # Precompute suffix masks
        m = len(v_colors)
        suffix_masks = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            v_i = v_colors[i][0]
            suffix_masks[i] = suffix_masks[i+1] | (1 << (v_i - 1))
        
        # Branch
        for idx in range(m):
            if time.time() - start_time > timeout:
                return
            
            v, c = v_colors[idx]
            
            # Bound check
            if clique_size + 1 + (max_color - c) <= best_size:
                break
            
            # Branch Include
            new_P_mask = comp_adj_mask[v] & suffix_masks[idx+1]
            
            if new_P_mask:
                new_P_list = get_vertices_by_degree(new_P_mask, comp_adj_mask)
                expand(new_P_list, clique_size + 1, clique + [v])
            else:
                if clique_size + 1 > best_size:
                    best_size = clique_size + 1
                    best_set = clique + [v]
                
    # Initial P: all vertices sorted by degree in complement graph
    initial_P_list = get_vertices_by_degree(full_mask, comp_adj_mask)
    expand(initial_P_list, 0, [])
    
    if not best_set:
        best_set = [1]
        
    return list(best_set)
    
    # Greedy Coloring for Upper Bound
    # Returns a list of colors for each vertex in P, and the max color used.
    # We want to color P such that no two adjacent vertices in P have the same color.
    # This gives an upper bound on the size of the MIS in P (since each color class is an independent set, 
    # wait. No. In a coloring of the graph, adjacent vertices have DIFFERENT colors.
    # An Independent Set can contain at most one vertex from each clique? No.
    # In a graph coloring, an Independent Set is NOT bounded by the number of colors.
    # Wait. The bound for MIS using coloring comes from the fact that if we color the graph with k colors,
    # the MIS can be at most... wait.
    # Tomita's algorithm uses coloring on the *remaining candidate set*.
    # The greedy coloring produces color classes $C_1, C_2, \dots, C_k$.
    # Each $C_i$ is an independent set in the original graph? No, $C_i$ is a set of vertices with the same color.
    # In a proper coloring, no two adjacent vertices have the same color.
    # So $C_i$ is an independent set.
    # The MIS of P is a subset of P.
    # Actually, the bound is: The size of the MIS of P is at most the sum of the sizes of the color classes? No.
    # The bound is: The MIS of P is at most the number of colors? No.
    # Let's re-read Tomita.
    # "We color the vertices of P greedily... Let the colors be 1..k.
    #  The vertices are ordered by color.
    #  We branch on vertices in this order.
    #  The bound is: if current_size + number_of_remaining_vertices_with_color >= c <= best_size, prune.
    #  Actually, the standard Tomita bound is:
    #  If we have colored P with k colors, and we are at vertex v with color c,
    #  then the maximum additional vertices we can add from the remaining part of P (after v) is at most k - c + 1?
    #  No.
    #  Let's look at the standard MAXCLQ (Max Clique) algorithm, as MIS on G is Max Clique on complement G'.
    #  In MAXCLQ, we color the candidate set P.
    #  If we assign colors 1..k to P, and we are processing vertices in order of increasing color,
    #  then any clique in the remaining candidates (after processing vertex i) can have size at most (k - color[i] + 1).
    #  Why? Because a clique can contain at most one vertex of each color.
    #  If we have already "used up" colors 1..color[i]-1 by taking vertices or pruning them,
    #  the remaining vertices have colors >= color[i].
    #  So the max clique size from the rest is bounded by the number of distinct colors remaining.
    
    # So we should solve MAX CLIQUE on the COMPLEMENT graph.
    # Complement graph G': Edge (a,b) if a and b are NOT connected in G.
    # i.e., (a*b) % (a+b) != 0.
    # We want Max Clique in G'.
    
    # Let's build the complement adjacency masks.
    # full_mask = (1 << n) - 1
    # comp_adj_mask[v] = full_mask ^ adj_mask[v] ^ (1 << (v-1))  # remove self
    
    full_mask = (1 << n) - 1
    comp_adj_mask = [0] * (n + 1)
    for v in range(1, n + 1):
        # adj_mask[v] has bits for neighbors in G.
        # We want bits for non-neighbors in G (excluding v).
        comp_adj_mask[v] = full_mask ^ adj_mask[v] ^ (1 << (v - 1))
        
    # Tomita MAXCLQ
    # P is a bitmask of candidates.
    # We maintain an array of vertices corresponding to bits in P? 
    # It's faster to just work with bitmasks and extract vertices when needed.
    
    def expand(P, clique_size, clique):
        nonlocal best_size, best_set
        
        if time.time() - start_time > timeout:
            return
            
        if not P:
            if clique_size > best_size:
                best_size = clique_size
                best_set = clique[:]
            return
            
        # Greedy Coloring of P
        # We need to order vertices by color.
        # Let's extract vertices from P.
        # To speed up, we can maintain a list of vertices in P.
        # But extracting from bitmask every time is O(n).
        # Alternative: Pass a list of vertices and use bitmasks for intersection.
        
        # Let's use a list-based approach for P to avoid repeated extraction.
        # P_list: list of vertices in P.
        # We will sort them by color.
        
        # Greedy Color
        # Returns: list of (vertex, color)
        # And the max color.
        
        # Optimization: We can cache coloring results? No, P changes.
        
        # Let's implement a fast greedy coloring on the list P_list.
        # We process vertices in P_list.
        # For each vertex v, we check which colors are used by its neighbors in P_list that have already been colored.
        
        # To make this fast, we can use bitmasks for colors?
        # Max color is likely < 20.
        
        # Let's stick to a simple loop but optimize the neighbor check.
        # Pre-compute for each vertex v, the mask of its neighbors in G' (comp_adj_mask[v]).
        
        # We need to know, for a vertex v, which colors are "blocked".
        # Blocked colors are colors of neighbors of v that are in P.
        
        # Algorithm:
        # 1. Initialize color of all vertices in P_list to 0.
        # 2. Iterate through P_list.
        # 3. For current vertex v, determine used_colors = { color[u] for u in P_list if u is neighbor of v in G' and u is already colored }
        # 4. Assign smallest positive integer not in used_colors.
        
        # To do step 3 fast:
        # We can maintain an array `color_of` of size n+1.
        # And for each color c, a bitmask `color_mask[c]` of vertices assigned color c.
        # Then for vertex v, `neighbor_mask = comp_adj_mask[v] & P`.
        # `used_colors` = { c | color_mask[c] & neighbor_mask != 0 }.
        
        # This is efficient if number of colors is small.
        
        # Let's implement this.
        
        # P_list is passed as a list.
        # We will sort P_list by the assigned color to determine branching order.
        # Tomita suggests branching on vertices in order of increasing color? 
        # Actually, usually we branch on the vertex with the *highest* color first? 
        # Or the last one?
        # In Tomita's max clique, the vertices are ordered by color.
        # We branch on vertices in the order they appear in the sorted list.
        # The bound is: if clique_size + (max_color - color[i] + 1) <= best_size, prune.
        
        # Let's refine the bound.
        # If we have colored P with colors 1..k.
        # Let the vertices be ordered $v_1, \dots, v_m$ such that $color(v_i) \le color(v_{i+1})$.
        # When we consider $v_i$, the remaining vertices are $v_{i+1}, \dots, v_m$.
        # All these vertices have color $\ge color(v_i)$.
        # The maximum clique in the subgraph induced by $v_{i+1}, \dots, v_m$ has size at most $k - color(v_i) + 1$.
        # (Because it can take at most one vertex of each color from $color(v_i)$ to $k$).
        
        # So, if `clique_size + 1 + (k - color(v_i))` <= best_size?
        # Wait. If we include $v_i$, we add 1 to clique size.
        # The remaining candidates are neighbors of $v_i$ in G' that are in $P \setminus \{v_1..v_i\}$.
        # The bound applies to the *entire* remaining set $P \setminus \{v_1..v_i\}$.
        # The max clique in that set is bounded by $k - color(v_i) + 1$.
        # So if `clique_size + 1 + (k - color(v_i))` <= best_size, we can prune the branch where we INCLUDE $v_i$?
        # No, we prune the branch if even the *best* possible extension is not better.
        # The best extension from including $v_i$ is $1 + \text{MIS}(N(v_i) \cap P_{remaining})$.
        # The bound $k - color(v_i) + 1$ is an upper bound on $\text{MIS}(P_{remaining})$.
        # But $N(v_i) \cap P_{remaining}$ is a subset of $P_{remaining}$.
        # So the bound holds.
        
        # Actually, the standard Tomita prune is:
        # If `clique_size + (k - color(v_i) + 1)` <= best_size, then we can stop processing this vertex and all subsequent vertices with the same or higher color?
        # No, we just skip this vertex?
        # If we don't include $v_i$, we move to $v_{i+1}$.
        # The bound for $v_{i+1}$ will be tighter (or equal).
        # So if `clique_size + (k - color(v_i) + 1)` <= best_size, we can break the loop entirely?
        # Yes, because for any $j \ge i$, $color(v_j) \ge color(v_i)$, so $k - color(v_j) + 1 \le k - color(v_i) + 1$.
        # Thus `clique_size + (k - color(v_j) + 1)` <= best_size for all $j \ge i$.
        # So we can break.
        
        # Implementation Details:
        # We need to maintain `color_mask` for the current P.
        
        # Let's restructure to pass P as a bitmask and extract vertices only when needed for coloring.
        # Or pass P as a list.
        # Given n=256, extracting 256 bits is fast.
        
        def get_vertices_from_mask(mask):
            verts = []
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length()
                verts.append(v)
                mask ^= lsb
            return verts

        # We will maintain P as a list of vertices to avoid repeated extraction.
        # When we branch, we compute new_P_list.
        
        # Let's rewrite expand to take P_list.
        
    # Redefine expand with P_list
    def expand(P_list, clique_size, clique):
        nonlocal best_size, best_set
        
        if time.time() - start_time > timeout:
            return
            
        if not P_list:
            if clique_size > best_size:
                best_size = clique_size
                best_set = clique[:]
            return
            
        # Greedy Coloring
        # color_of[v]
        # color_mask[c]
        
        # To speed up coloring, we can order P_list by degree in G' (descending) before coloring?
        # Tomita usually uses the order from the previous level or just index order.
        # Let's try sorting by degree in the induced subgraph? That's expensive.
        # Just use index order or the order they are in P_list.
        
        # We'll use a simple greedy:
        # Iterate P_list.
        # For each v, find min color not used by neighbors in P_list.
        
        # To find neighbors in P_list fast:
        # P_mask = bitmask of P_list.
        # neighbor_mask = comp_adj_mask[v] & P_mask.
        # Then check colors of bits in neighbor_mask.
        
        P_mask = 0
        for v in P_list:
            P_mask |= (1 << (v - 1))
            
        # We need to map vertex to index in P_list? No, just vertex.
        # We need to know which colors are used by neighbors.
        
        # Initialize colors
        # Since we re-color at each node, we can use a local dict or array.
        # n is small, array of size n+1 is fine.
        color_of = [0] * (n + 1)
        # Max colors expected is small (< 30).
        color_masks = [0] * 100  # 100 colors is plenty
        
        # We process vertices in P_list order.
        # To get a good bound, the order matters.
        # Tomita: "The vertices are colored in the order of their appearance in P".
        # P is often ordered by degree.
        # Let's sort P_list by degree in G' descending?
        # Degree in G' for v is popcount(comp_adj_mask[v] & P_mask).
        # This is O(|P|^2) if done naively.
        # Let's just use the current order for now.
        
        max_color = 0
        colored_vertices = []
        
        for v in P_list:
            # Find used colors
            neighbor_mask = comp_adj_mask[v] & P_mask
            # We only care about neighbors that are already colored?
            # No, greedy coloring typically considers the whole set?
            # Standard greedy: process v, check neighbors *already colored*.
            # But to make the bound valid, we need a proper coloring of the *entire* P.
            # The bound $k - color(v_i) + 1$ relies on the fact that all vertices in $P_{remaining}$ have colors $\ge color(v_i)$.
            # This is true if we assign colors such that colors are non-decreasing in the order of processing?
            # No, greedy coloring doesn't guarantee that.
            # However, Tomita's algorithm sorts the vertices by color *after* coloring.
            # And the bound is applied on the sorted list.
            
            # So:
            # 1. Color all vertices in P_list.
            # 2. Sort P_list by assigned color (ascending).
            # 3. Iterate through sorted list.
            # 4. Apply bound.
            
            # How to color efficiently?
            # For v, used_colors = { color_of[u] for u in P_list if u is neighbor of v and u is already processed }.
            # This is the standard greedy.
            
            # Let's optimize the "already processed" check.
            # We can just iterate through P_list again? No.
            # We can maintain `colored_mask` of vertices processed so far.
            
            pass # Placeholder
            
        # Let's implement the coloring properly.
        
        colored_mask = 0
        max_color = 0
        # We store (vertex, color)
        v_colors = []
        
        for v in P_list:
            # Find colors of colored neighbors
            neighbor_mask = comp_adj_mask[v] & colored_mask
            used_colors = set()
            temp = neighbor_mask
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length()
                used_colors.add(color_of[u])
                temp ^= lsb
                
            c = 1
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c > max_color:
                max_color = c
            v_colors.append((v, c))
            colored_mask |= (1 << (v - 1))
            
        # Sort by color
        v_colors.sort(key=lambda x: x[1])
        
        # Branch
        # P_list for children needs to be updated.
        # We iterate through v_colors.
        
        for v, c in v_colors:
            if time.time() - start_time > timeout:
                return
            
            # Bound check
            # If clique_size + 1 + (max_color - c) <= best_size?
            # Wait. The remaining vertices after v (in the sorted order) have colors >= c.
            # The max clique in the remaining set is bounded by (max_color - c + 1).
            # If we include v, we get 1 + MIS(remaining neighbors).
            # So if clique_size + 1 + (max_color - c) <= best_size?
            # No, the bound is on the *size* of the clique we can form from the remaining vertices.
            # So total size <= clique_size + 1 + (max_color - c).
            # If this <= best_size, we prune.
            
            if clique_size + 1 + (max_color - c) <= best_size:
                # Since colors are non-decreasing, for any subsequent vertex v', c' >= c.
                # So clique_size + 1 + (max_color - c') <= clique_size + 1 + (max_color - c) <= best_size.
                # We can break the loop.
                break
            
            # Include v
            # New P: neighbors of v in G' that are in P_list and come AFTER v in the sorted order?
            # No. In Tomita, when we branch on v, we remove v and all its non-neighbors from P.
            # The new P is $P \cap N(v)$.
            # But we also need to maintain the property that we don't revisit vertices.
            # The standard Tomita recursion passes $P \cap N(v)$.
            # However, we must ensure we don't pick vertices that were "before" v in some ordering?
            # Actually, the set $P$ passed to the next level is simply $P_{current} \cap N(v)$.
            # The ordering for the next level is re-determined by coloring.
            
            # So:
            # new_P_mask = P_mask & comp_adj_mask[v]
            # new_P_list = get_vertices_from_mask(new_P_mask)
            
            # But wait. If we just take $P \cap N(v)$, we might include vertices that were "before" v in the current sorted order.
            # Does this matter?
            # In Max Clique, the order of processing doesn't restrict the set, it just restricts the *search tree* to avoid duplicates.
            # The set of candidates for the next level is indeed $P \cap N(v)$.
            
            # However, to be consistent with the bound logic (which assumes we are processing in a specific order),
            # some implementations restrict to vertices that appear *after* v in the sorted list.
            # Let's check.
            # If we allow vertices before v, we might find the same clique in a different branch.
            # To avoid this, we usually restrict to vertices that are "later" in the ordering.
            # But the ordering changes at each level.
            # The standard Tomita algorithm uses the ordering from the *current* coloring to decide which vertices to keep.
            # Specifically, if we branch on $v_i$ (the i-th in sorted order), the new candidate set is $P \cap N(v_i) \cap \{v_{i+1}, \dots, v_m\}$.
            # This ensures that we only build cliques in a canonical order.
            
            # So we need to know which vertices in $P \cap N(v)$ appear after v in `v_colors`.
            
            # Let's create a set of vertices that are after v in v_colors.
            # Since v_colors is sorted by color, and we iterate through it,
            # we can just take the subset of `v_colors` from index i+1 onwards.
            
            # But we need the actual vertices.
            
            # Let's optimize:
            # We have `v_colors` list.
            # We are at index `idx`.
            # `v, c = v_colors[idx]`
            # The remaining vertices in the current level are `v_colors[idx+1:]`.
            # We need to intersect this set with $N(v)$.
            
            # Let's build a mask of the remaining vertices.
            # remaining_mask = 0
            # for u, cu in v_colors[idx+1:]:
            #     remaining_mask |= (1 << (u-1))
            
            # This is O(N) per branch. Total O(N^2) per node.
            # N=256. 256^2 = 65536.
            # If we have 1000 nodes, 65M ops. Might be slow in Python.
            
            # Alternative:
            # Precompute a prefix sum of masks?
            # Or just iterate.
            
            # Let's try to be clever.
            # We can maintain a bitmask `suffix_mask` that represents $v_{i+1} \dots v_m$.
            # Initially, suffix_mask for i=0 is mask of all P.
            # As we iterate i, we remove $v_i$ from suffix_mask?
            # No, we need the mask of $v_{i+1} \dots v_m$.
            
            # Let's just compute it.
            
            # Optimization:
            # Instead of building a list of remaining vertices, we can just use the mask.
            # new_P_mask = comp_adj_mask[v] & suffix_mask
            
            # Let's precompute suffix masks for v_colors.
            # suffix_masks[i] = mask of vertices v_colors[i] to v_colors[-1].
            # Then for branch at i, remaining = suffix_masks[i+1].
            
            suffix_masks = [0] * (len(v_colors) + 1)
            # suffix_masks[len] = 0
            for i in range(len(v_colors) - 1, -1, -1):
                v_i = v_colors[i][0]
                suffix_masks[i] = suffix_masks[i+1] | (1 << (v_i - 1))
                
            # Now loop
            for idx in range(len(v_colors)):
                if time.time() - start_time > timeout:
                    return
                
                v, c = v_colors[idx]
                
                # Bound
                if clique_size + 1 + (max_color - c) <= best_size:
                    break
                
                # Branch Include
                # New P is neighbors of v in G' that are in suffix_masks[idx+1]
                new_P_mask = comp_adj_mask[v] & suffix_masks[idx+1]
                
                if new_P_mask:
                    # Extract vertices
                    new_P_list = []
                    temp = new_P_mask
                    while temp:
                        lsb = temp & -temp
                        u = lsb.bit_length()
                        new_P_list.append(u)
                        temp ^= lsb
                    
                    # Recurse
                    expand(new_P_list, clique_size + 1, clique + [v])
                else:
                    # Leaf
                    if clique_size + 1 > best_size:
                        best_size = clique_size + 1
                        best_set = clique + [v]
                        
                # Branch Exclude is implicit by moving to next vertex
                # No explicit action needed, the loop continues.
                
    # Initial P
    initial_P_list = list(range(1, n + 1))
    expand(initial_P_list, 0, [])
    
    if not best_set:
        # Fallback
        best_set = [1]
        
    return list(best_set)

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))

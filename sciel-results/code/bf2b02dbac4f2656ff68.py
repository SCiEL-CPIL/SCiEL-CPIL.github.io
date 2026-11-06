import time
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85  # Safety margin
    
    # 1. Build compatibility graph
    # adj[i] is a bitmask where bit j is 1 if lcm(i, j) <= limit
    # We use 0-indexed for bitmasks, so node i (1..512) corresponds to bit (i-1)
    N = limit
    adj = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i * j // gcd(i, j) <= limit:
                # Set bit (j-1)
                mask |= (1 << (j - 1))
        adj[i] = mask
        
    # 2. Maximum Clique Solver (Tomita et al.)
    # We will search for the maximum clique in the compatibility graph.
    # Vertices are 1..N.
    
    best_clique_size = 0
    best_clique = []
    
    # Helper to get list of vertices from bitmask
    def mask_to_list(mask):
        return [i + 1 for i in range(N) if (mask >> i) & 1]

    # Greedy coloring of the subgraph induced by P (bitmask)
    # Returns list of (vertex, color) sorted by color descending, then vertex
    # Colors are 1..max_color
    def color_sort(P_mask):
        # We need to color the graph induced by P_mask.
        # Vertices in P_mask are candidates.
        # We want a proper coloring of the induced subgraph? 
        # No, for Max Clique, we need a coloring such that no two vertices of the same color are ADJACENT in G_comp.
        # This is a standard graph coloring of the induced subgraph G_comp[P].
        # The bound is: max clique size <= number of colors used.
        
        # To make it fast, we use a greedy approach.
        # We process vertices in some order. A good order is often by degree or just fixed.
        # Let's extract vertices from P_mask.
        vertices = mask_to_list(P_mask)
        
        # Sort vertices by degree in P descending? Or just use the order they appear?
        # Tomita suggests sorting by color in the previous step or just using a heuristic.
        # For simplicity and speed in Python, we'll use a simple greedy coloring.
        # We'll assign the smallest possible color.
        
        # To optimize, we can sort vertices by the number of neighbors in P (degree) descending.
        # This often leads to fewer colors.
        degs = []
        for v in vertices:
            # Count neighbors in P
            # adj[v] & P_mask gives neighbors.
            # We need to count bits.
            d = bin(adj[v] & P_mask).count('1')
            degs.append((d, v))
        
        # Sort by degree descending
        degs.sort(reverse=True)
        
        colors = {}
        max_color = 0
        # We need to track which colors are used by neighbors of v in P.
        # Actually, we just need to assign the smallest color not used by any neighbor of v in the induced subgraph.
        
        # Efficient way:
        # Keep a list of colors assigned to each vertex.
        # For each v, find min color not in {color[u] for u in neighbors(v) in P}.
        
        # Since N is small (512), this is fine.
        
        # To speed up, we can process in order of degs.
        for d, v in degs:
            # Find used colors among neighbors in P
            # neighbors = mask_to_list(adj[v] & P_mask)
            # This is slow. Let's use bit operations if possible, but we need the colors of specific neighbors.
            
            used_colors = set()
            # Iterate through neighbors in P
            # We can iterate through bits of (adj[v] & P_mask)
            mask = adj[v] & P_mask
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                u = idx + 1
                if u in colors:
                    used_colors.add(colors[u])
                mask ^= lsb
                
            # Find smallest available color
            c = 1
            while c in used_colors:
                c += 1
            
            colors[v] = c
            if c > max_color:
                max_color = c
        
        # Create list of (color, vertex)
        # Sort by color descending, then vertex (or any order)
        # We return the list of vertices ordered such that we process higher colors first.
        # In Tomita's algorithm, we process vertices in reverse order of their color assignment.
        # Specifically, we partition P into color classes C_1, C_2, ..., C_k.
        # We iterate from k down to 1.
        
        # Let's group by color
        color_classes = {}
        for v, c in colors.items():
            if c not in color_classes:
                color_classes[c] = []
            color_classes[c].append(v)
        
        # Sort colors descending
        sorted_colors = sorted(color_classes.keys(), reverse=True)
        
        # Build the ordered list of vertices
        # We want to return a list of (vertex, color) sorted by color descending.
        # Actually, the algorithm usually takes a list of vertices and their colors.
        # We'll return a list of tuples (vertex, color) sorted by color desc.
        
        ordered = []
        for c in sorted_colors:
            for v in color_classes[c]:
                ordered.append((v, c))
                
        return ordered

    def expand(C, P_mask):
        nonlocal best_clique_size, best_clique
        
        if time.time() > deadline:
            return
        
        if not P_mask:
            if len(C) > best_clique_size:
                best_clique_size = len(C)
                best_clique = C[:]
            return
        
        # Get ordered vertices and colors
        ordered = color_sort(P_mask)
        
        # Process in reverse order of the list? 
        # Tomita's algorithm:
        # While P is not empty:
        #   Pop v from P (usually the one with highest color)
        #   If |C| + color(v) <= best, break
        #   C.add(v)
        #   expand(C, P & adj[v])
        #   C.remove(v)
        #   P.remove(v)
        
        # The `ordered` list has vertices sorted by color descending.
        # We should process them in that order.
        
        # However, we need to maintain the set of remaining candidates P.
        # Let's convert P_mask to a list of vertices to make removal easier?
        # Or just iterate through the ordered list and check if the vertex is still in P_mask.
        
        # To be efficient, we'll iterate through the ordered list.
        # But we need to update P_mask as we go.
        
        # Let's copy the list of vertices from P_mask to a list for iteration?
        # The `ordered` list contains all vertices in P_mask.
        
        for v, c in ordered:
            if time.time() > deadline:
                return
            
            # Check if v is still a candidate (it might have been removed if we implemented a different loop structure, 
            # but here we are iterating over the initial set of P for this call. 
            # Wait, in standard Tomita, we remove v from P after processing.
            # Since we are iterating over a snapshot, we must ensure we don't process a vertex that is no longer in P.
            # But in this specific loop structure, we are popping from the *original* P set for this recursion level.
            # So every v in `ordered` is in the original P_mask.
            
            # Upper bound check
            if len(C) + c <= best_clique_size:
                break
            
            # Branch
            C.append(v)
            
            # New candidates are those in P_mask that are compatible with v
            # AND that come "after" v in some sense? 
            # In Tomita, we usually restrict to vertices that are "later" in the ordering to avoid duplicates?
            # No, the standard recursive call is expand(C + [v], P & adj[v]).
            # But we must ensure we don't revisit. The coloring bound handles the pruning.
            
            # Actually, the standard implementation processes vertices one by one from P.
            # When we pick v, the new P is (P \ {v}) & adj[v]? 
            # No, the new P is P & adj[v]. The vertices removed are those not adjacent to v.
            # The vertices adjacent to v remain in the candidate set for the next level.
            # However, we must remove v itself.
            
            new_P_mask = P_mask & adj[v]
            # Remove v from new_P_mask? 
            # adj[v] usually includes v (self-loop). We should check.
            # In our graph, lcm(v,v)=v <= 512, so v is compatible with v.
            # So bit v-1 is set in adj[v].
            # We should remove v from the candidate set for the next level to avoid adding it again?
            # A clique is a set of distinct vertices. So yes, remove v.
            new_P_mask &= ~(1 << (v - 1))
            
            expand(C, new_P_mask)
            
            C.pop()
            
            # Remove v from P_mask for the next iteration of the loop
            # Since we are iterating over `ordered` which is a list, we don't strictly need to update a local P variable 
            # if we just skip vertices that are no longer in the "active" set?
            # But `ordered` is fixed. 
            # The standard loop is:
            # while P:
            #   v = P.pop()
            #   ...
            #   P = P & adj[v]  (This updates the set of remaining candidates)
            
            # My current loop iterates over the initial P. 
            # This is incorrect for the "remove from P" part.
            # If I pick v, the remaining candidates for the *next* vertex in the loop should be restricted to those compatible with v?
            # No. The branching structure is:
            # Option 1: Include v. Then all future vertices must be compatible with v.
            # Option 2: Exclude v. Then future vertices can be anything in the remaining P.
            
            # The Tomita algorithm implements this by:
            # 1. Coloring P.
            # 2. Iterating through vertices in order of decreasing color.
            # 3. When processing v:
            #    - If bound fails, stop.
            #    - Include v: Recurse with P' = P & adj[v] (and v removed).
            #    - Exclude v: Remove v from P. Continue loop with the reduced P.
            
            # So, I need to maintain a local variable `current_P` that gets reduced.
            
            # Let's restructure the loop.
            
        # RESTRUCTURE:
        # We need a mutable set of candidates.
        # Let's use a list of vertices and a bitmask for fast intersection.
        
        # Actually, the code above was a draft. Let's write the correct `expand` body.
        
    # Corrected Expand
    def expand(C, P_mask):
        nonlocal best_clique_size, best_clique
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if len(C) > best_clique_size:
                best_clique_size = len(C)
                best_clique = C[:]
            return
        
        # Color the candidates
        ordered = color_sort(P_mask)
        
        # We will process vertices in the order of `ordered`.
        # `ordered` is a list of (v, color).
        # We maintain a bitmask `remaining_P` which starts as P_mask.
        remaining_P = P_mask
        
        for v, c in ordered:
            if time.time() > deadline:
                return
            
            # If v is not in remaining_P, it was excluded in a previous step?
            # No, in this loop structure, we process each vertex exactly once.
            # The "Exclude" branch is implicit: we just don't include it, and we remove it from remaining_P.
            
            # Check if v is still in remaining_P (it should be, unless we messed up)
            if not (remaining_P & (1 << (v - 1))):
                continue
            
            # Bound check
            if len(C) + c <= best_clique_size:
                # Since colors are non-increasing in `ordered`, all subsequent vertices have color <= c.
                # So we can break the loop entirely.
                break
            
            # Branch: Include v
            C.append(v)
            
            # New candidates: (remaining_P & adj[v]) with v removed
            # Note: remaining_P contains v.
            new_P_mask = remaining_P & adj[v]
            new_P_mask &= ~(1 << (v - 1)) # Remove v
            
            expand(C, new_P_mask)
            
            C.pop()
            
            # Branch: Exclude v
            # Remove v from remaining_P
            remaining_P &= ~(1 << (v - 1))

    # Initial call
    # P_mask is all nodes 1..512
    init_P_mask = (1 << N) - 1
    expand([], init_P_mask)
    
    return sorted(best_clique)

# Test run locally? No, just output the code.
# One optimization: The coloring function `color_sort` is called at every node of the search tree.
# This can be expensive. 
# However, N=512 is small enough that with good pruning, it should work.
# To speed up `color_sort`, we can cache colors? No, the subgraph changes.
# We can optimize `color_sort` by using a faster greedy coloring.
# The current implementation iterates through bits to find neighbors.
# `bin(mask).count('1')` is fast.
# The inner loop `while mask` iterates through set bits.
# This is O(E_in_subgraph).

# Let's add a simple heuristic to order vertices in `color_sort` better?
# The degree sort is already there.

# One potential issue: Recursion depth.
# Max depth is 512. Python default is 1000. Should be fine.
# But if we hit the limit, we might need to increase it.
import sys
sys.setrecursionlimit(2000)

# Final check on the logic.
# The graph is "compatibility".
# We want Max Clique.
# Tomita's algorithm finds Max Clique.
# The coloring bound is valid.

# Is there a faster way to build the graph?
# Precomputation is O(N^2) = 512^2 ~ 260k ops. Fast.

# Is there a faster way to color?
# We can use a bitset-based coloring if we precompute something?
# Maybe not necessary.

# Let's ensure the deadline check is robust.
# `time.time()` is called frequently.

# One more thing: The problem asks for a subset of [1, 512].
# The output is `sorted(best_clique)`.

# If the search takes too long, it will return the best found so far.
# This is a safe fallback.

# To improve the chance of finding the optimal solution quickly, 
# we can start with a good initial lower bound.
# We can run a greedy heuristic first to get a large clique.
# Then pass that size as `best_clique_size` to the exact solver.
# This will prune much more aggressively.

def greedy_clique(P_mask):
    # Greedy: pick vertex with max degree in P, add to clique, remove non-neighbors.
    C = []
    current_P = P_mask
    while current_P:
        # Find vertex with max degree in current_P
        best_v = -1
        max_deg = -1
        # Iterate through bits
        mask = current_P
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            v = idx + 1
            deg = bin(adj[v] & current_P).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
            mask ^= lsb
        
        if best_v == -1:
            break
            
        C.append(best_v)
        # Update P: keep only neighbors of best_v
        current_P &= adj[best_v]
        current_P &= ~(1 << (best_v - 1))
        
    return C

# In construct():
# 1. Build Graph
# 2. Run Greedy to get initial best
# 3. Run Exact Search with that bound

# Let's integrate this.

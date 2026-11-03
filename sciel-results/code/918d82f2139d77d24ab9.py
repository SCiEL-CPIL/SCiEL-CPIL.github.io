import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # Generate all weight-4 supports
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)
    
    # Build compatibility graph: edge if intersection size <= 2
    # Use bitmasks for fast intersection
    sup_masks = []
    for s in supports:
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        sup_masks.append(mask)
    
    # Build adjacency list for compatibility graph
    # compat[i] = set of j such that supports[i] and supports[j] are compatible
    compat = [set() for _ in range(N_sup)]
    
    for i in range(N_sup):
        if time.time() > deadline - 30.0:
            break
        for j in range(i + 1, N_sup):
            inter_count = bin(sup_masks[i] & sup_masks[j]).count('1')
            if inter_count <= 2:
                compat[i].add(j)
                compat[j].add(i)
    
    # Now find maximum clique in this graph using branch-and-bound with coloring
    # This is the key structural improvement: exact/near-exact optimization
    
    # Greedy coloring for upper bound
    def color_graph(vertices, adj):
        """
        Greedy coloring of the subgraph induced by vertices.
        Returns color classes and max color number.
        vertices: list of vertex indices
        adj: adjacency list (list of sets)
        """
        if not vertices:
            return 0, []
        
        colors = {}
        color_classes = []
        remaining = set(vertices)
        color_num = 0
        
        while remaining:
            color_num += 1
            current_class = set()
            # Greedy: pick vertices that are not adjacent to any in current_class
            # Process in some order for determinism
            for v in sorted(remaining):
                if v not in current_class:
                    # Check if v is compatible with all in current_class
                    # In the complement (clique problem), we need v to be adjacent to all in current_class
                    # Wait - for coloring in branch-and-bound for MAX CLIQUE:
                    # We color the INDUCED SUBGRAPH on remaining vertices.
                    # The coloring gives an upper bound on clique size.
                    # Two vertices get the same color if they are NOT adjacent in the original graph.
                    # So same color = incompatible.
                    
                    # v can join current_class if v is NOT adjacent to any vertex in current_class
                    # (i.e., v is incompatible with all in current_class)
                    can_join = True
                    for u in current_class:
                        if u in adj[v]:  # u and v are compatible (adjacent in original)
                            can_join = False
                            break
                    if can_join:
                        current_class.add(v)
            
            colors_to_assign = current_class
            for v in colors_to_assign:
                colors[v] = color_num
                remaining.discard(v)
            color_classes.append(colors_to_assign)
        
        return color_num, color_classes
    
    # Better: use the standard branch-and-bound for maximum clique
    # Reference: Tomita's algorithm with greedy coloring
    
    def max_clique_branch_and_bound(adj, n):
        """
        Find maximum clique using branch-and-bound with greedy coloring.
        adj: list of sets (adjacency in the compatibility graph)
        n: number of vertices
        Returns: set of vertices in maximum clique
        """
        best_clique = set()
        best_size = 0
        
        # Order vertices by degree (descending) for better bounds
        deg_order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        # Map from original index to position in ordered list
        pos_in_order = [0] * n
        for pos, orig in enumerate(deg_order):
            pos_in_order[orig] = pos
        
        # Rebuild adjacency in ordered indices
        # ordered_adj[i] = set of j (in ordered indices) such that deg_order[i] and deg_order[j] are adjacent
        ordered_adj = [set() for _ in range(n)]
        for i in range(n):
            orig_i = deg_order[i]
            for j in adj[orig_i]:
                if pos_in_order[j] > i:  # Only add once, for j after i
                    ordered_adj[i].add(pos_in_order[j])
                    ordered_adj[pos_in_order[j]].add(i)
        
        # Actually, let's keep it simpler: work with original indices but use ordering for branching
        
        # Greedy coloring function for a set of candidate vertices
        # Returns: list of color classes (each is a set), and max color
        def greedy_color(candidates):
            """
            Greedy coloring of the induced subgraph on candidates.
            In the context of max clique, we color the COMPATIBILITY graph.
            Two vertices get the same color if they are NOT adjacent (incompatible).
            The max color number is an upper bound on the clique size within candidates.
            """
            if not candidates:
                return [], 0
            
            colors = {}
            color_classes = []
            remaining = set(candidates)
            color_num = 0
            
            # Order candidates by degree within candidates (descending) for better coloring
            cand_list = sorted(remaining, key=lambda x: -len(adj[x] & remaining))
            
            for v in cand_list:
                if v not in remaining:
                    continue
                # Find the smallest color class where v can be added
                # v can be added to color class k if v is NOT adjacent to any vertex in that class
                placed = False
                for k, cls in enumerate(color_classes):
                    can_add = True
                    for u in cls:
                        if u in adj[v]:  # u and v are compatible, so they can't share a color
                            can_add = False
                            break
                    if can_add:
                        cls.add(v)
                        colors[v] = k
                        placed = True
                        break
                if not placed:
                    color_num += 1
                    color_classes.append({v})
                    colors[v] = color_num - 1
                remaining.discard(v)
            
            return color_classes, color_num
        
        # Branch and bound
        # We maintain a list P of candidate vertices (in order)
        # At each step, we color P and use the max color as upper bound
        
        # Use a stack-based approach
        # State: (candidates_list, current_clique)
        # candidates_list is ordered by the coloring order
        
        def search(P, clique):
            nonlocal best_clique, best_size
            
            if time.time() > deadline - 5.0:
                return
            
            if not P:
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = set(clique)
                return
            
            # Color P to get upper bound
            color_classes, max_color = greedy_color(P)
            
            # Upper bound: len(clique) + max_color
            if len(clique) + max_color <= best_size:
                return
            
            # Branch: process P in reverse order of coloring (last color class first)
            # For each vertex in P (in reverse color order), branch on including it
            
            # Reorder P by color (descending color number first)
            # Actually, standard approach: iterate through P in the order given by coloring
            # Process vertices in reverse order of their color assignment
            
            # Get the ordering from coloring
            # color_classes[0] has color 0, color_classes[max_color-1] has color max_color-1
            # We process from highest color to lowest
            
            # Build ordered list: vertices sorted by color descending, then by degree
            v_colors = {}
            for k, cls in enumerate(color_classes):
                for v in cls:
                    v_colors[v] = k
            
            # Process in order: highest color first
            ordered_P = sorted(P, key=lambda x: (-v_colors[x], -len(adj[x] & set(P))))
            
            for v in ordered_P:
                if time.time() > deadline - 5.0:
                    return
                
                # Upper bound check: clique size + number of vertices with color >= v's color
                # In the coloring, vertices with color >= v_colors[v] are at most (max_color - v_colors[v])
                # But we need a tighter bound: the number of color classes from v_colors[v] to max_color-1
                ub = len(clique) + (max_color - v_colors[v])
                if ub <= best_size:
                    break  # No more vertices can improve
                
                # Branch: include v
                new_clique = clique | {v}
                # New candidates: vertices in P that are compatible with v and come after v in order
                # Actually: vertices in P that are adjacent to v
                new_P = [u for u in ordered_P if u in adj[v]]
                # But we should only consider vertices that come after v in the ordering?
                # Standard: new_P = {u in P : u is adjacent to v}
                # The ordering helps with the bound
                
                search(new_P, new_clique)
                
                # Remove v from consideration (it's now excluded)
                # The next iteration will handle the remaining vertices
        
        # Initial call: all vertices as candidates, empty clique
        initial_P = list(range(n))
        search(initial_P, set())
        
        return best_clique
    
    # Run the branch-and-bound
    if time.time() < deadline - 20.0:
        best_clique = max_clique_branch_and_bound(compat, N_sup)
    else:
        best_clique = set()
    
    # If B&B didn't find anything or ran out of time, use greedy as fallback
    if not best_clique:
        # Greedy fallback
        rng = np.random.default_rng(42)
        best_size = 0
        best_iset = None
        
        for r in range(50):
            if time.time() > deadline - 5.0:
                break
            order = rng.permutation(N_sup)
            selected = set()
            for idx in order:
                if all(idx not in compat.get(s, set()) or s not in selected for s in selected):
                    # Check: idx is compatible with all in selected
                    # i.e., for all s in selected, s is in compat[idx]
                    compatible_with_all = True
                    for s in selected:
                        if s not in compat[idx]:
                            compatible_with_all = False
                            break
                    if compatible_with_all:
                        selected.add(idx)
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_iset = selected
        
        if best_iset is None:
            best_iset = set()
        else:
            best_clique = best_iset
    
    # Generate all weight-4 vectors for the selected supports
    vectors = []
    for i in best_clique:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
    
    if not vectors:
        # Fallback: return a small valid set
        # Just use e_1 + e_2 with both signs
        vectors = [
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
            np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
            np.array([1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
            np.array([-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        ]
    
    return np.array(vectors, dtype=np.int8)

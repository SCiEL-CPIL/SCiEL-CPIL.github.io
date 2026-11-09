"""Rewritten program for strong edge colouring with optimized local search and simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Target number of edges is roughly n*d/2, but the generator logic in the original
        # tries 4*n*d times. We replicate the exact logic to ensure same graph.
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e not in edges and degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def build_conflict_fast(edges, adj):
        """
        Build conflict graph.
        Two edges conflict if:
        1. They share a vertex.
        2. They are disjoint but joined by an edge (induced matching violation).
        """
        m = len(edges)
        # Map vertex to list of edge indices incident to it
        vert_to_edges = [[] for _ in range(max(adj.keys()) + 1 if adj else 1)]
        for i, (u, v) in enumerate(edges):
            vert_to_edges[u].append(i)
            vert_to_edges[v].append(i)
            
        conflict = [set() for _ in range(m)]
        
        # 1. Add conflicts for shared vertices
        for u in range(len(vert_to_edges)):
            e_list = vert_to_edges[u]
            for i in range(len(e_list)):
                for j in range(i + 1, len(e_list)):
                    ei = e_list[i]
                    ej = e_list[j]
                    conflict[ei].add(ej)
                    conflict[ej].add(ei)
        
        # 2. Add conflicts for induced matching violations
        # For each edge (u, v), check all edges (w, x) that are disjoint from (u, v)
        # and see if any of u-w, u-x, v-w, v-x is an edge.
        # Optimization: Iterate over edges, and for each, check against neighbors of its endpoints.
        edge_set = set(edges)
        
        for i, (u, v) in enumerate(edges):
            # Neighbors of u and v in the graph
            # Note: adj contains neighbors in the original graph
            neighbors_u = adj.get(u, set())
            neighbors_v = adj.get(v, set())
            
            # Any edge (w, x) that has w in neighbors_u or x in neighbors_u etc.
            # We need to check edges disjoint from (u,v).
            # Let's iterate through all edges j != i.
            # To speed up, we can just check the 4 potential connecting edges.
            # If (u, w) is an edge, and (v, x) is an edge, and (u,v) and (w,x) are disjoint, they conflict.
            # Actually, simpler:
            # For edge i=(u,v), it conflicts with j=(w,x) if:
            # - they share vertex (handled above)
            # - OR (u,w) in E, (u,x) in E, (v,w) in E, or (v,x) in E.
            
            # We can check this by looking at edges incident to neighbors.
            # But a simpler O(m^2) with bitsets or sets might be fast enough for m<300.
            # Let's stick to the set logic but optimize the loop.
            
            for j in range(i + 1, m):
                if j in conflict[i]:
                    continue # Already conflict via shared vertex
                
                w, x = edges[j]
                # Check induced matching condition
                if w in neighbors_u or x in neighbors_u or w in neighbors_v or x in neighbors_v:
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        return conflict

    def solve_instance(n, d, seed, time_limit=30):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        conflict = build_conflict_fast(edges, adj)
        
        # Initial Greedy Colouring
        assign = [-1] * m
        color_counts = {}
        
        for i in range(m):
            used_colors = set(assign[j] for j in conflict[i] if assign[j] != -1)
            c = 0
            while c in used_colors:
                c += 1
            assign[i] = c
            color_counts[c] = color_counts.get(c, 0) + 1
        
        best_assign = assign[:]
        best_count = len(color_counts)
        
        # Simulated Annealing
        rng = random.Random(seed + 99)
        start_time = time.time()
        
        # Temperature parameters
        T_start = 10.0
        T_end = 0.01
        alpha = 0.999 # Cooling rate
        
        current_count = best_count
        current_assign = best_assign[:]
        current_counts = dict(color_counts)
        
        T = T_start
        
        while time.time() - start_time < time_limit:
            # Pick random edge
            i = rng.randrange(m)
            old_c = current_assign[i]
            
            # Get conflicting colors
            used_colors = set(current_assign[j] for j in conflict[i])
            
            # Candidate colors: 
            # 1. Existing colors not in used_colors
            # 2. New color (old_count + 1) - usually bad, but allows escape?
            #    Actually, for minimization, we only move to existing colors or keep.
            #    Moving to a new color increases count, so we only do it if we are stuck?
            #    Standard SA on this problem usually restricts to valid moves (no conflict).
            
            active_colors = [c for c, cnt in current_counts.items() if cnt > 0]
            if not active_colors:
                break
            
            # Filter valid colors
            valid_colors = [c for c in active_colors if c not in used_colors]
            
            if not valid_colors:
                # No valid move to existing color. 
                # We could try to swap, but that's complex. 
                # For now, skip or try to introduce new color if T is high? 
                # Introducing new color increases energy. 
                # Let's just cool down.
                T *= alpha
                continue
            
            # Choose a target color
            # Bias towards colors that are "full" or just random?
            # Random is fine.
            new_c = rng.choice(valid_colors)
            
            if new_c == old_c:
                continue
            
            # Calculate Delta Energy
            # Energy = Number of distinct colors
            # If we move i from old_c to new_c:
            # - If current_counts[old_c] == 1, old_c becomes empty -> Count -1
            # - If current_counts[new_c] == 0 (impossible as new_c is active), Count +1
            # - Otherwise Count unchanged.
            
            delta_e = 0
            if current_counts[old_c] == 1:
                delta_e = -1
            # new_c is always active (count > 0), so no +1 from new_c becoming active.
            # Wait, if new_c was active, it stays active.
            # So delta is -1 if we empty old_c, else 0.
            
            # Acceptance
            if delta_e < 0:
                accept = True
            elif delta_e == 0:
                # Neutral move. Accept with prob 0.5 to explore?
                accept = rng.random() < 0.5
            else:
                # delta_e > 0 is impossible here because we only move to active colors.
                # Unless we allowed new colors. We don't.
                accept = False
            
            if accept:
                # Perform move
                current_counts[old_c] -= 1
                if current_counts[old_c] == 0:
                    del current_counts[old_c]
                
                current_counts[new_c] += 1
                current_assign[i] = new_c
                
                new_count = len(current_counts)
                
                if new_count < best_count:
                    best_count = new_count
                    best_assign = current_assign[:]
                    
            # Cool down
            T *= alpha
            if T < T_end:
                T = T_end
        
        # Final verification
        valid = True
        for i in range(m):
            c = best_assign[i]
            for j in conflict[i]:
                if best_assign[j] == c:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
             # Fallback to greedy
            assign = [0] * m
            for i in range(m):
                used = set(assign[j] for j in conflict[i])
                c = 0
                while c in used:
                    c += 1
                assign[i] = c
            best_assign = assign
            
        return best_assign
    
    out = []
    deadline = time.time() + 115 # 5s safety margin
    
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        remaining = deadline - time.time()
        if remaining < 10:
            # Fallback: Fast Greedy
            n, d, seed = spec
            edges = instance(n, d, seed)
            m = len(edges)
            if m == 0:
                out.append([])
                continue
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            conflict = build_conflict_fast(edges, adj)
            
            assign = [0] * m
            for i in range(m):
                used = set(assign[j] for j in conflict[i])
                c = 0
                while c in used:
                    c += 1
                assign[i] = c
            out.append(assign)
        else:
            # Allocate remaining time, reserving 5s for next
            t_limit = min(remaining - 5, 35)
            out.append(solve_instance(*spec, time_limit=t_limit))
            
    return out
# EVOLVE-BLOCK-END

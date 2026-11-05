"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Ensure we try enough times to get a connected-ish or sufficiently dense graph
        # The seed uses 4*n*d attempts.
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(n, edges):
        """
        Returns a dictionary mapping edge index to a set of conflicting edge indices.
        Two edges conflict if they share a vertex or if there is an edge connecting
        the two edges (distance 2 in the original graph).
        """
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        # Adjacency list of vertices
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        m = len(edges)
        conflicts = [set() for _ in range(m)]
        
        for i, (u, v) in enumerate(edges):
            # Conflict with edges sharing u or v
            for w in adj[u]:
                if w != v:
                    e2 = (min(u, w), max(u, w))
                    j = edge_to_idx[e2]
                    conflicts[i].add(j)
                    conflicts[j].add(i)
            for w in adj[v]:
                if w != u:
                    e2 = (min(v, w), max(v, w))
                    j = edge_to_idx[e2]
                    conflicts[i].add(j)
                    conflicts[j].add(i)
            
            # Conflict with edges at distance 2
            # For each neighbor w of u, for each neighbor x of w (x != u, x != v), 
            # the edge (w, x) conflicts with (u, v) if it doesn't share a vertex.
            # Actually, the definition is: two edges are joined by an edge.
            # So (u,v) and (w,x) conflict if there is an edge (u,w) or (u,x) or (v,w) or (v,x).
            # We already handled sharing vertices.
            # We need to handle the case where they don't share vertices but are connected.
            # This is covered by: for each neighbor w of u, for each neighbor x of w, 
            # if (w,x) is an edge and x is not v and w is not v... wait.
            
            # Let's re-evaluate.
            # Edge e1=(u,v). Edge e2=(w,x).
            # They conflict if:
            # 1. u==w or u==x or v==w or v==x (share vertex)
            # 2. There is an edge connecting {u,v} and {w,x}.
            #    i.e., (u,w) in E, (u,x) in E, (v,w) in E, or (v,x) in E.
            
            # My previous loop handled 1.
            # For 2, we need to check neighbors of u and v.
            # If (u,w) is an edge, then (u,w) is adjacent to (u,v).
            # Does (u,w) conflict with (u,v)? Yes, they share u.
            # Wait, the condition is "joined by an edge".
            # If e1=(u,v) and e2=(w,x).
            # If (u,w) is an edge, then e1 and e2 are joined by edge (u,w)?
            # No, e2 is (w,x). The edge joining e1 and e2 would be (u,w) if w is in e2 and u is in e1.
            # So if (u,w) is an edge, and w is an endpoint of e2, then they are joined.
            
            # So, for each neighbor w of u:
            #   If (u,w) is an edge, it conflicts with (u,v) by sharing u. (Handled)
            #   But we also need to check if any edge *starting from w* conflicts with (u,v) without sharing a vertex with (u,v).
            #   Let e2 = (w,x). If w is not u or v, and x is not u or v, then e1 and e2 don't share vertices.
            #   Do they conflict? They conflict if there is an edge between {u,v} and {w,x}.
            #   We know (u,w) is an edge. So yes, they are joined by (u,w).
            #   So, for each neighbor w of u, for each neighbor x of w:
            #      If (w,x) is an edge, and {w,x} is disjoint from {u,v}, then (u,v) and (w,x) conflict.
            
            # Let's implement this carefully to avoid duplicates and self-conflicts.
            for w in adj[u]:
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    e2 = (min(w, x), max(w, x))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        # Check if they share a vertex
                        if w != u and w != v and x != u and x != v:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
            for w in adj[v]:
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    e2 = (min(w, x), max(w, x))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if w != u and w != v and x != u and x != v:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
        
        return conflicts

    def greedy_color(conflicts, m):
        """
        Greedy coloring of the conflict graph.
        Returns a list of colors for each edge index.
        """
        # Order vertices by degree in conflict graph (descending)
        degrees = [len(c) for c in conflicts]
        order = sorted(range(m), key=lambda i: degrees[i], reverse=True)
        
        colors = [-1] * m
        used_colors = set()
        
        for i in order:
            # Find available colors
            taken = set()
            for j in conflicts[i]:
                if colors[j] != -1:
                    taken.add(colors[j])
            
            c = 0
            while c in taken:
                c += 1
            
            colors[i] = c
            used_colors.add(c)
        
        # Renumber colors to be 0..k-1
        max_c = max(colors) if colors else 0
        # We don't strictly need to renumber if we just count unique colors, 
        # but for local search, having contiguous indices is good.
        # However, greedy already assigns 0..k-1 if we start from 0.
        # But the order might skip numbers? No, we always pick the smallest available.
        # So colors will be a subset of 0..max_c.
        
        return colors

    def local_search_reduce(colors, conflicts, m):
        """
        Try to reduce the number of colors by removing the highest color class.
        Returns new colors and success flag.
        """
        max_c = max(colors) if colors else 0
        if max_c == 0:
            return colors, False
        
        # Get edges with max_c
        edges_to_move = [i for i in range(m) if colors[i] == max_c]
        
        # Try to move each edge to a lower color
        # We need to check if we can assign a color < max_c to each edge
        # such that no conflicts arise.
        
        # This is a subgraph coloring problem for the subgraph induced by edges_to_move
        # with the constraint that they can only use colors < max_c.
        # Since max_c is the highest, we have max_c available colors (0 to max_c-1).
        
        # Simple approach: try to move them one by one using a greedy strategy
        # or backtracking. Given small size, backtracking is feasible.
        
        # Let's try a simple greedy move:
        # For each edge in edges_to_move, try to assign the smallest color < max_c
        # that is not used by its neighbors.
        
        new_colors = colors[:]
        success = True
        
        # Order edges_to_move by degree in the conflict graph restricted to edges_to_move?
        # Or just try in original order.
        
        for i in edges_to_move:
            taken = set()
            for j in conflicts[i]:
                if new_colors[j] != -1:
                    taken.add(new_colors[j])
            
            # Find smallest c < max_c not in taken
            c = 0
            while c < max_c and c in taken:
                c += 1
            
            if c == max_c:
                # Cannot move this edge
                success = False
                break
            
            new_colors[i] = c
        
        if success:
            return new_colors, True
        else:
            return colors, False

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        conflicts = build_conflict_graph(n, edges)
        
        # Initial greedy coloring
        colors = greedy_color(conflicts, m)
        
        # Local search to reduce colors
        # We try to remove colors from the top down.
        # We might need multiple passes if removing one color makes another removable?
        # Usually, removing the highest color is the most direct.
        
        # To be more robust, we can try to improve the greedy ordering or use randomization.
        # Let's add a few random restarts for the greedy step if time permits.
        
        best_colors = colors
        best_k = len(set(best_colors))
        
        # Try a few random orderings for greedy
        # This is a structural change: diversification of the initial solution.
        
        for _ in range(5): # 5 restarts
            if time.time() > deadline:
                break
            # Shuffle the order slightly? Or use a different heuristic?
            # Let's try sorting by random priority + degree
            rng = random.Random(seed + _ + 1)
            priorities = [rng.random() * 0.1 + len(conflicts[i]) for i in range(m)]
            order = sorted(range(m), key=lambda i: priorities[i], reverse=True)
            
            c_colors = [-1] * m
            for i in order:
                taken = set()
                for j in conflicts[i]:
                    if c_colors[j] != -1:
                        taken.add(c_colors[j])
                c = 0
                while c in taken:
                    c += 1
                c_colors[i] = c
            
            # Local search on this new coloring
            curr_colors = c_colors
            curr_k = len(set(curr_colors))
            
            # Try to reduce
            changed = True
            while changed:
                if time.time() > deadline:
                    break
                max_c = max(curr_colors) if curr_colors else 0
                if max_c == 0:
                    break
                new_colors, success = local_search_reduce(curr_colors, conflicts, m)
                if success:
                    curr_colors = new_colors
                    curr_k -= 1
                else:
                    changed = False
            
            if curr_k < best_k:
                best_k = curr_k
                best_colors = curr_colors
        
        return best_colors

    out = []
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    
    for n, d, seed in specs:
        if time.time() > deadline:
            # Fallback to simple greedy if we are out of time
            edges = instance(n, d, seed)
            m = len(edges)
            if m == 0:
                out.append([])
                continue
            conflicts = build_conflict_graph(n, edges)
            colors = greedy_color(conflicts, m)
            out.append(colors)
        else:
            colors = solve_instance(n, d, seed)
            out.append(colors)
            
    return out
# EVOLVE-BLOCK-END

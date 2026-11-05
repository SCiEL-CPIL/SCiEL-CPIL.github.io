"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges, adj):
        """
        Build conflict graph for induced matching.
        Two edges conflict if:
        1. They share a vertex.
        2. They are joined by an edge (i.e., there is an edge in G connecting a vertex of e1 to a vertex of e2).
        """
        n_edges = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(n_edges)]
        
        # For each edge, find all conflicting edges
        for i, (u, v) in enumerate(edges):
            # 1. Edges sharing u or v
            for nb in adj[u]:
                e = (min(u, nb), max(u, nb))
                if e in edge_to_idx:
                    j = edge_to_idx[e]
                    if j != i:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
            for nb in adj[v]:
                e = (min(v, nb), max(v, nb))
                if e in edge_to_idx:
                    j = edge_to_idx[e]
                    if j != i:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
            
            # 2. Edges joined by an edge: (u, w) or (v, w) where w is neighbor of u or v
            # Actually, the condition is: for edge (a,b) and (c,d), they conflict if 
            # any of a,c; a,d; b,c; b,d is an edge in G, OR they share a vertex.
            # We already handled sharing vertices.
            # Now handle "joined by an edge".
            # For edge (u,v), it conflicts with (c,d) if (u,c) in E, (u,d) in E, (v,c) in E, or (v,d) in E.
            
            # Iterate over all edges again? No, that's O(M^2).
            # Better: for each neighbor w of u, any edge (w, x) conflicts with (u,v) if x is not v and x is not u?
            # Wait, if (w,x) is an edge, and (u,w) is an edge, then (u,v) and (w,x) are joined by edge (u,w).
            # So (u,v) conflicts with (w,x) for all x such that (w,x) is an edge and (w,x) != (u,v).
            
            for w in adj[u]:
                # Any edge starting from w conflicts with (u,v) because (u,w) connects them.
                # Except (u,v) itself (already handled by shared vertex, but let's be careful).
                for x in adj[w]:
                    e2 = (min(w, x), max(w, x))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
                            
            for w in adj[v]:
                # Any edge starting from w conflicts with (u,v) because (v,w) connects them.
                for x in adj[w]:
                    e2 = (min(w, x), max(w, x))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
                            
        return conflicts

    def max_induced_matching(conflicts, active_edges):
        """
        Find a maximum induced matching in the subgraph induced by active_edges.
        We model this as a maximum matching in a bipartite graph?
        No, induced matching is NP-hard in general, but for bounded degree it's small.
        However, we can use a greedy heuristic that is "local optimum" or use a specific algorithm.
        
        Actually, the prompt suggests: "iteratively finding a maximum induced matching in the remaining conflict graph using a bipartite matching reduction on the line graph's square".
        
        Let's re-read the inspiration: "bipartite matching reduction on the line graph's square".
        
        Wait, an induced matching in G is a matching in G such that no two edges are joined by another edge.
        This is equivalent to a matching in the "induced matching conflict graph" (let's call it H) where vertices are edges of G, and edges in H connect conflicting edges.
        We want to find a set of vertices in H that are independent (no edges between them) and form a matching in G.
        But "form a matching in G" is already implied by "no two edges share a vertex".
        
        So the problem is: Find a Maximum Independent Set in the conflict graph H, restricted to vertices that don't share vertices in G?
        No, the conflict graph H already encodes both "share vertex" and "joined by edge".
        So a set of edges is an induced matching in G iff they form an Independent Set in H.
        
        Finding Maximum Independent Set is NP-hard.
        However, H is the square of the line graph? Or related?
        
        The inspiration mentions "bipartite matching reduction".
        If H were bipartite, we could use Konig's theorem. But H is not necessarily bipartite.
        
        Let's try a different approach:
        Since the graphs are small (n=40, 60, 80) and bounded degree, we can use a greedy algorithm that picks the "best" edge, removes it and all its conflicts, and repeats.
        To make it "Maximum", we can try all possible first edges? No, too slow.
        
        Let's use a heuristic that approximates maximum induced matching:
        1. Sort edges by some priority (e.g., degree in conflict graph, ascending).
        2. Pick the first available edge.
        3. Remove it and all its neighbors from the active set.
        4. Repeat.
        
        This is the standard greedy for Independent Set.
        To improve, we can use "Iterated Local Search" or "Simulated Annealing" on the set of chosen edges.
        
        Given the time limit, let's try a robust greedy with multiple random restarts and local improvement.
        """
        if not active_edges:
            return []
        
        best_match = []
        rng = random.Random(42)
        
        # Try multiple random orderings
        for _ in range(20):
            # Randomize order slightly or use a specific heuristic
            # Heuristic: pick edge with minimum degree in the *current* conflict subgraph
            # This is similar to "Minimum Degree First"
            
            current_active = set(active_edges)
            match = []
            
            while current_active:
                # Find edge with minimum conflict count within current_active
                # This tends to pick "isolated" edges which allow more future edges
                min_conf = float('inf')
                best_edge = None
                
                # Optimization: if time is tight, just pick random
                if len(current_active) > 100:
                    # Random pick for speed
                    best_edge = rng.choice(list(current_active))
                else:
                    for e in current_active:
                        # Count conflicts in current_active
                        conf_count = len(conflicts[e] & current_active)
                        if conf_count < min_conf:
                            min_conf = conf_count
                            best_edge = e
                            if min_conf == 0:
                                break
                
                if best_edge is None:
                    break
                    
                match.append(best_edge)
                # Remove best_edge and all its conflicts
                to_remove = {best_edge} | (conflicts[best_edge] & current_active)
                current_active -= to_remove
            
            if len(match) > len(best_match):
                best_match = match
                
        return best_match

    def color_instance(n, d, seed):
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflicts = build_conflict_graph(edges, adj)
        n_edges = len(edges)
        
        colour = [-1] * n_edges
        active_edges = set(range(n_edges))
        current_color = 0
        
        while active_edges:
            # Check time
            if time.time() > deadline - 10:
                # Fallback: simple greedy for remaining
                for e in sorted(active_edges):
                    colour[e] = current_color
                    # Remove conflicts
                    to_remove = {e} | (conflicts[e] & active_edges)
                    active_edges -= to_remove
                break
            
            # Find maximum induced matching in active_edges
            matching = max_induced_matching(conflicts, active_edges)
            
            if not matching:
                # Should not happen if active_edges is not empty, but safety
                break
                
            # Assign color
            for e in matching:
                colour[e] = current_color
                
            # Remove matching and all their conflicts from active
            to_remove = set()
            for e in matching:
                to_remove.add(e)
                to_remove.update(conflicts[e])
            active_edges -= to_remove
            
            current_color += 1
            
            # Safety: if we are using too many colors, stop and use greedy
            if current_color > 100:
                 for e in active_edges:
                     colour[e] = current_color
                 break

        # Final check: ensure all edges colored
        for i in range(n_edges):
            if colour[i] == -1:
                colour[i] = current_color
                
        return colour

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 20:
            # Not enough time, use simple greedy for remaining
            edges = instance(*spec)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        else:
            out.append(color_instance(*spec))
    
    return out
# EVOLVE-BLOCK-END

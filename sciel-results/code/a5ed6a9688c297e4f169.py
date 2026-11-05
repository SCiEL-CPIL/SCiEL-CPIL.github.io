"""Strong edge colourings via Local Search on Conflict Graph."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        # Generate random graph with max degree d
        # The seed program uses a specific generation method, we must replicate it exactly.
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges, n):
        """
        Build a conflict graph where nodes are edge indices.
        Two nodes are connected if the corresponding edges in the original graph
        share a vertex or are joined by an edge (i.e., they cannot have the same color).
        """
        m = len(edges)
        # Map vertex to list of incident edge indices
        vertex_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_to_edges[u].append(i)
            vertex_to_edges[v].append(i)
        
        # Build adjacency list for conflict graph
        # conflict_adj[i] is a set of edge indices j that conflict with i
        conflict_adj = [set() for _ in range(m)]
        
        # 1. Edges sharing a vertex conflict
        for u in range(n):
            inc_edges = vertex_to_edges[u]
            for i in range(len(inc_edges)):
                for j in range(i + 1, len(inc_edges)):
                    e_i = inc_edges[i]
                    e_j = inc_edges[j]
                    conflict_adj[e_i].add(e_j)
                    conflict_adj[e_j].add(e_i)
        
        # 2. Edges joined by an edge conflict
        # If edge e1=(u,v) and edge e2=(w,x) have an edge (v,w) or (u,x) etc in the graph, they conflict.
        # Actually, "joined by an edge" means there is a path of length 1 between an endpoint of e1 and an endpoint of e2.
        # So for edge (u,v), any edge incident to a neighbor of u (other than v) or a neighbor of v (other than u) conflicts.
        # Let's iterate over edges and their neighbors.
        
        # First, build vertex adjacency for original graph
        vertex_adj = [set() for _ in range(n)]
        for u, v in edges:
            vertex_adj[u].add(v)
            vertex_adj[v].add(u)
        
        for i, (u, v) in enumerate(edges):
            # Neighbors of u (excluding v)
            for nu in vertex_adj[u]:
                if nu == v:
                    continue
                # Any edge incident to nu conflicts with edge i
                for j in vertex_to_edges[nu]:
                    if j != i:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)
            # Neighbors of v (excluding u)
            for nv in vertex_adj[v]:
                if nv == u:
                    continue
                for j in vertex_to_edges[nv]:
                    if j != i:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)
                        
        return conflict_adj

    def solve_instance(edges, n, d, seed, time_limit):
        m = len(edges)
        if m == 0:
            return []
        
        conflict_adj = build_conflict_graph(edges, n)
        
        # Greedy initialization
        # Order edges by degree in conflict graph (highest first) to potentially reduce colors?
        # Or just random. Let's try a few random initializations.
        
        best_colours = None
        best_count = float('inf')
        
        rng = random.Random(seed)
        
        max_restarts = 50
        restart_count = 0
        
        while restart_count < max_restarts and time.time() < time_limit:
            # 1. Greedy Color
            # Random order
            order = list(range(m))
            rng.shuffle(order)
            
            colours = [0] * m
            # For each edge, find the smallest available color
            # We maintain a set of used colors? No, just check conflicts.
            # To speed up, we can track which colors are "blocked" for each edge.
            # But for small m, direct check is fine.
            
            for idx in order:
                # Find smallest color c such that no neighbor of idx has color c
                blocked = set()
                for nb in conflict_adj[idx]:
                    if colours[nb] != -1: # Only if already colored? In greedy, we process in order, so neighbors might be uncolored.
                        blocked.add(colours[nb])
                
                c = 0
                while c in blocked:
                    c += 1
                colours[idx] = c
            
            current_count = max(colours) + 1 if m > 0 else 0
            
            # 2. Local Search to minimize colors
            # We try to eliminate colors one by one, starting from the highest.
            # Or we can use a general hill climbing.
            # Let's try "Color Elimination" heuristic.
            
            # Sort colors by frequency? Or just try to remove highest color.
            # A more robust method: Simulated Annealing or Random Walk.
            # Given the small size, let's try a simple iterative improvement:
            # While we can reduce the number of colors:
            #   For each color c (from high to low):
            #     Try to reassign all edges with color c to other colors.
            #     If possible, remove color c.
            
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                # Get current max color
                max_c = max(colours) if m > 0 else -1
                
                # Try to remove colors from max_c down to 1
                for c_remove in range(max_c, 0, -1):
                    if time.time() > time_limit:
                        break
                    
                    # Check if color c_remove is actually used
                    edges_with_c = [i for i in range(m) if colours[i] == c_remove]
                    if not edges_with_c:
                        continue
                    
                    # Try to reassign these edges
                    # We need to find a valid assignment for these edges using colors < c_remove
                    # This is a small subgraph coloring problem.
                    # Since we are removing one color, we can try to reassign them one by one.
                    # If we fail, we might need to backtrack or swap.
                    # Simple approach: Try to reassign each edge in edges_with_c to a random valid color < c_remove.
                    # If all succeed, we removed a color.
                    
                    # To make it more effective, we can try to "push" edges to lower colors.
                    # Let's try a randomized reassignment.
                    
                    # Save old state
                    old_colours = colours[:]
                    
                    # Try to reassign
                    success = True
                    # Shuffle to avoid bias
                    rng.shuffle(edges_with_c)
                    
                    for idx in edges_with_c:
                        # Find available colors < c_remove
                        available = []
                        for c_new in range(c_remove):
                            # Check if c_new is valid for idx
                            valid = True
                            for nb in conflict_adj[idx]:
                                if colours[nb] == c_new:
                                    valid = False
                                    break
                            if valid:
                                available.append(c_new)
                        
                        if not available:
                            success = False
                            break
                        else:
                            # Pick a random available color
                            colours[idx] = rng.choice(available)
                    
                    if success:
                        improved = True
                        # We removed color c_remove.
                        # We can break and restart the loop to check for more removals
                        break
                    
                    # If failed, restore state
                    colours = old_colours
            
            # Count colors
            current_count = max(colours) + 1 if m > 0 else 0
            
            if current_count < best_count:
                best_count = current_count
                best_colours = colours[:]
            
            restart_count += 1
            
            # Optional: Try a more aggressive local search if stuck?
            # For now, multiple restarts should suffice for these small graphs.
            
        if best_colours is None:
            # Fallback
            best_colours = [0] * m
            for i in range(m):
                blocked = set()
                for nb in conflict_adj[i]:
                    if best_colours[nb] != -1:
                        blocked.add(best_colours[nb])
                c = 0
                while c in blocked:
                    c += 1
                best_colours[i] = c
        
        return best_colours

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        
        # Allocate time per instance
        # Total 140s. 3 instances. ~45s each.
        inst_start = time.time()
        inst_limit = inst_start + 45.0
        
        colours = solve_instance(edges, n, d, seed, inst_limit)
        results.append(colours)
        
        # Check global deadline
        if time.time() > deadline:
            # If we run out of time, we just return what we have.
            # The loop above handles individual limits.
            pass

    return results
# EVOLVE-BLOCK-END

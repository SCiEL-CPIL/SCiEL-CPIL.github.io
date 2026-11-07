"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    def find_bicolored_cycle(assignment, num_colors):
        """
        Returns a list of vertices forming a bicolored cycle, or None if no such cycle exists.
        We check all pairs of colors.
        """
        for c1 in range(num_colors):
            for c2 in range(c1 + 1, num_colors):
                # Build induced subgraph on vertices with color c1 or c2
                # Use DFS to find a cycle
                visited = [False] * n
                # We only care about vertices with color c1 or c2
                active_vertices = [i for i in range(n) if assignment[i] in (c1, c2)]
                if not active_vertices:
                    continue
                
                # Map active vertices to indices 0..k-1 for easier DFS? 
                # Or just use the original indices and a visited array.
                # Let's use a standard DFS cycle detection on the induced subgraph.
                
                # To make it fast, we can just iterate edges in the induced subgraph.
                # A graph is a forest iff E = V - C (components).
                # Or simply: if we find a back edge in DFS, it's a cycle.
                
                parent = [-1] * n
                color_state = [0] * n # 0: unvisited, 1: visiting, 2: visited
                
                def dfs(u):
                    color_state[u] = 1
                    for v in adj[u]:
                        if assignment[v] not in (c1, c2):
                            continue
                        if color_state[v] == 0:
                            parent[v] = u
                            cycle = dfs(v)
                            if cycle:
                                return cycle
                        elif color_state[v] == 1:
                            # Found a back edge u-v where v is in current stack
                            # Reconstruct cycle
                            cycle = [u, v]
                            curr = u
                            while curr != v:
                                curr = parent[curr]
                                cycle.append(curr)
                                if curr == v:
                                    break
                            return cycle
                    color_state[u] = 2
                    return None

                for v in active_vertices:
                    if color_state[v] == 0:
                        parent[v] = -1
                        cycle = dfs(v)
                        if cycle:
                            return cycle
        return None

    def local_search(num_colors, max_iters=100000):
        # Initialize random coloring
        assignment = [random.randint(0, num_colors - 1) for _ in range(n)]
        
        for _ in range(max_iters):
            cycle = find_bicolored_cycle(assignment, num_colors)
            if cycle is None:
                # Check proper coloring constraint: adjacent vertices must differ
                # The cycle detection assumes we only look at edges between c1 and c2.
                # But we also need to ensure no edge connects same color.
                # If there is an edge (u,v) with assignment[u] == assignment[v], it's a "cycle" of length 2?
                # No, the problem says "Adjacent vertices must differ".
                # So we must also check for monochromatic edges.
                valid = True
                for u, v in edges:
                    if assignment[u] == assignment[v]:
                        valid = False
                        # Change v
                        # Pick a random different color
                        other_colors = [c for c in range(num_colors) if c != assignment[u]]
                        assignment[v] = random.choice(other_colors)
                        break
                if valid:
                    return assignment
                continue
            
            # We found a bicolored cycle.
            # Pick a random vertex in the cycle and change its color.
            # To break the cycle, we should change it to a color NOT in {c1, c2}.
            # But we need to know which colors c1, c2 are.
            # The cycle vertices all have colors from some pair {c1, c2}.
            # Let's identify the pair.
            cycle_colors = set(assignment[v] for v in cycle)
            # It should be exactly 2 colors.
            if len(cycle_colors) == 2:
                c1, c2 = tuple(cycle_colors)
                # Pick a vertex to recolor
                v = random.choice(cycle)
                # Choose a new color not in {c1, c2} if possible
                available = [c for c in range(num_colors) if c not in (c1, c2)]
                if available:
                    assignment[v] = random.choice(available)
                else:
                    # If num_colors == 2, we can't escape the pair.
                    # But we are trying num_colors >= 3 usually.
                    # If num_colors == 2, we just flip?
                    assignment[v] = 1 - assignment[v] if num_colors == 2 else random.randint(0, num_colors-1)
            else:
                # Should not happen for a bicolored cycle
                v = random.choice(cycle)
                assignment[v] = random.randint(0, num_colors - 1)
        
        return None

    # Try 3 colors first
    for nc in [3, 4, 5]:
        # Run multiple restarts
        for restart in range(50):
            sol = local_search(nc, max_iters=5000)
            if sol is not None:
                # Final verification
                valid = True
                for u, v in edges:
                    if sol[u] == sol[v]:
                        valid = False
                        break
                if valid:
                    # Verify no bicolored cycles
                    if find_bicolored_cycle(sol, nc) is None:
                        return sol
    
    # Fallback: Use 4 colors with a simple pattern if local search failed?
    # Or just return a valid 4-coloring if we can find one quickly.
    # Given the graph is 4-regular, 4 colors should be easy.
    # Let's try a greedy assignment with 4 colors.
    assignment = [-1] * n
    # Order vertices by degree (all 4) or just index
    for i in range(n):
        # Pick a color not used by neighbors
        used = set(assignment[nb] for nb in adj[i] if assignment[nb] != -1)
        for c in range(4):
            if c not in used:
                assignment[i] = c
                break
        else:
            # If no color found (shouldn't happen with 4 colors for 4-reg graph? 
            # Actually, 4-reg graph can require 4 colors, but greedy might fail if order is bad.
            # But with 4 colors, and max degree 4, greedy always works?
            # No, greedy uses at most Delta+1 colors. So 5 colors guaranteed.
            # With 4 colors, it might fail.
            assignment[i] = 0 # Fallback, might be invalid
            
    # Verify this greedy solution
    if find_bicolored_cycle(assignment, 4) is None and all(assignment[u] != assignment[v] for u, v in edges):
        return assignment
        
    # Absolute fallback
    return list(range(n))
# EVOLVE-BLOCK-END

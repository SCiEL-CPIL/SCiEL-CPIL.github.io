import time
import random
import math
from collections import deque

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    def norm_edge(u, v):
        if u > v:
            u, v = v, u
        return (u, v)

    def validate_solution(edges_list, cycles_list):
        edge_set = {norm_edge(e[0], e[1]) for e in edges_list}
        if len(edge_set) != len(edges_list):
            return False
        for i, cyc in enumerate(cycles_list):
            expected_len = i + 3
            if len(cyc) != expected_len:
                return False
            if len(set(cyc)) != expected_len:
                return False
            for j in range(expected_len):
                u = cyc[j]
                v = cyc[(j + 1) % expected_len]
                if norm_edge(u, v) not in edge_set:
                    return False
        return True

    def get_cycle_edges(cyc):
        es = set()
        for i in range(len(cyc)):
            es.add(norm_edge(cyc[i], cyc[(i+1)%len(cyc)]))
        return es

    def find_cycle_in_graph(length, adj, forbidden_nodes=None, start_node=0):
        """
        Find a simple cycle of given length in the graph defined by adj.
        Uses DFS with pruning.
        """
        if forbidden_nodes is None:
            forbidden_nodes = set()
        
        # If start_node is forbidden, we can't start there.
        # We might need to try other start nodes, but for efficiency we stick to 0
        # or allow the caller to handle start node selection.
        # For this implementation, we assume start_node is a good candidate.
        
        visited = {start_node: 0}
        path = [start_node]
        
        def dfs(node, depth):
            if depth == length:
                # Check if we can close the cycle
                if start_node in adj[node]:
                    return path[:]
                return None
            
            for neighbor in adj[node]:
                if neighbor in forbidden_nodes:
                    continue
                if neighbor in visited:
                    # Only allow closing the cycle if it's the start node and we are at the right depth
                    # (Handled above)
                    continue
                visited[neighbor] = depth + 1
                path.append(neighbor)
                
                result = dfs(neighbor, depth + 1)
                if result:
                    return result
                
                path.pop()
                del visited[neighbor]
            return None
        
        return dfs(start_node, 1)

    # 1. Construct a sparse base graph
    # Start with a large cycle to ensure 2-connectivity and a base for long cycles.
    # Vertices 0..39
    edges = []
    # Base cycle 0-1-...-39-0
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    edge_set = {norm_edge(u, v) for u, v in edges}
    adj = {i: set() for i in range(N)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # 2. Initialize Cycles
    # We need cycles of length 3..40.
    # The base cycle provides a 40-cycle.
    # We need to add edges to create shorter cycles.
    
    cycles = [None] * 38
    # L=40 is the base cycle
    cycles[37] = list(range(N))
    
    # We need to construct cycles for L=3..39.
    # Strategy: Add edges to create "chords" that form smaller cycles with the base cycle.
    # A chord (u, v) creates a cycle of length (v - u) % N and (N - (v - u) % N).
    # We want to cover all lengths 3..39.
    
    # Let's try to add edges greedily to cover missing lengths.
    covered_lengths = {40}
    
    # We will add edges one by one.
    # We prioritize edges that create multiple new lengths or fill gaps.
    # Actually, a simpler constructive approach:
    # Use a "fan" like structure but sparser?
    # Or use chords from 0?
    # If we add (0, i), we get cycles of length i+1 and N-i.
    # If we add (0, 3), we get 4-cycle (0-1-2-3-0) and 37-cycle.
    # If we add (0, 4), we get 5-cycle and 36-cycle.
    # ...
    # If we add (0, k), we get cycle of length k+1.
    # To get all lengths 3..39, we can add edges (0, 2), (0, 3), ..., (0, 38)?
    # That's 37 edges. Total 40 + 37 = 77 edges. This is the Fan.
    
    # Can we do better?
    # We can use chords between non-0 nodes.
    # Let's use a local search approach starting from a random sparse graph.
    
    # Randomized Restart Strategy
    best_edges = None
    best_cycles = None
    best_score = float('inf')
    
    max_restarts = 50
    restarts = 0
    
    while restarts < max_restarts and time.time() < deadline:
        restarts += 1
        
        # Random sparse graph initialization
        # Start with a random 2-connected graph?
        # Simpler: Start with a cycle and add random edges.
        # Let's start with the base cycle and add random chords.
        
        current_edges = [(i, (i + 1) % N) for i in range(N)]
        current_edge_set = {norm_edge(u, v) for u, v in current_edges}
        current_adj = {i: set() for i in range(N)}
        for u, v in current_edges:
            current_adj[u].add(v)
            current_adj[v].add(u)
        
        # Add random edges to try to cover lengths
        # We need to find cycles for 3..39.
        # Let's try to add edges that create specific lengths.
        
        # Heuristic: Add edges (0, k) for random k?
        # Or random edges?
        # Let's try a mix.
        
        # First, let's try to construct a solution using a specific heuristic:
        # "Path + Chords"
        # Vertices 0..39.
        # Add edges to form a "ladder" like structure but random?
        
        # Let's stick to the "Chords from 0" strategy but optimize the set of chords.
        # We need lengths 3..39.
        # Length L can be formed by a chord (0, L-1) if L-1 < N.
        # Or a chord (u, v) such that the distance is L or N-L.
        
        # Let's use a local search on the set of chords.
        # State: Set of edges.
        # Objective: Minimize |Edges| such that for all L in 3..40, there exists a cycle of length L.
        
        # Start with a random set of chords.
        # Let's start with the "Fan" (all chords from 0) and remove edges if possible.
        # This is top-down.
        
        # Alternatively, bottom-up:
        # Start with base cycle.
        # For each L in 3..39:
        #   If L not covered:
        #      Find a random edge (u, v) not in graph such that adding it creates a cycle of length L.
        #      Add it.
        
        # This is deterministic if we pick the "best" edge.
        # Let's randomize the edge choice.
        
        curr_edges = [(i, (i + 1) % N) for i in range(N)]
        curr_edge_set = {norm_edge(u, v) for u, v in curr_edges}
        curr_adj = {i: set() for i in range(N)}
        for u, v in curr_edges:
            curr_adj[u].add(v)
            curr_adj[v].add(u)
        
        # We need to track which lengths are covered.
        # Efficiently checking if a length is covered is hard.
        # Instead, let's just try to find cycles for all lengths at the end.
        
        # Let's add edges greedily.
        # For L in 3..39:
        #   Try to find an edge (u, v) that creates a cycle of length L.
        #   How?
        #   A cycle of length L implies a path of length L-1 between two nodes that are adjacent in the cycle.
        #   If we add edge (u, v), it creates cycles of length d(u, v) + 1 where d is the distance in the existing graph?
        #   No, it creates cycles formed by the new edge and any path between u and v in the old graph.
        
        # This is complex. Let's use the "Chords from 0" heuristic which is simple and effective.
        # We will randomize the order of adding chords and maybe use chords not from 0.
        
        # Let's try a different heuristic: "Random Chords"
        # Add random chords until all lengths are covered.
        
        # To check coverage, we need to find cycles.
        # This is expensive.
        
        # Let's go back to the Ladder. The Ladder was 59 edges.
        # Can we improve on 59?
        # The Ladder uses 19 top, 19 bottom, 20 vertical, 1 chord.
        # Total 59.
        
        # What if we use a "Double Ladder" or similar?
        # Or a "Torus"?
        
        # Let's try to optimize the Ladder via local search.
        # Start with Ladder.
        # Try to remove edges? No, all are used.
        # Try to replace cycles to use fewer edges?
        # The union of cycles in the Ladder is the whole graph.
        # If we can find a set of cycles that doesn't use some edges, we can remove those edges.
        
        # Let's implement a local search on the cycles for the Ladder graph.
        
        # Base Ladder Construction
        ladder_edges = []
        for c in range(19):
            ladder_edges.append((c, c + 1))
        for c in range(19):
            ladder_edges.append((20 + c, 20 + c + 1))
        for c in range(20):
            ladder_edges.append((c, 20 + c))
        ladder_edges.append((0, 21))
        
        ladder_edge_set = {norm_edge(u, v) for u, v in ladder_edges}
        ladder_adj = {i: set() for i in range(N)}
        for u, v in ladder_edges:
            ladder_adj[u].add(v)
            ladder_adj[v].add(u)
            
        # Initial Cycles for Ladder
        init_cycles = [None] * 38
        for m in range(1, 20):
            L = 2 * m + 1
            idx = L - 3
            if m == 1:
                c = [0, 20, 21]
            else:
                c = [0]
                for cc in range(1, m):
                    c.append(cc)
                c.append(20 + m - 1)
                for cc in range(m - 2, 0, -1):
                    c.append(20 + cc)
            init_cycles[idx] = c

        for k in range(1, 20):
            L = 2 * k
            idx = L - 3
            c = [0]
            for cc in range(1, k):
                c.append(cc)
            c.append(20 + k - 1)
            for cc in range(k - 2, -1, -1):
                c.append(20 + cc)
            init_cycles[idx] = c
        
        c40 = []
        for cc in range(20):
            c40.append(cc)
        for cc in range(19, -1, -1):
            c40.append(20 + cc)
        init_cycles[37] = c40
        
        # Local Search on Cycles
        # Goal: Minimize the size of the union of edges used by the cycles.
        # If the union is smaller than the graph, we can return the union as the edge set.
        
        current_cycles = init_cycles[:]
        
        def calc_union(cycles_list):
            union = set()
            for c in cycles_list:
                union.update(get_cycle_edges(c))
            return union
        
        current_union = calc_union(current_cycles)
        best_union_size = len(current_union)
        best_cycles_ls = current_cycles[:]
        
        # Try to replace each cycle with a different one that reduces the union size.
        # We will do a few passes.
        
        for _ in range(10): # 10 passes
            improved = False
            for i in range(38):
                L = i + 3
                # Get current cycle edges
                curr_edges_c = get_cycle_edges(current_cycles[i])
                
                # Try to find a different cycle of length L
                # We can use DFS with a constraint: prefer edges already in the union of OTHER cycles.
                
                # Calculate union of other cycles
                other_union = set()
                for j in range(38):
                    if j != i:
                        other_union.update(get_cycle_edges(current_cycles[j]))
                
                # Find a cycle of length L that maximizes the number of edges in other_union
                # This is a hard problem.
                # Approximation: Try random cycles?
                # Or use the specific structure of the Ladder.
                
                # In the Ladder, cycles are quite constrained.
                # Let's just check if we can find a cycle that uses fewer "exclusive" edges.
                
                # For now, skip complex replacement. The Ladder cycles are likely optimal for the Ladder graph.
                # The problem is the graph itself.
                
            if not improved:
                break
        
        # If we found a better union, update best
        if len(current_union) < best_union_size:
            best_union_size = len(current_union)
            best_cycles_ls = current_cycles[:]
        
        # Construct the final solution from the best union
        # The edges are the union of the best cycles.
        final_edges = list(best_union)
        final_cycles = best_cycles_ls
        
        # Validate
        if validate_solution(final_edges, final_cycles):
            if len(final_edges) < best_score:
                best_score = len(final_edges)
                best_edges = final_edges
                best_cycles = final_cycles
        
        # If we haven't found anything better than 59, we might try a different base graph.
        # But for now, the Ladder is our main candidate.
        
        # Let's also try the "Fan" as a fallback if it's better (it's 77, so no).
        
        # What if we try to construct a graph with ~50 edges?
        # A random 3-regular graph has 60 edges.
        # A random 2-regular graph is a union of cycles.
        
        # Let's try a "Random 3-Regular" graph and check if it's pancyclic.
        # If it is, and has < 59 edges, it's a win.
        # But checking pancyclicity is hard.
        
        # We'll stick to the Ladder for now.
        
        if time.time() > deadline - 5:
            break

    # If no solution found, return the Ladder
    if best_edges is None:
        # Return the standard Ladder
        edges = []
        for c in range(19):
            edges.append((c, c + 1))
        for c in range(19):
            edges.append((20 + c, 20 + c + 1))
        for c in range(20):
            edges.append((c, 20 + c))
        edges.append((0, 21))
        
        cycles = [None] * 38
        for m in range(1, 20):
            L = 2 * m + 1
            idx = L - 3
            if m == 1:
                c = [0, 20, 21]
            else:
                c = [0]
                for cc in range(1, m):
                    c.append(cc)
                c.append(20 + m - 1)
                for cc in range(m - 2, 0, -1):
                    c.append(20 + cc)
            cycles[idx] = c

        for k in range(1, 20):
            L = 2 * k
            idx = L - 3
            c = [0]
            for cc in range(1, k):
                c.append(cc)
            c.append(20 + k - 1)
            for cc in range(k - 2, -1, -1):
                c.append(20 + cc)
            cycles[idx] = c
        
        c40 = []
        for cc in range(20):
            c40.append(cc)
        for cc in range(19, -1, -1):
            c40.append(20 + cc)
        cycles[37] = c40
        
        return {'edges': [list(e) for e in edges], 'cycles': cycles}

    return {'edges': [list(e) for e in best_edges], 'cycles': best_cycles}

# EVOLVE-BLOCK-END

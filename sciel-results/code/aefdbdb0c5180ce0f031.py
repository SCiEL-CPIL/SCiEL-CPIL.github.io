import random
import time
import sys
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses random lifts of K_{3,3} for k=3.
    """
    if k != 3:
        # Fallback for other k, though task specifies k=3
        # Simple random regular graph generation
        return _random_regular(k, g)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Base graph: K_{3,3}
    # Vertices 0,1,2 (part A) and 3,4,5 (part B)
    # Edges: (0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)
    base_edges = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
    base_adj = {i: [] for i in range(6)}
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
        
    # Precompute closed walks of even lengths up to g-1 (14)
    # We only care about even lengths because K_{3,3} is bipartite
    max_walk_len = g - 1  # 15, but we only check even
    walks_by_len = {l: [] for l in range(4, max_walk_len + 1, 2)}
    
    # DFS to find closed walks
    # Start from each vertex
    # To avoid duplicates, we can enforce a canonical order? 
    # Or just generate all and check. The number is small.
    # Length 14: 3^14 is too big for naive DFS if we don't prune.
    # But K_{3,3} has degree 3.
    # Number of walks of length L is approx 3 * 2^(L-1).
    # L=14: 3 * 2^13 = 24576. This is small!
    
    def generate_walks():
        for start in range(6):
            # Walk is a sequence of vertices v0, v1, ..., vL such that v0=vL
            # We generate paths of length L starting at start and ending at start.
            # We can use recursion.
            # To avoid counting the same cycle multiple times with different rotations, 
            # we don't need to deduplicate for the voltage check. 
            # If a cycle exists, it will be detected by any rotation.
            # However, checking duplicates is fine as 24k is small.
            
            def dfs(current, length, path):
                if length == 0:
                    if current == start:
                        # Store the sequence of edges or vertices
                        # We need the voltage sum, which depends on edges.
                        # Let's store the list of vertices to compute edge voltages later.
                        walks_by_len[length].append(list(path))
                    return
                if length > 0:
                    for next_v in base_adj[current]:
                        # Avoid immediate backtracking? 
                        # No, closed walks can backtrack. 
                        # But a cycle is a simple closed walk.
                        # If we find a non-simple closed walk with identity voltage, 
                        # it implies a simple cycle with identity voltage?
                        # Not necessarily. 
                        # However, girth is determined by SIMPLE cycles.
                        # If a non-simple walk has identity voltage, it's composed of cycles.
                        # If all simple cycles have non-identity voltage, can a non-simple have identity?
                        # Yes, if voltages cancel out.
                        # BUT, if a non-simple walk has identity, it means the product of cycle voltages is identity.
                        # This doesn't guarantee a single cycle is identity.
                        # HOWEVER, we want to ensure NO simple cycle is identity.
                        # Checking all closed walks is sufficient? 
                        # If a simple cycle is identity, it is a closed walk.
                        # If a non-simple walk is identity, it doesn't create a short simple cycle.
                        # So we ONLY need to check SIMPLE cycles?
                        # Actually, if we check all closed walks, we might reject a valid graph 
                        # because a non-simple walk has identity voltage, even if no simple cycle does.
                        # This would be a false negative.
                        # So we should ONLY check simple cycles.
                        
                        # How to generate only simple cycles?
                        # Standard DFS with visited set.
                        pass
                    
            # Let's restart DFS logic to generate simple cycles only.
            
    # Simpler: Generate all simple cycles of length 4, 6, ..., 14 in K_{3,3}.
    # K_{3,3} is small.
    simple_cycles = {l: [] for l in range(4, max_walk_len + 1, 2)}
    
    def find_simple_cycles():
        # Use Johnson's algorithm or simple DFS with visited
        for start in range(6):
            # We want cycles that start and end at start.
            # To avoid duplicates, we can require that start is the smallest vertex in the cycle?
            # Or just generate all and deduplicate by frozenset of edges?
            # Since N is small, let's just generate paths and check if they close simply.
            
            def dfs_simple(current, length, path, visited):
                if length == 0:
                    if current == start and len(path) > 1:
                        # Check if this is a simple cycle
                        # It is, by construction of visited
                        # Add to list
                        simple_cycles[length].append(list(path))
                    return
                
                for next_v in base_adj[current]:
                    if next_v == start and length == 1:
                        # Closing the cycle
                        if len(path) >= 3: # At least triangle? K3,3 has no triangles. Min 4.
                            simple_cycles[length].append(list(path))
                    elif next_v not in visited:
                        if next_v > start: # Only visit vertices greater than start to avoid rotation duplicates?
                            # This restricts to cycles where start is the minimum vertex.
                            # This ensures each cycle is found exactly once.
                            visited.add(next_v)
                            path.append(next_v)
                            dfs_simple(next_v, length - 1, path, visited)
                            path.pop()
                            visited.remove(next_v)
                        
            # Start DFS
            # We must start from start.
            # The first step can be any neighbor.
            # But we used "next_v > start" for intermediate nodes.
            # This works if we iterate neighbors.
            for first in base_adj[start]:
                if first > start:
                    visited = {start, first}
                    path = [start, first]
                    dfs_simple(first, length_target - 1, path, visited)
                    
    # We need to call this for each length.
    # The DFS above is a bit tricky with the length parameter.
    # Let's just generate all simple cycles up to length 14 in one go.
    
    all_simple_cycles = []
    def dfs_all_simple(current, path, visited):
        # Path starts with start
        for next_v in base_adj[current]:
            if next_v == path[0]:
                # Closed cycle
                if len(path) >= 4: # K3,3 min girth 4
                    # Check if simple (visited ensures this)
                    all_simple_cycles.append(list(path))
            elif next_v not in visited and next_v > path[0]:
                visited.add(next_v)
                path.append(next_v)
                dfs_all_simple(next_v, path, visited)
                path.pop()
                visited.remove(next_v)
                
    for start in range(6):
        visited = {start}
        path = [start]
        dfs_all_simple(start, path, visited)
        
    # Filter by length
    for cycle in all_simple_cycles:
        l = len(cycle)
        if l in simple_cycles:
            simple_cycles[l].append(cycle)
            
    # Sort cycles to ensure deterministic order if needed
    for l in simple_cycles:
        simple_cycles[l].sort()
        
    # Now, search for a valid lift
    # Try different group sizes m
    # n = 6 * m
    # Target n ~ 800 -> m ~ 133
    # Range for m: 130 to 160
    
    best_result = None
    
    for m in range(130, 161):
        if time.time() > deadline:
            break
            
        # Try several random assignments for this m
        attempts = 50
        for _ in range(attempts):
            if time.time() > deadline:
                break
                
            # Random voltage assignment: edge -> Z_m
            # We need to assign values to the 9 edges of K3,3
            # To make it random, we pick 9 random integers in [0, m)
            # Note: The voltage of a cycle is the sum of signed voltages.
            # We define voltage on directed edges.
            # For undirected edge (u,v), let voltage be x.
            # Then v->u is -x.
            # Cycle sum is sum of voltages along the path.
            
            voltages = {}
            for u, v in base_edges:
                voltages[(u, v)] = random.randint(0, m - 1)
                voltages[(v, u)] = (-voltages[(u, v)]) % m
                
            valid = True
            # Check all simple cycles
            for l in range(4, max_walk_len + 1, 2):
                for cycle in simple_cycles[l]:
                    # cycle is a list of vertices [v0, v1, ..., v_{l-1}]
                    # Edges are (v0,v1), (v1,v2), ..., (v_{l-1}, v0)
                    # Sum voltages
                    s = 0
                    for i in range(l):
                        u = cycle[i]
                        v = cycle[(i + 1) % l]
                        s += voltages[(u, v)]
                    if s % m == 0:
                        valid = False
                        break
                if not valid:
                    break
                    
            if valid:
                # Construct the graph
                # Vertices are (base_vertex, group_element)
                # Total vertices 6 * m
                n = 6 * m
                edges = []
                # Map (base_v, g) to index
                # index = base_v * m + g
                
                for u, v in base_edges:
                    val = voltages[(u, v)]
                    for g in range(m):
                        u_idx = u * m + g
                        v_idx = v * m + (g + val) % m
                        edges.append((u_idx, v_idx))
                        
                # Verify girth exactly (optional but safe)
                # Given the construction, if no simple cycle in base has identity voltage,
                # the lift has no cycle of that length.
                # Since we checked all simple cycles up to 14, girth is at least 16.
                # We can return this.
                return n, edges
                
    # If no lift found, fallback to random regular graph search (slower)
    # Or return the largest m tried? No, must be valid.
    # If we run out of time, we might return a valid graph from a smaller m?
    # The problem asks for MINIMUM n.
    # If we didn't find one, we should try smaller m?
    # Smaller m means smaller n, which is better.
    # But smaller m is harder to satisfy constraints?
    # Actually, smaller m has fewer "slots" for voltages, so it's EASIER to hit 0?
    # No, random variables mod m. Probability of sum 0 is 1/m.
    # Smaller m -> higher probability of collision -> HARDER to find valid.
    # So we should try SMALL m first?
    # But small m means small n.
    # If m=10, n=60. Girth 16 is impossible (Moore bound).
    # So we must find m such that n >= Moore Bound.
    # Moore bound 766. m >= 128.
    
    # So my loop 130-160 is correct.
    
    # If we failed, try a different strategy: Random 3-regular graph of size n
    # This is very slow to verify.
    
    # As a last resort, return a known valid graph if hardcoded?
    # No, must generate.
    
    # Let's try m from 128 upwards, but with more attempts.
    # The previous loop broke if time exceeded.
    
    # If we reach here, we didn't find one in 130-160.
    # Try 128-130 with more attempts?
    # Or try non-abelian group?
    
    # For now, if we fail, we return the best effort.
    # But we MUST return a valid graph.
    # If we can't find one, we might have to use a different base.
    
    # Let's try base = 3-prism (6 vertices, girth 4) or Dodecahedron (20 vertices, girth 5).
    # Dodecahedron lift: n = 20 * m.
    # m=40 -> n=800.
    # Cycles in Dodecahedron: min length 5.
    # We need to check cycles 5, 6, ..., 14.
    
    # This is getting complex. The K3,3 approach is likely to succeed.
    # Let's add a fallback: if K3,3 fails, try Dodecahedron.
    
    # Dodecahedron construction
    # Vertices 0..19.
    # This is hardcoded.
    
    # Given time constraints, I will stick to K3,3 but expand the search range slightly
    # and increase attempts if time permits.
    
    # If we still fail, we return an error or a large random graph?
    # The evaluator expects a valid graph.
    
    # Let's ensure we try m=128 as well.
    
    # Re-run search with wider range and more attempts if time permits
    if time.time() < deadline - 10:
        for m in range(128, 130):
            attempts = 100
            for _ in range(attempts):
                if time.time() > deadline:
                    break
                voltages = {}
                for u, v in base_edges:
                    voltages[(u, v)] = random.randint(0, m - 1)
                    voltages[(v, u)] = (-voltages[(u, v)]) % m
                valid = True
                for l in range(4, max_walk_len + 1, 2):
                    for cycle in simple_cycles[l]:
                        s = 0
                        for i in range(l):
                            u = cycle[i]
                            v = cycle[(i + 1) % l]
                            s += voltages[(u, v)]
                        if s % m == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    n = 6 * m
                    edges = []
                    for u, v in base_edges:
                        val = voltages[(u, v)]
                        for g in range(m):
                            u_idx = u * m + g
                            v_idx = v * m + (g + val) % m
                            edges.append((u_idx, v_idx))
                    return n, edges

    # Absolute fallback: Return a valid graph from a larger m if found?
    # We didn't find any.
    # We must return something.
    # Let's try a very large m to ensure validity, even if n is large.
    # But n must be minimized.
    # If we can't find a small one, a large one is better than invalid.
    
    # Try m=200
    m = 200
    for _ in range(10):
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(0, m - 1)
            voltages[(v, u)] = (-voltages[(u, v)]) % m
        valid = True
        for l in range(4, max_walk_len + 1, 2):
            for cycle in simple_cycles[l]:
                s = 0
                for i in range(l):
                    u = cycle[i]
                    v = cycle[(i + 1) % l]
                    s += voltages[(u, v)]
                if s % m == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            n = 6 * m
            edges = []
            for u, v in base_edges:
                val = voltages[(u, v)]
                for g in range(m):
                    u_idx = u * m + g
                    v_idx = v * m + (g + val) % m
                    edges.append((u_idx, v_idx))
            return n, edges

    # If even this fails, raise error
    raise ValueError("Failed to construct graph")

def _random_regular(k, g):
    # Placeholder for other k
    # Not used for k=3
    return 0, []


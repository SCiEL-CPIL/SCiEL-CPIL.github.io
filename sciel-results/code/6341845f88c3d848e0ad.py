import time
import random
from itertools import combinations, product

def _count_distinct_cycle_lengths(adj, n):
    """
    Enumerate all simple cycles in an undirected graph and return
    the set of distinct cycle lengths.
    For speed, we only explore cycles where the start vertex is the
    smallest vertex in the cycle (to avoid counting each cycle multiple times).
    This is a standard trick: only start DFS from vertex v, and only visit
    vertices with index > v.
    """
    lengths = set()
    for start in range(n):
        # DFS: path is a list of vertices, stack-based for efficiency
        # We require all vertices in the path to be > start (except start itself)
        # and the last vertex must connect back to start.
        # Use iterative DFS to avoid recursion depth issues, though n=12 is small.
        
        # State: (current, path)
        # We start with path = [start]
        stack = [(start, [start])]
        while stack:
            curr, path = stack.pop()
            for nb in adj[curr]:
                if nb == start:
                    if len(path) >= 3:  # Cycle must have length >= 3 (actually >=4 due to triangle-free)
                        lengths.add(len(path))
                elif nb > start:
                    # Only visit vertices with index > start to ensure uniqueness
                    if nb not in path:
                        stack.append((nb, path + [nb]))
    return lengths

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    n = 12
    min_deg = 3
    
    # We know K_{3,3} x K_{3,3} works and gives score 2.5 (value 2).
    # Can we get value 1?
    # Value 1 means all simple cycles have the same length.
    # If the graph is bipartite, cycles are even.
    # If all cycles are length 4, the graph is a "4-cycle graph" (e.g., a grid or specific structure).
    # If all cycles are length 6, etc.
    
    # Let's try to find a graph where all cycles have the same length.
    # A known example: The incidence graph of a projective plane? No, that has many lengths.
    # What about a graph composed of disjoint cycles of the same length?
    # E.g., three disjoint 4-cycles.
    # Vertices: 12.
    # Cycles: (0-1-2-3-0), (4-5-6-7-4), (8-9-10-11-8).
    # Degrees: All vertices have degree 2.
    # Constraint: Every vertex must have degree at least 3.
    # So disjoint cycles are not enough unless we add chords or connect them.
    
    # If we add chords to a 4-cycle, we might create triangles (bad) or 3-cycles (bad) or change cycle lengths.
    # Adding a diagonal to a 4-cycle (0-1-2-3) creates two triangles (0-1-2 and 1-2-3? No, 0-2 diagonal creates 0-1-2 and 0-2-3 which are triangles? No, 0-1-2 is a triangle if 0-2 is an edge. Yes.)
    # So we cannot add diagonals to 4-cycles in a triangle-free graph.
    
    # What about a graph where all cycles are length 6?
    # Or a mix that results in only one distinct length?
    
    # Let's search for a triangle-free graph with min degree 3 on 12 vertices
    # that minimizes distinct cycle lengths.
    
    # Current best: 2 (lengths 4 and 6 from K3,3).
    # Target: 1.
    
    # Strategy: Randomized local search / simulated annealing or just random generation + check.
    # Since n=12 is small, we can be somewhat exhaustive or use a good heuristic.
    
    # Heuristic: Start with K_{3,3} x K_{3,3}.
    # Try to remove edges? No, degree must be >=3. K3,3 has degree 3. Removing any edge drops degree to 2.
    # So we must add edges?
    # Adding an edge within a part of K3,3 creates triangles?
    # K3,3 parts are A={0,1,2}, B={3,4,5}. Edges are all a-b.
    # If we add edge (0,1), we create triangle 0-1-3 (since 0-3 and 1-3 exist). Triangle!
    # So we cannot add edges within parts of K3,3 without creating triangles.
    
    # What if we use a different base graph?
    # The Petersen graph has 10 vertices, degree 3, girth 5.
    # Cycle lengths in Petersen: 5, 6, 8, 9, 10. (Many distinct lengths).
    
    # What about a graph composed of disjoint components that all have only one cycle length?
    # But components must have min degree 3.
    # Smallest triangle-free graph with min degree 3 is K_{3,3} (6 vertices).
    # Next?
    # If we take a single component, it must have min degree 3.
    # If we take two components, each must have min degree 3.
    # We have 12 vertices.
    # Option A: Two K_{3,3}. Cycle lengths: {4, 6}. Distinct: 2.
    # Option B: One component of 12 vertices.
    #    If we can find a 12-vertex triangle-free min-degree-3 graph with only one cycle length, we win.
    #    Or one component with cycle lengths {4} and another with {4}?
    #    Smallest triangle-free min-degree-3 graph with only 4-cycles?
    #    A 4-regular bipartite graph on 8 vertices? (Cube graph Q3).
    #    Q3 has 8 vertices, degree 3.
    #    Cycle lengths in Q3: 4, 6, 8. Distinct: 3.
    #    If we take two Q3s (8+8=16 vertices, too many).
    #    We have 12 vertices.
    
    #    Can we have a 12-vertex graph with only 4-cycles?
    #    If all cycles are 4, the graph is a "linear" combination of 4-cycles?
    #    Consider the incidence graph of a set system where all cycles are 4.
    #    This implies the graph is a "4-cycle-free" in the sense that it has NO cycles of length != 4.
    #    Wait, "all cycles are length 4".
    #    If a graph has only 4-cycles, it is a disjoint union of 4-cycles?
    #    No, you can have two 4-cycles sharing an edge?
    #    If they share an edge, you get a smaller cycle or a triangle?
    #    Example: (0-1-2-3-0) and (1-2-4-5-1).
    #    Cycle 1-2-3-0-1? No, 0-1 is an edge. 1-2 is an edge. 2-3 is an edge. 3-0 is an edge.
    #    Is there a cycle 1-2-4-5-1? Yes.
    #    Is there a cycle 0-1-2-3-0? Yes.
    #    Is there a cycle 0-1-5-4-2-3-0? Length 6.
    #    So sharing edges creates longer cycles.
    #    Thus, a graph with only 4-cycles must be a disjoint union of 4-cycles.
    #    But disjoint 4-cycles have degree 2. We need degree >= 3.
    #    So we cannot have a graph with ONLY 4-cycles and min degree 3.
    
    #    What about ONLY 6-cycles?
    #    Disjoint 6-cycles have degree 2.
    #    Sharing edges creates other lengths.
    
    #    So it seems impossible to have min degree 3 and only one cycle length if the graph is simple and undirected?
    #    Let's check if there's a counterexample.
    #    "Cubic graphs with girth 6"?
    #    The Heawood graph has 14 vertices. Girth 6.
    #    Cycle lengths in Heawood: 6, 8, 10, 12, 14?
    #    It's a bipartite graph.
    #    It likely has multiple cycle lengths.
    
    #    What about the Clebsch graph? 16 vertices, girth 5. Not bipartite.
    
    #    It is a known result that any graph with min degree >= 3 and girth >= 4 will have at least two distinct cycle lengths?
    #    Actually, if a graph is "cycle-regular" (all cycles same length), it's a very strong constraint.
    #    For min degree 3, I suspect it's impossible to have only one cycle length.
    #    Therefore, the minimum distinct cycle lengths is likely 2.
    
    #    If the minimum is 2, the current score of 2.5 (value 2) is optimal.
    #    The score is seed/value. Seed is 5.0?
    #    If value is 2, score is 2.5.
    #    If we can't get value 1, we are at the optimum.
    
    #    However, I should double check if I can construct a graph with value 1.
    #    Maybe a non-bipartite graph?
    #    If girth is 5, min degree 3.
    #    Petersen graph: lengths 5, 6, 8, 9, 10.
    #    Can we prune it?
    
    #    Let's try a random search to see if we can find a graph with 1 distinct cycle length.
    #    If we find one, we return it. If not, we return the K3,3xK3,3.
    
    def is_valid(edges):
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            if u == v: return False
            if (v, u) in edge_set: return False
            edge_set.add((u, v))
            edge_set.add((v, u))
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) < min_deg:
                return False
        
        # Check triangle-free
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and (u, w) in edge_set:
                        return False
                    if w == u:
                        continue
                    # Check if u-w is an edge
                    if w in adj[u]:
                        # Triangle u-v-w
                        return False
        
        # Check if it has a cycle
        # If min degree >= 3, it must have a cycle.
        
        return True

    def get_cycle_lengths(edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return _count_distinct_cycle_lengths(adj, n)

    # Try to find a graph with 1 distinct cycle length
    # Randomized restart: generate random triangle-free graphs with min degree 3?
    # This is hard to generate directly.
    # Instead, start from K3,3xK3,3 and try to mutate?
    # Or start from a random graph and repair?
    
    # Given the time limit, and the high likelihood that 2 is the minimum,
    # I will just return the known optimal (or near-optimal) solution.
    # But let's do a quick search for 1 distinct length just in case.
    
    # Search strategy:
    # 1. Generate random simple graphs on 12 vertices.
    # 2. Filter: min degree >= 3, triangle-free.
    # 3. Count distinct cycle lengths.
    # 4. If 1, return.
    # 5. Else, return best found (min distinct lengths).
    
    best_edges = None
    best_val = float('inf')
    
    # Initial best: K3,3 x K3,3
    init_edges = []
    for u in range(3):
        for v in range(3, 6):
            init_edges.append([u, v])
    for u in range(6, 9):
        for v in range(9, 12):
            init_edges.append([u, v])
    
    if is_valid(init_edges):
        lens = get_cycle_lengths(init_edges)
        best_val = len(lens)
        best_edges = init_edges
    
    # Random search
    # We need to generate random graphs that are likely to satisfy constraints.
    # Start with a random permutation of vertices, assign high probability to edges in a bipartite-like structure?
    # Or just random edges with p=0.5, then check.
    
    num_trials = 1000
    for i in range(num_trials):
        if time.time() > deadline:
            break
            
        # Generate random graph
        # To increase chance of min degree 3, use p=0.6
        # To increase chance of triangle-free, this is hard.
        # Let's just try random edges.
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.6:
                    edges.append([u, v])
        
        if is_valid(edges):
            lens = get_cycle_lengths(edges)
            val = len(lens)
            if val < best_val:
                best_val = val
                best_edges = edges
                if best_val == 1:
                    break
    
    # If we found a valid graph, return it.
    # Otherwise, return the initial one (which is valid).
    
    if best_edges is None:
        # Should not happen, as init_edges is valid
        return init_edges
        
    # Ensure types are correct
    return [list(e) for e in best_edges]

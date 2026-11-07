"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
EDGES = P['edges']
EDGE_SET = set(frozenset(e) for e in EDGES)
ADJ = {i: set() for i in range(N)}
for u, v in EDGES:
    ADJ[u].add(v)
    ADJ[v].add(u)

# Pre-compute all simple cycles of length 3 to 7
# We represent a cycle as a tuple of vertices in order, normalized so the smallest vertex is first
# and the second smallest is also fixed to break rotational symmetry? 
# Actually, for edge checking, we just need the set of edges.
# For uniqueness in lists, we can use a canonical form.

def get_canonical_cycle(cycle):
    if not cycle:
        return None
    # Rotate to start with min vertex
    min_idx = cycle.index(min(cycle))
    c = tuple(cycle[min_idx:] + cycle[:min_idx])
    # Check reverse? A cycle (0,1,2) is the same as (0,2,1). 
    # We just need to ensure we don't add duplicates to the pool if possible, 
    # but for search, having duplicates is fine, just less efficient.
    # However, for the "remove and repack" logic, we just need the edge set.
    return c

def find_cycles_in_subgraph(available_edges_set, max_len=7):
    """
    Find all simple cycles of length <= max_len in the graph formed by available_edges_set.
    Returns a list of (cycle_vertices, edge_set)
    """
    found = []
    # Build adjacency for the subgraph
    sub_adj = {i: set() for i in range(N)}
    for e in available_edges_set:
        u, v = tuple(e)
        sub_adj[u].add(v)
        sub_adj[v].add(u)
    
    # DFS from each node
    for start in range(N):
        if len(sub_adj[start]) == 0:
            continue
        # visited array for path
        path = [start]
        visited = {start}
        
        def dfs(current, depth):
            if depth >= max_len:
                return
            if depth >= 3:
                # Check if we can close the cycle to start
                if start in sub_adj[current]:
                    # Found a cycle
                    c = tuple(path)
                    # Check if this cycle is already in found? 
                    # To avoid massive duplicate lists in small subgraphs, 
                    # we might skip dedup here for speed, or use a set of edge-frozensets.
                    cycle_edges = set()
                    for i in range(len(path)):
                        u, v = path[i], path[(i+1)%len(path)]
                        cycle_edges.add(frozenset([u,v]))
                    
                    # Simple check: is this edge set unique in 'found'? 
                    # This is O(K) where K is number of found cycles. 
                    # In a small subgraph, K is small.
                    if not any(existing[1] == cycle_edges for existing in found):
                        found.append((c, cycle_edges))
            
            # Explore neighbors
            for neighbor in sub_adj[current]:
                if neighbor == start:
                    continue # Handled above
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, depth + 1)
                    path.pop()
                    visited.remove(neighbor)
        
        dfs(start, 1)
        
    return found

def greedy_pack(available_edges):
    """
    Greedily pack cycles from available_edges.
    Returns a list of edge_sets for the cycles found.
    """
    if not available_edges:
        return []
    
    cycles_found = []
    current_available = set(available_edges)
    
    # Find all cycles in the current graph
    all_cycles = find_cycles_in_subgraph(current_available, max_len=7)
    
    # Sort by length (shorter is better)
    all_cycles.sort(key=lambda x: len(x[0]))
    
    # Randomize to avoid local optima? 
    # Let's shuffle within same lengths
    for i in range(len(all_cycles)):
        pass # We can't easily sort and then randomize groups in one go without overhead
    # Instead, just pick randomly from the shortest available each time?
    # Or pick randomly from all, but weight by 1/length?
    
    # Let's try: While there are cycles:
    # 1. Find all cycles in current_available
    # 2. Pick one randomly (prefer shorter)
    # 3. Remove its edges
    # 4. Repeat
    
    # Optimization: Pre-finding all cycles every iteration is expensive.
    # But N=21 is small.
    
    while True:
        # Find cycles in current_available
        # To save time, we can just look for triangles first?
        # No, generic is safer.
        
        # If the number of edges is small, find_cycles is fast.
        candidates = find_cycles_in_subgraph(current_available, max_len=7)
        
        if not candidates:
            break
        
        # Heuristic: Pick a random candidate, but bias towards shorter ones.
        # Probability proportional to 1/length
        lengths = [len(c[0]) for c in candidates]
        probs = [1.0/l for l in lengths]
        
        # Normalize
        total_prob = sum(probs)
        probs = [p/total_prob for p in probs]
        
        chosen_idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
        chosen_cycle, chosen_edges = candidates[chosen_idx]
        
        cycles_found.append(chosen_edges)
        
        # Remove edges
        for e in chosen_edges:
            current_available.remove(e)
            
        if not current_available:
            break
            
    return cycles_found

def local_search(cycles_edge_sets, available_edges_original):
    """
    Try to improve the solution by unpacking a cycle and repacking.
    Returns a new list of edge sets if improved.
    """
    current_copies = [set(c) for c in cycles_edge_sets]
    # The "used" edges are the union of all cycles
    used_edges = set()
    for c in current_copies:
        used_edges.update(c)
    
    # The "free" edges are original - used
    free_edges = available_edges_original - used_edges
    
    improved = False
    # Try each cycle
    for i in range(len(current_copies)):
        # Remove cycle i
        cycle_to_remove = current_copies[i]
        # New free edges: free_edges + cycle_to_remove
        new_free = set(free_edges)
        new_free.update(cycle_to_remove)
        
        # Repack new_free
        new_cycles = greedy_pack(new_free)
        
        # The rest of the cycles remain
        other_cycles = [current_copies[j] for j in range(len(current_copies)) if j != i]
        
        # Check if new_cycles is valid (it is, by construction)
        # Check if we improved the count
        if len(new_cycles) > 1:
            # We replaced 1 cycle with >1 cycles. This is an improvement in count.
            # We should also check if we can pack *more* by repacking the *entire* set?
            # No, just swapping is good.
            # Construct new list
            new_solution = other_cycles + new_cycles
            
            # Verify validity: No edge overlap
            seen = set()
            valid = True
            for c in new_solution:
                for e in c:
                    if e in seen:
                        valid = False
                        break
                    seen.add(e)
                if not valid:
                    break
            # Verify all edges are from original
            if valid and all(e in available_edges_original for e in seen):
                return new_solution
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    best_count = 0
    best_cycles = []
    
    # Ensure we return a valid format: list of lists of ints
    # The input P['edges'] defines the graph.
    # The task requires: "All cycles must use exactly the same vertex set... in the pinned 21-vertex host"
    # Wait, "use exactly the same vertex set" usually means the *union* of vertices in all cycles must be the same?
    # Or does it mean each cycle must be a Hamiltonian cycle?
    # "each represented by a list of distinct vertices... All cycles must use exactly the same vertex set of size at least three"
    # This phrasing is tricky.
    # Interpretation 1: The set of vertices appearing in Cycle 1 == Set of vertices in Cycle 2 == ...
    # Interpretation 2: The union of all cycles covers the same vertex set?
    # "Packing edge-disjoint cycles on one vertex set"
    # Usually, "on a vertex set" means the graph is defined on that set.
    # "All cycles must use exactly the same vertex set" -> This likely means every cycle must be a **Hamiltonian Cycle** on the 21 vertices?
    # If they must be Hamiltonian, each cycle uses 21 edges.
    # Total edges 72. Max cycles = floor(72/21) = 3.
    # The seed returns 1 cycle (range(21)), which is a Hamiltonian cycle if edges exist.
    # Let's check if the seed cycle [0..20] is a valid cycle in the graph.
    # Edges: (0,1), (1,2)...(20,0).
    # (0,1) Yes. (1,2) Yes. (2,3) Yes. ... (20,0) Yes.
    # So the seed finds 1 Hamiltonian cycle.
    
    # If the requirement is that *every* cycle must visit *all* 21 vertices, then we are looking for the maximum number of edge-disjoint Hamiltonian cycles.
    # This is a much harder and more constrained problem.
    # If the requirement is just that the cycles are on the host graph (which has 21 vertices), and "use exactly the same vertex set" is a clumsy way of saying "defined on the same host", then we want max edge-disjoint simple cycles.
    
    # Let's re-read carefully: "All cycles must use exactly the same vertex set of size at least three in the pinned 21-vertex host."
    # If I have a cycle 0-1-2-0 and another cycle 0-3-4-0, they do NOT use the same vertex set.
    # The first uses {0,1,2}, the second {0,3,4}.
    # The host vertex set is {0..20}.
    # Does "use exactly the same vertex set" mean "Use the host vertex set"? i.e. Each cycle is a Hamiltonian Cycle?
    # OR does it mean "The set of vertices used by Cycle 1 is the same as the set of vertices used by Cycle 2"?
    # If it's the latter, and size >= 3, it implies all cycles have the same number of vertices and are isomorphic in terms of vertex support?
    # Given "Maximize number of cycles", and the seed is 1 Hamiltonian cycle.
    # If small cycles were allowed, the score would likely be much higher than 1.
    # The seed score is 1.0000.
    # If I can find 2 Hamiltonian cycles, score becomes 2.0.
    # If I can find 3, score 3.0.
    # If small cycles were allowed, I could easily find 10+ cycles (72/3 = 24).
    # The fact that the seed is 1 suggests the constraint is tight.
    # "All cycles must use exactly the same vertex set" -> I will bet this means **Hamiltonian Cycles**.
    # Why? Because otherwise "use exactly the same vertex set" is a very strange constraint for packing small cycles (e.g. all triangles on disjoint vertex sets? No, "same" set).
    # If all cycles must use the *same* set of vertices, and they are edge-disjoint, they must all be Hamiltonian cycles on that specific set.
    # Since the host has 21 vertices, and "size at least three", and it's a "pinned 21-vertex host", it implies the vertex set IS the 21 vertices.
    
    # So, Task: Find max number of edge-disjoint Hamiltonian cycles in the given graph.
    
    # Let's verify if 2 Hamiltonian cycles exist.
    # This is a specific subgraph.
    
    # Strategy for Hamiltonian Cycles:
    # 1. Check if a second Hamiltonian cycle exists disjoint from the first.
    # 2. Use a solver or heuristic to find them.
    
    # Heuristic:
    # 1. Find one Hamiltonian cycle (Seed has one).
    # 2. Remove its edges.
    # 3. Try to find a Hamiltonian cycle in the remaining graph.
    # 4. Repeat.
    
    # Since N=21 is small, we can use a randomized DFS for Hamiltonian cycles.
    
    def find_hamiltonian_cycle(available_edges_set):
        """
        Find a Hamiltonian cycle in the graph defined by available_edges_set.
        Returns the cycle as a list of vertices, or None.
        Uses Warnsdorff's heuristic for better performance.
        """
        # Build adjacency
        adj = {i: set() for i in range(N)}
        for e in available_edges_set:
            u, v = tuple(e)
            adj[u].add(v)
            adj[v].add(u)
            
        # Check degrees: all must be >= 2
        for i in range(N):
            if len(adj[i]) < 2:
                return None
        
        # Start at 0
        start = 0
        path = [start]
        visited = {start}
        
        def warnsdorff_order(current, remaining):
            """Sort neighbors by Warnsdorff's rule: fewest onward options first."""
            def score(neighbor):
                # Count unvisited neighbors of this neighbor (excluding current)
                count = 0
                for nb in adj[neighbor]:
                    if nb not in visited and nb != current:
                        count += 1
                return count
            # Filter to only unvisited neighbors
            candidates = [n for n in adj[current] if n not in visited]
            if not candidates:
                return []
            # Sort by score (ascending - prefer fewer options)
            candidates.sort(key=lambda x: (score(x), x))
            return candidates
        
        def dfs(current):
            if len(path) == N:
                # Check if edge back to start exists
                if start in adj[current]:
                    return True
                return False
            
            # Get neighbors sorted by Warnsdorff's rule
            neighbors = warnsdorff_order(current, set(range(N)) - visited)
            
            # Add some randomness: with small probability, pick a random neighbor
            if neighbors and random.random() < 0.1:
                neighbors = neighbors[:]
                random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    if dfs(neighbor):
                        return True
                    path.pop()
                    visited.remove(neighbor)
            return False
        
        # Try with different random seeds for robustness
        for attempt in range(20):
            if dfs(start):
                # Reconstruct edges
                cycle_edges = set()
                for i in range(N):
                    u, v = path[i], path[(i+1)%N]
                    cycle_edges.add(frozenset([u,v]))
                return cycle_edges
            # Reset for next attempt
            path = [start]
            visited = {start}
        
        return None

    # Initialize with seed cycle
    seed_cycle_edges = set()
    seed_path = list(range(N))
    for i in range(N):
        u, v = seed_path[i], seed_path[(i+1)%N]
        seed_cycle_edges.add(frozenset([u,v]))
    
    # Verify seed is valid
    valid_seed = all(e in EDGE_SET for e in seed_cycle_edges)
    
    if not valid_seed:
        # Find any ham cycle?
        # For now, assume seed is valid.
        pass

    best_count = 0
    best_cycles_list = []
    
    # We will run a search loop.
    # Each iteration:
    # 1. Start with empty used edges.
    # 2. Greedily find Hamiltonian cycles.
    
    max_attempts = 50
    attempt = 0
    while time.time() < deadline and attempt < max_attempts:
        attempt += 1
        used_edges = set()
        current_cycles = []
        
        available = set(EDGE_SET)
        
        # Try to find as many as possible
        while True:
            # Check time before each cycle search
            if time.time() > deadline:
                break
                
            # Find a ham cycle in available
            h_cycle = find_hamiltonian_cycle(available)
            
            if h_cycle is None:
                break
                
            current_cycles.append(h_cycle)
            used_edges.update(h_cycle)
            available = set(EDGE_SET - used_edges)
            
            # Check if we've used all edges (perfect packing)
            if not available:
                break
            
        # Update best
        if len(current_cycles) > best_count:
            best_count = len(current_cycles)
            best_cycles_list = current_cycles
            
            # If we reached the theoretical max (3), break
            if best_count >= 3:
                break
            
    # Construct return value
    # Each cycle is a list of vertices.
    # We have edge sets. We need to convert back to vertex lists.
    
    def edges_to_vertex_list(edge_set):
        """
        Reconstruct the vertex list from a set of edges forming a single cycle.
        """
        # Build adjacency for just this cycle
        adj = {i: [] for i in range(N)}
        for e in edge_set:
            u, v = tuple(e)
            adj[u].append(v)
            adj[v].append(u)
        
        # Each vertex should have degree 2 in a cycle
        for i in range(N):
            if len(adj[i]) != 2:
                return None
        
        # Start at the smallest vertex
        start = min(range(N), key=lambda x: x)
        if len(adj[start]) < 2:
            return None
        
        path = [start]
        current = start
        visited = {start}
        
        for _ in range(N-1):
            next_v = None
            for n in adj[current]:
                if n not in visited:
                    next_v = n
                    break
            if next_v is None:
                # Try the other neighbor
                for n in adj[current]:
                    if n == start and len(path) == N:
                        next_v = n
                        break
                    elif n not in visited:
                        next_v = n
                        break
            if next_v is None:
                return None
            path.append(next_v)
            visited.add(next_v)
            current = next_v
        
        # Check close: last vertex must connect to start
        if start not in adj[current]:
            return None
            
        return path
        
    final_result = []
    for e_set in best_cycles_list:
        v_list = edges_to_vertex_list(e_set)
        if v_list:
            final_result.append(v_list)
            
    # If no cycles found (shouldn't happen), return seed?
    if not final_result:
        return [list(range(N))]
        
    return final_result

# EVOLVE-BLOCK-END

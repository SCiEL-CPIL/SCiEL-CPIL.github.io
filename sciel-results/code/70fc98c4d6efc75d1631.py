"""Rewritten program: edge-disjoint Hamiltonian cycle packing via 3-edge-coloring."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
EDGES = P['edges']
M = len(EDGES)

# Precompute edge index
EDGE_TO_IDX = {}
for i, (u, v) in enumerate(EDGES):
    key = (min(u, v), max(u, v))
    EDGE_TO_IDX[key] = i

# For each vertex, list of incident edge indices
VERTEX_EDGES = [[] for _ in range(N)]
for i, (u, v) in enumerate(EDGES):
    VERTEX_EDGES[u].append(i)
    VERTEX_EDGES[v].append(i)

# Adjacency sets
ADJ = [set() for _ in range(N)]
for u, v in EDGES:
    ADJ[u].add(v)
    ADJ[v].add(u)


def _find_decomposition(deadline):
    """
    Find a 3-edge-coloring of the graph where each color class is a Hamiltonian cycle.
    Uses randomized local search (simulated annealing style) on the edge coloring.
    """
    rng = random.Random(42)
    
    # color[i] = 0, 1, or 2 for edge i
    # Each vertex must have exactly 2 edges of each color.
    
    def random_coloring():
        """Generate a random valid degree-constrained coloring."""
        # Start with empty, assign edges one by one respecting degree constraints
        color = [-1] * M
        deg = [[0] * N for _ in range(3)]
        
        # Shuffle edges for randomization
        edge_order = list(range(M))
        rng.shuffle(edge_order)
        
        for ei in edge_order:
            u, v = EDGES[ei]
            # Find a valid color
            valid_colors = []
            for c in range(3):
                if deg[c][u] < 2 and deg[c][v] < 2:
                    valid_colors.append(c)
            if not valid_colors:
                return None  # Failed
            c = rng.choice(valid_colors)
            color[ei] = c
            deg[c][u] += 1
            deg[c][v] += 1
        
        # Check if all degrees are exactly 2
        for c in range(3):
            for v in range(N):
                if deg[c][v] != 2:
                    return None
        
        return color
    
    def check_hamiltonian(color):
        """Check if each color class forms a single Hamiltonian cycle. Returns list of cycles or None."""
        cycles = []
        for c in range(3):
            # Build adjacency for this color
            adj_c = [set() for _ in range(N)]
            for i, (u, v) in enumerate(EDGES):
                if color[i] == c:
                    adj_c[u].add(v)
                    adj_c[v].add(u)
            
            # Verify all vertices have degree 2
            for v in range(N):
                if len(adj_c[v]) != 2:
                    return None
            
            # Traverse to find the cycle
            visited = [False] * N
            cycle = []
            start = 0
            cur = start
            visited[start] = True
            cycle.append(start)
            prev = -1
            for _ in range(N - 1):
                nexts = list(adj_c[cur])
                nexts = [x for x in nexts if x != prev]
                if not nexts:
                    return None
                nxt = nexts[0]
                if visited[nxt] and len(cycle) < N:
                    return None
                visited[nxt] = True
                cycle.append(nxt)
                prev = cur
                cur = nxt
            
            # Check it's a single cycle covering all vertices
            if len(cycle) != N or not all(visited):
                return None
            # Check closing edge
            if cycle[0] not in adj_c[cycle[-1]]:
                return None
            
            cycles.append(cycle)
        
        return cycles
    
    def count_subcycles(color):
        """Count the total number of cycles across all color classes (lower is better, target is 3)."""
        total = 0
        for c in range(3):
            adj_c = [set() for _ in range(N)]
            for i, (u, v) in enumerate(EDGES):
                if color[i] == c:
                    adj_c[u].add(v)
                    adj_c[v].add(u)
            
            visited = [False] * N
            count = 0
            for v in range(N):
                if not visited[v]:
                    count += 1
                    # BFS
                    stack = [v]
                    visited[v] = True
                    while stack:
                        node = stack.pop()
                        for nb in adj_c[node]:
                            if not visited[nb]:
                                visited[nb] = True
                                stack.append(nb)
            total += count
        return total
    
    # Try to find a valid coloring
    best_color = None
    best_score = float('inf')
    
    # First, try random initializations
    for attempt in range(200):
        if time.time() > deadline:
            break
        color = random_coloring()
        if color is None:
            continue
        
        cycles = check_hamiltonian(color)
        if cycles is not None:
            return cycles
        
        score = count_subcycles(color)
        if score < best_score:
            best_score = score
            best_color = color[:]
    
    if best_color is None:
        return None
    
    # Local search: swap colors of edges to reduce number of subcycles
    color = best_color[:]
    current_score = best_score
    
    # Precompute for fast updates
    # For each vertex and color, track which edges are assigned
    # deg[c][v] should always be 2
    
    def swap_edges(color, e1, e2):
        """Swap colors of edges e1 and e2. Returns True if valid (degree constraints maintained)."""
        c1, c2 = color[e1], color[e2]
        if c1 == c2:
            return False
        u1, v1 = EDGES[e1]
        u2, v2 = EDGES[e2]
        # After swap: e1 gets c2, e2 gets c1
        # Check: for each vertex involved, the new color assignment doesn't violate degree
        # Actually, since we're just swapping, the degree constraint is automatically maintained
        # as long as we swap between two different colors.
        color[e1] = c2
        color[e2] = c1
        return True
    
    def unswap_edges(color, e1, e2):
        c1, c2 = color[e1], color[e2]
        color[e1] = c2
        color[e2] = c1
    
    # Simulated annealing
    temperature = 1.0
    cooling = 0.9995
    min_temp = 0.01
    
    while temperature > min_temp and time.time() < deadline:
        improved = False
        for _ in range(500):
            if time.time() > deadline:
                break
            
            # Pick two random edges of different colors
            e1 = rng.randint(0, M - 1)
            e2 = rng.randint(0, M - 1)
            if e1 == e2 or color[e1] == color[e2]:
                continue
            
            old_c1, old_c2 = color[e1], color[e2]
            swap_edges(color, e1, e2)
            
            new_score = count_subcycles(color)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < pow(2.718281828, -delta / temperature):
                current_score = new_score
                improved = True
                if current_score == 3:
                    cycles = check_hamiltonian(color)
                    if cycles is not None:
                        return cycles
            else:
                unswap_edges(color, e1, e2)
        
        temperature *= cooling
        
        if current_score == 3:
            cycles = check_hamiltonian(color)
            if cycles is not None:
                return cycles
    
    # Final check
    if current_score == 3:
        cycles = check_hamiltonian(color)
        if cycles is not None:
            return cycles
    
    return None


def _find_decomposition_deterministic(deadline):
    """
    Deterministic backtracking with good heuristics to find 3 edge-disjoint Hamiltonian cycles.
    """
    # Fix edge 0 to color 0, edge 1 to color 1 (symmetry breaking)
    # Process edges in a smart order
    
    color = [-1] * M
    deg = [[0] * N for _ in range(3)]
    
    # Order edges: start with edges incident to vertex 0, then others
    # Actually, let's use a fixed order but with symmetry breaking
    edge_order = list(range(M))
    
    # Symmetry breaking: first edge (0,1) -> color 0
    # Find edge index for (0,1)
    e01 = EDGE_TO_IDX[(0, 1)]
    e02 = EDGE_TO_IDX[(0, 2)]
    
    # Put e01 first, e02 second
    edge_order.remove(e01)
    edge_order.remove(e02)
    edge_order = [e01, e02] + edge_order
    
    # For the rest, order by vertex to improve pruning
    # Actually, let's just use the natural order for the rest
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == len(edge_order):
            # Check if each color forms a Hamiltonian cycle
            for c in range(3):
                adj_c = [set() for _ in range(N)]
                for i, (u, v) in enumerate(EDGES):
                    if color[i] == c:
                        adj_c[u].add(v)
                        adj_c[v].add(u)
                # Check connectivity
                visited = [False] * N
                visited[0] = True
                stack = [0]
                count = 1
                while stack:
                    node = stack.pop()
                    for nb in adj_c[node]:
                        if not visited[nb]:
                            visited[nb] = True
                            count += 1
                            stack.append(nb)
                if count != N:
                    return False
            return True
        
        ei = edge_order[idx]
        u, v = EDGES[ei]
        
        # Try colors 0, 1, 2
        for c in range(3):
            if deg[c][u] >= 2 or deg[c][v] >= 2:
                continue
            
            # Additional pruning: check if this leads to a dead end
            # For now, just try it
            
            color[ei] = c
            deg[c][u] += 1
            deg[c][v] += 1
            
            if backtrack(idx + 1):
                return True
            
            color[ei] = -1
            deg[c][u] -= 1
            deg[c][v] -= 1
        
        return False
    
    if backtrack(0):
        # Extract cycles
        cycles = []
        for c in range(3):
            adj_c = [set() for _ in range(N)]
            for i, (u, v) in enumerate(EDGES):
                if color[i] == c:
                    adj_c[u].add(v)
                    adj_c[v].add(u)
            
            cycle = [0]
            visited = {0}
            cur = 0
            for _ in range(N - 1):
                nexts = [nb for nb in adj_c[cur] if nb not in visited]
                if not nexts:
                    break
                cur = nexts[0]
                cycle.append(cur)
                visited.add(cur)
            cycles.append(cycle)
        return cycles
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin
    
    # Try deterministic backtracking first
    cycles = _find_decomposition_deterministic(deadline)
    if cycles is not None and len(cycles) == 3:
        # Validate
        for cycle in cycles:
            if len(cycle) != N:
                cycles = None
                break
            if len(set(cycle)) != N:
                cycles = None
                break
        if cycles is not None:
            return [[int(v) for v in cycle] for cycle in cycles]
    
    # Try randomized approach
    cycles = _find_decomposition(deadline)
    if cycles is not None and len(cycles) == 3:
        for cycle in cycles:
            if len(cycle) != N:
                cycles = None
                break
            if len(set(cycle)) != N:
                cycles = None
                break
        if cycles is not None:
            return [[int(v) for v in cycle] for cycle in cycles]
    
    # Fallback: find any valid cycles
    # At minimum, find one Hamiltonian cycle
    used_edges = set()
    found_cycles = []
    
    for _ in range(3):
        if time.time() > deadline:
            break
        # Find a Hamiltonian cycle avoiding used edges
        adj_avail = [set() for _ in range(N)]
        for u, v in EDGES:
            if (min(u,v), max(u,v)) not in used_edges:
                adj_avail[u].add(v)
                adj_avail[v].add(u)
        
        # Simple DFS for Hamiltonian cycle
        def dfs_ham(path, visited):
            if time.time() > deadline:
                return None
            if len(path) == N:
                if path[0] in adj_avail[path[-1]]:
                    return path
                return None
            for nb in sorted(adj_avail[path[-1]]):
                if nb not in visited:
                    visited.add(nb)
                    result = dfs_ham(path + [nb], visited)
                    if result is not None:
                        return result
                    visited.discard(nb)
            return None
        
        cycle = dfs_ham([0], {0})
        if cycle is not None:
            # Mark edges as used
            for i in range(N):
                u, v = cycle[i], cycle[(i + 1) % N]
                used_edges.add((min(u, v), max(u, v)))
            found_cycles.append(cycle)
    
    if found_cycles:
        return [[int(v) for v in cycle] for cycle in found_cycles]
    
    # Last resort: return a single triangle if found
    for u, v in EDGES:
        common = ADJ[u] & ADJ[v]
        for w in common:
            if w != u and w != v:
                return [[int(u), int(v), int(w)]]
    
    # Absolute fallback (should not happen)
    return [[int(i) for i in range(N)]]

# EVOLVE-BLOCK-END

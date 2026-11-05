import time
import random
import numpy as np
from itertools import combinations

def construct():
    """Construct a dense triangle-free graph on 16 vertices that is not 3-colorable."""
    deadline = time.time() + 115  # Safety margin before 150s limit
    
    n = 16
    all_pairs = list(combinations(range(n), 2))
    
    def is_triangle_free(adj_bits):
        """Check triangle-free using bitmasks."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    common = adj_bits[u] & adj_bits[v]
                    if common:
                        return False
        return True
    
    def greedy_3_colorable(adj_bits):
        """Fast greedy 3-coloring check. Returns True if greedily colorable, False otherwise.
        This is a necessary condition for 3-colorability (if greedy fails, graph might still be 3-colorable,
        but if greedy succeeds, graph IS 3-colorable)."""
        color = [-1] * n
        # Order by degree descending
        degrees = [bin(adj_bits[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: -degrees[i])
        
        for v in order:
            used = set()
            for c in range(3):
                for u in range(n):
                    if color[u] == c and (adj_bits[v] >> u) & 1:
                        used.add(c)
                        break
            for c in range(3):
                if c not in used:
                    color[v] = c
                    break
            else:
                return False  # Can't color this vertex greedily
        return True  # Successfully greedily colored
    
    def is_3_colorable(adj_bits):
        """Check 3-colorability using backtracking with bitmasks and forward checking."""
        # Fast pre-check: if greedy coloring succeeds, it's 3-colorable
        if greedy_3_colorable(adj_bits):
            return True
        
        color = [-1] * n
        color_masks = [0, 0, 0]
        
        # Order vertices by degree (descending) for better pruning
        degrees = [bin(adj_bits[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: -degrees[i])
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            forbidden = 0
            for c in range(3):
                if color_masks[c] & adj_bits[v]:
                    forbidden |= (1 << c)
            
            # Try colors in order of least constraining value
            available = []
            for c in range(3):
                if not (forbidden & (1 << c)):
                    # Count how many future vertices this would constrain
                    constraint = 0
                    for future_idx in range(idx + 1, n):
                        fv = order[future_idx]
                        if (adj_bits[v] >> fv) & 1:
                            constraint += 1
                    available.append((constraint, c))
            
            available.sort()
            
            for _, c in available:
                color[v] = c
                color_masks[c] |= (1 << v)
                if backtrack(idx + 1):
                    return True
                color[v] = -1
                color_masks[c] &= ~(1 << v)
            return False
        
        return backtrack(0)
    
    def make_maximal(adj_bits, edges):
        """Make the triangle-free graph maximal by adding all possible edges."""
        changed = True
        while changed:
            changed = False
            for u, v in all_pairs:
                if (adj_bits[u] >> v) & 1:
                    continue
                common = adj_bits[u] & adj_bits[v]
                if not common:
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edges.append((u, v))
                    changed = True
        return adj_bits, edges
    
    def random_maximal_triangle_free_graph(density):
        """Generate a random maximal triangle-free graph."""
        adj_bits = [0] * n
        edges = []
        
        pairs = all_pairs[:]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if random.random() < density:
                common = adj_bits[u] & adj_bits[v]
                if not common:
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edges.append((u, v))
        
        # Make it maximal
        adj_bits, edges = make_maximal(adj_bits, edges)
        
        return adj_bits, edges
    
    def local_search_add_edges(adj_bits, edges, time_limit):
        """Try to add more edges while maintaining triangle-freeness and non-3-colorability."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Get candidate edges (non-edges that don't create triangles)
            candidates = []
            for u, v in all_pairs:
                if not ((adj_bits[u] >> v) & 1):
                    common = adj_bits[u] & adj_bits[v]
                    if not common:
                        candidates.append((u, v))
            
            if not candidates:
                break
            
            # Shuffle for randomness
            random.shuffle(candidates)
            
            for u, v in candidates:
                if time.time() > time_limit:
                    break
                
                # Try adding edge
                common = adj_bits[u] & adj_bits[v]
                if common:
                    continue
                
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edges.append((u, v))
                
                if not is_3_colorable(adj_bits):
                    improved = True
                else:
                    # Revert
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    edges.pop()
        
        return adj_bits, edges
    
    def simulated_annealing(adj_bits, edges, time_limit):
        """Simulated annealing to find denser triangle-free non-3-colorable graphs."""
        current_adj = adj_bits[:]
        current_edges = edges[:]
        best_adj = adj_bits[:]
        best_edges = edges[:]
        best_count = len(edges)
        
        # Temperature schedule
        t0 = 1.0
        t_end = 0.01
        cooling = 0.995
        
        temperature = t0
        
        while temperature > t_end and time.time() < time_limit:
            # Generate a neighbor: add or remove a random edge
            action = random.random()
            
            if action < 0.5:
                # Try to add an edge
                candidates = []
                for u, v in all_pairs:
                    if not ((current_adj[u] >> v) & 1):
                        common = current_adj[u] & current_adj[v]
                        if not common:
                            candidates.append((u, v))
                
                if candidates:
                    u, v = random.choice(candidates)
                    new_adj = current_adj[:]
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                    
                    # Check triangle-free
                    if is_triangle_free(new_adj):
                        new_edges = current_edges[:] + [(u, v)]
                        is_nc = not is_3_colorable(new_adj)
                        
                        if is_nc:
                            # Always accept if valid and non-3-colorable
                            current_adj = new_adj
                            current_edges = new_edges
                            if len(current_edges) > best_count:
                                best_adj = current_adj[:]
                                best_edges = current_edges[:]
                                best_count = len(current_edges)
                        else:
                            # Reject (would make it 3-colorable)
                            pass
            else:
                # Try to remove an edge (to allow other additions later)
                if current_edges:
                    u, v = random.choice(current_edges)
                    new_adj = current_adj[:]
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
                    
                    new_edges = [e for e in current_edges if e != (u, v)]
                    is_nc = not is_3_colorable(new_adj)
                    
                    # Accept removal with some probability (to escape local optima)
                    delta = len(new_edges) - len(current_edges)  # -1
                    if is_nc and delta >= 0:
                        current_adj = new_adj
                        current_edges = new_edges
                    elif not is_nc and random.random() < np.exp(delta / temperature):
                        # Accept worse solution with probability
                        current_adj = new_adj
                        current_edges = new_edges
                    elif is_nc and random.random() < np.exp(delta / temperature):
                        # Accept even if it becomes 3-colorable (rare)
                        current_adj = new_adj
                        current_edges = new_edges
            
            temperature *= cooling
        
        return best_adj, best_edges
    
    best_edges = []
    best_edge_count = 0
    
    random.seed(42)
    
    # Phase 1: Try various random maximal triangle-free graphs
    for trial in range(10000):
        if time.time() > deadline - 30:
            break
        
        density = 0.1 + 0.8 * (trial % 20) / 20
        
        adj_bits, edges = random_maximal_triangle_free_graph(density)
        edge_count = len(edges)
        
        if edge_count <= best_edge_count:
            continue
        
        if not is_triangle_free(adj_bits):
            continue
        
        if not is_3_colorable(adj_bits):
            best_edges = edges[:]
            best_edge_count = edge_count
    
    # Phase 2: Local search to add more edges
    if best_edges:
        adj_bits = [0] * n
        for u, v in best_edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        adj_bits, best_edges = local_search_add_edges(adj_bits, best_edges, deadline - 15)
        best_edge_count = len(best_edges)
    
    # Phase 3: Simulated annealing from best solution
    if best_edges and time.time() < deadline - 10:
        adj_bits = [0] * n
        for u, v in best_edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        sa_adj, sa_edges = simulated_annealing(adj_bits, best_edges, deadline - 5)
        if len(sa_edges) > best_edge_count:
            best_edges = sa_edges
            best_edge_count = len(sa_edges)
    
    # Phase 4: Final local search polish
    if best_edges and time.time() < deadline - 2:
        adj_bits = [0] * n
        for u, v in best_edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        adj_bits, best_edges = local_search_add_edges(adj_bits, best_edges, deadline - 1)
        best_edge_count = len(best_edges)
    
    return [[u, v] for u, v in best_edges]

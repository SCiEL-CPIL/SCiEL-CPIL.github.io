import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 16
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def groetzsch_edges():
        """Return edges of the Grötzsch graph (Mycielski of C5) on vertices 0..10."""
        edges = [(0,1),(1,2),(2,3),(3,4),(4,0)]
        edges += [(5,1),(5,4),(6,2),(6,0),(7,3),(7,1),(8,4),(8,2),(9,0),(9,3)]
        edges += [(10,5),(10,6),(10,7),(10,8),(10,9)]
        return edges
    
    def build_adj(edges, n):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def can_add_edge(adj, u, v, n):
        """Check if adding edge (u,v) keeps graph triangle-free."""
        return (adj[u] & adj[v]) == 0
    
    def is_not_3colorable(adj, n):
        """Check if graph is NOT 3-colorable using bitmask backtracking."""
        color = [-1] * n
        # Order vertices by degree descending
        order = sorted(range(n), key=lambda v: -adj[v].bit_count())
        
        # For fast forbidden color check, maintain color_masks
        color_masks = [0, 0, 0]  # color_masks[c] = bitmask of vertices with color c
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Forbidden colors: any color used by a neighbor
            forbidden = 0
            vmask = adj[v]
            for c in range(3):
                if vmask & color_masks[c]:
                    forbidden |= (1 << c)
            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    color_masks[c] |= (1 << v)
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
                    color_masks[c] &= ~(1 << v)
            return False
        
        return not backtrack(0)
    
    def greedy_fill(start_edges, n, deadline, seed):
        """Greedy fill: add edges in degree-ordered random order."""
        random.seed(seed)
        adj = build_adj(start_edges, n)
        cur_edges = list(start_edges)
        
        existing = set()
        for u, v in cur_edges:
            existing.add((u, v))
            existing.add((v, u))
        
        # Candidates ordered by sum of degrees (ascending - lower degree vertices first)
        candidates = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in existing:
                    deg_sum = adj[i].bit_count() + adj[j].bit_count()
                    candidates.append((deg_sum, i, j))
        
        # Shuffle with slight bias toward lower degree sums
        random.shuffle(candidates)
        candidates.sort(key=lambda x: (x[0], random.random()))
        
        for deg_sum, u, v in candidates:
            if time.time() > deadline:
                break
            if can_add_edge(adj, u, v, n):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                cur_edges.append((u, v))
                existing.add((u, v))
                existing.add((v, u))
        
        return cur_edges, adj
    
    def local_search_swap(edges, adj, n, deadline, seed):
        """Local search: try edge swaps to improve."""
        random.seed(seed)
        cur_adj = [adj[i] for i in range(n)]
        cur_edges = list(edges)
        best_count = len(cur_edges)
        
        existing = set()
        for u, v in cur_edges:
            existing.add((u, v))
            existing.add((v, u))
        
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in existing]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(non_edges)
            random.shuffle(cur_edges)
            
            for u, v in cur_edges[:]:  # Try removing each edge
                if time.time() > deadline:
                    break
                
                # Remove edge (u,v)
                cur_adj[u] &= ~(1 << v)
                cur_adj[v] &= ~(1 << u)
                cur_edges.remove((u, v))
                existing.discard((u, v))
                existing.discard((v, u))
                
                # Try adding a non-edge
                found = False
                for a, b in non_edges[:]:
                    if time.time() > deadline:
                        break
                    if can_add_edge(cur_adj, a, b, n):
                        cur_adj[a] |= (1 << b)
                        cur_adj[b] |= (1 << a)
                        cur_edges.append((a, b))
                        existing.add((a, b))
                        existing.add((b, a))
                        non_edges.remove((a, b))
                        found = True
                        improved = True
                        if len(cur_edges) > best_count:
                            best_count = len(cur_edges)
                        break
                
                if not found:
                    # Restore original edge
                    cur_adj[u] |= (1 << v)
                    cur_adj[v] |= (1 << u)
                    cur_edges.append((u, v))
                    existing.add((u, v))
                    existing.add((v, u))
                    non_edges.append((u, v))
        
        return cur_edges, cur_adj
    
    best_edges = []
    best_count = 0
    
    # Phase 1: Greedy fill from Grötzsch core with multiple seeds
    core_edges = groetzsch_edges()
    core_adj = build_adj(core_edges, n)
    
    if not is_not_3colorable(core_adj, n):
        return core_edges
    
    phase1_deadline = start_time + 60.0
    for seed in range(50):
        if time.time() > phase1_deadline:
            break
        
        edges, adj = greedy_fill(core_edges, n, phase1_deadline, seed)
        
        if is_not_3colorable(adj, n):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = [list(e) for e in edges]
    
    # Phase 2: Local search from best solutions
    phase2_deadline = start_time + 140.0
    if best_edges and time.time() < phase2_deadline:
        best_adj = build_adj(best_edges, n)
        for seed in range(30):
            if time.time() > phase2_deadline:
                break
            
            edges2, adj2 = local_search_swap(best_edges, best_adj, n, phase2_deadline, seed + 10000)
            
            if is_not_3colorable(adj2, n):
                if len(edges2) > best_count:
                    best_count = len(edges2)
                    best_edges = [list(e) for e in edges2]
                    best_adj = adj2
    
    if not best_edges:
        return core_edges
    
    return best_edges
# EVOLVE-BLOCK-END

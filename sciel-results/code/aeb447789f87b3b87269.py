import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 20
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m_total = len(all_edges)
    
    # Map edge to index
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
        
    # Precompute for each edge (u,v), which other edges (x,y) would create a K2,3 if both present?
    # Actually, K2,3 constraint: No two vertices have 3 common neighbors.
    # Let's define the constraint on the set of edges S.
    # For any pair of vertices (u, v), let N(u) be neighbors.
    # Constraint: |N(u) ∩ N(v)| <= 2 for all u != v.
    
    # This is a global constraint on the set of edges.
    # However, we can check validity incrementally.
    
    def popcount(x):
        return bin(x).count('1')
        
    class CSP:
        def __init__(self):
            self.adj = [0] * n
            self.edges = []
            self.edge_set = set()
            self.best_edges = []
            self.best_count = 0
            self.nodes_visited = 0
            
        def can_add(self, u, v):
            # Check if adding (u,v) violates K2,3
            # 1. Common neighbors of u and v must be <= 2
            common = self.adj[u] & self.adj[v]
            if popcount(common) > 2:
                return False
            
            # 2. For each neighbor w of v (w != u), 
            #    if we add v to N(u), then v becomes a common neighbor of u and w.
            #    So |N(u) ∩ N(w)| must be <= 1 currently.
            nv = self.adj[v]
            w = nv
            while w:
                low = w & (-w)
                wi = low.bit_length() - 1
                w ^= low
                if wi == u:
                    continue
                common_uw = self.adj[u] & self.adj[wi]
                if popcount(common_uw) > 1:
                    return False
            
            # 3. For each neighbor w of u (w != v),
            #    if we add u to N(v), then u becomes a common neighbor of v and w.
            #    So |N(v) ∩ N(w)| must be <= 1 currently.
            nu = self.adj[u]
            w = nu
            while w:
                low = w & (-w)
                wi = low.bit_length() - 1
                w ^= low
                if wi == v:
                    continue
                common_vw = self.adj[v] & self.adj[wi]
                if popcount(common_vw) > 1:
                    return False
            
            return True
            
        def add_edge(self, u, v):
            self.adj[u] |= (1 << v)
            self.adj[v] |= (1 << u)
            self.edges.append((u, v))
            self.edge_set.add((u, v))
            
        def remove_edge(self, u, v):
            self.adj[u] &= ~(1 << v)
            self.adj[v] &= ~(1 << u)
            self.edges.remove((u, v))
            self.edge_set.remove((u, v))
            
        def get_missing_edges(self):
            missing = []
            for u in range(n):
                for v in range(u+1, n):
                    if not (self.adj[u] & (1 << v)):
                        missing.append((u, v))
            return missing
            
        def upper_bound(self):
            # Simple upper bound: remaining possible edges.
            # We can improve this by counting "tight" pairs.
            # A pair (u,v) with 2 common neighbors is "tight". 
            # It cannot accept any more common neighbors.
            # This doesn't directly limit total edges easily without complex counting.
            # Let's use a looser bound: m_total - len(self.edges)
            return m_total - len(self.edges)
            
        def solve(self, edges_to_consider, depth=0):
            if time.time() > deadline:
                return
                
            self.nodes_visited += 1
            if self.nodes_visited % 1000 == 0:
                # print(f"Depth {depth}, Edges {len(self.edges)}, Best {self.best_count}")
                pass
                
            if len(self.edges) > self.best_count:
                self.best_count = len(self.edges)
                self.best_edges = list(self.edges)
                
            if not edges_to_consider:
                return
                
            # Heuristic: Try edges that are "most constrained" or have highest degree potential?
            # Let's just iterate in a fixed order but allow skipping.
            # To make it a search, we branch: Include edge OR Exclude edge.
            # But excluding edges is usually not useful for maximization unless it unlocks other edges.
            # Standard Max-CSP approach: 
            # 1. Pick an edge e.
            # 2. Branch 1: Add e (if valid). Recurse on remaining edges.
            # 3. Branch 2: Skip e. Recurse on remaining edges.
            # Pruning: If len(current_edges) + upper_bound(remaining) <= best_count, prune.
            
            # Optimization: Instead of binary search on all edges, use a greedy-like selection
            # but with backtracking.
            # Let's pick the first edge in the list.
            u, v = edges_to_consider[0]
            remaining = edges_to_consider[1:]
            
            # Branch 1: Include (u,v)
            if self.can_add(u, v):
                self.add_edge(u, v)
                # Recurse
                self.solve(remaining, depth + 1)
                self.remove_edge(u, v)
                
                # Pruning check for Branch 2
                # If even adding all remaining edges doesn't beat best, skip Branch 2?
                # len(self.edges) + len(remaining) <= self.best_count
                if len(self.edges) + len(remaining) <= self.best_count:
                    return
                    
            # Branch 2: Exclude (u,v)
            # Only useful if adding (u,v) blocked many other edges.
            # This is where the search power comes from.
            self.solve(remaining, depth + 1)
            
        def solve_greedy_backtrack(self, edges_to_consider):
            # A hybrid: Greedy fill, but if stuck, backtrack?
            # Actually, simple backtracking on 190 edges is too slow if not pruned well.
            # Let's use a "Limited Discrepancy Search" or "Iterated Local Search" on the CSP state?
            # No, let's stick to pure Backtracking with strong pruning.
            
            # Reorder edges to improve pruning?
            # Edges that are part of many potential K2,3s should be considered first.
            # For now, use the provided order.
            self.solve(edges_to_consider)

    # Strategy: 
    # 1. Start with a good seed (from a fast greedy) to raise the lower bound (best_count).
    # 2. Use Backtracking Search to try to exceed that bound.
    
    # Step 1: Fast Greedy Seed
    def greedy_seed():
        adj = [0] * n
        edges = []
        # Order edges by something? Random or fixed.
        # Let's use a deterministic "minimum common neighbors" heuristic for the seed.
        # But for speed, just random shuffle and greedy add.
        rng = random.Random(123)
        shuffled_edges = all_edges[:]
        rng.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            # Check validity
            common = adj[u] & adj[v]
            if popcount(common) > 2:
                continue
            
            valid = True
            nv = adj[v]
            w = nv
            while w:
                low = w & (-w)
                wi = low.bit_length() - 1
                w ^= low
                if wi == u: continue
                if popcount(adj[u] & adj[wi]) > 1:
                    valid = False
                    break
            if not valid: continue
            
            nu = adj[u]
            w = nu
            while w:
                low = w & (-w)
                wi = low.bit_length() - 1
                w ^= low
                if wi == v: continue
                if popcount(adj[v] & adj[wi]) > 1:
                    valid = False
                    break
            if not valid: continue
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append((u, v))
            
        return edges, adj

    seed_edges, seed_adj = greedy_seed()
    print(f"Greedy seed edges: {len(seed_edges)}")
    
    # Step 2: Backtracking Search to improve
    # We need to be careful with recursion depth and time.
    # n=20, m=190. Depth 190 is fine for Python if we use sys.setrecursionlimit.
    sys.setrecursionlimit(300)
    
    csp = CSP()
    # Initialize with seed? No, backtracking usually starts from empty or a specific state.
    # If we start from empty, we might waste time.
    # If we start from seed, we need to "unmake" decisions? 
    # Better: Run backtracking from empty, but use the seed count as the initial best_count.
    # This allows aggressive pruning.
    
    csp.best_count = len(seed_edges)
    csp.best_edges = list(seed_edges)
    
    # To make backtracking effective, we should order edges such that "conflicting" edges are considered earlier.
    # Or, we can use the greedy order but allow skipping.
    
    # Let's try a different approach: 
    # The search space is huge. Pure backtracking might be too slow.
    # Let's use "Branch and Bound" with a stronger heuristic for edge ordering.
    # Sort edges by how "central" they are? 
    # Actually, let's just run the backtracking on the full list of edges.
    # The pruning `len(self.edges) + len(remaining) <= self.best_count` is very effective 
    # if `self.best_count` is close to the optimum.
    
    # If greedy found 190 edges (max possible is 190, but K2,3 free max is lower, maybe ~80-100?),
    # let's check typical ex(20, K2,3). 
    # ex(n, K2,3) is approx n^2/4 for large n? No, K2,3 is a tree? No, K2,3 is bipartite.
    # Zarankiewicz problem. z(20, 20; 2, 3).
    # Approx n^(3/2) ? 
    # If seed is around 80, and max is 90, backtracking has to search a lot.
    
    # Let's optimize the `solve` function.
    # Use an array for adjacency to speed up `can_add`? 
    # The current `can_add` is O(n) due to bit loops.
    
    # Let's try to run the backtracking.
    # To avoid timeout, we will stop if time is up.
    
    # One issue: `edges_to_consider` is a list slice, which is O(n) copy.
    # Use an index pointer instead.
    
    class OptimizedCSP:
        def __init__(self):
            self.adj = [0] * n
            self.edges = []
            self.best_edges = []
            self.best_count = 0
            self.edge_list = all_edges # Fixed order
            
        def can_add(self, u, v):
            common = self.adj[u] & self.adj[v]
            if popcount(common) > 2:
                return False
            
            nv = self.adj[v]
            w = nv
            while w:
                low = w & (-w)
                wi = low.bit_length() - 1
                w ^= low
                if wi == u:
                    continue
                common_uw = self.adj[u] & self.adj[wi]
                if popcount(common_uw) > 1:
                    return False
            
            nu = self.adj[u]
            w = nu
            while w:
                low = w & (-w)
                wi = low.bit_length() - 1
                w ^= low
                if wi == v:
                    continue
                common_vw = self.adj[v] & self.adj[wi]
                if popcount(common_vw) > 1:
                    return False
            
            return True
            
        def add_edge(self, u, v):
            self.adj[u] |= (1 << v)
            self.adj[v] |= (1 << u)
            self.edges.append((u, v))
            
        def remove_edge(self, u, v):
            self.adj[u] &= ~(1 << v)
            self.adj[v] &= ~(1 << u)
            self.edges.pop()
            
        def solve(self, idx):
            if time.time() > deadline:
                return
                
            if len(self.edges) > self.best_count:
                self.best_count = len(self.edges)
                self.best_edges = list(self.edges)
                
            if idx >= m_total:
                return
                
            u, v = self.edge_list[idx]
            
            # Branch 1: Include
            if self.can_add(u, v):
                self.add_edge(u, v)
                self.solve(idx + 1)
                self.remove_edge(u, v)
                
                # Pruning
                if len(self.edges) + (m_total - idx - 1) <= self.best_count:
                    return
            
            # Branch 2: Exclude
            self.solve(idx + 1)
            
    opt_csp = OptimizedCSP()
    opt_csp.best_count = len(seed_edges)
    opt_csp.best_edges = list(seed_edges)
    
    # Run the solver
    opt_csp.solve(0)
    
    if len(opt_csp.best_edges) > len(seed_edges):
        return [list(e) for e in opt_csp.best_edges]
    else:
        return [list(e) for e in seed_edges]

# Keep the interface
def construct():
    # The logic above is inside construct? 
    # Yes, I defined everything inside construct in the thought process.
    # I need to wrap the code above into the function.
    
    start_time = time.time()
    deadline = start_time + 145
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m_total = len(all_edges)
    
    def popcount(x):
        return bin(x).count('1')
        
    def greedy_seed():
        adj = [0] * n
        edges = []
        rng = random.Random(123)
        shuffled_edges = all_edges[:]
        rng.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            common = adj[u] & adj[v]
            if popcount(common) > 2:
                continue
            
            valid = True
            nv = adj[v]
            w = nv
            while w:
                low = w & (-w)
                wi = low.bit_length() - 1
                w ^= low
                if wi == u: continue
                if popcount(adj[u] & adj[wi]) > 1:
                    valid = False
                    break
            if not valid: continue
            
            nu = adj[u]
            w = nu
            while w:
                low = w & (-w)
                wi = low.bit_length() - 1
                w ^= low
                if wi == v: continue
                if popcount(adj[v] & adj[wi]) > 1:
                    valid = False
                    break
            if not valid: continue
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append((u, v))
            
        return edges, adj

    seed_edges, seed_adj = greedy_seed()
    
    sys.setrecursionlimit(300)
    
    class OptimizedCSP:
        def __init__(self):
            self.adj = [0] * n
            self.edges = []
            self.best_edges = []
            self.best_count = 0
            self.edge_list = all_edges
            
        def can_add(self, u, v):
            common = self.adj[u] & self.adj[v]
            if popcount(common) > 2:
                return False
            
            nv = self.adj[v]
            w = nv
            while w:
                low = w & (-w)
                wi = low.bit_length() - 1
                w ^= low
                if wi == u:
                    continue
                common_uw = self.adj[u] & self.adj[wi]
                if popcount(common_uw) > 1:
                    return False
            
            nu = self.adj[u]
            w = nu
            while w:
                low = w & (-w)
                wi = low.bit_length() - 1
                w ^= low
                if wi == v:
                    continue
                common_vw = self.adj[v] & self.adj[wi]
                if popcount(common_vw) > 1:
                    return False
            
            return True
            
        def add_edge(self, u, v):
            self.adj[u] |= (1 << v)
            self.adj[v] |= (1 << u)
            self.edges.append((u, v))
            
        def remove_edge(self, u, v):
            self.adj[u] &= ~(1 << v)
            self.adj[v] &= ~(1 << u)
            self.edges.pop()
            
        def solve(self, idx):
            if time.time() > deadline:
                return
                
            if len(self.edges) > self.best_count:
                self.best_count = len(self.edges)
                self.best_edges = list(self.edges)
                
            if idx >= m_total:
                return
                
            u, v = self.edge_list[idx]
            
            # Branch 1: Include
            if self.can_add(u, v):
                self.add_edge(u, v)
                self.solve(idx + 1)
                self.remove_edge(u, v)
                
                # Pruning
                if len(self.edges) + (m_total - idx - 1) <= self.best_count:
                    return
            
            # Branch 2: Exclude
            self.solve(idx + 1)
            
    opt_csp = OptimizedCSP()
    opt_csp.best_count = len(seed_edges)
    opt_csp.best_edges = list(seed_edges)
    
    # Run the solver
    opt_csp.solve(0)
    
    if len(opt_csp.best_edges) > len(seed_edges):
        return [list(e) for e in opt_csp.best_edges]
    else:
        return [list(e) for e in seed_edges]

# The actual construct function for the module
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m_total = len(all_edges)
    
    def popcount(x):
        return bin(x).count('1')
        
    def greedy_seed():
        adj = [0] * n
        edges = []
        rng = random.Random(123)
        shuffled_edges = all_edges[:]
        rng.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            common = adj[u] & adj[v]
            if popcount(common) > 2:
                continue
            
            valid = True
            nv = adj[v]
            w = nv
            while w:
                low = w & (-w)
                wi = low.bit_length() - 1
                w ^= low
                if wi == u: continue
                if popcount(adj[u] & adj[wi]) > 1:
                    valid = False
                    break
            if not valid: continue
            
            nu = adj[u]
            w = nu
            while w:
                low = w & (-w)
                wi = low.bit_length() - 1
                w ^= low
                if wi == v: continue
                if popcount(adj[v] & adj[wi]) > 1:
                    valid = False
                    break
            if not valid: continue
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append((u, v))
            
        return edges, adj

    seed_edges, seed_adj = greedy_seed()
    
    sys.setrecursionlimit(300)
    
    class OptimizedCSP:
        def __init__(self):
            self.adj = [0] * n
            self.edges = []
            self.best_edges = []
            self.best_count = 0
            self.edge_list = all_edges
            
        def can_add(self, u, v):
            common = self.adj[u] & self.adj[v]
            if popcount(common) > 2:
                return False
            
            nv = self.adj[v]
            w = nv
            while w:
                low = w & (-w)
                wi = low.bit_length() - 1
                w ^= low
                if wi == u:
                    continue
                common_uw = self.adj[u] & self.adj[wi]
                if popcount(common_uw) > 1:
                    return False
            
            nu = self.adj[u]
            w = nu
            while w:
                low = w & (-w)
                wi = low.bit_length() - 1
                w ^= low
                if wi == v:
                    continue
                common_vw = self.adj[v] & self.adj[wi]
                if popcount(common_vw) > 1:
                    return False
            
            return True
            
        def add_edge(self, u, v):
            self.adj[u] |= (1 << v)
            self.adj[v] |= (1 << u)
            self.edges.append((u, v))
            
        def remove_edge(self, u, v):
            self.adj[u] &= ~(1 << v)
            self.adj[v] &= ~(1 << u)
            self.edges.pop()
            
        def solve(self, idx):
            if time.time() > deadline:
                return
                
            if len(self.edges) > self.best_count:
                self.best_count = len(self.edges)
                self.best_edges = list(self.edges)
                
            if idx >= m_total:
                return
                
            u, v = self.edge_list[idx]
            
            # Branch 1: Include
            if self.can_add(u, v):
                self.add_edge(u, v)
                self.solve(idx + 1)
                self.remove_edge(u, v)
                
                # Pruning
                if len(self.edges) + (m_total - idx - 1) <= self.best_count:
                    return
            
            # Branch 2: Exclude
            self.solve(idx + 1)
            
    opt_csp = OptimizedCSP()
    opt_csp.best_count = len(seed_edges)
    opt_csp.best_edges = list(seed_edges)
    
    # Run the solver
    opt_csp.solve(0)
    
    if len(opt_csp.best_edges) > len(seed_edges):
        return [list(e) for e in opt_csp.best_edges]
    else:
        return [list(e) for e in seed_edges]

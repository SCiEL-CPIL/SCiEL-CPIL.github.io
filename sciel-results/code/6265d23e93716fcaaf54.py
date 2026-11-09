"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a C5-free graph with no independent set of size 9, maximizing n.
    
    Strategy: 
    1. Start with known constructions (8*K4 = 32 vertices)
    2. Use local search to try to add vertices or modify graph for n=33,34,35
    3. Fast C5 detection via matrix multiplication
    4. Exact independence number check via branch and bound
    """
    deadline = time.time() + 130  # 130s to leave 20s safety margin
    
    def make_baseline():
        """Return 8 disjoint K4s: n=32, C5-free, alpha=8."""
        n = 32
        edges = []
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}
    
    def has_c5_fast(adj, n):
        """Check if graph has a 5-cycle using matrix multiplication.
        A 5-cycle exists iff trace(A^5) > 0 where A is adjacency matrix.
        Actually, we need to check for simple 5-cycles. 
        Number of closed walks of length 5 = trace(A^5).
        Each 5-cycle contributes 10 to trace(A^5) (5 starting points * 2 directions).
        So if trace(A^5) > 0, there's a 5-cycle (or shorter cycles that close in 5 steps).
        But for simple graphs, closed walks of length 5 that aren't 5-cycles would involve
        backtracking, which contributes even numbers. Let me think more carefully.
        
        Actually, for a simple graph with no multiple edges, trace(A^5) counts closed walks
        of length 5. A 5-cycle contributes 10. A triangle with a repeated edge would contribute
        but in a simple graph, you can't have repeated edges in a walk unless you go back and forth.
        
        Let me just use the direct enumeration approach but optimized.
        """
        # Use the fact that C5 exists iff there are vertices a,b,c,d,e with 
        # a-b-c-d-e-a all edges, all distinct.
        # For each pair (a,b) that is an edge, count paths of length 3 from b to a
        # that don't use a or b as intermediate vertices.
        
        # Alternative: for each vertex v, look at pairs of neighbors (u,w) of v.
        # If u and w have a common neighbor x (not v), and x has a neighbor y (not v,u,w),
        # and y is adjacent to v, we have a 5-cycle. This is getting complex.
        
        # Let's use matrix approach: trace(A^5) / 10 gives number of 5-cycles
        # if there are no shorter cycles contributing. But triangles contribute to A^3,
        # and A^5 includes walks that go through triangles.
        
        # Safer: just enumerate. For n<=35, C(n,5) = 324632 max.
        # For each 5-set, check if it contains a C5.
        
        adj_np = adj  # already numpy or list
        
        for combo in combinations(range(n), 5):
            v0, v1, v2, v3, v4 = combo
            # Check all 12 distinct 5-cycles on these 5 vertices
            # The cycles are: 0-1-2-3-4-0 and its rotations/reversals
            # But we just need to check if ANY permutation forms a cycle.
            # A 5-cycle on vertices {a,b,c,d,e} means there's a cyclic ordering.
            # Check if the induced subgraph on these 5 vertices is exactly a C5
            # (each vertex has degree 2 in the induced subgraph, and it's connected).
            
            # Get degrees in induced subgraph
            degs = [0, 0, 0, 0, 0]
            # Edge (v0,v1)
            if adj[v0][v1]:
                degs[0] += 1; degs[1] += 1
            if adj[v0][v2]:
                degs[0] += 1; degs[2] += 1
            if adj[v0][v3]:
                degs[0] += 1; degs[3] += 1
            if adj[v0][v4]:
                degs[0] += 1; degs[4] += 1
            if adj[v1][v2]:
                degs[1] += 1; degs[2] += 1
            if adj[v1][v3]:
                degs[1] += 1; degs[3] += 1
            if adj[v1][v4]:
                degs[1] += 1; degs[4] += 1
            if adj[v2][v3]:
                degs[2] += 1; degs[3] += 1
            if adj[v2][v4]:
                degs[2] += 1; degs[4] += 1
            if adj[v3][v4]:
                degs[3] += 1; degs[4] += 1
            
            if degs != [2, 2, 2, 2, 2]:
                continue
            
            # Check connectivity (it's a 2-regular graph on 5 vertices, so it's either C5 or C3+C2, but C2 not possible in simple graph, so must be C5)
            # Actually 2-regular on 5 vertices: only C5 is possible (since 5 is odd, can't split into cycles of length >= 3)
            return True
        
        return False
    
    def has_independent_set_k(adj, n, k):
        """Check if there's an independent set of size k using branch and bound."""
        if k <= 0:
            return True
        if n < k:
            return False
        
        # Use recursive backtracking with pruning
        # State: (vertices_available, remaining_k)
        # vertices_available is a list of vertex indices
        
        def backtrack(avail, rem):
            if rem == 0:
                return True
            if len(avail) < rem:
                return False
            
            # Pick a vertex to branch on
            v = avail[0]
            rest = avail[1:]
            
            # Option 1: don't include v
            if backtrack(rest, rem):
                return True
            
            # Option 2: include v, remove v and all its neighbors from avail
            new_avail = [u for u in rest if not adj[v][u]]
            if backtrack(new_avail, rem - 1):
                return True
            
            return False
        
        return backtrack(list(range(n)), k)
    
    def alpha_exact(adj, n):
        """Compute exact independence number."""
        # Binary search or just check if alpha >= 9
        # For our purposes, we just need alpha < 9, i.e., alpha <= 8
        if has_independent_set_k(adj, n, 9):
            return 9  # at least 9
        else:
            # Find the actual alpha
            for k in range(8, 0, -1):
                if has_independent_set_k(adj, n, k):
                    return k
            return 1
    
    def is_valid(adj, n):
        """Check if graph is C5-free and has alpha < 9."""
        if has_c5_fast(adj, n):
            return False
        if has_independent_set_k(adj, n, 9):
            return False
        return True
    
    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges
    
    # Start with baseline
    baseline = make_baseline()
    best_result = baseline
    best_n = 32
    
    # Build adjacency for baseline
    adj = [[False] * 32 for _ in range(32)]
    for e in baseline['edges']:
        adj[e[0]][e[1]] = adj[e[1]][e[0]] = True
    
    # Try to improve: local search on the edge set
    # For each target n from 35 down to 33:
    for target_n in [35, 34, 33]:
        if time.time() > deadline:
            break
        
        # Start from the 32-vertex graph and add vertices
        # New vertices start with no edges, then we add edges to reduce alpha
        for trial in range(30):
            if time.time() > deadline:
                break
            
            # Initialize: copy the 32-vertex graph, add new vertices
            adj_t = [[False] * target_n for _ in range(target_n)]
            for i in range(32):
                for j in range(32):
                    adj_t[i][j] = adj[i][j]
            
            # New vertices 32..target_n-1 start with random edges
            random.seed(trial * 1000 + target_n)
            for i in range(32, target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.3:
                        adj_t[i][j] = adj_t[j][i] = True
                # Also add some edges from new vertices to old ones
                for j in range(32):
                    if random.random() < 0.2:
                        adj_t[i][j] = adj_t[j][i] = True
            
            # Local search: try to fix C5s and reduce alpha
            for iteration in range(2000):
                if time.time() > deadline:
                    break
                
                # Check current state
                c5_exists = has_c5_fast(adj_t, target_n)
                indep_9 = has_independent_set_k(adj_t, target_n, 9)
                
                if not c5_exists and not indep_9:
                    # Valid!
                    edges = edges_from_adj(adj_t, target_n)
                    result = {'n': target_n, 'edges': edges}
                    if target_n > best_n:
                        best_result = result
                        best_n = target_n
                    break
                
                # Try a random edge flip
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                
                old_val = adj_t[u][v]
                adj_t[u][v] = adj_t[v][u] = not old_val
                
                # Check if this move is acceptable (don't make things much worse)
                # Simple: accept if it doesn't create a C5 when we didn't have one,
                # or if it reduces alpha issues
                new_c5 = has_c5_fast(adj_t, target_n)
                new_indep = has_independent_set_k(adj_t, target_n, 9)
                
                # Accept if we improved or stayed the same
                if (c5_exists and not new_c5) or (not indep_9 and new_indep):
                    # Revert - this made things worse
                    adj_t[u][v] = adj_t[v][u] = old_val
                elif c5_exists and new_c5 and indep_9 and new_indep:
                    # Both still bad, keep the change (random walk)
                    pass
                else:
                    # Improvement or neutral, keep
                    pass
        
        if best_n >= target_n:
            break
    
    return best_result
# EVOLVE-BLOCK-END

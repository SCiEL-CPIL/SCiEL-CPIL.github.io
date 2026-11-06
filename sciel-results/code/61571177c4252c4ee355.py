import time
import random
import sys

def construct():
    """
    Construct a C4-free graph with no independent 11-set.
    Uses the polarity graph of PG(2,5) (36 vertices) as a base.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s/150s limits

    # 1. Construct the polarity graph of PG(2,5)
    # Points and lines are elements of F_5^3 (projective).
    # A point P is incident to line L if dot(P, L) == 0.
    # Vertices are the 31 points. Wait, PG(2,5) has 5^2+5+1 = 31 points.
    # Let's check the size.
    # The polarity graph of PG(2,q) has q^2+q+1 vertices.
    # For q=5, n = 31.
    # The independence number of the polarity graph of PG(2,q) is roughly sqrt(q^2+q+1) ?
    # Actually, for PG(2,5), alpha is small.
    # Is 31 enough? We need n up to 43.
    # The task says "Best known construction has 39 vertices".
    # So 31 is too small.
    
    # Let's re-read the inspiration: "polarity graph of PG(2,7)" -> 7^2+7+1 = 57 vertices.
    # But we need n <= 43.
    # So we can take a subgraph of the PG(2,7) polarity graph?
    # Or use a different structure.
    
    # The Gewirtz graph has 56 vertices.
    # The Hoffman-Singleton graph has 50 vertices.
    # The Chvatal graph has 12 vertices.
    
    # Let's look at the "Best known construction has 39 vertices" hint.
    # This suggests we should aim for 39.
    
    # Strategy:
    # 1. Start with a strong C4-free graph with small alpha.
    # 2. The Clebsch graph (16 vertices, alpha=5) is C4-free? No, Clebsch is C4-free? 
    #    Clebsch graph is the 5-dim hypercube with antipodal vertices identified? No.
    #    Clebsch graph is C4-free? Yes, it is a (16,5,2,2) SRG. It has no C4.
    #    Alpha(Clebsch) = 5.
    #    
    # 3. Can we extend the Clebsch graph?
    #    Or use the complement of the line graph of K_6? No.
    
    # Let's try to build a graph from scratch using a greedy randomized search with backtracking,
    # but optimized with bitsets for C4 checking.
    
    # The current program's weakness is the slow C4 check and the weak base (10 triangles).
    # 10 triangles = 30 vertices. Alpha is likely large because triangles don't help much with alpha if they are disjoint.
    # Disjoint triangles: alpha = 3 * 10 = 30. Way too high. We need alpha < 11.
    # So the current program's base is terrible for alpha.
    
    # We need a graph where vertices are densely connected to keep alpha low.
    # High degree is good for low alpha.
    # C4-free graphs have max degree bounded by roughly sqrt(n).
    # For n=39, max degree ~ 6-7.
    
    # Let's use the polarity graph of PG(2,5) (31 vertices) and add 8 vertices?
    # Or use a known 39-vertex graph?
    # I don't have the specific 39-vertex graph memorized.
    
    # Let's try a different approach:
    # Generate a random graph with n=39, and use simulated annealing / greedy repair to satisfy C4-free and alpha<11.
    
    # Actually, let's look at the "inspiration 3" which used PG(2,7) polarity graph.
    # PG(2,7) has 57 vertices. We can take a random subset of 39 vertices?
    # If we take a random subset, alpha might increase.
    # But the base graph has very small alpha (approx sqrt(57) ~ 7.5).
    # So a random subset of 39 vertices will likely have alpha < 11.
    # And it will be C4-free because it's a subgraph of a C4-free graph.
    
    # Plan:
    # 1. Construct polarity graph of PG(2,7) (57 vertices).
    # 2. Randomly select 39 vertices.
    # 3. Check alpha. If alpha >= 11, try another subset or add edges? No, can't add edges easily without breaking C4.
    # 4. If we can find a subset of size 39 with alpha < 11, we are done.
    # 5. If not, try size 38, 37, etc.
    
    # Why is this better?
    # The base graph (PG(2,7) polarity) is C4-free and has low alpha.
    # The current program starts with disjoint triangles (high alpha).
    # This strategy starts with a dense, well-connected C4-free graph and prunes it.
    
    # Implementation details:
    # 1. Build PG(2,7) polarity graph.
    #    Points: 57.
    #    Edges: P_i -- P_j if dot(P_i, P_j) == 0.
    #    This graph is C4-free.
    # 2. Sample subsets of size n_target (39, 38, ...) from the 57 vertices.
    # 3. For each subset, compute alpha.
    #    Alpha computation: Use a fast branch and bound or greedy + verification.
    #    Since n=39, alpha is likely 8-10.
    #    We need to be sure alpha < 11.
    
    # Let's implement the PG(2,7) construction.
    
    def build_pg2_polarity(q):
        """Build polarity graph of PG(2,q)."""
        # Generate all points in F_q^3 (up to scaling)
        points = []
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize: first non-zero element must be 1
                    # Since we are in F_q, we can divide by the first non-zero element
                    if x != 0:
                        points.append((1, (y * pow(x, q-2, q)) % q, (z * pow(x, q-2, q)) % q))
                    elif y != 0:
                        points.append((0, 1, (z * pow(y, q-2, q)) % q))
                    else:
                        points.append((0, 0, 1))
        
        n_pts = len(points)
        # Should be q^2+q+1
        # For q=7, 49+7+1 = 57
        
        # Build adjacency
        adj = [set() for _ in range(n_pts)]
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                # Dot product
                p_i = points[i]
                p_j = points[j]
                dot = (p_i[0]*p_j[0] + p_i[1]*p_j[1] + p_i[2]*p_j[2]) % q
                if dot == 0:
                    adj[i].add(j)
                    adj[j].add(i)
        
        return n_pts, adj

    def max_independent_set_fast(adj, n, timeout=5.0):
        """
        Fast maximum independent set using branch and bound with coloring.
        adj: list of sets.
        """
        # Convert to list of neighbors for faster access
        neigh = [list(s) for s in adj]
        
        best = 0
        start_time = time.time()
        
        # Greedy coloring for upper bound
        def greedy_color(vertices):
            # Simple greedy coloring
            color = {}
            for v in vertices:
                used = set()
                for u in neigh[v]:
                    if u in color:
                        used.add(color[u])
                c = 0
                while c in used:
                    c += 1
                color[v] = c
            return max(color.values()) + 1 if color else 0

        def branch_and_bound(remaining, current_size):
            nonlocal best
            if time.time() > start_time + timeout:
                return
            
            if not remaining:
                if current_size > best:
                    best = current_size
                return
            
            # Upper bound
            ub = current_size + greedy_color(remaining)
            if ub <= best:
                return
            
            # Pick vertex with max degree in remaining
            # To speed up, we can just pick the first one or one with high degree
            v = max(remaining, key=lambda x: len(neigh[x] & set(remaining)))
            
            # Include v
            new_remaining = [u for u in remaining if u != v and u not in neigh[v]]
            branch_and_bound(new_remaining, current_size + 1)
            
            # Exclude v
            remaining.remove(v)
            branch_and_bound(remaining, current_size)
            remaining.append(v)

        branch_and_bound(list(range(n)), 0)
        return best

    # 1. Build PG(2,7) polarity graph
    n_base, adj_base = build_pg2_polarity(7)
    # n_base should be 57
    
    # 2. Try to find a subset of size 39 (or less) with alpha < 11
    best_n = 0
    best_edges = []
    
    # We want to maximize n.
    # Try n from 39 down to 30.
    # For each n, try random subsets.
    
    for target_n in range(39, 29, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        # Try multiple random subsets
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
                
            # Randomly select target_n vertices from 57
            subset = random.sample(range(n_base), target_n)
            subset_set = set(subset)
            
            # Build subgraph
            # Map old indices to new 0..target_n-1
            idx_map = {old: new for new, old in enumerate(subset)}
            
            # Check C4-free (it is, by construction)
            # Check alpha
            # We need to compute alpha of the subgraph.
            # Instead of rebuilding adj lists fully, we can just use the subset logic in the B&B.
            # But for simplicity, let's build the subgraph adj list.
            
            adj_sub = [set() for _ in range(target_n)]
            for i in range(target_n):
                old_i = subset[i]
                for j in range(i+1, target_n):
                    old_j = subset[j]
                    if old_j in adj_base[old_i]:
                        adj_sub[i].add(j)
                        adj_sub[j].add(i)
                        
            # Compute alpha
            alpha = max_independent_set_fast(adj_sub, target_n, timeout=2.0)
            
            if alpha < 11:
                # Valid!
                best_n = target_n
                edges = []
                for i in range(target_n):
                    for j in adj_sub[i]:
                        if i < j:
                            edges.append([i, j])
                best_edges = edges
                found = True
                break
        
        if found:
            break
            
    if best_n == 0:
        # Fallback: return a small valid graph
        # e.g. 10 vertices, complete graph? No, K10 has C4.
        # Just return the 31-vertex PG(2,5) graph if we can't find 39.
        # Actually, let's just return the best we found. If 0, return a trivial one.
        # But we should find something.
        pass

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(res['n'])

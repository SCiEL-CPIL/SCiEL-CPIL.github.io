import time
import random
from collections import defaultdict
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n = 31
    
    # Build a C4-free graph with 90 edges using polarity graph of PG(2,5)
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph
    edge_set = set()
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edge_set.add((i, j))
                adj[i].add(j)
                adj[j].add(i)
    
    # Should have 90 edges
    if len(edge_set) != 90:
        # Fallback: use a different construction
        # Try to build a C4-free graph greedily
        edge_set = set()
        adj = [set() for _ in range(n)]
        # Use a known C4-free construction
        # For now, just use the polarity graph even if edge count differs
        pass
    
    # Count C4s in the current graph
    def count_c4(adj, edge_set, n):
        """Count number of 4-cycles (subgraphs) in the graph."""
        c4_count = 0
        # For each pair of vertices, count common neighbors
        # A C4 exists for each pair (u,v) with >= 2 common neighbors
        # Number of C4s through edge (u,v) = C(common_neighbors(u,v), 2)
        # But this counts each C4 exactly twice (once for each diagonal pair)
        # Actually, a C4 has 2 diagonals, so we count each C4 twice
        # Wait, let me think again.
        # A 4-cycle v1-v2-v3-v4-v1 has diagonals (v1,v3) and (v2,v4)
        # For diagonal (v1,v3): common neighbors include v2 and v4
        # For diagonal (v2,v4): common neighbors include v1 and v3
        # So each C4 is counted exactly twice (once per diagonal)
        # Therefore: C4 count = (1/2) * sum over all pairs (u,v) of C(common(u,v), 2)
        
        for u in range(n):
            for v in range(u+1, n):
                common = len(adj[u] & adj[v])
                if common >= 2:
                    c4_count += common * (common - 1) // 2
        return c4_count // 2
    
    # Start with polarity graph + one edge
    # Find the edge to add that minimizes new C4s
    best_edge = None
    best_total = float('inf')
    
    # Precompute adjacency as boolean matrix for speed
    is_adj = [[False]*n for _ in range(n)]
    for (u, v) in edge_set:
        is_adj[u][v] = True
        is_adj[v][u] = True
    
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                non_edges.append((i, j))
    
    # For adding edge (i,j), new C4s = number of paths i-a-b-j where a~i, b~j, a~b, a!=b
    # This equals the number of common "2-paths" between i and j
    # More precisely: sum over a in N(i) of |N(a) ∩ N(j)|, but excluding a=j and b=i
    # Since (i,j) is not an edge, a!=j automatically (a~i doesn't imply a!=j unless j~i)
    # Actually if a=j, then j~i which contradicts (i,j) not being an edge. So a!=j automatic.
    # Similarly b!=i automatic.
    # So new C4s = number of pairs (a,b) with a in N(i), b in N(j), a~b, a!=b
    
    # To speed up, for each non-edge (i,j), compute:
    # cost = sum_{a in N(i)} |N(a) ∩ N(j) \ {a}|
    # But we need a!=b, and a~b, a in N(i), b in N(j)
    
    # Let's compute costs for all non-edges
    adj_lists = [sorted(adj[i]) for i in range(n)]
    
    # For efficiency, precompute N(a) as a set for each a
    # cost(i,j) = |{(a,b) : a in N(i), b in N(j), a~b, a != b}|
    #           = sum_{a in N(i)} |{b in N(j) : b~a, b != a}|
    #           = sum_{a in N(i)} (|N(a) ∩ N(j)| - (1 if a in N(j) else 0))
    #           But if a in N(j), then a~j, and we're counting b=a which we exclude
    #           However, if a in N(i) and a in N(j), then a is a common neighbor
    #           In that case, |N(a) ∩ N(j)| includes... wait, a is in N(j) means j~a
    #           We're counting b in N(j) such that b~a. If b=a, then a~a which is false (no self-loops)
    #           So a is never in N(a), so b=a is never counted! 
    #           Wait, b ranges over N(j), and we check if b~a. If b=a, then we check a~a = False.
    #           So actually cost = sum_{a in N(i)} |N(a) ∩ N(j)|
    #           Because a is never in N(a), so b=a never satisfies b~a.
    
    # So cost(i,j) = sum_{a in N(i)} |N(a) ∩ N(j)|
    
    # Let's compute this for all non-edges
    edge_costs = []
    for (i, j) in non_edges:
        cost = 0
        for a in adj_lists[i]:
            # Count neighbors of a that are also in N(j)
            common = len(adj[a] & adj[j])
            cost += common
        edge_costs.append((cost, i, j))
    
    edge_costs.sort()
    
    # Evaluate top candidates more carefully (the formula above should be exact)
    # But let me verify with a direct count for the top few
    
    evaluated = 0
    for cost, i, j in edge_costs:
        if time.time() > deadline:
            break
        
        # Direct verification
        actual_cost = 0
        for a in adj_lists[i]:
            for b in adj_lists[j]:
                if a != b and is_adj[a][b]:
                    actual_cost += 1
        
        if actual_cost < best_total:
            best_total = actual_cost
            best_edge = (i, j)
        
        evaluated += 1
        if evaluated > 200:  # Only check top 200
            break
    
    if best_edge is not None:
        i, j = best_edge
        edge_set.add((i, j))
        adj[i].add(j)
        adj[j].add(i)
        is_adj[i][j] = True
        is_adj[j][i] = True
    
    # Now we have 91 edges. Let's try local search to improve.
    # Local move: remove edge (u,v) and add edge (x,y) where (x,y) is a non-edge
    # This maintains 91 edges. We want to minimize total C4s.
    
    def count_c4_fast(adj, n):
        """Fast C4 count using common neighbor counts."""
        c4_count = 0
        for u in range(n):
            for v in range(u+1, n):
                common = len(adj[u] & adj[v])
                if common >= 2:
                    c4_count += common * (common - 1) // 2
        return c4_count // 2
    
    current_c4 = count_c4_fast(adj, n)
    best_c4 = current_c4
    best_edge_set = set(edge_set)
    best_adj = [set(adj[i]) for i in range(n)]
    
    # Local search: try edge swaps
    # For each edge (u,v) in the graph, try removing it and adding a non-edge (x,y)
    # This is O(E * non_edges) which is too slow. Use heuristic.
    
    # Better: for each edge (u,v), compute the change in C4 count if removed
    # Then for each non-edge (x,y), compute change if added
    # Net change = delta_remove(u,v) + delta_add(x,y) - interaction
    
    # Actually, let's just do a simple hill climbing with limited iterations
    
    random.seed(42)
    current_edges = list(edge_set)
    current_adj = [set(adj[i]) for i in range(n)]
    current_c4 = count_c4_fast(current_adj, n)
    
    # Precompute all non-edges
    all_pairs = set()
    for i in range(n):
        for j in range(i+1, n):
            all_pairs.add((i, j))
    
    non_edge_list = []
    for (i, j) in all_pairs:
        if (i, j) not in edge_set:
            non_edge_list.append((i, j))
    
    # For each edge, compute its "C4 contribution" - how many C4s it's part of
    # An edge (u,v) is in a C4 if there exist a,b such that u-a-b-v-u or u-b-a-v-u
    # i.e., a in N(u)\{v}, b in N(v)\{u}, a~b
    # Number of C4s containing edge (u,v) = number of pairs (a,b) with a in N(u)\{v}, b in N(v)\{u}, a~b, a!=b
    
    # For local search, we'll try random swaps
    max_iterations = 5000
    iteration = 0
    no_improve_count = 0
    max_no_improve = 200
    
    while iteration < max_iterations and no_improve_count < max_no_improve:
        if time.time() > deadline - 10:
            break
        
        # Pick a random edge to remove
        edge_to_remove = random.choice(current_edges)
        u, v = edge_to_remove
        
        # Pick a random non-edge to add
        if not non_edge_list:
            break
        edge_to_add = random.choice(non_edge_list)
        x, y = edge_to_add
        
        # Check that the swap is valid (both vertices distinct, etc.)
        # Remove (u,v) and add (x,y)
        # Compute new C4 count efficiently
        
        # For efficiency, let's just compute the full C4 count for candidate
        # This is O(n^2 * avg_degree) which for n=31 should be fast
        
        # Temporarily remove (u,v)
        current_adj[u].discard(v)
        current_adj[v].discard(u)
        # Temporarily add (x,y)
        current_adj[x].add(y)
        current_adj[y].add(x)
        
        new_c4 = count_c4_fast(current_adj, n)
        
        # Revert
        current_adj[u].add(v)
        current_adj[v].add(u)
        current_adj[x].discard(y)
        current_adj[y].discard(x)
        
        if new_c4 < current_c4:
            # Accept the move
            current_edges.remove(edge_to_remove)
            current_edges.append(edge_to_add)
            non_edge_list.remove(edge_to_add)
            non_edge_list.append(edge_to_remove)
            
            current_adj[u].discard(v)
            current_adj[v].discard(u)
            current_adj[x].add(y)
            current_adj[y].add(x)
            
            current_c4 = new_c4
            no_improve_count = 0
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edge_set = set(current_edges)
                best_adj = [set(current_adj[i]) for i in range(n)]
        else:
            no_improve_count += 1
        
        iteration += 1
    
    # Use the best found
    if best_c4 < current_c4:
        edge_set = best_edge_set
        adj = best_adj
    
    # Convert to list of edges
    edges = [[u, v] for (u, v) in edge_set]
    
    # Ensure exactly 91 distinct edges
    if len(edges) != 91:
        # Fix up
        edge_set = set()
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            if u != v:
                edge_set.add((min(u,v), max(u,v)))
                adj[u].add(v)
                adj[v].add(u)
        edges = [[u, v] for (u, v) in edge_set]
        
        if len(edges) < 91:
            for i in range(n):
                for j in range(i+1, n):
                    if len(edges) >= 91:
                        break
                    if (i, j) not in edge_set:
                        edge_set.add((i, j))
                        adj[i].add(j)
                        adj[j].add(i)
                        edges.append([i, j])
                if len(edges) >= 91:
                    break
        
        if len(edges) > 91:
            edges = edges[:91]
            edge_set = set(tuple(sorted(e)) for e in edges)
    
    # Verify
    assert len(edges) == 91
    assert len(set(tuple(sorted(e)) for e in edges)) == 91
    for e in edges:
        assert 0 <= e[0] <= 30
        assert 0 <= e[1] <= 30
        assert e[0] != e[1]
    
    return edges

import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n = 16
    # Precompute Q3 structure
    # Q3 has 8 vertices, 12 edges
    q3_edges = [
        (0, 1), (0, 2), (0, 3), 
        (1, 4), (1, 5), 
        (2, 4), (2, 6), 
        (3, 5), (3, 6), 
        (4, 7), (5, 7), (6, 7)
    ]
    
    def make_adj_list(adj):
        """Convert adjacency matrix to list of neighbor lists"""
        return [[j for j in range(n) if adj[i][j]] for i in range(n)]
    
    def check_q3_with_edge(u, v, adj, adj_list):
        """Check if adding edge (u,v) creates a Q3 subgraph"""
        # In Q3 with edge (0,1):
        # 0's other neighbors: 2, 3
        # 1's other neighbors: 4, 5
        # 2,3 must be neighbors of u (not v)
        # 4,5 must be neighbors of v (not u)
        # 6,7 are the remaining vertices
        
        neu = [x for x in adj_list[u] if x != v]
        nev = [x for x in adj_list[v] if x != u]
        
        if len(neu) < 2 or len(nev) < 2:
            return False
        
        # Try all pairs for roles 2,3 from neu and 4,5 from nev
        for r2, r3 in itertools.combinations(neu, 2):
            for r4, r5 in itertools.combinations(nev, 2):
                # Must be 4 distinct vertices
                if len({r2, r3, r4, r5}) < 4:
                    continue
                # Check edges r2-r4 and r3-r5
                if not adj[r2][r4] or not adj[r3][r5]:
                    continue
                # Find r6: common neighbor of r2 and r3, not in {u,v,r2,r3,r4,r5}
                # Find r7: common neighbor of r4 and r5, not in {u,v,r2,r3,r4,r5}
                # r6-r7 must be an edge
                excluded = {u, v, r2, r3, r4, r5}
                for r6 in adj_list[r2]:
                    if r6 in excluded:
                        continue
                    if not adj[r3][r6]:
                        continue
                    for r7 in adj_list[r4]:
                        if r7 in excluded or r7 == r6:
                            continue
                        if not adj[r5][r7]:
                            continue
                        if adj[r6][r7]:
                            return True
        return False
    
    def count_edges(adj):
        return sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])
    
    def greedy_construct(order_seed=None):
        """Greedy construction with optional random ordering"""
        adj = [[False] * n for _ in range(n)]
        
        # Start with K3,13
        for u in range(3):
            for v in range(3, 16):
                adj[u][v] = True
                adj[v][u] = True
        
        adj_list = make_adj_list(adj)
        degrees = [len(adj_list[i]) for i in range(n)]
        
        # Get missing edges
        missing_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    missing_edges.append((i, j))
        
        # Sort by degree sum, or randomize
        if order_seed is not None:
            random.seed(order_seed)
            random.shuffle(missing_edges)
        else:
            missing_edges.sort(key=lambda e: degrees[e[0]] + degrees[e[1]])
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
            
            # Tentatively add
            adj[u][v] = True
            adj[v][u] = True
            adj_list[u].append(v)
            adj_list[v].append(u)
            
            # Check if Q3 exists involving (u,v)
            if check_q3_with_edge(u, v, adj, adj_list):
                # Revert
                adj[u][v] = False
                adj[v][u] = False
                adj_list[u].remove(v)
                adj_list[v].remove(u)
            else:
                degrees[u] += 1
                degrees[v] += 1
        
        return adj, count_edges(adj)
    
    def local_search(adj, max_time=30):
        """Local search: try edge swaps to improve edge count"""
        adj_list = make_adj_list(adj)
        current_count = count_edges(adj)
        best_adj = [row[:] for row in adj]
        best_count = current_count
        search_deadline = time.time() + max_time
        
        # Get current edges and missing edges
        current_edges = set()
        missing_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    current_edges.add((i, j))
                else:
                    missing_edges.append((i, j))
        
        # Try to add missing edges
        for u, v in missing_edges:
            if time.time() > search_deadline:
                break
            
            # Tentatively add
            adj[u][v] = True
            adj[v][u] = True
            adj_list[u].append(v)
            adj_list[v].append(u)
            
            if not check_q3_with_edge(u, v, adj, adj_list):
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = [row[:] for row in adj]
            else:
                # Revert
                adj[u][v] = False
                adj[v][u] = False
                adj_list[u].remove(v)
                adj_list[v].remove(u)
        
        # Try edge swaps: remove one edge, try to add two
        current_edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    current_edges.add((i, j))
        
        missing_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    missing_edges.append((i, j))
        
        random.shuffle(missing_edges)
        
        for edge_to_remove in list(current_edges):
            if time.time() > search_deadline:
                break
            
            e1, e2 = edge_to_remove
            # Remove the edge
            adj[e1][e2] = False
            adj[e2][e1] = False
            adj_list[e1].remove(e2)
            adj_list[e2].remove(e1)
            current_count -= 1
            
            # Try to add missing edges
            added = 0
            added_edges = []
            for u, v in missing_edges:
                if u == e1 and v == e2:
                    continue
                # Check if this edge was already present (it shouldn't be, but just in case)
                if adj[u][v]:
                    continue
                    
                adj[u][v] = True
                adj[v][u] = True
                adj_list[u].append(v)
                adj_list[v].append(u)
                
                if not check_q3_with_edge(u, v, adj, adj_list):
                    added += 1
                    added_edges.append((u, v))
                    current_count += 1
                else:
                    adj[u][v] = False
                    adj[v][u] = False
                    adj_list[u].remove(v)
                    adj_list[v].remove(u)
                
                if added >= 2:
                    break
            
            # If we added more than we removed, keep the change
            if added > 1:
                if current_count > best_count:
                    best_count = current_count
                    best_adj = [row[:] for row in adj]
            else:
                # Revert the removal
                adj[e1][e2] = True
                adj[e2][e1] = True
                adj_list[e1].append(e2)
                adj_list[e2].append(e1)
                current_count += 1
                # Revert added edges
                for u, v in added_edges:
                    adj[u][v] = False
                    adj[v][u] = False
                    adj_list[u].remove(v)
                    adj_list[v].remove(u)
                    current_count -= 1
        
        return best_adj, best_count
    
    # Try multiple orderings
    best_adj = None
    best_count = 0
    
    # First, deterministic ordering
    adj, cnt = greedy_construct(order_seed=None)
    if cnt > best_count:
        best_count = cnt
        best_adj = [row[:] for row in adj]
    
    # Try several random orderings
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 40:  # Leave time for local search
            break
        adj, cnt = greedy_construct(order_seed=42 + i)
        if cnt > best_count:
            best_count = cnt
            best_adj = [row[:] for row in adj]
    
    # Apply local search to the best result
    if best_adj is not None and time.time() < deadline - 5:
        best_adj, best_count = local_search(best_adj, max_time=25)
    
    # Return edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i][j]:
                edges.append([i, j])
                
    return edges

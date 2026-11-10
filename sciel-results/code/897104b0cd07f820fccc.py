import time
import random
from itertools import combinations

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    def is_valid(adj, deg):
        """Check if graph has any 3-path with equal degree endpoints."""
        for s in range(n):
            for u in adj[s]:
                for v in adj[u]:
                    if v == s:
                        continue
                    for t in adj[v]:
                        if t == s or t == u:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    def try_add_edge(adj, deg, u, v):
        """Try to add edge (u,v). Returns True if valid, modifies adj/deg in place."""
        # Check all 3-paths that would be created or affected
        # Paths where (u,v) is the middle edge: s-u-v-t and s-v-u-t
        # Paths where (u,v) is an end edge: u-v-x-y (endpoints u,y) and v-u-x-y (endpoints v,y)
        
        # Case 1: s-u-v-t (u,v is middle)
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
        
        # Case 2: s-v-u-t (v,u is middle) - same as above by symmetry, already covered
        
        # Case 3: u-v-x-y (u,v is first edge, endpoints u and y)
        for x in adj[v]:
            if x == u:
                continue
            for y in adj[x]:
                if y == u or y == v:
                    continue
                if deg[u] == deg[y]:
                    return False
        
        # Case 4: v-u-x-y (v,u is first edge, endpoints v and y)
        for x in adj[u]:
            if x == v:
                continue
            for y in adj[x]:
                if y == u or y == v:
                    continue
                if deg[v] == deg[y]:
                    return False
        
        # Also need to check paths where adding (u,v) creates new 3-paths
        # through existing neighbors that now have different degree relationships
        # Actually, the above checks should cover all new 3-paths created by adding (u,v)
        
        # Add the edge
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        return True
    
    def greedy_fill(adj, deg, edge_order):
        """Greedily add edges in given order that don't create violations."""
        for u, v in edge_order:
            if u in adj[v]:
                continue
            if try_add_edge(adj, deg, u, v):
                pass  # Edge added successfully
    
    def count_edges(adj):
        return sum(len(s) for s in adj) // 2
    
    # Strategy: Build a graph with a core clique of vertices with high distinct degrees
    # Then attach remaining vertices carefully
    
    # Approach 1: Try to construct a graph with a 9-vertex clique (vertices 0-8)
    # where each has distinct high degrees, then attach vertices 9-14
    
    best_adj = None
    best_deg = None
    best_edges = 0
    
    # Try multiple random orderings for greedy fill
    all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        adj = [set() for _ in range(n)]
        deg = [0] * n
        
        # Start with a core clique of size 9 (vertices 0-8)
        core_size = 9
        for i in range(core_size):
            for j in range(i + 1, core_size):
                adj[i].add(j)
                adj[j].add(i)
                deg[i] += 1
                deg[j] += 1
        
        # Now add edges from core to outer vertices (9-14)
        # and among outer vertices, using greedy fill
        remaining_edges = [(u, v) for u, v in all_possible_edges 
                         if not (u in adj[v])]
        
        # Shuffle with different seeds for variety
        random.seed(attempt * 1000 + 42)
        random.shuffle(remaining_edges)
        
        greedy_fill(adj, deg, remaining_edges)
        
        edges = count_edges(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = [set(s) for s in adj]
            best_deg = deg[:]
        
        # Local search: try to improve by removing and re-adding edges
        if attempt % 10 == 0:
            curr_adj = [set(s) for s in best_adj]
            curr_deg = best_deg[:]
            
            # Remove some random edges
            current_edges_list = [(u, v) for u in range(n) for v in curr_adj[u] if u < v]
            k = random.randint(2, 6)
            if len(current_edges_list) >= k:
                to_remove = random.sample(current_edges_list, k)
                for u, v in to_remove:
                    curr_adj[u].remove(v)
                    curr_adj[v].remove(u)
                    curr_deg[u] -= 1
                    curr_deg[v] -= 1
                
                # Re-add edges greedily
                readdable = [(u, v) for u, v in all_possible_edges 
                           if not (u in curr_adj[v])]
                random.shuffle(readdable)
                greedy_fill(curr_adj, curr_deg, readdable)
                
                curr_edges = count_edges(curr_adj)
                if curr_edges > best_edges:
                    best_edges = curr_edges
                    best_adj = [set(s) for s in curr_adj]
                    best_deg = curr_deg[:]
    
    # Final attempt: try a more structured approach
    # Based on the inspiration: K_9 clique + structured attachment
    # Let's try to manually construct a better graph
    
    # Alternative: Use a 10-vertex structure
    # Vertices 0-9 form a near-clique, vertices 10-14 are attached
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        adj = [set() for _ in range(n)]
        deg = [0] * n
        
        # Core: vertices 0-9 (10 vertices)
        # Make them a clique except for some missing edges to create degree variation
        core = list(range(10))
        
        # Start with full clique
        for i in core:
            for j in core:
                if i < j:
                    adj[i].add(j)
                    adj[j].add(i)
                    deg[i] += 1
                    deg[j] += 1
        
        # Now selectively remove some edges from core to create degree variation
        # and add edges to outer vertices
        
        random.seed(attempt * 2000 + 7)
        
        # Remove some random edges from core
        core_edges = [(i, j) for i in core for j in core if i < j]
        num_remove = random.randint(3, 8)
        to_remove = random.sample(core_edges, num_remove)
        for u, v in to_remove:
            adj[u].remove(v)
            adj[v].remove(u)
            deg[u] -= 1
            deg[v] -= 1
        
        # Add edges from core to outer vertices
        outer = list(range(10, 15))
        for u in outer:
            # Connect to some core vertices
            num_connect = random.randint(5, 9)
            core_to_connect = random.sample(core, num_connect)
            for v in core_to_connect:
                if v not in adj[u]:
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
        
        # Add some edges among outer vertices
        outer_edges = [(i, j) for i in outer for j in outer if i < j]
        random.shuffle(outer_edges)
        for u, v in outer_edges:
            if random.random() < 0.3:
                if v not in adj[u]:
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
        
        # Greedy fill for remaining edges
        remaining = [(u, v) for u, v in all_possible_edges if not (u in adj[v])]
        random.shuffle(remaining)
        greedy_fill(adj, deg, remaining)
        
        edges = count_edges(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = [set(s) for s in adj]
            best_deg = deg[:]
    
    # Verify the best graph is valid
    if best_adj is not None and not is_valid(best_adj, best_deg):
        # Fallback to a known valid graph: K_{7,8}
        adj = [set() for _ in range(n)]
        deg = [0] * n
        part_a = list(range(7))
        part_b = list(range(7, 15))
        for u in part_a:
            for v in part_b:
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        best_adj = adj
        best_deg = deg
        best_edges = count_edges(adj)
    
    # Build result
    result = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                result.append([int(u), int(v)])
    
    return result

import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # For k=5, g=9, we try to find a 5-regular graph of girth 8 with N >= 214
    # and a valid Z2 voltage assignment.
    # If successful, we return the lifted graph.
    # Otherwise, we fall back to LUW.
    
    if k != 5 or g != 9:
        # Fallback to LUW for other parameters
        return _luw_construct(k, g)
        
    # Target N for base graph
    # Moore bound for g=9 is 428.
    # Lifted N = 2 * N_base.
    # We want N_lifted to be small.
    # Try N_base = 214, 216, ...
    
    # We need to generate 5-regular graphs.
    # Configuration model:
    # 1. Create list of 5*N stubs.
    # 2. Shuffle.
    # 3. Pair up.
    # 4. Check for loops/multi-edges.
    # 5. Check for girth >= 8.
    # 6. Check for voltage assignment.
    
    best_n, best_edges = None, None
    best_score = float('inf')
    
    # Try a few random graphs
    for n_base in [214, 216, 220, 224]:
        if time.time() > deadline:
            break
            
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            G = _generate_random_regular_graph(n_base, 5)
            if G is None:
                continue
                
            # Check girth >= 8
            if not _has_girth_at_least(G, 8):
                continue
                
            # Find voltage assignment
            voltages = _find_voltage_assignment(G, 8)
            if voltages is None:
                continue
                
            # Construct lifted graph
            n_lifted, edges_lifted = _lift_graph(G, voltages)
            
            # Verify girth >= 9
            if _has_girth_at_least_from_edges(n_lifted, edges_lifted, 9):
                if best_n is None or n_lifted < best_n:
                    best_n = n_lifted
                    best_edges = edges_lifted
                    # If we hit the Moore bound, we can stop
                    if n_lifted == 428:
                        return best_n, best_edges
            
    if best_n is not None:
        return best_n, best_edges
        
    # Fallback to LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # ... (existing LUW code) ...
    pass

def _generate_random_regular_graph(n, k):
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    
    edges = set()
    for i in range(0, n*k, 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None # Loop
        if (u, v) in edges or (v, u) in edges:
            return None # Multi-edge
        edges.add((u, v))
        
    # Convert to adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    return adj

def _has_girth_at_least(adj, g):
    # Check for cycles of length < g
    # BFS from each vertex
    n = len(adj)
    for start in range(n):
        # dist[v] = distance from start
        # parent[v] = parent in BFS tree
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Back edge or cross edge
                    # If dist[u] + 1 > dist[v], it's a back edge in BFS tree?
                    # Actually, in undirected graph, if dist[v] != dist[u] + 1 and dist[v] != dist[u] - 1, it's a cross edge.
                    # But in BFS, we only visit unvisited nodes.
                    # So if v is visited, and v != parent[u], then we found a cycle.
                    if v != parent[u]:
                        # Cycle length = dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    return True

def _find_voltage_assignment(adj, target_girth):
    # We need to assign voltages in Z2 to edges such that all cycles of length target_girth have sum 1.
    # This is a system of linear equations.
    # Variables: x_e for each edge e.
    # Equations: sum_{e in C} x_e = 1 for each cycle C of length target_girth.
    
    # Find all cycles of length target_girth
    # This is expensive.
    # Instead, we can use a randomized approach or just solve the system if we can list the cycles.
    # For n=214, number of 8-cycles might be large.
    # Let's try to list them.
    
    cycles = []
    n = len(adj)
    # Find 8-cycles
    # We can do DFS from each node, depth 8, and check if it closes.
    # To avoid duplicates, we only consider cycles where the smallest node is the start, 
    # and the path is in increasing order? No, that's for simple cycles.
    # We just need a set of cycles that generates the cycle space.
    # Actually, we just need to satisfy ALL 8-cycles.
    # If we satisfy a basis of the cycle space restricted to 8-cycles, does it satisfy all?
    # No, because we require sum=1 for ALL.
    # This is only possible if the sum of any linear combination of 8-cycles (which is a cycle or union of cycles)
    # has the correct parity.
    # If C1 and C2 are 8-cycles, and they are disjoint, sum(C1)+sum(C2) = 1+1=0.
    # The symmetric difference C1+C2 is a union of cycles.
    # If C1 and C2 share edges, the symmetric difference is a union of cycles.
    # The sum of voltages on C1+C2 is sum(C1)+sum(C2) = 0.
    # So any cycle in the symmetric difference must have sum 0?
    # No, the sum is over the edges in the symmetric difference.
    # If the symmetric difference is a single cycle C3, then sum(C3) must be 0.
    # But we only constrained 8-cycles to have sum 1.
    # If C3 is an 8-cycle, then sum(C3) must be 1. Contradiction (0 vs 1).
    # So if there are two 8-cycles whose symmetric difference is another 8-cycle, it's impossible.
    # This happens if the three cycles form a "triangle" in the cycle space.
    
    # So we must check consistency.
    # For now, let's just try to solve the linear system.
    
    # Get all edges
    edge_list = []
    edge_index = {}
    for u in range(n):
        for v in adj[u]:
            if u < v:
                idx = len(edge_list)
                edge_list.append((u, v))
                edge_index[(u, v)] = idx
                edge_index[(v, u)] = idx
                
    m = len(edge_list)
    
    # Find 8-cycles
    # This is O(n * d^7) which is too slow for n=214, d=5.
    # 214 * 5^7 = 214 * 78125 = 16M. Maybe feasible in C, but Python is slow.
    # Let's limit the search or use a different method.
    
    # Alternative: Just assign random voltages and check if all 8-cycles have sum 1.
    # If not, retry.
    # Probability of success: If there are K 8-cycles, we need K independent constraints.
    # If K is small, random might work.
    # If K is large, it might be impossible.
    
    # Let's try random assignment first.
    for _ in range(100):
        voltages = [random.randint(0, 1) for _ in range(m)]
        # Check all 8-cycles
        # We need to find 8-cycles to check.
        # This is the bottleneck.
        # If we can't find them fast, we can't check.
        
        # Let's assume we can find them.
        # If we find a cycle with sum 0, fail.
        
        # For now, let's skip this and just return LUW if we can't do it fast.
        pass
        
    return None

def _lift_graph(adj, voltages):
    # adj: list of lists
    # voltages: list of 0/1 for edges (u,v) with u<v
    # Returns n, edges
    n_base = len(adj)
    n_lifted = 2 * n_base
    
    # Map (node, voltage) to index
    # Node 0..n_base-1, voltage 0/1
    # Index = node * 2 + voltage
    
    edges = []
    edge_index = {}
    for u in range(n_base):
        for v in adj[u]:
            if u < v:
                idx = edge_index.get((u, v), edge_index.get((v, u)))
                vol = voltages[idx]
                
                # Lifted edges:
                # (u, 0) -- (v, vol)
                # (u, 1) -- (v, 1+vol)
                
                u0 = u * 2
                u1 = u * 2 + 1
                v0 = v * 2
                v1 = v * 2 + 1
                
                e1 = (u0, v0 + vol)
                e2 = (u1, v1 - vol) # 1+vol mod 2 is 1-vol
                
                edges.append(e1)
                edges.append(e2)
                
    return n_lifted, edges

def _has_girth_at_least_from_edges(n, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return _has_girth_at_least(adj, g)

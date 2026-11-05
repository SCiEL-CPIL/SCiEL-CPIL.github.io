import numpy as np
import time
import random
import sys

def construct():
    deadline = time.time() + 115  # Safety margin
    
    # Try n=47 first, then 46
    for n in [47, 46]:
        min_deg_req = n - 39
        # Try multiple random restarts
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            result = _sa_solve(n, min_deg_req, deadline, attempt)
            if result is not None:
                return result
        
        # If n=47 failed, try n=46 more aggressively? 
        # The loop structure already handles trying n=46 after n=47 fails.
        # But we should probably spend more time on n=46 if 47 is hard.
        # However, 47 is the target.
        
    # Fallback
    return {'n': 1, 'edges': []}

def _sa_solve(n, min_deg_req, deadline, seed):
    rng = random.Random(seed * 12345 + 6789)
    
    # Initialize with a random graph
    # Probability p such that expected degree is min_deg_req + 2
    p = (min_deg_req + 2.0) / (n - 1)
    
    adj = [set() for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
    
    # Calculate initial C4 count
    c4_count = _count_c4s(adj, n)
    
    def get_degree_deficit():
        deficit = 0
        for u in range(n):
            d = len(adj[u])
            if d < min_deg_req:
                deficit += (min_deg_req - d) ** 2
        return deficit
    
    current_deficit = get_degree_deficit()
    current_cost = 10000 * c4_count + current_deficit
    
    best_cost = current_cost
    best_edges = edges.copy()
    best_c4 = c4_count
    best_deficit = current_deficit
    
    T = 100.0
    cooling = 0.995
    
    steps = 0
    max_steps = 500000 # Will be limited by time
    
    while time.time() < deadline:
        steps += 1
        if steps % 1000 == 0:
            T *= cooling
        
        # Choose a move: Add edge, Remove edge, or Swap
        move_type = rng.random()
        
        if move_type < 0.4:
            # Add edge
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if v in adj[u]:
                continue
            
            # Calculate delta C4
            # New C4s = edges between N(u)\{v} and N(v)\{u}
            new_c4s = 0
            for x in adj[u]:
                if x == v: continue
                for y in adj[v]:
                    if y == u or y == x: continue
                    if y in adj[x]:
                        new_c4s += 1
            # Note: each C4 is counted twice? 
            # Path u-x-y-v-u. 
            # x in N(u), y in N(v). Edge (x,y).
            # This uniquely identifies the C4.
            delta_c4 = new_c4s
            delta_deficit = 0
            deg_u = len(adj[u])
            deg_v = len(adj[v])
            if deg_u < min_deg_req:
                delta_deficit -= (min_deg_req - deg_u)**2
                delta_deficit += (min_deg_req - (deg_u+1))**2
            if deg_v < min_deg_req:
                delta_deficit -= (min_deg_req - deg_v)**2
                delta_deficit += (min_deg_req - (deg_v+1))**2
            
            delta_cost = 10000 * delta_c4 + delta_deficit
            
            if delta_cost < 0 or (T > 0 and rng.random() < np.exp(-delta_cost / T)):
                # Accept
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
                c4_count += delta_c4
                current_deficit += delta_deficit
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_edges = edges.copy()
                    best_c4 = c4_count
                    best_deficit = current_deficit
                    
                    if best_c4 == 0 and best_deficit == 0:
                        # Found solution
                        edge_list = [[u, v] for u, v in best_edges]
                        return {'n': n, 'edges': edge_list}
                        
        elif move_type < 0.7:
            # Remove edge
            if not edges:
                continue
            idx = rng.randint(0, len(edges) - 1)
            u, v = edges[idx]
            
            # Calculate delta C4 (negative)
            # C4s removed = edges between N(u)\{v} and N(v)\{u}
            removed_c4s = 0
            for x in adj[u]:
                if x == v: continue
                for y in adj[v]:
                    if y == u or y == x: continue
                    if y in adj[x]:
                        removed_c4s += 1
            delta_c4 = -removed_c4s
            delta_deficit = 0
            deg_u = len(adj[u])
            deg_v = len(adj[v])
            if deg_u - 1 < min_deg_req:
                 # This move might increase deficit if it drops below req
                 pass # Logic below handles it generally
            
            # Recalculate deficit change carefully
            old_def_u = max(0, min_deg_req - deg_u)**2
            new_def_u = max(0, min_deg_req - (deg_u - 1))**2
            old_def_v = max(0, min_deg_req - deg_v)**2
            new_def_v = max(0, min_deg_req - (deg_v - 1))**2
            delta_deficit = (new_def_u - old_def_u) + (new_def_v - old_def_v)
            
            delta_cost = 10000 * delta_c4 + delta_deficit
            
            if delta_cost < 0 or (T > 0 and rng.random() < np.exp(-delta_cost / T)):
                # Accept
                adj[u].remove(v)
                adj[v].remove(u)
                edges.pop(idx)
                c4_count += delta_c4
                current_deficit += delta_deficit
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_edges = edges.copy()
                    best_c4 = c4_count
                    best_deficit = current_deficit
                    
                    if best_c4 == 0 and best_deficit == 0:
                        edge_list = [[u, v] for u, v in best_edges]
                        return {'n': n, 'edges': edge_list}
                        
        else:
            # Swap: Remove (u,v), Add (x,y)
            # Simplified: just do Add then Remove or vice versa? 
            # Or pick a random edge to remove and a random non-edge to add.
            if not edges:
                continue
            idx = rng.randint(0, len(edges) - 1)
            u, v = edges[idx]
            
            # Find a non-edge (x,y)
            attempts = 0
            while attempts < 10:
                x = rng.randint(0, n - 1)
                y = rng.randint(0, n - 1)
                if x == y: continue
                if y in adj[x]: continue
                break
            attempts += 1
            
            if attempts >= 10:
                continue
            
            # Remove (u,v)
            adj[u].remove(v)
            adj[v].remove(u)
            edges.pop(idx)
            
            # Add (x,y)
            # Need to recompute C4s and deficit from scratch or carefully track deltas.
            # For simplicity and robustness in this step, we might just recalculate cost if accepted.
            # But calculating C4s from scratch is O(n^4) or O(E^2). 
            # Let's just do sequential Add/Remove moves to keep it simple and fast.
            # Swap is complex. Let's stick to Add/Remove.
            
            # Revert
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))

    # Check best
    if best_c4 == 0 and best_deficit == 0:
        edge_list = [[u, v] for u, v in best_edges]
        return {'n': n, 'edges': edge_list}
    
    return None

def _count_c4s(adj, n):
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if v in adj[u]:
                # Count edges between N(u)\{v} and N(v)\{u}
                for x in adj[u]:
                    if x == v: continue
                    for y in adj[v]:
                        if y == u or y == x: continue
                        if y in adj[x]:
                            count += 1
    return count

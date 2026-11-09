import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Target n values, descending
    best_n = 0
    best_edges = []
    
    for target_n in [47, 46]:
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        found = False
        
        # Run Simulated Annealing
        # We use multiple restarts if SA fails
        for restart in range(5):
            if time.time() > deadline:
                break
            
            random.seed(restart * 100 + target_n)
            
            # Initialize graph
            # Target degree ~ min_deg_needed + 2
            p = (min_deg_needed + 2) / target_n
            adj_mask = [0] * target_n
            deg = [0] * target_n
            edges = set()
            
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < p:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
                        edges.add((u, v))
            
            # Calculate initial C4 count
            # C4 count = sum over all edges (u,v) of C(|N(u)&N(v)|, 2) / 2 ?
            # No, each C4 is counted 4 times (once for each edge) in sum over edges C(k,2)?
            # Let's check. A C4 u-v-w-x-u.
            # Edge (u,v): common neighbors w,x. C(2,2)=1.
            # Edge (v,w): common neighbors u,x. C(2,2)=1.
            # Edge (w,x): common neighbors v,u. C(2,2)=1.
            # Edge (x,u): common neighbors w,v. C(2,2)=1.
            # Sum = 4.
            # So Total C4s = Sum_{(u,v) in E} C(|N(u)&N(v)|, 2) / 4.
            
            def calc_c4_count():
                count = 0
                for (u, v) in edges:
                    common = adj_mask[u] & adj_mask[v]
                    k = bin(common).count('1')
                    if k >= 2:
                        count += k * (k - 1) // 2
                return count // 4
            
            c4_count = calc_c4_count()
            
            # SA Parameters
            temp = 10.0
            cool_rate = 0.999
            min_temp = 0.01
            max_iter = 500000 # Adjust based on time
            
            # Precompute edge list for random selection?
            # We can just pick u, v randomly and check if it's an edge or not.
            # Or maintain a list of edges.
            
            while temp > min_temp and time.time() < deadline:
                # Pick random u, v
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                
                # Ensure u < v for consistent edge representation
                if u > v:
                    u, v = v, u
                
                is_edge = (u, v) in edges
                
                # Calculate delta C4
                # If adding: new C4s = C(k, 2) where k = |N(u)&N(v)|
                # If removing: removed C4s = C(k, 2)
                # Note: N(u) and N(v) do NOT include each other if not edge.
                # If edge, N(u) includes v, N(v) includes u.
                # Common neighbors are those in N(u) \ {v} and N(v) \ {u}.
                # But since u!=v, N(u)&N(v) naturally excludes u and v?
                # adj_mask[u] has bit v if edge. adj_mask[v] has bit u if edge.
                # Intersection will have bit v if v in N(u) and v in N(v)? No, v is not in N(v).
                # So intersection is correct.
                
                common = adj_mask[u] & adj_mask[v]
                k = bin(common).count('1')
                c4_delta = k * (k - 1) // 2
                
                if is_edge:
                    # Remove edge
                    delta_c4 = -c4_delta
                else:
                    # Add edge
                    delta_c4 = c4_delta
                
                # Degree change
                # If add: deg[u]++, deg[v]++
                # If remove: deg[u]--, deg[v]--
                
                # Cost function: C4s + Penalty for min degree
                # Current min degree
                min_deg = min(deg)
                deficit = max(0, min_deg_needed - min_deg)
                current_cost = c4_count + 100 * deficit
                
                # New min degree
                new_deg = deg[:]
                if is_edge:
                    new_deg[u] -= 1
                    new_deg[v] -= 1
                else:
                    new_deg[u] += 1
                    new_deg[v] += 1
                
                new_min_deg = min(new_deg)
                new_deficit = max(0, min_deg_needed - new_min_deg)
                new_cost = (c4_count + delta_c4) + 100 * new_deficit
                
                delta_cost = new_cost - current_cost
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / temp):
                    # Accept move
                    if is_edge:
                        edges.remove((u, v))
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                        deg[u] -= 1
                        deg[v] -= 1
                    else:
                        edges.add((u, v))
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
                    
                    c4_count += delta_c4
                    temp *= cool_rate
                
                # Check for solution
                if c4_count == 0 and min(deg) >= min_deg_needed:
                    found = True
                    break
            
            if found:
                final_edges = [[u, v] for (u, v) in edges]
                if target_n > best_n:
                    best_n = target_n
                    best_edges = final_edges
                break # Break restart loop
        
        if best_n == 47:
            break # Can't do better than 47
            
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback
        return {'n': 39, 'edges': []}

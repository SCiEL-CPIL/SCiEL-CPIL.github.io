# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO CVRP.
    
    Strategy:
      1. Run Clarke-Wright savings to construct feasible routes.
      2. Extract edges from those routes as "reference edges".
      3. Base heuristic = inverse distance.
      4. Boost reference edges strongly (they are proven good connections).
      5. Add demand-efficiency weighting: edges connecting nodes whose
         combined demand is a good fraction of capacity get extra boost.
      6. Depot edges get a slight boost for high-demand customers (big rocks first).
    """
    start_time = time.time()
    deadline = start_time + 280.0

    n = distance_matrix.shape[0]
    if distance_matrix.shape[1] != n:
        return np.full((n, n), 1e-9)

    demands = np.asarray(demands, dtype=np.float64)
    if demands.shape[0] != n:
        demands = demands[:n]

    dist_safe = np.maximum(distance_matrix, 1e-9)
    
    # Base heuristic: inverse distance with mild power to emphasize short edges
    power = 1.2
    heuristic = 1.0 / (dist_safe ** power)

    # --- Clarke-Wright route construction to get reference edges ---
    ref_edges = set()
    
    if n > 1 and time.time() < deadline - 10:
        # Initialize: each customer is its own route
        # route_info: dict rid -> {'start': int, 'end': int, 'demand': float, 'customers': list}
        routes = {}
        cust_to_route = {}
        
        for i in range(1, n):
            routes[i] = {
                'start': i,
                'end': i,
                'demand': demands[i],
                'customers': [i]
            }
            cust_to_route[i] = i
        
        # Compute savings
        d0 = distance_matrix[0]
        savings = d0[:, None] + d0[None, :] - distance_matrix
        
        # Build list of (savings, i, j) for 1 <= i < j < n
        pairs = []
        for i in range(1, n):
            for j in range(i + 1, n):
                s = savings[i, j]
                if s > 0:  # only consider positive savings
                    pairs.append((s, i, j))
        
        pairs.sort(key=lambda x: x[0], reverse=True)
        
        for sav, i, j in pairs:
            if time.time() > deadline - 5:
                break
            
            rid_i = cust_to_route.get(i)
            rid_j = cust_to_route.get(j)
            
            if rid_i is None or rid_j is None or rid_i == rid_j:
                continue
            
            route_i = routes.get(rid_i)
            route_j = routes.get(rid_j)
            
            if route_i is None or route_j is None:
                continue
            
            # Capacity check
            if route_i['demand'] + route_j['demand'] > capacity:
                continue
            
            # Determine endpoint positions
            i_is_start = (route_i['start'] == i)
            i_is_end = (route_i['end'] == i)
            j_is_start = (route_j['start'] == j)
            j_is_end = (route_j['end'] == j)
            
            # We need i and j to be at opposite ends (one start, one end)
            # to connect them directly. If not, we can reverse one route.
            
            merged = False
            new_route = None
            
            # Case 1: i is end of A, j is start of B -> direct connect
            if i_is_end and j_is_start:
                new_route = {
                    'start': route_i['start'],
                    'end': route_j['end'],
                    'demand': route_i['demand'] + route_j['demand'],
                    'customers': route_i['customers'] + route_j['customers']
                }
                merged = True
            
            # Case 2: i is start of A, j is end of B -> reverse both, then connect
            elif i_is_start and j_is_end:
                # Reverse A: new start = old end, new end = i
                # Reverse B: new start = j, new end = old start
                new_route = {
                    'start': route_i['end'],
                    'end': route_j['start'],
                    'demand': route_i['demand'] + route_j['demand'],
                    'customers': list(reversed(route_i['customers'])) + list(reversed(route_j['customers']))
                }
                merged = True
            
            # Case 3: i is end of A, j is end of B -> reverse B, then connect
            elif i_is_end and j_is_end:
                # Reverse B: new start = j, new end = old start of B
                new_route = {
                    'start': route_i['start'],
                    'end': route_j['start'],
                    'demand': route_i['demand'] + route_j['demand'],
                    'customers': route_i['customers'] + list(reversed(route_j['customers']))
                }
                merged = True
            
            # Case 4: i is start of A, j is start of B -> reverse A, then connect
            elif i_is_start and j_is_start:
                # Reverse A: new start = old end of A, new end = i
                new_route = {
                    'start': route_i['end'],
                    'end': route_j['end'],
                    'demand': route_i['demand'] + route_j['demand'],
                    'customers': list(reversed(route_i['customers'])) + route_j['customers']
                }
                merged = True
            
            if merged:
                # Record the connecting edge
                ref_edges.add((i, j))
                ref_edges.add((j, i))
                
                # Also record all edges within the new route
                for k in range(len(new_route['customers']) - 1):
                    a = new_route['customers'][k]
                    b = new_route['customers'][k + 1]
                    ref_edges.add((a, b))
                    ref_edges.add((b, a))
                
                # Update route structure
                routes[rid_i] = new_route
                for c in new_route['customers']:
                    cust_to_route[c] = rid_i
                del routes[rid_j]
    
    # --- Build final heuristic ---
    if ref_edges:
        # Boost reference edges
        for (a, b) in ref_edges:
            if 0 <= a < n and 0 <= b < n:
                heuristic[a, b] *= 3.0  # strong boost for proven edges
    
    # Demand-efficiency weighting:
    # Edges where demand[i] + demand[j] is a good fraction of capacity are better
    # This encourages grouping customers that fill the truck well
    demand_sum = demands[:, None] + demands[None, :]
    # Efficiency: peaks when demand_sum is close to capacity
    # Use a smooth function: exp(-((demand_sum - capacity*0.8)^2) / (2 * (capacity*0.3)^2))
    # But simpler: linear boost that peaks near capacity
    efficiency = 1.0 + 0.3 * np.clip(demand_sum / capacity, 0, 1.5)
    
    # Apply efficiency boost to customer-customer edges (not depot)
    depot_mask = np.zeros((n, n), dtype=bool)
    depot_mask[0, :] = True
    depot_mask[:, 0] = True
    
    heuristic[~depot_mask] *= efficiency[~depot_mask]
    
    # Depot edges: boost high-demand customers (big rocks first)
    # When going to/from depot, prefer higher demand customers
    if n > 1:
        # For depot -> customer edges: boost by demand
        demand_boost = 1.0 + 0.2 * (demands[1:] / capacity)
        heuristic[0, 1:] *= demand_boost
        heuristic[1:, 0] *= demand_boost
    
    # Ensure positivity and finiteness
    heuristic = np.maximum(heuristic, 1e-9)
    
    if not np.all(np.isfinite(heuristic)):
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
        heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END

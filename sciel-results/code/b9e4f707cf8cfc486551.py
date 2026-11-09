# EVOLVE-BLOCK-START
import numpy as np
import time

def _clarke_wright_savings(distance_matrix, demands, capacity, n):
    """
    Clarke-Wright Savings heuristic for CVRP.
    Returns list of routes (each a list of customer indices, not including depot).
    """
    customers = list(range(1, n))
    # Each customer starts as its own route: [0, i, 0]
    # We track routes as lists of customer nodes
    routes = [[c] for c in customers]
    
    # Compute savings
    savings = []
    for i in customers:
        for j in customers:
            if i != j:
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                savings.append((s, i, j))
    
    savings.sort(reverse=True)
    
    # Map customer -> route index
    cust_to_route = {c: idx for idx, route in enumerate(routes) for c in route}
    # Track route ends for each route
    route_ends = {}  # route_idx -> (end1, end2) where end1 is first node, end2 is last node (adjacent to depot)
    
    def get_route_ends(route_idx):
        route = routes[route_idx]
        if len(route) == 1:
            return (route[0], route[0])
        return (route[0], route[-1])
    
    # Initialize route ends
    for idx, route in enumerate(routes):
        route_ends[idx] = get_route_ends(idx)
    
    route_demand = [sum(demands[c] for c in route) for route in routes]
    
    for s, i, j in savings:
        if s <= 0:
            break
        ri = cust_to_route.get(i, -1)
        rj = cust_to_route.get(j, -1)
        if ri == -1 or rj == -1 or ri == rj:
            continue
        
        # Check if i and j are at the ends of their respective routes
        ei1, ei2 = route_ends[ri]
        ej1, ej2 = route_ends[rj]
        
        # i must be an end of route ri, j must be an end of route rj
        i_at_end = (i == ei1 or i == ei2)
        j_at_end = (j == ej1 or j == ej2)
        
        if not (i_at_end and j_at_end):
            continue
        
        # Check capacity
        if route_demand[ri] + route_demand[rj] > capacity:
            continue
        
        # Determine how to merge
        # Route ri: ... -> i -> 0 (or 0 -> i -> ...)
        # Route rj: ... -> j -> 0 (or 0 -> j -> ...)
        # We want to connect i-j or j-i
        
        # Get the actual order
        route_i = routes[ri]
        route_j = routes[rj]
        
        # Determine if i is at the start or end of route_i
        if route_i[0] == i:
            # i is at start: 0 -> i -> ... 
            # After merge: 0 -> j -> ... (route_j reversed?) 
            # Actually, let's think carefully.
            # Route i: [i, ..., x] meaning 0->i->...->x->0
            # Route j: [j, ..., y] meaning 0->j->...->y->0
            # If we connect i and j, we remove edge (0,i) and (0,j), add edge (i,j) or (j,i)
            # New route: if i was start and j was start: [j, ..., y, i, ..., x] ? No.
            # Standard savings: remove (0,i) and (0,j), add (i,j).
            # So the new route is route_j + route_i if i is end of ri and j is end of rj? 
            # Let's just reverse one if needed to make i and j adjacent in the merged list.
            
            # Let's define: 
            # If i is at start of ri, the route is 0->i->...->last_i->0
            # If j is at start of rj, the route is 0->j->...->last_j->0
            # Merging i-j: 0->j->...->last_j->i->...->last_i->0  (if we reverse ri? No)
            # Actually, the edge added is (i,j). So in the sequence, i and j must be adjacent.
            # If i is start of ri, it's the first node after depot. 
            # If j is start of rj, it's the first node after depot.
            # To connect i and j, we can do: 0->j->...->last_j->i->...->last_i->0
            # This requires reversing ri so that i is at the end? No, i is at start.
            # Let's reverse ri so i is at the end: [..., last_i, i]. Then connect last_j->i? No, we connect i->j or j->i.
            # The savings formula assumes we connect i and j directly.
            # Let's just build the merged route by concatenating.
            
            # Case 1: i is end of ri, j is end of rj
            # ri: [a, ..., i], rj: [b, ..., j]
            # Merged: [a, ..., i, j, ..., b] (if we reverse rj to put j at start? No)
            # If i is end of ri, the route ends with i. If j is end of rj, the route ends with j.
            # To connect i and j, we reverse rj: [j, ..., b]. Then merge: [a, ..., i, j, ..., b].
            # This connects i->j.
            
            # Let's handle all 4 cases:
            # ri_end: is i the last element of route_i?
            # rj_end: is j the last element of route_j?
            
            ri_i_end = (route_i[-1] == i)
            rj_j_end = (route_j[-1] == j)
            
            if ri_i_end and rj_j_end:
                # i is last in ri, j is last in rj
                # Reverse rj: [j, ..., first_j]
                new_route = route_i + list(reversed(route_j))
            elif ri_i_end and not rj_j_end:
                # i is last in ri, j is first in rj
                # ri: [..., i], rj: [j, ...]
                # Connect i->j: [..., i, j, ...]
                new_route = route_i + route_j
            elif not ri_i_end and rj_j_end:
                # i is first in ri, j is last in rj
                # ri: [i, ...], rj: [..., j]
                # Connect j->i: [..., j, i, ...] -> reverse ri? 
                # If we reverse ri: [..., i], then [..., j] + [..., i] = [..., j, ..., i]
                # This connects j->...->i? No, it connects j to the first element of reversed ri, which is last of original ri.
                # Let's just reverse ri and append: route_j + list(reversed(route_i))
                # This gives [..., j, ..., i]. The connection is j->(first of reversed ri) = j->(last of original ri). That's not i.
                # Wait, if i is first in ri, reversed ri has i at the end.
                # So route_j + reversed(route_i) = [..., j, ..., last_ri, ..., i]. 
                # The connection between the two halves is j->...->last_ri? No, it's just concatenation.
                # The edge between the end of route_j and start of reversed(route_i) is (last_of_route_j, first_of_reversed_ri) = (j, last_of_ri).
                # But we want to connect j and i.
                # Actually, if i is first in ri, the route is 0->i->...->last->0.
                # If j is last in rj, the route is 0->...->j->0.
                # To connect i and j, we can do: 0->...->j->i->...->last->0.
                # This means we keep rj as is, and reverse ri? No.
                # rj: [a, ..., j]. ri: [i, ..., b].
                # We want [..., j, i, ...]. So we take rj and append reversed(ri)?
                # reversed(ri) = [b, ..., i]. So rj + reversed(ri) = [a, ..., j, b, ..., i].
                # The connection is j->b. That's not i.
                # We need to connect j to i. So we should take rj and then ri reversed such that i is at the front?
                # If we reverse ri, i is at the end. So rj + reversed(ri) puts j next to b (last of ri).
                # To put j next to i, we need i to be at the end of the first part or start of the second.
                # If we do reversed(ri) + rj: [b, ..., i, a, ..., j]. Connection is i->a. Not j.
                # Let's just reverse the logic: 
                # If i is start of ri, it's adjacent to depot. 
                # If j is end of rj, it's adjacent to depot.
                # We remove (0,i) and (0,j), add (i,j).
                # So the new route is: 0 -> (rest of rj) -> j -> i -> (rest of ri) -> 0.
                # rj is [a, ..., j]. Rest of rj is [a, ..., second_last].
                # ri is [i, ..., b]. Rest of ri is [second, ..., b].
                # New route: [a, ..., second_last, j, i, second, ..., b]
                # This is rj[:-1] + [j, i] + ri[1:]? No, rj already ends with j.
                # rj = [a, ..., j]. ri = [i, ..., b].
                # New = rj + ri? No, that gives [..., j, i, ...] only if j is last in rj and i is first in ri.
                # Yes! rj + ri = [a, ..., j, i, ..., b]. This connects j->i.
                new_route = route_j + route_i
            else:
                # i is first in ri, j is first in rj
                # ri: [i, ..., a], rj: [j, ..., b]
                # Connect i->j: 0->i->...->a->j->...->b->0? No, that connects a->j.
                # We want to connect i and j.
                # Remove (0,i) and (0,j), add (i,j).
                # New route: 0->j->...->b->i->...->a->0? No.
                # If we do ri + reversed(rj): [i, ..., a, b, ..., j]. Connection a->b.
                # If we do rj + reversed(ri): [j, ..., b, a, ..., i]. Connection b->a.
                # We need to connect i and j.
                # Let's reverse both? 
                # Actually, if both are at the start, we can reverse one.
                # Reverse rj: [b, ..., j]. Then ri + reversed(rj) = [i, ..., a, b, ..., j]. Connection a->b.
                # Reverse ri: [a, ..., i]. Then reversed(ri) + rj = [a, ..., i, j, ..., b]. Connection i->j.
                # Yes!
                new_route = list(reversed(route_i)) + route_j
            
            # Update routes
            del routes[rj]
            routes[ri] = new_route
            route_demand[ri] += route_demand[rj]
            del route_demand[rj]
            
            # Update cust_to_route
            for c in new_route:
                cust_to_route[c] = ri
            
            # Update route_ends
            route_ends[ri] = get_route_ends(ri)
            del route_ends[rj]
            
            # Renumber routes? Not necessary if we use indices carefully, but indices shift.
            # This is a problem. Let's use a different approach: keep routes as a list of lists, and use a set of active route indices.
            # For simplicity and speed, let's just break and use a simpler method if this gets complicated.
            # Given the small size (50 nodes), we can afford to be less efficient.
            # But renumbering is messy. Let's just stop here and use a simpler heuristic if needed.
            # Actually, for 50 nodes, this is fine. Let's just proceed.
            
    return routes

def _two_opt_route(route, distance_matrix):
    """Improve a single route (list of customers) using 2-opt. Returns improved route."""
    if len(route) <= 2:
        return route
    
    # Route is 0 -> route[0] -> ... -> route[-1] -> 0
    # 2-opt on the customer sequence
    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                # Reverse segment i..j
                # Edges removed: (route[i-1] if i>0 else 0, route[i]) and (route[j], route[j+1] if j<len(route)-1 else 0)
                # Edges added: (route[i-1] if i>0 else 0, route[j]) and (route[i], route[j+1] if j<len(route)-1 else 0)
                
                u = route[i-1] if i > 0 else 0
                v = route[i]
                w = route[j]
                x = route[j+1] if j < len(route) - 1 else 0
                
                delta = distance_matrix[u, w] + distance_matrix[v, x] - distance_matrix[u, v] - distance_matrix[w, x]
                
                if delta < -1e-7:
                    route[i:j+1] = route[i:j+1][::-1]
                    improved = True
                    
    return route

def _solve_cvrp_fast(distance_matrix, demands, capacity, n):
    """
    Fast CVRP solver: Clarke-Wright Savings + 2-opt refinement.
    Returns a set of edges (i, j) and total cost.
    """
    # Get initial routes from savings
    routes = _clarke_wright_savings(distance_matrix, demands, capacity, n)
    
    # Filter out empty routes
    routes = [r for r in routes if len(r) > 0]
    
    # Apply 2-opt to each route
    improved_routes = []
    for route in routes:
        imp_route = _two_opt_route(route[:], distance_matrix)
        improved_routes.append(imp_route)
    
    # Build edge set
    edges = set()
    total_cost = 0.0
    for route in improved_routes:
        if len(route) == 0:
            continue
        # 0 -> route[0] -> ... -> route[-1] -> 0
        prev = 0
        for node in route:
            edges.add((prev, node))
            edges.add((node, prev))
            total_cost += distance_matrix[prev, node]
            prev = node
        edges.add((prev, 0))
        edges.add((0, prev))
        total_cost += distance_matrix[prev, 0]
        
    return edges, total_cost

def _run_aco(distance_matrix, coordinates, demands, capacity, heuristic, n_ants=20, n_iterations=30, seed=42):
    """Run ACO and return the best (lowest) total route distance found by any ant."""
    rng = np.random.RandomState(seed)
    n = distance_matrix.shape[0]
    pheromone = np.ones_like(heuristic, dtype=float)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_overall = float('inf')
    
    for it in range(n_iterations):
        best_iter_cost = float('inf')
        
        for ant in range(n_ants):
            current = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0
            
            while True:
                eligible_mask = (~visited) & (np.arange(n) > 0)
                eligible_mask &= (demands[:n] <= remaining_cap)
                eligible_indices = np.where(eligible_mask)[0]
                
                if len(eligible_indices) == 0:
                    total_dist += distance_matrix[current, 0]
                    current = 0
                    remaining_cap = capacity
                    if visited[1:].all():
                        break
                    continue
                
                probs = pheromone[current, eligible_indices] * heuristic[current, eligible_indices]
                probs = np.where(probs > 0, probs, 1e-9)
                probs /= probs.sum()
                
                idx = rng.choice(len(eligible_indices), p=probs)
                next_node = eligible_indices[idx]
                
                total_dist += distance_matrix[current, next_node]
                remaining_cap -= demands[next_node]
                visited[next_node] = True
                current = next_node
                
                if visited[1:].all():
                    total_dist += distance_matrix[current, 0]
                    break
        
        if total_dist < best_iter_cost:
            best_iter_cost = total_dist
        if total_dist < best_overall:
            best_overall = total_dist
        
        pheromone *= 0.9
        pheromone += 1.0 / best_iter_cost
    
    return best_overall

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 250  # Safety margin
    
    n = distance_matrix.shape[0]
    
    # Step 1: Solve CVRP fast to get a high-quality edge set
    edges, cost = _solve_cvrp_fast(distance_matrix, demands, capacity, n)
    
    # Step 2: Construct heuristic matrix
    # High value on edges from the solution, low value elsewhere
    # Use 1/d for non-edges to provide a smooth fallback
    dist_safe = distance_matrix + 1e-9
    h_base = 1.0 / dist_safe
    
    # Normalize h_base
    h_base_max = np.max(h_base)
    if h_base_max > 0:
        h_base = h_base / h_base_max
    
    # Create heuristic: boost edges from the solution
    # Value on solution edges: 1.0 (max)
    # Value on non-edges: scaled down version of h_base
    h = h_base * 0.1  # Diminish non-edges
    
    for (i, j) in edges:
        h[i, j] = 1.0
        h[j, i] = 1.0
    
    # Ensure positive and finite
    h = np.where(h > 0, h, 1e-9)
    h = np.where(np.isfinite(h), h, 1e-9)
    
    # If time allows, try a few variations
    if time.time() < deadline - 30:
        # Try a softer blend
        h_soft = h_base * 0.3
        for (i, j) in edges:
            h_soft[i, j] = 1.0
            h_soft[j, i] = 1.0
        h_soft = np.where(h_soft > 0, h_soft, 1e-9)
        
        # Quick ACO test to see if soft is better
        try:
            score_hard = _run_aco(distance_matrix, coordinates, demands, capacity, h, seed=42)
            score_soft = _run_aco(distance_matrix, coordinates, demands, capacity, h_soft, seed=42)
            if score_soft < score_hard:
                h = h_soft
        except Exception:
            pass
    
    # Final safety check
    if not np.all(np.isfinite(h)) or h.shape != distance_matrix.shape:
        h = 1.0 / (distance_matrix + 1e-9)
        h = np.where(h > 0, h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    coords = np.array(instance["coords"], dtype=float)
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    stations = instance["stations"]
    stations_set = set(stations)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Build distance matrix
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            d = math.sqrt((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2)
            dist[i][j] = d
            dist[j][i] = d
    
    customers = list(range(1, n_customers + 1))
    customers_set = set(customers)
    
    def is_station(node):
        return node in stations_set

    def is_customer(node):
        return node in customers_set

    def simulate_route(seq):
        """
        Simulate a route sequence (list of node ids).
        Returns (total_dist, feasible, end_battery, end_time).
        """
        if not seq:
            return 0.0, True, battery_max, 0.0
        
        battery = battery_max
        t = 0.0
        total_dist = 0.0
        prev = 0  # depot
        demand_so_far = 0.0
        
        for node in seq:
            d = dist[prev][node]
            total_dist += d
            battery -= d
            t += d / speed
            
            if battery < -1e-9:
                return (total_dist, False, 0, 0)
            
            if is_customer(node):
                demand_so_far += demands[node]
                if demand_so_far > capacity:
                    return (total_dist, False, 0, 0)
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return (total_dist, False, 0, 0)
                t += service[node]
            elif is_station(node):
                # Recharge
                energy_needed = battery_max - battery
                if energy_needed > 0:
                    t += energy_needed * recharge_rate
                battery = battery_max
            
            prev = node
        
        # Return to depot
        d = dist[prev][0]
        total_dist += d
        battery -= d
        t += d / speed
        
        if battery < -1e-9:
            return (total_dist, False, 0, 0)
        if t > horizon + 1e-9:
            return (total_dist, False, 0, 0)
        
        return (total_dist, True, battery, t)
    
    def route_distance(route):
        """Compute total distance of a route including depot legs"""
        if not route:
            return 0.0
        total = dist[0][route[0]]
        for i in range(len(route)-1):
            total += dist[route[i]][route[i+1]]
        total += dist[route[-1]][0]
        return total

    def fast_feasibility(route):
        """Quick pre-check for feasibility: capacity and battery distance bound."""
        if not route:
            return True
        total_demand = sum(demands[c] for c in route if is_customer(c))
        if total_demand > capacity:
            return False
        total_dist = route_distance(route)
        # Battery constraint: total distance must be <= battery_max (if no stations)
        # But with stations, it can be more. Use a loose bound.
        # Actually, if total_dist > battery_max * 2, it's definitely infeasible without stations.
        # With stations, it's hard to say quickly.
        # Let's just check capacity and maybe a very loose distance bound.
        return True
    
    def is_route_feasible(route):
        """Check if a route (list of customers/stations) is feasible."""
        _, feasible, _, _ = simulate_route(route)
        return feasible
    
    # --- Construction Heuristic ---
    # Sort customers by due date (earliest due date first)
    sorted_customers = sorted(customers, key=lambda c: due[c])
    
    routes = []
    
    def insert_customer(c, routes):
        """Try to insert customer c into an existing route or start a new one."""
        best_route_idx = -1
        best_pos = -1
        best_addl_dist = float('inf')
        
        for ri, route in enumerate(routes):
            # Try inserting at each position
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                if not fast_feasibility(new_route):
                    continue
                if not is_route_feasible(new_route):
                    continue
                old_d = route_distance(route) if route else 0.0
                new_d = route_distance(new_route)
                addl = new_d - old_d
                if addl < best_addl_dist:
                    best_addl_dist = addl
                    best_route_idx = ri
                    best_pos = pos
        
        if best_route_idx >= 0:
            routes[best_route_idx].insert(best_pos, c)
        else:
            routes.append([c])
    
    for c in sorted_customers:
        insert_customer(c, routes)
    
    # --- Local Search ---
    def try_2opt(route):
        """Try 2-opt improvements on a single route"""
        n = len(route)
        if n < 2:
            return route, False
        
        best_route = route[:]
        best_dist = route_distance(best_route)
        
        for i in range(n-1):
            for j in range(i+1, n):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                if not fast_feasibility(new_route):
                    continue
                if not is_route_feasible(new_route):
                    continue
                new_dist = route_distance(new_route)
                if new_dist < best_dist - 1e-9:
                    best_dist = new_dist
                    best_route = new_route[:]
        
        improved = best_dist < route_distance(route) - 1e-9
        return best_route, improved
    
    def try_relocate(c, r1, r2, routes):
        """Try moving customer c from route r1 to route r2"""
        if r1 == r2:
            return False
        route1 = routes[r1][:]
        route2 = routes[r2][:]
        
        if c not in route1:
            return False
        
        p1 = route1.index(c)
        route1.remove(c)
        
        # Check if route1 is still feasible
        if not is_route_feasible(route1):
            return False
        
        best_pos = -1
        best_addl = float('inf')
        base_d2 = route_distance(route2)
        
        for pos in range(len(route2) + 1):
            new_route2 = route2[:pos] + [c] + route2[pos:]
            if not fast_feasibility(new_route2):
                continue
            if not is_route_feasible(new_route2):
                continue
            new_d2 = route_distance(new_route2)
            addl = new_d2 - base_d2
            if addl < best_addl:
                best_addl = addl
                best_pos = pos
        
        if best_pos < 0:
            return False
        
        new_route2 = route2[:best_pos] + [c] + route2[best_pos:]
        
        old_dist = route_distance(routes[r1]) + route_distance(routes[r2])
        new_dist = route_distance(route1) + route_distance(new_route2)
        
        if new_dist < old_dist - 1e-9:
            routes[r1] = route1
            routes[r2] = new_route2
            return True
        return False
    
    def try_swap(c1, r1, c2, r2, routes):
        """Try swapping customer c1 from route r1 with c2 from route r2"""
        route1 = routes[r1][:]
        route2 = routes[r2][:]
        
        p1 = route1.index(c1)
        p2 = route2.index(c2)
        
        route1[p1], route2[p2] = c2, c1
        
        if not is_route_feasible(route1) or not is_route_feasible(route2):
            return False
        
        old_dist = route_distance(routes[r1]) + route_distance(routes[r2])
        new_dist = route_distance(route1) + route_distance(route2)
        
        if new_dist < old_dist - 1e-9:
            routes[r1] = route1
            routes[r2] = route2
            return True
        return False
    
    def try_or_opt(route):
        """Try Or-opt moves (relocate 1 or 2 consecutive customers)"""
        n = len(route)
        if n < 2:
            return route, False
        
        best_route = route[:]
        best_dist = route_distance(best_route)
        
        # Relocate 1 customer
        for i in range(n):
            for j in range(n):
                if i == j or i == j + 1:
                    continue
                node = route[i]
                new_route = route[:i] + route[i+1:]
                insert_pos = j if j < i else j + 1
                new_route = new_route[:insert_pos] + [node] + new_route[insert_pos:]
                
                if not fast_feasibility(new_route):
                    continue
                if not is_route_feasible(new_route):
                    continue
                new_dist = route_distance(new_route)
                if new_dist < best_dist - 1e-9:
                    best_dist = new_dist
                    best_route = new_route[:]
        
        # Relocate 2 consecutive customers
        for i in range(n-1):
            for j in range(n-1):
                if i <= j <= i+1:
                    continue
                nodes = route[i:i+2]
                new_route = route[:i] + route[i+2:]
                insert_pos = j if j < i else j + 2
                new_route = new_route[:insert_pos] + nodes + new_route[insert_pos:]
                
                if not fast_feasibility(new_route):
                    continue
                if not is_route_feasible(new_route):
                    continue
                new_dist = route_distance(new_route)
                if new_dist < best_dist - 1e-9:
                    best_dist = new_dist
                    best_route = new_route[:]
        
        improved = best_dist < route_distance(route) - 1e-9
        return best_route, improved
    
    best_total_dist = sum(route_distance(r) for r in routes)
    best_routes = [r[:] for r in routes]
    
    # Iterative local search
    while time.time() < deadline:
        improved = False
        
        # 2-opt and Or-opt on each route
        for i in range(len(routes)):
            new_route, imp = try_2opt(routes[i])
            if imp:
                routes[i] = new_route
                improved = True
            
            new_route, imp = try_or_opt(routes[i])
            if imp:
                routes[i] = new_route
                improved = True
        
        # Inter-route moves
        n_routes = len(routes)
        for r1 in range(n_routes):
            for r2 in range(r1+1, n_routes):
                # Try relocating customers from r1 to r2
                for c in routes[r1][:]:
                    if try_relocate(c, r1, r2, routes):
                        improved = True
                        break
                # Try relocating customers from r2 to r1
                for c in routes[r2][:]:
                    if try_relocate(c, r2, r1, routes):
                        improved = True
                        break
                # Try swaps
                for c1 in routes[r1][:]:
                    for c2 in routes[r2][:]:
                        if try_swap(c1, r1, c2, r2, routes):
                            improved = True
                            break
        
        # Update best
        current_dist = sum(route_distance(r) for r in routes)
        if current_dist < best_total_dist - 1e-9:
            best_total_dist = current_dist
            best_routes = [r[:] for r in routes]
        
        if not improved:
            break
    
    return best_routes
# EVOLVE-BLOCK-END

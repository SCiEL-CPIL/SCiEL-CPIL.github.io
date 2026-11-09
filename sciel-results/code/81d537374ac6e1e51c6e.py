# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _dist_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _check_route(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
    """Check if a route is feasible. Returns (feasible, total_distance, total_time)."""
    if not route:
        return True, 0.0, 0.0
    
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > capacity:
        return False, 0.0, 0.0
    
    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0  # depot
    
    for v in route:
        d = dist[prev][v]
        battery -= d
        time_t += d / speed
        total_dist += d
        
        if battery < -1e-9:
            return False, 0.0, 0.0
        
        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0, 0.0
            time_t += service[v]
        else:
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max
        
        prev = v
    
    d = dist[prev][0]
    battery -= d
    time_t += d / speed
    total_dist += d
    
    if battery < -1e-9:
        return False, 0.0, 0.0
    if time_t > horizon + 1e-9:
        return False, 0.0, 0.0
    
    return True, total_dist, time_t


def _route_distance(route, dist):
    """Quick distance calculation for a route."""
    if not route:
        return 0.0
    total = 0.0
    prev = 0
    for v in route:
        total += dist[prev][v]
        prev = v
    total += dist[prev][0]
    return total


def _cluster_customers(coords, n_customers, n_clusters=5):
    """Simple k-means-like clustering to group spatially close customers."""
    customers = list(range(1, n_customers + 1))
    random.shuffle(customers)
    
    # Initialize centroids
    centroids = [np.array(coords[c], dtype=float) for c in customers[:n_clusters]]
    assignments = {}
    
    for _ in range(10):
        new_assignments = {}
        for c in customers:
            best_cluster = 0
            best_dist = float('inf')
            for i, cent in enumerate(centroids):
                d = np.linalg.norm(np.array(coords[c]) - cent)
                if d < best_dist:
                    best_dist = d
                    best_cluster = i
            new_assignments[c] = best_cluster
        
        if new_assignments == assignments:
            break
        assignments = new_assignments
        
        # Update centroids
        for i in range(n_clusters):
            members = [c for c in customers if assignments[c] == i]
            if members:
                centroids[i] = np.mean([np.array(coords[c]) for c in members], axis=0)
    
    clusters = [[] for _ in range(n_clusters)]
    for c in customers:
        clusters[assignments[c]].append(c)
    
    return clusters


def _construct_cluster_routes(instance, dist, time_limit=5):
    """Construct initial routes using spatial clustering + time-window-aware insertion."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Cluster customers spatially
    n_clusters = min(5, n_customers)
    clusters = _cluster_customers(coords, n_customers, n_clusters)
    
    routes = []
    unassigned = set(range(1, n_customers + 1))
    
    for cluster in clusters:
        if not cluster:
            continue
        
        # Sort by due date within cluster
        cluster.sort(key=lambda c: due[c])
        
        current_route = []
        current_demand = 0
        
        for c in cluster:
            if c not in unassigned:
                continue
            if time.time() > deadline:
                break
            
            if current_demand + demands[c] > capacity:
                if current_route:
                    routes.append(current_route)
                    current_route = []
                    current_demand = 0
                else:
                    routes.append([c])
                    unassigned.remove(c)
                    continue
            
            # Try to insert c into current_route at best feasible position
            best_pos = -1
            best_cost = float('inf')
            
            for pos in range(len(current_route) + 1):
                new_route = current_route[:pos] + [c] + current_route[pos:]
                feasible, d, t = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                if feasible and d < best_cost:
                    best_cost = d
                    best_pos = pos
            
            if best_pos != -1:
                current_route = current_route[:best_pos] + [c] + current_route[best_pos:]
                current_demand += demands[c]
                unassigned.remove(c)
            else:
                # Can't insert, start new route with just this customer
                if current_route:
                    routes.append(current_route)
                current_route = [c]
                current_demand = demands[c]
                unassigned.remove(c)
        
        if current_route:
            routes.append(current_route)
    
    # Check for any missing customers and add them as singletons if possible
    # This should ideally be empty if clusters cover all customers
    for c in sorted(unassigned):
        routes.append([c])
    
    return routes


class _RouteState:
    """Maintains prefix/suffix feasibility info for fast insertion checks."""
    
    def __init__(self, route, dist, instance):
        self.route = list(route)
        self.dist = dist
        self.instance = instance
        self.battery_max = instance["battery"]
        self.capacity = instance["capacity"]
        self.speed = instance["speed"]
        self.recharge_rate = instance["recharge_rate"]
        self.horizon = instance["horizon"]
        self.n_customers = instance["n_customers"]
        self.demands = instance["demands"]
        self.ready = instance["ready"]
        self.due = instance["due"]
        self.service = instance["service"]
        
        self.total_dist = 0.0
        self.total_demand = 0
        self._recompute()
    
    def _recompute(self):
        """Recompute all prefix/suffix states."""
        n = len(self.route)
        d = self.dist
        
        # Prefix: state after visiting route[0..i]
        # prefix_time[i] = time at end of service/charge at node i
        # prefix_batt[i] = battery after leaving node i (i.e., at start of travel to i+1)
        # prefix_dist[i] = cumulative distance up to and including edge to node i
        
        self.prefix_time = [0.0] * n
        self.prefix_batt = [0.0] * n
        self.prefix_dist = [0.0] * n
        
        battery = self.battery_max
        t = 0.0
        total_d = 0.0
        total_dem = 0
        
        for i, v in enumerate(self.route):
            prev = 0 if i == 0 else self.route[i-1]
            dd = d[prev][v]
            battery -= dd
            t += dd / self.speed
            total_d += dd
            
            if v <= self.n_customers:
                total_dem += self.demands[v]
                if t < self.ready[v]:
                    t = self.ready[v]
                # t <= due[v] must hold for feasibility
                t += self.service[v]
            else:
                energy_needed = self.battery_max - battery
                t += energy_needed * self.recharge_rate
                battery = self.battery_max
            
            self.prefix_time[i] = t
            self.prefix_batt[i] = battery
            self.prefix_dist[i] = total_d
        
        # Suffix: state before arriving at node i from depot side
        # We need: if we insert at position pos, the travel from route[pos-1] to new node,
        # then from new node to route[pos], etc.
        # suffix_time[i] = time needed from node i to depot (including service at i)
        # suffix_batt[i] = battery needed at node i to reach depot
        
        self.suffix_time = [0.0] * n
        self.suffix_batt = [0.0] * n
        self.suffix_dist = [0.0] * n
        
        # Compute from end
        # suffix at last node: travel to depot
        # We'll compute "cost to finish" from each position
        
        # Let's define: from position i (at node route[i] after service), 
        # what's the time and battery to get back to depot?
        # This is tricky with charging stations. Let's just store the full route check.
        
        self.total_dist = total_d + d[self.route[-1]][0] if n > 0 else 0.0
        self.total_demand = total_dem
        
        # For fast insertion, we'll just do full check but with early termination
        # Actually, let's precompute a "forward" and "backward" pass
    
    def check_insertion(self, customer, pos):
        """Quick check if inserting customer at pos is feasible. Returns (feasible, new_dist)."""
        route = self.route
        d = self.dist
        n = len(route)
        
        # New route would be route[:pos] + [customer] + route[pos:]
        # Check demand
        if self.total_demand + self.demands[customer] > self.capacity:
            return False, float('inf')
        
        # Estimate distance change
        if n == 0:
            new_dist = d[0][customer] + d[customer][0]
        else:
            if pos == 0:
                old_edge = d[0][route[0]]
                new_edges = d[0][customer] + d[customer][route[0]]
            elif pos == n:
                old_edge = d[route[-1]][0]
                new_edges = d[route[-1]][customer] + d[customer][0]
            else:
                old_edge = d[route[pos-1]][route[pos]]
                new_edges = d[route[pos-1]][customer] + d[customer][route[pos]]
            delta = new_edges - old_edge
            new_dist = self.total_dist + delta
        
        # Quick feasibility: check if new_dist is reasonable and battery allows
        # For full check, we'd need to simulate. Let's do a lightweight simulation.
        battery = self.battery_max
        t = 0.0
        prev = 0
        
        for i, v in enumerate(route):
            if i == pos:
                # Insert customer before v
                dd = d[prev][customer]
                battery -= dd
                t += dd / self.speed
                if battery < -1e-9:
                    return False, float('inf')
                if t < self.ready[customer]:
                    t = self.ready[customer]
                if t > self.due[customer] + 1e-9:
                    return False, float('inf')
                t += self.service[customer]
                prev = customer
                # Now continue to v
                dd2 = d[customer][v]
                battery -= dd2
                t += dd2 / self.speed
                if battery < -1e-9:
                    return False, float('inf')
                # Continue with v
                if v <= self.n_customers:
                    if t < self.ready[v]:
                        t = self.ready[v]
                    if t > self.due[v] + 1e-9:
                        return False, float('inf')
                    t += self.service[v]
                else:
                    energy_needed = self.battery_max - battery
                    t += energy_needed * self.recharge_rate
                    battery = self.battery_max
                prev = v
            else:
                dd = d[prev][v]
                battery -= dd
                t += dd / self.speed
                if battery < -1e-9:
                    return False, float('inf')
                
                if v <= self.n_customers:
                    if t < self.ready[v]:
                        t = self.ready[v]
                    if t > self.due[v] + 1e-9:
                        return False, float('inf')
                    t += self.service[v]
                else:
                    energy_needed = self.battery_max - battery
                    t += energy_needed * self.recharge_rate
                    battery = self.battery_max
                prev = v
        
        # Check return to depot
        dd = d[prev][0]
        battery -= dd
        t += dd / self.speed
        if battery < -1e-9:
            return False, float('inf')
        if t > self.horizon + 1e-9:
            return False, float('inf')
        
        return True, new_dist


def _local_search(routes, dist, instance, time_limit=40):
    """
    Efficient local search with 2-Opt, Or-Opt, and Relocate moves.
    Uses fast distance delta calculations to avoid full route re-evaluation
    unless a potential improvement is found.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute route distances
    route_dists = [_route_distance(r, dist) for r in routes]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. 2-Opt within routes
        for i in range(len(routes)):
            r = routes[i]
            n = len(r)
            if n < 2:
                continue
            
            for a in range(n - 1):
                for b in range(a + 1, n):
                    # Calculate delta distance for reversing r[a:b+1]
                    prev_a = r[a-1] if a > 0 else 0
                    next_b = r[b+1] if b < n - 1 else 0
                    
                    old_d = dist[prev_a][r[a]] + dist[r[b]][next_b]
                    new_d = dist[prev_a][r[b]] + dist[r[a]][next_b]
                    
                    if new_d < old_d - 1e-9:
                        new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                        feasible, d, t = _check_route(new_r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                        if feasible:
                            routes[i] = new_r
                            route_dists[i] = d
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # 2. Or-Opt within routes (move 1 or 2 consecutive nodes)
        for i in range(len(routes)):
            r = routes[i]
            n = len(r)
            if n < 3:
                continue
            
            for L in [1, 2]:
                if n < L + 1:
                    continue
                for start in range(n - L):
                    moving = r[start:start+L]
                    if any(m > n_customers for m in moving):
                        continue
                    
                    rest = r[:start] + r[start+L:]
                    best_pos = -1
                    best_dist = float('inf')
                    
                    for pos in range(len(rest) + 1):
                        if pos == start or (start > 0 and pos == start + L) or (start == 0 and pos == L):
                            continue
                        
                        new_r = rest[:pos] + moving + rest[pos:]
                        d_new = _route_distance(new_r, dist)
                        if d_new < route_dists[i] - 1e-9:
                            feasible, d, t = _check_route(new_r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                            if feasible:
                                best_pos = pos
                                best_dist = d
                                break
                    
                    if best_pos != -1:
                        new_r = rest[:best_pos] + moving + rest[best_pos:]
                        routes[i] = new_r
                        route_dists[i] = best_dist
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue

        # 3. Relocate: Move customer from route i to route j
        for i in range(len(routes)):
            r_i = routes[i]
            if not r_i:
                continue
            for pos_i in range(len(r_i)):
                c = r_i[pos_i]
                if c > n_customers:
                    continue
                
                for j in range(len(routes)):
                    if i == j:
                        continue
                    r_j = routes[j]
                    
                    demand_i = sum(instance["demands"][v] for v in r_i if v <= n_customers)
                    demand_j = sum(instance["demands"][v] for v in r_j if v <= n_customers)
                    
                    if demand_j + instance["demands"][c] > capacity:
                        continue
                    
                    best_pos_j = -1
                    best_dist_j = float('inf')
                    
                    if c in r_j:
                        continue
                        
                    for pos_j in range(len(r_j) + 1):
                        new_r_j = r_j[:pos_j] + [c] + r_j[pos_j:]
                        d_j_new = _route_distance(new_r_j, dist)
                        if d_j_new < best_dist_j:
                            feasible, d, t = _check_route(new_r_j, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                            if feasible:
                                best_dist_j = d
                                best_pos_j = pos_j
                                break
                    
                    if best_pos_j != -1:
                        new_r_j = r_j[:best_pos_j] + [c] + r_j[best_pos_j:]
                        new_r_i = r_i[:pos_i] + r_i[pos_i+1:]
                        
                        d_i_new = _route_distance(new_r_i, dist) if new_r_i else 0.0
                        d_j_new = best_dist_j
                        
                        total_new = d_i_new + d_j_new
                        total_old = route_dists[i] + route_dists[j]
                        
                        if total_new < total_old - 1e-9:
                            if new_r_i:
                                feasible_i, _, _ = _check_route(new_r_i, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                            else:
                                feasible_i = True
                            
                            if feasible_i:
                                routes[i] = new_r_i
                                routes[j] = new_r_j
                                route_dists[i] = d_i_new
                                route_dists[j] = d_j_new
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break
            
            if not r_i:
                routes.pop(i)
                route_dists.pop(i)
                break
        
        if improved:
            continue
            
    return routes


def _solve_instance(instance, time_limit=45):
    """Solve a single instance."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    dist = _dist_matrix(coords)
    
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]
    
    construct_time = min(3, time_limit * 0.1)
    routes = _construct_cluster_routes(instance, dist, construct_time)
    
    all_served = set()
    for r in routes:
        all_served.update(r)
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        routes.append([c])
    
    ls_time = deadline - time.time() - 2
    if ls_time > 1:
        routes = _local_search(routes, dist, instance, ls_time)
    
    valid_routes = []
    for r in routes:
        feasible, _, _ = _check_route(r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            valid_routes.append(r)
            
    # Deduplicate: Ensure each customer appears at most once
    seen = set()
    cleaned_routes = []
    for r in valid_routes:
        new_r = []
        for c in r:
            if c not in seen:
                new_r.append(c)
                seen.add(c)
            else:
                # Customer already served, skip this occurrence
                pass
        if new_r:
            # Re-check feasibility after removal (removal makes it easier, so it should still be feasible)
            feasible, _, _ = _check_route(new_r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            if feasible:
                cleaned_routes.append(new_r)
    
    valid_routes = cleaned_routes
    
    # Check for missing customers
    all_served = set()
    for r in valid_routes:
        all_served.update(r)
    
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        # Try to insert into existing routes first
        inserted = False
        for i in range(len(valid_routes)):
            r = valid_routes[i]
            for pos in range(len(r) + 1):
                new_r = r[:pos] + [c] + r[pos:]
                feasible, _, _ = _check_route(new_r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                if feasible:
                    valid_routes[i] = new_r
                    inserted = True
                    break
            if inserted:
                break
        if not inserted:
            # Create new route
            new_r = [c]
            feasible, _, _ = _check_route(new_r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            if feasible:
                valid_routes.append(new_r)
            else:
                # If even a singleton is infeasible (e.g. battery/time), we have a problem.
                # But usually singletons are feasible.
                valid_routes.append(new_r)
    
    return valid_routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    per_instance_time = 45
    routes = _solve_instance(instance, per_instance_time)
    return routes
# EVOLVE-BLOCK-END

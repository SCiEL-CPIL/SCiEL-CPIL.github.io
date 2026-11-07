<<<<<<< SEARCH
def plan(instance):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    n_customers = instance["n_customers"]
    D = _build_dist_matrix(instance["coords"])

    # Generate initial solution using Clarke-Wright savings
    routes = _clarke_wright_savings(instance, D)

    if not _is_valid_solution(routes, instance, D):
        # Fallback: one customer per route
        customers = list(range(1, n_customers + 1))
        customers.sort(key=lambda v: instance["due"][v])
        routes = [[v] for v in customers]

    best_routes = [list(r) for r in routes]
    best_dist = _total_distance(best_routes, D)

    # Apply local search
    if time.time() < deadline - 10:
        for route_idx in range(len(best_routes)):
            best_routes[route_idx] = _2opt(best_routes[route_idx], instance, D, max_iters=30)
        if _is_valid_solution(best_routes, instance, D):
            d = _total_distance(best_routes, D)
            if d < best_dist:
                best_dist = d

    if time.time() < deadline - 10:
        best_routes = _or_opt(best_routes, instance, D, max_iters=20)
        if _is_valid_solution(best_routes, instance, D):
            d = _total_distance(best_routes, D)
            if d < best_dist:
                best_dist = d

    if time.time() < deadline - 10:
        best_routes = _cross_route_swap(best_routes, instance, D, max_iters=10)
        if _is_valid_solution(best_routes, instance, D):
            d = _total_distance(best_routes, D)
            if d < best_dist:
                best_dist = d

    # Additional passes if time remains
    while time.time() < deadline - 5:
        improved = False
        for route_idx in range(len(best_routes)):
            new_route = _2opt(best_routes[route_idx], instance, D, max_iters=20)
            if new_route != best_routes[route_idx]:
                best_routes[route_idx] = new_route
                if _is_valid_solution(best_routes, instance, D):
                    d = _total_distance(best_routes, D)
                    if d < best_dist - 1e-9:
                        best_dist = d
                        improved = True

        if time.time() < deadline - 5:
            new_routes = _or_opt(best_routes, instance, D, max_iters=10)
            if _is_valid_solution(new_routes, instance, D):
                d = _total_distance(new_routes, D)
                if d < best_dist - 1e-9:
                    best_routes = new_routes
                    best_dist = d
                    improved = True

        if time.time() < deadline - 5:
            new_routes = _cross_route_swap(best_routes, instance, D, max_iters=5)
            if _is_valid_solution(new_routes, instance, D):
                d = _total_distance(new_routes, D)
                if d < best_dist - 1e-9:
                    best_routes = new_routes
                    best_dist = d
                    improved = True

        if not improved:
            break

    return best_routes
=======
def plan(instance):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin

    n_customers = instance["n_customers"]
    D = _build_dist_matrix(instance["coords"])

    # Generate initial solution using regret-based insertion
    routes = _regret_insertion(instance, D)

    if not _is_valid_solution(routes, instance, D):
        # Fallback: one customer per route
        customers = list(range(1, n_customers + 1))
        customers.sort(key=lambda v: instance["due"][v])
        routes = [[v] for v in customers]

    best_routes = [list(r) for r in routes]
    best_dist = _total_distance(best_routes, D)

    # Apply local search
    if time.time() < deadline - 15:
        for route_idx in range(len(best_routes)):
            best_routes[route_idx] = _2opt(best_routes[route_idx], instance, D, max_iters=30)
        if _is_valid_solution(best_routes, instance, D):
            d = _total_distance(best_routes, D)
            if d < best_dist:
                best_dist = d

    if time.time() < deadline - 15:
        best_routes = _or_opt(best_routes, instance, D, max_iters=20)
        if _is_valid_solution(best_routes, instance, D):
            d = _total_distance(best_routes, D)
            if d < best_dist:
                best_dist = d

    if time.time() < deadline - 15:
        best_routes = _cross_route_swap(best_routes, instance, D, max_iters=10)
        if _is_valid_solution(best_routes, instance, D):
            d = _total_distance(best_routes, D)
            if d < best_dist:
                best_dist = d

    # Tabu search for deeper optimization
    if time.time() < deadline - 10:
        new_routes = _tabu_search(best_routes, instance, D, max_iters=150, tabu_tenure=10)
        if _is_valid_solution(new_routes, instance, D):
            d = _total_distance(new_routes, D)
            if d < best_dist - 1e-9:
                best_routes = new_routes
                best_dist = d

    # Additional passes if time remains
    while time.time() < deadline - 5:
        improved = False
        for route_idx in range(len(best_routes)):
            new_route = _2opt(best_routes[route_idx], instance, D, max_iters=20)
            if new_route != best_routes[route_idx]:
                best_routes[route_idx] = new_route
                if _is_valid_solution(best_routes, instance, D):
                    d = _total_distance(best_routes, D)
                    if d < best_dist - 1e-9:
                        best_dist = d
                        improved = True

        if time.time() < deadline - 5:
            new_routes = _or_opt(best_routes, instance, D, max_iters=10)
            if _is_valid_solution(new_routes, instance, D):
                d = _total_distance(new_routes, D)
                if d < best_dist - 1e-9:
                    best_routes = new_routes
                    best_dist = d
                    improved = True

        if time.time() < deadline - 5:
            new_routes = _cross_route_swap(best_routes, instance, D, max_iters=5)
            if _is_valid_solution(new_routes, instance, D):
                d = _total_distance(new_routes, D)
                if d < best_dist - 1e-9:
                    best_routes = new_routes
                    best_dist = d
                    improved = True

        if time.time() < deadline - 5:
            new_routes = _tabu_search(best_routes, instance, D, max_iters=50, tabu_tenure=8)
            if _is_valid_solution(new_routes, instance, D):
                d = _total_distance(new_routes, D)
                if d < best_dist - 1e-9:
                    best_routes = new_routes
                    best_dist = d
                    improved = True

        if not improved:
            break

    return best_routes
>>>>>>> REPLACE

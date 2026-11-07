<<<<<<< SEARCH
def plan(instance):
    global _DEADLINE
    _DEADLINE = time.time() + 150.0 - 12.0  # 138s effective

    # Solve single instance
    routes = _solve_instance(instance)

    # Verify all customers are served
    served = set()
    for r in routes:
        for v in r:
            if v > 0:
                served.add(v)
    expected = set(range(1, instance["n_customers"] + 1))
    if served != expected:
        # Fallback: one customer per route
        routes = [[v] for v in range(1, instance["n_customers"] + 1)]

    return routes
=======
def plan(instance):
    global _DEADLINE
    _DEADLINE = time.time() + 150.0 - 12.0  # 138s effective

    # Solve single instance
    routes = _solve_instance(instance)

    # Verify all customers are served
    served = set()
    for r in routes:
        for v in r:
            if v > 0:
                served.add(v)
    expected = set(range(1, instance["n_customers"] + 1))
    if served != expected:
        # Fallback: one customer per route
        routes = [[v] for v in range(1, instance["n_customers"] + 1)]

    return routes
>>>>>>> REPLACE

<<<<<<< SEARCH
    if best_n is None:
        # Should not happen; fall back to a guaranteed-valid voltage lift with a large m
        return _voltage_construct(k, g, max(k, 5) * 4, deadline)
=======
    if best_n is None:
        # Fallback: try progressively larger m values until one works or time runs out.
        # Large m reduces the probability of short cycles in the lift.
        current_m = 200
        while time.time() < deadline - 10.0:
            try:
                n, edges = _voltage_construct(k, g, current_m, deadline)
                if n is not None:
                    return n, edges
            except Exception:
                pass
            current_m += 100
        # Last resort: try a very large m
        return _voltage_construct(k, g, 1000, deadline)
>>>>>>> REPLACE

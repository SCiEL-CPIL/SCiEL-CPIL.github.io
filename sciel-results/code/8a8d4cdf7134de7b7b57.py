    # Phase 3: Multiple random restarts with greedy
    if time_left() > 20:
        random.seed(42)
        restarts_done = 0
        while time_left() > 15:
            restarts_done += 1
            # ... same logic ...

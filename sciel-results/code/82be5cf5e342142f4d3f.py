    def construct():
        deadline = time.time() + 50 # Safety margin
        # ...
        for step in range(50000): # Adjust count based on speed
            if time.time() > deadline:
                break
            # ...
            reduced_candidate = greedy_remove(candidate, deadline, max_passes=5)
            # ...
    
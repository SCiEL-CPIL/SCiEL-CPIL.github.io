    def solve(n_target, time_limit=100):
        start_time = time.time()
        for attempt in range(1000):
            if time.time() - start_time > time_limit:
                break
            # ...

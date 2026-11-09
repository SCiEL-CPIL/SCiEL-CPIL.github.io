        while time.time() < deadline:
            for _ in range(steps_per_restart):
                if time.time() > deadline:
                    break
                # ... step logic ...

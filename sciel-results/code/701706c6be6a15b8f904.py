    def counted_f(t, y):
        nonlocal call_count
        call_count += 1
        if time.time() > deadline:
            raise TimeoutError("Time limit exceeded")
        return f(t, y)

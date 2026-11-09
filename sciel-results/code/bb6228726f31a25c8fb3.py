        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            # ...
            call_count += 1
            return fx

    def local_search(coeffs, time_limit):
        current_coeffs = coeffs.copy()
        current_norm = _fast_norm(current_coeffs)
        start_search_time = time.time()
        
        while True:
            if time.time() > time_limit or (time.time() - start_search_time) > 5.0:
                break
            # ... flip logic ...
            
        return current_coeffs, current_norm

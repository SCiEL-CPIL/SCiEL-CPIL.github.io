import time

# Inside construct:
    def solve_instance_with_timeout(n, edges, lists, timeout=10.0):
        # ... setup adj, domains, assigned, colors ...
        
        start_time = time.time()
        
        # We need to signal timeout to the recursive function.
        # Using a global or nonlocal flag.
        
        timed_out = [False]
        
        def backtrack(remaining):
            if timed_out[0]:
                return False
            if time.time() - start_time > timeout:
                timed_out[0] = True
                return False
            
            if remaining == 0:
                return True
            
            # ... MRV ...
            
            for c in domains[v_idx]:
                # ... assign, forward check ...
                if valid:
                    if backtrack(remaining - 1):
                        return True
                # ... undo ...
            
            return False
        
        if backtrack(n) and not timed_out[0]:
            return colors
        
        # Fallback greedy
        # ...

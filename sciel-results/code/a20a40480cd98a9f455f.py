    # After finding a good k, try to extend
    def try_extend(s, k):
        # Get the set of phis
        phis = [phi(s + i) for i in range(k)]
        seen = set(phis)
        
        # Extend left
        while s > 1:
            ns = s - 1
            p_ns = phi(ns)
            if p_ns in seen:
                break
            seen.add(p_ns)
            s = ns
            k += 1
        
        # Extend right
        while s + k <= max_n:
            ns = s + k
            p_ns = phi(ns)
            if p_ns in seen:
                break
            seen.add(p_ns)
            k += 1
            
        return s, k

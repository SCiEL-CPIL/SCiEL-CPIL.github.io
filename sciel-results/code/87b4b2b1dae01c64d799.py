    for d in range(d_min, d_min + 6):
        if time.time() > deadline - 20.0:
            break
        try:
            n, edges = _luw(k, d)
        except Exception:
            continue
        
        # Trust the LUW girth guarantee for the first candidate to save time
        if d == d_min:
            best = (n, edges)
            break
            
        if time.time() > deadline - 10.0:
            if best is None:
                best = (n, edges)
            break
            
        if _check_girth(n, edges, g):
            best = (n, edges)
            break
   
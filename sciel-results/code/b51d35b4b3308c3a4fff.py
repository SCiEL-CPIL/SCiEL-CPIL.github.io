<<<<<<< SEARCH
    # Try heuristic for a few seconds
    best_result = None
    # Try N from 500 to 1200
    for N in [500, 600, 700, 800, 900, 1000, 1100, 1200]:
        if time.time() > deadline - 10:
            break
        # Try a few times for each N
        for _ in range(5):
            res = try_heuristic(N)
            if res:
                best_result = res
                break
        if best_result:
            break
            
    if best_result:
        return best_result
        
    # Fallback to LUW
    return build_luw()
=======
    # Skip heuristic search to ensure fast, reliable LUW construction
    # The LUW construction for k=5, g=9 (d=4) yields n=1250, which is a significant
    # improvement over the seed's d=5 (n=6250).
    return build_luw()
>>>>>>> REPLACE

    # Try ranks from low to high
    for r in range(1, 25):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iter=20000, seed=r)
        if result is not None:
            return result
        
        # Try a few more seeds
        for seed in [r+100, r+200, r+300]:
            if time.time() > deadline - 10:
                break
            result = try_rank(r, max_iter=10000, seed=seed)
            if result is not None:
                return result

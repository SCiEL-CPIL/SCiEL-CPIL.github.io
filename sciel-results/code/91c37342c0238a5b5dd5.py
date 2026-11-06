def construct():
    import time
    start_time = time.time()
    deadline = 90.0  # Safety margin for 60s limit? The prompt says 60s evaluator, 90s kill. 
    # Actually "evaluator KILLS ... after 90s". "Each evaluation must finish within 60s".
    # So we should aim for < 50s.
    
    n = 256
    S = []
    # We will store S as a list, and maybe a set for fast lookup if needed, 
    # but for sum checks we just iterate.
    
    # To speed up, we can maintain a set of all possible sums of pairs in S?
    # If we add x, we need to check if any sum in Sums is divisible by x.
    # When we add x, new sums are x + y for y in S.
    # We need to check if x divides any existing sum.
    
    sums = set()
    
    for x in range(n, 0, -1):
        if time.time() - start_time > 45:
            break
        
        # Check if x divides any sum in 'sums'
        # sums contains b+c for b,c in S with b <= c
        # We need to check if any s in sums is a multiple of x.
        
        valid = True
        # Optimization: only check sums up to 2*n? 
        # Actually sums are <= 2*n.
        # We can just iterate through sums.
        # Number of sums is O(|S|^2). |S| <= 128. Sums <= 8000.
        # Checking 8000 sums for 256 numbers is 2M ops.
        
        # We can be smarter: 
        # A sum s is divisible by x if s % x == 0.
        # We can just loop.
        
        for s in sums:
            if s % x == 0:
                valid = False
                break
        
        if valid:
            S.append(x)
            # Update sums
            # New sums are x + y for y in S (including x itself? b<=c, so x+x is valid if x is in S)
            # Wait, S currently contains x.
            # The pairs are (x, y) for y in S.
            # Since we are adding x to S, the new pairs are (x, x) and (x, y) for y in S \ {x}.
            # Actually, S before adding x was S_old.
            # Now S = S_old U {x}.
            # New sums are x + y for y in S_old, and x + x.
            
            for y in S:
                if y == x:
                    sums.add(x + x)
                else:
                    # y is in S_old, so y > x
                    sums.add(x + y)
            # Note: x+x is added once. x+y is added for each y.
            # Do we need to remove duplicates? Set handles it.
            
    return S

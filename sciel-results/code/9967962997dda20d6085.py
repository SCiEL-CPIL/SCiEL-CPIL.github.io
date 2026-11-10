    for L in range(1, N + 1):
        # Check time limit
        if time.time() > deadline:
            break
            
        # Expand R from L upwards
        R = L - 1
        # We check if we can add y = R+1
        while R < N:
            y = R + 1
            # The new element is y.
            # The interval would be [L, y].
            # It is valid iff inv[y] is NOT in [L, y].
            if L <= inv[y] <= y:
                break
            R = y

        def is_feasible_push(r, W, H):
            # Try multiple random inits
            for _ in range(20):
                x = np.random.uniform(r, W - r, N)
                y = np.random.uniform(r, H - r, N)
                
                # Relaxation
                for _ in range(500):
                    moved = False
                    # Overlaps
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < 2*r:
                                # Push apart
                                if dist == 0:
                                    dx = np.random.uniform(-0.1, 0.1)
                                    dy = np.random.uniform(-0.1, 0.1)
                                    dist = np.sqrt(dx*dx + dy*dy)
                                # Move each by half the overlap
                                push = (2*r - dist) / 2.0
                                ux = dx / dist
                                uy = dy / dist
                                x[i] += push * ux
                                y[i] += push * uy
                                x[j] -= push * ux
                                y[j] -= push * uy
                                moved = True
                    # Bounds
                    for i in range(N):
                        if x[i] < r:
                            x[i] = r
                            moved = True
                        if x[i] > W - r:
                            x[i] = W - r
                            moved = True
                        if y[i] < r:
                            y[i] = r
                            moved = True
                        if y[i] > H - r:
                            y[i] = H - r
                            moved = True
                    
                    if not moved:
                        break
                
                # Check final state
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < 2*r - 1e-6:
                            valid = False
                            break
                    if not valid: break
                    if x[i] < r - 1e-6 or x[i] > W - r + 1e-6 or y[i] < r - 1e-6 or y[i] > H - r + 1e-6:
                        valid = False
                        break
                
                if valid:
                    return True, np.column_stack((x, y))
            
            return False, None

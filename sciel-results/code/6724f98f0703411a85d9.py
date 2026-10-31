        # Add small jitter
        jitter = 0.01 * min(dx, dy)
        xs += np.random.uniform(-jitter, jitter, N)
        ys += np.random.uniform(-jitter, jitter, N)
        
        # Re-clamp to ensure bounds are respected initially
        xs = np.clip(xs, r_init, w - r_init)
        ys = np.clip(ys, r_init, h - r_init)

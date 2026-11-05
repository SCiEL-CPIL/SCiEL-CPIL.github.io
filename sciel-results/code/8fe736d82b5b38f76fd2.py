<<<<<<< SEARCH
    # Coordinate-descent-style local refinement: perturb individual coordinates
    # to reduce the max auto-convolution peak.
    if time.time() - start < DEADLINE - 3:
        rng2 = np.random.RandomState(1)
        # First pass: random perturbations
        for _ in range(150):
            if time.time() - start > DEADLINE - 3:
                break
            h = best_h + rng2.normal(0.0, 0.005, N)
            val, valid = _evaluate(h)
            if valid and val < best_value:
                best_value = val
                best_h = h.copy()
        # Second pass: targeted coordinate adjustments on the spike region
        for _ in range(200):
            if time.time() - start > DEADLINE - 3:
                break
            h = best_h.copy()
            # Perturb coordinates in the first 50 positions more aggressively
            idx = rng2.randint(0, 50)
            h[idx] += rng2.uniform(-0.02, 0.02)
            # Also perturb some negative tail coordinates
            if idx < 50:
                idx2 = rng2.randint(50, N)
                h[idx2] += rng2.uniform(-0.005, 0.005)
            val, valid = _evaluate(h)
            if valid and val < best_value:
                best_value = val
                best_h = h.copy()
=======
    # Coordinate-descent-style local refinement: perturb individual coordinates
    # to reduce the max auto-convolution peak.
    if time.time() - start < DEADLINE - 3:
        rng2 = np.random.RandomState(1)
        # First pass: random perturbations with adaptive step size
        step_size = 0.01
        for _ in range(200):
            if time.time() - start > DEADLINE - 3:
                break
            h = best_h + rng2.normal(0.0, step_size, N)
            val, valid = _evaluate(h)
            if valid and val < best_value:
                best_value = val
                best_h = h.copy()
                step_size *= 0.9  # Shrink step on success
            else:
                step_size *= 1.05  # Grow step on failure
                step_size = min(step_size, 0.05)
        
        # Second pass: targeted coordinate adjustments on the spike region
        for _ in range(250):
            if time.time() - start > DEADLINE - 3:
                break
            h = best_h.copy()
            # Perturb coordinates in the first 50 positions more aggressively
            idx = rng2.randint(0, 50)
            h[idx] += rng2.uniform(-0.02, 0.02)
            # Also perturb some negative tail coordinates
            if idx < 50:
                idx2 = rng2.randint(50, N)
                h[idx2] += rng2.uniform(-0.005, 0.005)
            val, valid = _evaluate(h)
            if valid and val < best_value:
                best_value = val
                best_h = h.copy()
        
        # Third pass: try to reduce the peak by adjusting the coordinates
        # that contribute most to the max autoconvolution
        conv = np.convolve(best_h, best_h)
        max_idx = np.argmax(conv)
        # The max autoconvolution at lag max_idx comes from pairs (i, i+max_idx)
        # We can try to reduce these specific products
        for _ in range(100):
            if time.time() - start > DEADLINE - 3:
                break
            h = best_h.copy()
            # Find the pair contributing to the peak
            if max_idx < N:
                # Reduce h[i] and h[i+max_idx] for the dominant pair
                i = rng2.randint(0, N - max_idx)
                h[i] *= 0.95
                h[i + max_idx] *= 0.95
            val, valid = _evaluate(h)
            if valid and val < best_value:
                best_value = val
                best_h = h.copy()
>>>>>>> REPLACE


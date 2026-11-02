<<<<<<< SEARCH
        # ALS Iterations
        prev_err = np.inf
        for it in range(300):
            if time.time() > deadline:
                break
            
            als_step(U, V, W)
            err = compute_error(U, V, W)
            
            if err < 1e-10:
                break
            
            # If stuck, add perturbation
            if abs(err - prev_err) < 1e-12 and it > 20:
                U += 0.3 * rng.randn(n * m, R)
                V += 0.3 * rng.randn(m * p, R)
                W += 0.3 * rng.randn(p * n, R)
            
            prev_err = err
        
        # Gradient polishing
        if compute_error(U, V, W) < 0.1:
            for grad_it in range(100):
                if time.time() > deadline:
                    break
                lr = 0.01 * (0.99 ** grad_it)  # Decaying learning rate
                gradient_step(U, V, W, lr)
                err = compute_error(U, V, W)
                if err < 1e-10:
                    break
=======
        # ALS Iterations
        prev_err = np.inf
        for it in range(500):
            if time.time() > deadline:
                break
            
            als_step(U, V, W)
            err = compute_error(U, V, W)
            
            if err < 1e-10:
                break
            
            # If stuck, add perturbation
            if abs(err - prev_err) < 1e-12 and it > 20:
                U += 0.3 * rng.randn(n * m, R)
                V += 0.3 * rng.randn(m * p, R)
                W += 0.3 * rng.randn(p * n, R)
            
            prev_err = err
        
        # Gradient polishing
        if compute_error(U, V, W) < 0.1:
            for grad_it in range(200):
                if time.time() > deadline:
                    break
                lr = 0.01 * (0.99 ** grad_it)  # Decaying learning rate
                gradient_step(U, V, W, lr)
                err = compute_error(U, V, W)
                if err < 1e-10:
                    break
>>>>>>> REPLACE

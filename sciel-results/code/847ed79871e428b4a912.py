<<<<<<< SEARCH
    run_count = 0
    while time.time() < deadline:
        code_indices = greedy_randomized()
        if len(code_indices) < best_len:
            best_len = len(code_indices)
            # Convert indices back to words
            best_code = [words[i] for i in code_indices]
            # Optional: If we hit 186, we can stop or try to break it.
            # 186 is the known best. If we find 186, we can continue to see if we can get 185.
            # But 156 is lower bound.
            # If we find 186, it's a great score.
            # Let's keep going to try for better.
        
        run_count += 1
=======
    run_count = 0
    while time.time() < deadline:
        # Use Boltzmann greedy
        # We can vary T slightly or keep it fixed.
        # Let's try a few temperatures.
        T = 1.0 + np.random.random() * 1.0 # T in [1.0, 2.0]
        code_indices = greedy_boltzmann(T)
        
        # Prune
        code_indices = prune_code(code_indices)
        
        if len(code_indices) < best_len:
            best_len = len(code_indices)
            # Convert indices back to words
            best_code = [words[i] for i in code_indices]
        
        run_count += 1
>>>>>>> REPLACE

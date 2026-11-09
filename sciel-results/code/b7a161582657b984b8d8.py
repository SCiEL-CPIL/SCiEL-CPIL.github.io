<<<<<<< SEARCH
        best_partition = None
        best_count = float('inf')
        
        # Determine how many restarts we can afford
        # Assume 1ms per run. 140s is huge. But let's cap it to 50 restarts per instance to be safe and quick.
        # Actually, n=90 is small. Let's do 20 restarts.
        max_restarts = 20
        
        for r in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Use a specific seed for the perturbation RNG to be reproducible if needed, 
            # or just random.
            perturb_seed = spec[3] * 100 + r
            perturb_rng = Random(perturb_seed)
            
            res = solve_instance(*spec, perturb_rng)
            count = len(res)
            
            if count < best_count:
                best_count = count
                best_partition = res
                
            # If we found a very good solution, maybe stop early?
            # Theoretical min for split graph is ~3n^2/16. 
            # For n=90, 3*8100/16 = 1518. 
            # If we get close to that, we can stop.
            # But we don't know the exact min.
            # Let's just run all restarts.
=======
        best_partition = None
        best_count = float('inf')
        
        # More restarts with varied noise levels
        max_restarts = 40
        
        for r in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Vary noise level: some runs with low noise, some with high
            noise_scale = 0.3 + (r % 10) * 0.1  # 0.3 to 1.2
            perturb_seed = spec[3] * 1000 + r * 7 + int(noise_scale * 100)
            perturb_rng = Random(perturb_seed)
            
            # Monkey-patch random to scale noise
            orig_random = perturb_rng.random
            perturb_rng.random = lambda: orig_random() * noise_scale
            
            res = solve_instance(*spec, perturb_rng)
            count = len(res)
            
            if count < best_count:
                best_count = count
                best_partition = res
                
            # Early stop if we hit theoretical lower bound
            # For split graphs: at least ceil(n^2/6) cliques needed
            lower_bound = (n * n) // 6 + n // 2
            if best_count <= lower_bound + 5:
                break
>>>>>>> REPLACE

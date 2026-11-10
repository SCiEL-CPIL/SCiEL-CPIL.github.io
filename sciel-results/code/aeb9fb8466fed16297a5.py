    # In construct loop:
    last_best_r = -1
    stagnant_cycles = 0
    
    while ...:
       ...
       if best_state.r == last_best_r:
           stagnant_cycles += 1
           if stagnant_cycles > 2:
               # Perturb best_state
               perturbed = best_state.copy()
               # Add noise to 10 random points
               for _ in range(10):
                   idx = random.randint(0, perturbed.N-1)
                   noise = np.random.randn(2) * 0.02
                   perturbed.move_point(idx, perturbed.c[idx] + noise)
               # Use perturbed as start for next SA
               # But we need to pass it to SA. 
               # Currently we pass best_state.copy().
               # Let's modify SA to accept a start state.
       else:
           stagnant_cycles = 0
           last_best_r = best_state.r

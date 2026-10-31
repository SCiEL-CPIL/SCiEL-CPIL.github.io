def solve():
    # Init pos
    pos = initial_hex_lattice()
    S = 4.0 # Start high
    
    # Optimize
    for step in range(10000):
        if time.time() > START_TIME + TIME_LIMIT: break
        
        # 1. Check validity
        valid, penalty = check_validity(pos, S)
        
        if valid:
            # Try to shrink S
            S_new = S * 0.99
            # Check if still valid with S_new (positions same)
            # Actually, shrinking S makes boundary tighter, so likely invalid.
            # We should shrink S only if we can adjust positions.
            # So, set S = S_new and let the force loop fix positions.
            S = S_new
            # Add a small random perturbation to escape local minima
            pos = add_noise(pos)
        else:
            # Apply forces
            forces = calculate_forces(pos, S)
            pos = update_positions(pos, forces)
            
    return pos, S

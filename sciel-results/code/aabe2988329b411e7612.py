<<<<<<< SEARCH
    # Non-standard patterns
    # HLLH pattern
    initial_designs.append((["H", "L", "L", "H"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    initial_designs.append((["H", "L", "L", "H", "L", "L", "H"], 
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_L), 
                            centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    
    # LHL pattern
    initial_designs.append((["L", "H", "L"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]))
=======
    # Non-standard patterns
    # HLLH pattern
    initial_designs.append((["H", "L", "L", "H"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    initial_designs.append((["H", "L", "L", "H", "L", "L", "H"], 
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_L), 
                            centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    
    # LHL pattern
    initial_designs.append((["L", "H", "L"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Add a few random valid designs to diversify search
    import random
    random.seed(42) # Fixed seed for reproducibility in this function context
    for _ in range(10):
        n_rand = random.randint(3, 12)
        rand_mats = []
        rand_thicks = []
        for i in range(n_rand):
            m = random.choice(["L", "H"])
            rand_mats.append(m)
            # Random thickness between 50 and 300 nm
            t = random.uniform(50.0, 300.0)
            rand_thicks.append(t)
        # Check total thickness
        if sum(rand_thicks) <= max_total_nm:
            initial_designs.append((rand_mats, rand_thicks))
>>>>>>> REPLACE

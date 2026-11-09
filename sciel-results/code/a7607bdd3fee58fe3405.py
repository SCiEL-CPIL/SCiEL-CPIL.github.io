def _beam_search(sequence, dimension, rng, width=50, time_budget=60.0):
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0,) * dimension
    # State: (score, energy, chain, occupied)
    beam = [(0.0, 0, [start], {start: 0})]
    
    start_time = time.time()
    
    for i in range(1, n):
        next_beam = []
        for score, energy, chain, occupied in beam:
            last_point = chain[-1]
            x, y, z = last_point
            for site in _neighbours(last_point):
                if site in occupied:
                    continue
                
                new_energy = energy
                if sequence[i] == "H":
                    sx, sy, sz = site
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            new_energy -= 1
                
                free_count = 0
                sx, sy, sz = site
                for q in _neighbours(site):
                    if q not in occupied:
                        free_count += 1
                
                new_score = -new_energy + 0.5 * free_count + rng.random() * 0.2
                
                new_chain = chain + [site]
                new_occupied = occupied.copy()
                new_occupied[site] = i
                
                next_beam.append((new_score, new_energy, new_chain, new_occupied))
                
                if time.time() - start_time > time_budget:
                    if next_beam:
                        next_beam.sort(key=lambda x: x[1])
                        return next_beam[0][2]
                    return chain
        
        if not next_beam:
            if beam:
                return beam[0][2]
            return []
        
        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:width]
        
        if time.time() - start_time > time_budget:
             if beam:
                 beam.sort(key=lambda x: x[1])
                 return beam[0][2]
             return []

    if beam:
        beam.sort(key=lambda x: x[1])
        return beam[0][2]
    return []

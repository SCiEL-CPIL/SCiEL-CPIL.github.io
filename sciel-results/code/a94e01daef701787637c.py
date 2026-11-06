<<<<<<< SEARCH
    # --- Phase 3: Swap-based local search ---
    current = best_network[:]
    for iteration in range(10000):
        if (time.time() - start_time) > DEADLINE - 5:
            break
        if len(current) <= 70:
            break
            
        # Try swapping two comparators
        if len(current) < 2:
            break
        i1 = random.randint(0, len(current) - 1)
        i2 = random.randint(0, len(current) - 1)
        if i1 == i2:
            continue
            
        candidate = current[:]
        candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
        
        if verify_network(candidate):
            current = candidate
            if len(current) < best_size:
                best_network = current[:]
                best_size = len(current)
=======
    # --- Phase 3: Simulated Annealing ---
    current = best_network[:]
    temp = 5.0
    cooling = 0.9995
    for iteration in range(20000):
        if (time.time() - start_time) > DEADLINE - 15:
            break
        if len(current) <= 70:
            break
        
        # Generate neighbor: random removal + optional reinsertion
        if random.random() < 0.7:
            # Remove one comparator
            idx = random.randint(0, len(current) - 1)
            candidate = current[:]
            removed_comp = candidate.pop(idx)
            # Maybe reinsert at different position
            if random.random() < 0.5 and len(candidate) > 0:
                new_pos = random.randint(0, len(candidate))
                candidate.insert(new_pos, removed_comp)
        else:
            # Swap two comparators
            if len(current) < 2:
                continue
            i1 = random.randint(0, len(current) - 1)
            i2 = random.randint(0, len(current) - 1)
            if i1 == i2:
                continue
            candidate = current[:]
            candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
        
        # Evaluate
        if verify_network(candidate):
            current = candidate
            if len(current) < best_size:
                best_network = current[:]
                best_size = len(current)
        else:
            # Compute violation count as energy
            energy = count_violations(candidate)
            # Accept if energy is lower or with probability exp(-delta/temp)
            if energy < 0:
                current = candidate
            else:
                # delta_energy = new_size - old_size (we want to minimize size)
                delta_energy = len(candidate) - len(current)
                if delta_energy <= 0:
                    current = candidate
                    if len(current) < best_size:
                        best_network = current[:]
                        best_size = len(current)
                else:
                    import math
                    if random.random() < math.exp(-delta_energy / temp):
                        current = candidate
        
        temp *= cooling
>>>>>>> REPLACE


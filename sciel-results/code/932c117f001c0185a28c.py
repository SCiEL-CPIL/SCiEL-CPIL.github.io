<<<<<<< SEARCH
def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts
=======
def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _local_search(sequence, chain, rng, time_budget):
    """Simulated annealing to improve a conformation via move, pivot, and crank moves."""
    if not chain:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    start_time = time.time()
    T = 2.0
    T_min = 0.01
    cooling_rate = 0.995
    
    while T > T_min and (time.time() - start_time) < time_budget:
        # Try a random move
        n = len(current_chain)
        move_type = rng.random()
        
        if move_type < 0.4:
            # Single step move: try moving a random residue to a neighbouring free site
            i = rng.randint(1, n - 2)  # not first or last
            old_pos = current_chain[i]
            for new_pos in _neighbours(old_pos):
                if new_pos in [current_chain[j] for j in range(n) if j != i]:
                    continue
                # Check self-avoidance
                valid = True
                for j in range(n):
                    if j != i and current_chain[j] == new_pos:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute energy delta
                old_energy = current_energy
                new_chain = current_chain[:]
                new_chain[i] = new_pos
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - old_energy
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
                break
        elif move_type < 0.7:
            # Pivot move: rotate a segment
            if n < 4:
                T *= cooling_rate
                continue
            i = rng.randint(1, n - 3)
            length = min(3, n - i - 1)
            if length < 2:
                T *= cooling_rate
                continue
            
            # Extract segment
            segment = current_chain[i:i+length+1]
            # Try rotating by 90, 180, 270 degrees around current_chain[i]
            center = current_chain[i]
            for angle in [90, 180, 270]:
                # Rotate segment around center
                new_segment = [center]
                for k in range(1, len(segment)):
                    dx = segment[k][0] - center[0]
                    dy = segment[k][1] - center[1]
                    if angle == 90:
                        new_segment.append((center[0] - dy, center[1] + dx))
                    elif angle == 180:
                        new_segment.append((center[0] - dx, center[1] - dy))
                    else:  # 270
                        new_segment.append((center[0] + dy, center[1] - dx))
                
                # Check self-avoidance
                new_chain = current_chain[:]
                new_chain[i:i+length+1] = new_segment
                valid = True
                seen = set()
                for pos in new_chain:
                    if pos in seen:
                        valid = False
                        break
                    seen.add(pos)
                if not valid:
                    continue
                
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
                break
        
        T *= cooling_rate
    
    return best_chain
>>>>>>> REPLACE


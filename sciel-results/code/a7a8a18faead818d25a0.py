<<<<<<< SEARCH
def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours2d(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours2d(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours2d(site) if q not in occupied)
            # Heuristic: prefer high gain, then high free space (to avoid trapping)
            score = gain * 2.0 + free * 0.5 + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain
=======
def _generate_motifs(sequence, motif_len=5):
    """
    Generate all valid self-avoiding walk motifs of given length.
    Returns list of (motif_chain, contact_count) where motif_chain is list of relative positions.
    """
    motifs = []
    # Generate all SAWs of length motif_len starting at (0,0)
    def _gen_saw(pos, visited, remaining):
        if remaining == 0:
            # Calculate contacts within this motif
            contacts = 0
            chain = list(visited)
            for i in range(len(chain)):
                for j in range(i + 2, len(chain)):
                    x1, y1 = chain[i]
                    x2, y2 = chain[j]
                    if abs(x1 - x2) + abs(y1 - y2) == 1:
                        if sequence[0] == 'H' and sequence[1] == 'H':
                            # Need to check actual residue types, but for now just count geometric contacts
                            # This is a simplification; we'll refine in placement
                            pass
            # Normalize: make sure first point is (0,0) and orientation is canonical
            # For simplicity, just store the relative positions
            motifs.append(list(visited))
            return
        
        x, y = pos
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            new_pos = (x + dx, y + dy)
            if new_pos not in visited:
                visited.append(new_pos)
                _gen_saw(new_pos, visited, remaining - 1)
                visited.pop()
    
    _gen_saw((0, 0), [(0, 0)], motif_len)
    return motifs

def _beam_search_fold(sequence, dimension, beam_width=20, time_limit=30.0):
    """
    Hierarchical beam search that places the chain in segments.
    Each beam entry is (chain, occupied, energy).
    """
    start_time = time.time()
    n = len(sequence)
    
    # Start with first residue
    initial_chain = [(0, 0)]
    initial_occupied = {(0, 0): 0}
    beams = [(initial_chain, initial_occupied, 0)]
    
    # We'll grow the chain residue by residue but with beam search
    for i in range(1, n):
        if time.time() - start_time > time_limit:
            break
        
        new_beams = []
        for chain, occupied, energy in beams:
            last_pos = chain[-1]
            for site in _neighbours2d(last_pos):
                if site in occupied:
                    continue
                
                # Check self-avoidance: site must not be adjacent to any non-adjacent residue
                valid = True
                for q in _neighbours2d(site):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate energy change
                delta = 0
                if sequence[i] == 'H':
                    for q in _neighbours2d(site):
                        j = occupied.get(q)
                        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                            delta -= 1
                
                new_energy = energy + delta
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                
                new_beams.append((new_chain, new_occupied, new_energy))
        
        if not new_beams:
            break
        
        # Keep only top beam_width by energy
        new_beams.sort(key=lambda x: x[2])
        beams = new_beams[:beam_width]
        
        if not beams:
            break
    
    if not beams:
        return None, 0
    
    # Return the best beam
    best_chain, best_occupied, best_energy = beams[0]
    return best_chain, best_energy
>>>>>>> REPLACE


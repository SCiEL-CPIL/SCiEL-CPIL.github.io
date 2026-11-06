"""HP lattice protein folding on the square lattice – scratch rewrite."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: negative count of H-H contacts (non-consecutive neighbours)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, temperature=0.3):
    """Randomized greedy chain growth with temperature-controlled exploration."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Count new H-H contacts this site would form
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            # Count free neighbours (space for future growth)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.15 * free + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best option
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _local_search(sequence, chain, dimension, rng, max_iter=500, deadline=None):
    """Local search via single-residue moves and small segment moves."""
    n = len(sequence)
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    # Precompute which residues are H
    is_h = [c == 'H' for c in sequence]
    
    def try_move(chain, occupied, i):
        """Try to move residue i to a neighbouring site. Returns (new_chain, new_occupied, new_energy, success)."""
        p = chain[i]
        old_energy = _energy(sequence, chain)
        
        for q in _neighbours(p):
            if q == chain[i-1] if i > 0 else False:
                continue
            if q == chain[i+1] if i < n-1 else False:
                continue
            if q in occupied:
                continue
            
            # Check if moving is valid (no self-avoidance violation)
            chain[i] = q
            del occupied[p]
            occupied[q] = i
            
            new_energy = _energy(sequence, chain)
            
            if new_energy < best_energy:
                return True, new_energy
            else:
                # Revert
                chain[i] = p
                del occupied[q]
                occupied[p] = i
        
        return False, None
    
    iter_count = 0
    no_improve = 0
    while iter_count < max_iter:
        if deadline and time.time() > deadline:
            break
        
        improved = False
        # Try moving a random residue
        i = rng.randint(0, n-1)
        p = best_chain[i]
        
        for q in _neighbours(p):
            # Check not adjacent in chain
            if i > 0 and q == best_chain[i-1]:
                continue
            if i < n-1 and q == best_chain[i+1]:
                continue
            if q in occupied:
                continue
            
            # Tentatively move
            old_pos = best_chain[i]
            best_chain[i] = q
            del occupied[old_pos]
            occupied[q] = i
            
            new_energy = _energy(sequence, best_chain)
            
            if new_energy < best_energy:
                best_energy = new_energy
                improved = True
                no_improve = 0
                break
            else:
                # Revert
                best_chain[i] = old_pos
                del occupied[q]
                occupied[old_pos] = i
        
        if not improved:
            no_improve += 1
            if no_improve > 50:
                break
        iter_count += 1
    
    return best_chain, best_energy


def _segment_move(sequence, chain, dimension, rng, deadline=None):
    """Try to relocate a segment of the chain to a new location."""
    n = len(sequence)
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    for _ in range(20):
        if deadline and time.time() > deadline:
            break
        
        # Pick a segment
        seg_len = rng.randint(2, min(15, n//3))
        start = rng.randint(0, n - seg_len)
        end = start + seg_len - 1
        
        # Remove segment
        removed = best_chain[start:end+1]
        new_chain = best_chain[:start] + best_chain[end+1:]
        
        # Try to reinsert segment at various positions
        for ins_pos in range(len(new_chain) - seg_len + 1):
            if abs(ins_pos - start) < 2:  # Skip nearby positions
                continue
            
            # Try to grow segment from ins_pos
            candidate = list(new_chain[:ins_pos])
            occupied = {p: i for i, p in enumerate(candidate)}
            
            # Add segment residues
            valid = True
            for k in range(seg_len):
                if k == 0:
                    site = removed[k]
                else:
                    # Find neighbour of previous site
                    prev = candidate[-1]
                    found = False
                    for q in _neighbours(prev):
                        if q not in occupied:
                            # Check if this matches removed[k]'s relative position
                            # For simplicity, just try all neighbours
                            site = q
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                
                if site in occupied:
                    valid = False
                    break
                candidate.append(site)
                occupied[site] = len(candidate) - 1
            
            if not valid:
                continue
            
            # Complete the rest of the chain
            for j in range(ins_pos + seg_len, len(new_chain)):
                site = new_chain[j]
                if site in occupied:
                    valid = False
                    break
                candidate.append(site)
                occupied[site] = len(candidate) - 1
            
            if not valid:
                continue
            
            # Check self-avoidance
            if len(set(candidate)) != len(candidate):
                continue
            
            energy = _energy(sequence, candidate)
            if energy < best_energy:
                best_chain = candidate
                best_energy = energy
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth + local search + segment moves."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Deadline: 180s total, use 170s for safety
    start_time = time.time()
    deadline = start_time + 170
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple greedy growths with different temperatures
    temps = [0.1, 0.3, 0.5, 0.8, 1.2, 2.0]
    for temp in temps:
        if time.time() > deadline - 30:
            break
        for _ in range(30):
            if time.time() > deadline - 30:
                break
            chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
            if chain is None:
                continue
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    # Phase 2: Local search on best chain
    if best_chain is not None:
        if time.time() > deadline - 60:
            # Not enough time for local search, return best
            pass
        else:
            best_chain, best_energy = _local_search(
                sequence, best_chain, dimension, rng, 
                max_iter=1000, deadline=deadline - 30
            )
            
            # Phase 3: Segment moves
            if time.time() < deadline - 20:
                best_chain, best_energy = _segment_move(
                    sequence, best_chain, dimension, rng, 
                    deadline=deadline - 10
                )
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

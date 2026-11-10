"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 175.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _greedy_grow(sequence, dimension, rng, temperature=0.0):
    """
    Randomized greedy chain growth.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _min_conflicts_sa(sequence, chain, rng, max_iters=10000, start_temp=5.0, cool_rate=0.999):
    """
    Stochastic Local Search (min-conflicts + SA) on the global chain representation.
    Uses kink flips and short regrows with fast incremental energy updates.
    """
    n = len(sequence)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = start_temp
    no_improve = 0

    # Precompute H positions for faster checks
    is_h = [c == 'H' for c in sequence]

    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        if temp < 0.05:
            break

        move_type = rng.random()
        
        if move_type < 0.7:
            # Kink flip: move residue i to a neighbor that maintains connectivity
            # Pick i with some bias towards the middle to allow larger rearrangements
            i = rng.randint(1, n - 2)
            p = chain[i]
            prev, nxt = chain[i-1], chain[i+1]
            
            candidates = []
            for np in _neighbours(prev):
                if np == p:
                    continue
                if abs(np[0]-nxt[0]) + abs(np[1]-nxt[1]) != 1:
                    continue
                if np in occupied and occupied[np] != i:
                    continue
                candidates.append(np)
            
            if candidates:
                np = rng.choice(candidates)
                
                # Fast energy delta calculation for kink flip
                # Remove contacts involving i
                delta = 0
                if is_h[i]:
                    for q in _neighbours(p):
                        j = occupied.get(q)
                        if j is not None and abs(j - i) > 1 and is_h[j]:
                            delta += 1
                # Add contacts involving i at new position
                if is_h[i]:
                    for q in _neighbours(np):
                        j = occupied.get(q)
                        if j is not None and abs(j - i) > 1 and is_h[j]:
                            delta -= 1
                
                new_energy = current_energy - delta
                
                if new_energy < current_energy or (temp > 0 and rng.random() < (10 ** (delta / max(temp, 0.1)))):
                    del occupied[p]
                    chain[i] = np
                    occupied[np] = i
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
        else:
            # Short regrow: regrow last 3-6 residues
            length = rng.randint(3, min(6, n-3))
            split = n - length
            prefix = chain[:split]
            suffix_seq = sequence[split:]
            
            # Regrow with simple greedy
            new_suffix = []
            occupied_copy = {p: i for i, p in enumerate(prefix)}
            last = prefix[-1]
            success = True
            for k, res in enumerate(suffix_seq):
                global_idx = split + k
                options = []
                for site in _neighbours(last):
                    if site in occupied_copy:
                        continue
                    gain = 0
                    if res == 'H':
                        for q in _neighbours(site):
                            j = occupied_copy.get(q)
                            if j is not None and j < global_idx - 1 and is_h[j]:
                                gain += 1
                    free = sum(1 for q in _neighbours(site) if q not in occupied_copy)
                    score = gain * 5.0 + free * 0.1 + rng.random() * temp
                    options.append((score, site))
                
                if not options:
                    success = False
                    break
                
                best_score, best_site = max(options, key=lambda x: x[0])
                occupied_copy[best_site] = global_idx
                new_suffix.append(best_site)
                last = best_site
            
            if success:
                new_chain = prefix + new_suffix
                new_energy = _compute_energy(sequence, new_chain)
                
                delta = new_energy - current_energy
                if new_energy < current_energy or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
                    chain = new_chain
                    occupied = {p: i for i, p in enumerate(chain)}
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
        
        temp *= cool_rate
        
        if no_improve > 500:
            break

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence using a portfolio of greedy growths followed by global stochastic local search.
    """
    rng = random.Random(42)
    best_chain = None
    best_energy = 0

    n = len(sequence)
    
    # Phase 1: Multiple greedy restarts to get diverse starting points
    phase1_end = time.time() + 40.0
    for attempt in range(10000):
        if time.time() > phase1_end:
            break

        temp = rng.uniform(0.0, 3.0)
        chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue

        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Global stochastic local search (min-conflicts + SA) on the best found
    if best_chain is not None:
        phase2_end = time.time() + 120.0
        # Run multiple passes with different temperatures
        for sa_pass in range(5):
            if time.time() > phase2_end:
                break
            remaining_time = phase2_end - time.time()
            if remaining_time < 5.0:
                break
            
            start_temp = 5.0 - sa_pass * 0.5
            cool_rate = 0.999 - sa_pass * 0.0001
            sa_chain, sa_energy = _min_conflicts_sa(
                sequence, best_chain, rng, max_iters=20000, 
                start_temp=start_temp, cool_rate=cool_rate)
            if sa_energy < best_energy:
                best_chain = sa_chain
                best_energy = sa_energy

    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _grow(sequence, dimension, rng, strategy="greedy"):
    """Randomized chain growth with different strategies."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            if strategy == "greedy":
                score = gain + 0.1 * free + rng.random() * 0.5
            elif strategy == "hydrophobic":
                score = gain * 2 + rng.random() * 0.3
            else:  # random
                score = rng.random()
            options.append((score, site))
        if not options:
            return None
        options.sort(reverse=True)
        # Pick from top few with some randomness
        k = min(3, len(options))
        site = options[rng.randint(0, k - 1)][1]
        occupied[site] = i
        chain.append(site)
    return chain


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


def _local_search(sequence, chain, rng, deadline):
    """Simulated annealing with single-residue and segment moves."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    best = list(chain)
    current = list(chain)
    best_e = _energy(sequence, best)
    current_e = best_e
    occupied = {p: i for i, p in enumerate(current)}
    
    # Precompute H positions for quick lookup
    h_positions = set(i for i, c in enumerate(sequence) if c == "H")
    
    start_time = time.time()
    initial_temp = 3.0
    min_temp = 0.01
    cooling_rate = 0.995
    
    temp = initial_temp
    no_improve_count = 0
    
    def _delta_energy(seq, pos, old_site, new_site, occ):
        """Compute energy delta for moving residue at pos from old_site to new_site."""
        delta = 0
        if seq[pos] == 'H':
            # Lost contacts: neighbors of old_site that are H and not pos
            for q in _neighbours(old_site):
                j = occ.get(q)
                if j is not None and j != pos:
                    if seq[j] == 'H':
                        delta -= 1
            # Gained contacts: neighbors of new_site that are H and not pos
            for q in _neighbours(new_site):
                j = occ.get(q)
                if j is not None and j != pos:
                    if seq[j] == 'H':
                        delta += 1
        return delta
    
    def _try_single_move(i):
        """Try moving residue i to a valid neighboring site."""
        old = current[i]
        prev = current[i - 1]
        next_pos = current[i + 1]
        
        # Find valid candidates: neighbor of prev AND neighbor of next_pos
        candidates = set()
        for site in _neighbours(prev):
            if site in _neighbours(next_pos):
                if site not in occupied or occupied[site] == i:
                    if site != old:
                        candidates.add(site)
        
        if not candidates:
            return False
        
        for site in candidates:
            delta = _delta_energy(sequence, i, old, site, occupied)
            # Accept if better or with probability exp(-delta/temp)
            if delta > 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
                # Apply move
                occupied[old] = -1
                occupied[site] = i
                current[i] = site
                return True, delta
        return False
    
    def _try_segment_reversal(i, length):
        """Reverse segment from i to i+length-1."""
        if i + length >= n or i < 1:
            return False
        
        # Check if reversal is valid: the endpoints must connect
        start_idx = i
        end_idx = i + length - 1
        
        # The segment is current[start_idx:end_idx+1]
        # After reversal, current[start_idx] connects to current[start_idx-1]
        # and current[end_idx] connects to current[end_idx+1]
        
        # Check connectivity after reversal
        # Before: ... -> A -> [B C D] -> E -> ...
        # After:  ... -> A -> [D C B] -> E -> ...
        # Need: A adjacent to D (was A adjacent to B)
        # Need: B adjacent to E (was D adjacent to E)
        
        a = current[start_idx - 1]
        b = current[start_idx]
        d = current[end_idx]
        e = current[end_idx + 1] if end_idx + 1 < n else None
        
        # Check if a is adjacent to d (after reversal, d becomes first)
        if not any(a2 == d2 for a2 in _neighbours(a) for d2 in [d]):
            return False
        # Check if b is adjacent to e (after reversal, b becomes last)
        if e is not None:
            if not any(b2 == e2 for b2 in _neighbours(b) for e2 in [e]):
                return False
        
        # Check self-avoidance: all sites in reversed segment must be free
        # (they are already occupied by the segment, so just check no conflicts)
        # The sites remain the same, just the order changes, so self-avoidance is preserved
        
        # Compute energy delta
        old_sites = current[start_idx:end_idx + 1]
        new_sites = list(reversed(old_sites))
        
        # Remove old assignments
        for idx in range(start_idx, end_idx + 1):
            occupied[current[idx]] = -1
        
        # Compute delta by comparing contacts
        delta = 0
        for idx in range(start_idx, end_idx + 1):
            old_site = old_sites[idx - start_idx]
            new_site = new_sites[idx - start_idx]
            if old_site == new_site:
                continue
            if sequence[idx] == 'H':
                # Lost contacts at old_site
                for q in _neighbours(old_site):
                    j = occupied.get(q)
                    if j is not None and j != idx:
                        if sequence[j] == 'H':
                            delta -= 1
                # Gained contacts at new_site
                for q in _neighbours(new_site):
                    j = occupied.get(q)
                    if j is not None and j != idx:
                        if sequence[j] == 'H':
                            delta += 1
        
        # Apply new assignments
        for idx in range(start_idx, end_idx + 1):
            occupied[new_sites[idx - start_idx]] = idx
            current[idx] = new_sites[idx - start_idx]
        
        return True, delta
    
    while time.time() < deadline and temp > min_temp:
        improved_this_iter = False
        
        # Try single moves
        for i in range(1, n - 1):
            if time.time() >= deadline:
                break
            result = _try_single_move(i)
            if result:
                success, delta = result
                if success:
                    current_e -= delta
                    if current_e < best_e:
                        best_e = current_e
                        best = list(current)
                        no_improve_count = 0
                        improved_this_iter = True
                    else:
                        no_improve_count += 1
                    break
        
        # Try segment reversals
        if not improved_this_iter:
            max_seg_len = min(4, n // 3)
            for length in range(2, max_seg_len + 1):
                if time.time() >= deadline:
                    break
                for i in range(1, n - length):
                    result = _try_segment_reversal(i, length)
                    if result:
                        success, delta = result
                        if success:
                            current_e -= delta
                            if current_e < best_e:
                                best_e = current_e
                                best = list(current)
                                no_improve_count = 0
                                improved_this_iter = True
                            else:
                                no_improve_count += 1
                            break
                if improved_this_iter:
                    break
        
        # Cool down
        temp *= cooling_rate
        
        # If stuck for a while, reset current to best
        if no_improve_count > 100:
            current = list(best)
            current_e = best_e
            occupied = {p: i for i, p in enumerate(current)}
            no_improve_count = 0
            temp = max(temp * 2, 0.5)  # Reheat
    
    return best, best_e


def fold(sequence, dimension):
    """Best conformation via multiple growth strategies + simulated annealing."""
    start_time = time.time()
    total_deadline = start_time + 12.0  # Leave 30s margin for 26 sequences
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate diverse initial chains
    strategies = ["greedy", "hydrophobic", "random"]
    initial_chains = []
    
    # Do multiple growths with different strategies
    for strategy in strategies:
        for _ in range(3):
            if time.time() > total_deadline - 5.0:
                break
            chain = _grow(sequence, dimension, rng, strategy)
            if chain is not None:
                e = _energy(sequence, chain)
                if best_chain is None or e < best_energy:
                    best_chain = chain
                    best_energy = e
                initial_chains.append((chain, e))
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search on the best initial chain
    remaining_time = total_deadline - time.time()
    if remaining_time > 1.0:
        best_chain, best_energy = _local_search(
            sequence, best_chain, rng, total_deadline - 0.5
        )
    
    # Phase 3: If time remains, try local search on second-best initial chain
    if len(initial_chains) > 1:
        initial_chains.sort(key=lambda x: x[1])
        if initial_chains[1][0] != best_chain and time.time() < total_deadline - 2.0:
            chain2, _ = initial_chains[1]
            chain2, e2 = _local_search(
                sequence, chain2, rng, total_deadline - 0.5
            )
            if e2 < best_energy:
                best_chain, best_energy = chain2, e2
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

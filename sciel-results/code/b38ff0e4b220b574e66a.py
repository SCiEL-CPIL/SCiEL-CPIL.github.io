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


def _is_valid_walk(chain):
    if not chain:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i - 1]
            dist = sum(abs(a - b) for a, b in zip(p, prev))
            if dist != 1:
                return False
    return True


def _greedy_grow(sequence, dimension, rng, bias=0.0):
    """Greedy chain growth with configurable bias toward contacts vs free space."""
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
            score = gain + bias * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _regrow_segment(sequence, chain, start_idx, end_idx, rng, occupied_set, dimension):
    """Remove segment [start_idx, end_idx) and regrow it. Returns new chain or None."""
    # Save the segment to remove
    removed = chain[start_idx:end_idx]
    # Remove from occupied
    for pos in removed:
        if pos in occupied_set:
            del occupied_set[pos]
    
    # Regrow from start_idx
    new_chain = chain[:start_idx]
    current_occupied = set(occupied_set)
    
    for i in range(start_idx, end_idx):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site in current_occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in current_occupied and sequence[list(current_occupied)[list(current_occupied).index(q)]] == "H"
                           if q not in removed)
            free = sum(1 for q in _neighbours(site) if q not in current_occupied)
            score = gain + 0.1 * free + rng.random() * 0.3
            options.append((score, site))
        
        if not options:
            # Restore original
            for j, pos in enumerate(removed):
                occupied_set[pos] = start_idx + j
            return None
        
        site = max(options)[1]
        current_occupied.add(site)
        new_chain.append(site)
    
    # Update occupied_set
    for i, pos in enumerate(removed):
        if pos in occupied_set:
            del occupied_set[pos]
    for i, pos in enumerate(new_chain[start_idx:end_idx]):
        occupied_set[pos] = start_idx + i
    
    chain[start_idx:end_idx] = new_chain[start_idx:end_idx]
    return chain


def _kink_move(sequence, chain, idx, rng, occupied_set, dimension):
    """Try to move residue at idx to a nearby unoccupied position."""
    if idx <= 0 or idx >= len(chain) - 1:
        return None
    
    pos = chain[idx]
    prev = chain[idx - 1]
    next_pos = chain[idx + 1]
    
    # Find valid neighbors of prev that aren't next_pos or occupied
    candidates = []
    for site in _neighbours(prev):
        if site == next_pos:
            continue
        if site in occupied_set and occupied_set[site] != idx:
            continue
        # Check that the step from site to next_pos is valid (unit step)
        dist = sum(abs(a - b) for a, b in zip(site, next_pos))
        if dist != 1:
            continue
        candidates.append(site)
    
    if not candidates:
        return None
    
    new_pos = rng.choice(candidates)
    
    # Update
    del occupied_set[pos]
    occupied_set[new_pos] = idx
    chain[idx] = new_pos
    return chain


def _local_search(sequence, chain, rng, occupied_set, max_iter=5000, deadline=None):
    """Local search with kink moves and occasional regrow moves."""
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_energy = best_energy
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Choose move type: 70% kink, 30% regrow
        if rng.random() < 0.7:
            # Kink move
            idx = rng.randint(1, len(chain) - 2)
            old_chain = list(chain)
            old_occupied = dict(occupied_set)
            
            result = _kink_move(sequence, chain, idx, rng, occupied_set, 2)
            if result is None:
                # Restore
                chain[:] = old_chain
                occupied_set.clear()
                occupied_set.update(old_occupied)
                continue
            
            new_energy = _energy(sequence, chain)
            delta = new_energy - current_energy
            
            # Accept if better or with small probability if worse
            if delta < 0 or rng.random() < math.exp(-delta / 0.5):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
            else:
                # Restore
                chain[:] = old_chain
                occupied_set.clear()
                occupied_set.update(old_occupied)
        else:
            # Regrow a short segment
            seg_len = rng.choice([3, 4, 5])
            if len(chain) < seg_len + 2:
                continue
            start_idx = rng.randint(1, len(chain) - seg_len - 1)
            end_idx = start_idx + seg_len
            
            old_chain = list(chain)
            old_occupied = dict(occupied_set)
            
            result = _regrow_segment(sequence, chain, start_idx, end_idx, rng, occupied_set, 2)
            if result is None:
                chain[:] = old_chain
                occupied_set.clear()
                occupied_set.update(old_occupied)
                continue
            
            new_energy = _energy(sequence, chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / 0.8):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
            else:
                chain[:] = old_chain
                occupied_set.clear()
                occupied_set.update(old_occupied)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using diverse initialization + local search + regrow SA."""
    deadline = time.time() + 190  # Safety margin before 210s kill
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Diverse greedy initializations
    n_init = 30
    for i in range(n_init):
        if time.time() > deadline - 60:
            break
        bias = rng.uniform(-0.5, 0.5)
        chain = _greedy_grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search on best initializations
    # Run local search on top few initializations
    inits = []
    for i in range(20):
        if time.time() > deadline - 90:
            break
        bias = rng.uniform(-0.5, 0.5)
        chain = _greedy_grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        inits.append((energy, chain))
        if len(inits) > 5:
            inits.sort(key=lambda x: x[0])
            inits = inits[:5]
    
    inits.sort(key=lambda x: x[0])
    
    for energy, chain in inits[:3]:
        if time.time() > deadline - 30:
            break
        
        occupied_set = {p: i for i, p in enumerate(chain)}
        improved_chain, improved_energy = _local_search(
            sequence, chain, rng, occupied_set,
            max_iter=2000, deadline=deadline - 30
        )
        
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Phase 3: Regrow-based simulated annealing on best chain
    if time.time() < deadline - 20:
        occupied_set = {p: i for i, p in enumerate(best_chain)}
        sa_chain, sa_energy = _local_search(
            sequence, best_chain, rng, occupied_set,
            max_iter=5000, deadline=deadline - 20
        )
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
    
    # Final validation
    if not _is_valid_walk(best_chain):
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

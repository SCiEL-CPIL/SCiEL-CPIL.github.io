"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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
            # Weight gain heavily, free space as tiebreaker
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
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


def _delta_energy(sequence, chain, i, new_pos):
    """Compute change in energy when moving residue i to new_pos (must be valid/self-avoiding)."""
    if sequence[i] != "H":
        # Only H residues contribute to contacts
        # But moving an H away/breaks contacts, moving P doesn't directly
        # Actually, moving any residue can break or form H-H contacts
        pass
    old_pos = chain[i]
    if old_pos == new_pos:
        return 0
    # Build set of occupied positions excluding i for checking
    # For delta: contacts involving i that change
    where = {p: idx for idx, p in enumerate(chain)}
    # Remove i from where
    del where[old_pos]
    
    # Count H-H contacts involving residue i at old_pos
    old_contacts = 0
    if sequence[i] == "H":
        for q in _neighbours(old_pos):
            j = where.get(q)
            if j is not None and sequence[j] == "H":
                # Check j != i+1 and j != i-1 (adjacent in chain)
                if abs(j - i) > 1:
                    old_contacts += 1
    
    # Count H-H contacts involving residue i at new_pos
    new_contacts = 0
    if sequence[i] == "H":
        for q in _neighbours(new_pos):
            j = where.get(q)
            if j is not None and sequence[j] == "H":
                if abs(j - i) > 1:
                    new_contacts += 1
    
    # Also need to check if moving i breaks contacts between other residues? No,
    # only contacts involving i change since only i moves.
    # But wait: if another H residue was contacting i at old_pos, that contact is lost.
    # And if another H residue contacts i at new_pos, that's gained.
    # The above captures that.
    
    delta = -(new_contacts - old_contacts)  # energy = -contacts, so delta_energy = -(delta_contacts)
    return delta


def _anneal(sequence, chain, dimension, rng, deadline):
    """Simulated annealing with geometric cooling. Returns best chain found."""
    n = len(chain)
    if n < 4:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    # Temperature schedule
    T0 = max(1.0, abs(current_energy) * 0.5)
    T_min = 0.01
    # Number of steps: aim for ~2000-5000 steps depending on time
    # We'll run until deadline with safety margin
    margin = 5.0  # seconds before deadline
    
    # Geometric cooling: T = T0 * (T_min/T0)^(step/total_steps)
    # We'll estimate total steps based on available time
    time_left = deadline - time.time()
    if time_left <= margin:
        return best_chain
    
    # Estimate steps per second: roughly 5000-20000 simple moves/sec
    est_steps_per_sec = 8000
    total_steps = int((time_left - margin) * est_steps_per_sec)
    total_steps = max(1000, min(total_steps, 100000))  # cap
    
    cooling_factor = (T_min / T0) ** (1.0 / total_steps)
    T = T0
    
    for step in range(total_steps):
        if time.time() >= deadline - margin:
            break
        T *= cooling_factor
        if T < T_min:
            T = T_min
        
        # Choose a move
        move_type = rng.random()
        if move_type < 0.5:
            # Flip: move one residue to a random neighbor
            i = rng.randint(1, n - 2)  # not endpoints
            old_pos = current_chain[i]
            neighbors = list(_neighbours(old_pos))
            new_pos = rng.choice(neighbors)
            if new_pos == old_pos:
                continue
            # Check self-avoidance: new_pos must not be occupied by any other residue
            # and must not be adjacent to i-1 or i+1 in a way that creates collision
            # Actually, new_pos just must not be in the set of other positions
            other_positions = set(current_chain[:i] + current_chain[i+1:])
            if new_pos in other_positions:
                continue
            # Also check that the step from i-1 to new_pos and new_pos to i+1 are valid unit steps
            # They should be since we're only moving i, but the chain connectivity must hold
            # i-1 -> new_pos must be a unit step
            prev = current_chain[i-1]
            diff = [new_pos[a] - prev[a] for a in range(dimension)]
            if sum(abs(d) for d in diff) != 1:
                continue
            # i+1 -> new_pos must be a unit step
            nxt = current_chain[i+1]
            diff2 = [nxt[a] - new_pos[a] for a in range(dimension)]
            if sum(abs(d) for d in diff2) != 1:
                continue
            
            # Compute delta energy
            delta = _delta_energy(sequence, current_chain, i, new_pos)
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_chain[i] = new_pos
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        
        elif move_type < 0.8:
            # 180-degree rotation of a sub-chain
            # Pick a segment [a, b] and rotate it 180 degrees around its midpoint
            # This is complex; let's do a simpler move: pivot
            # Pivot move: pick a pivot point i, rotate the tail [i+1:] by 90 or 180 degrees
            i = rng.randint(1, n - 3)
            # Try rotating the tail starting at i+1 by 90 degrees around current_chain[i]
            pivot = current_chain[i]
            tail = current_chain[i+1:]
            # Generate all 90-degree rotations of the tail around pivot
            # In 3D, 90-degree rotations around each axis
            rotations = []
            for axis in range(dimension):
                for sign in (1, -1):
                    # Rotate tail by 90*sign degrees around axis through pivot
                    new_tail = []
                    for pos in tail:
                        rel = [pos[a] - pivot[a] for a in range(dimension)]
                        if axis == 0:
                            # Rotate around x-axis
                            new_rel = [rel[0], rel[1] * (1 if sign == 1 else -1), rel[2] * (sign if sign == 1 else -1)]
                            # Actually, 90 deg rotation around x: (x,y,z) -> (x, -z, y) for +90
                            # Let me just use simple coordinate swaps
                            pass
                    # This is getting complicated. Let me use a simpler move.
                    break
                break
            # Skip rotation, do a different move type
            # Instead: try moving a short segment (2-3 residues)
            seg_len = rng.choice([2, 3])
            if i + seg_len >= n:
                continue
            start_idx = i
            end_idx = i + seg_len
            # Try translating this segment by one unit in a random direction
            # This is like a collective flip
            # For simplicity, just do single flips more often
            continue
        
        else:
            # Reversal: reverse a segment of the chain
            # This is a valid move if the segment can be reversed without self-intersection
            a = rng.randint(0, n - 2)
            b = rng.randint(a + 1, n - 1)
            # Reverse chain[a:b+1]
            # Check if the reversed segment is self-avoiding and doesn't collide
            new_chain = list(current_chain)
            new_chain[a:b+1] = reversed(new_chain[a:b+1])
            # Check self-avoidance
            if len(set(new_chain)) != n:
                continue
            # Check unit steps
            valid = True
            for k in range(n - 1):
                diff = [new_chain[k+1][d] - new_chain[k][d] for d in range(dimension)]
                if sum(abs(d) for d in diff) != 1:
                    valid = False
                    break
            if not valid:
                continue
            # Compute energy
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
    
    return best_chain


import math


def fold(sequence, dimension):
    """Best conformation using multi-start growth + simulated annealing with deadline guard."""
    deadline = time.time() + 240.0  # 240s budget with safety margin
    
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0  # 0 = no contacts (unfolded)
    
    n = len(sequence)
    
    # Phase 1: Multiple growth starts
    num_growths = min(50, max(10, int(30.0 / max(1, n) * 500)))  # scale with length
    for g in range(num_growths):
        if time.time() >= deadline - 60:  # leave time for annealing
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    
    # Phase 2: Anneal the best chain and some others
    time_left = deadline - time.time()
    if time_left > 10:
        # Anneal the best chain
        annealed = _anneal(sequence, best, dimension, rng, deadline)
        anneal_energy = _energy(sequence, annealed)
        if anneal_energy < best_energy:
            best, best_energy = annealed, anneal_energy
        
        # If time remains, anneal a second best growth
        if time.time() < deadline - 30:
            # Generate one more growth and anneal it
            chain2 = _grow(sequence, dimension, rng)
            if chain2 is not None:
                e2 = _energy(sequence, chain2)
                if e2 < best_energy:
                    best, best_energy = chain2, e2
                annealed2 = _anneal(sequence, best, dimension, rng, deadline)
                e2a = _energy(sequence, annealed2)
                if e2a < best_energy:
                    best, best_energy = annealed2, e2a
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 170.0  # Safety margin before 180s wall-clock kill


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _init_contact_tracker(sequence, chain):
    """Initialize contact tracking for incremental energy updates."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contact_set = set()
    
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    if i < j:
                        contact_set.add((i, j))
    
    return where, len(contact_set)


def _update_contact_on_move(sequence, chain, where, contact_count, old_pos, new_pos, idx):
    """
    Update contact count when residue idx moves from old_pos to new_pos.
    Returns new contact count.
    """
    # Remove contacts involving idx at old_pos
    old_contacts_removed = 0
    ox, oy = old_pos
    for q in ((ox+1, oy), (ox-1, oy), (ox, oy+1), (ox, oy-1)):
        j = where.get(q)
        if j is not None and abs(j - idx) > 1 and sequence[j] == 'H' and sequence[idx] == 'H':
            old_contacts_removed += 1
    
    # Add contacts involving idx at new_pos
    new_contacts_added = 0
    nx, ny = new_pos
    for q in ((nx+1, ny), (nx-1, ny), (nx, ny+1), (nx, ny-1)):
        j = where.get(q)
        if j is not None and abs(j - idx) > 1 and sequence[j] == 'H' and sequence[idx] == 'H':
            new_contacts_added += 1
    
    return contact_count - old_contacts_removed + new_contacts_added


def _greedy_grow(sequence, dimension, rng, temperature=0.0, strategy='default'):
    """
    Randomized greedy chain growth with multiple strategies.
    strategy: 'default', 'compact', 'cluster', 'random'
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    h_positions = []

    for i in range(1, n):
        last = chain[-1]
        options = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            sx, sy = site
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
                        
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            score = gain * 10.0 + free * 0.1
            
            if strategy == 'compact' and i > n * 0.2:
                # Bias toward center of mass
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                dist = abs(sx - cx) + abs(sy - cy)
                score -= dist * 0.3
            
            elif strategy == 'cluster' and h_positions:
                # Bias toward existing H positions (but not too close)
                min_dist = min(abs(sx - hx) + abs(sy - hy) for hx, hy in h_positions)
                if 2 <= min_dist <= 4:
                    score += 0.5
                elif min_dist < 2:
                    score -= 0.3
            
            elif strategy == 'random':
                score += rng.random() * 2.0
            
            score += rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        chain.append(best_site)
        
        if sequence[i] == 'H':
            h_positions.append(best_site)

    return chain


def _greedy_grow_from_fixed(suffix_seq, start_pos, prefix, full_sequence, rng, temperature=0.0):
    """
    Greedy growth for suffix, with proper access to full sequence for H checks.
    """
    if not suffix_seq:
        return []

    occupied = {p: i for i, p in enumerate(prefix)}
    chain = [start_pos]
    
    for i, res in enumerate(suffix_seq):
        global_idx = len(prefix) + i
        last = chain[-1]
        options = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            sx, sy = site
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < global_idx - 1 and full_sequence[j] == 'H':
                        gain += 1
                
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = global_idx
        chain.append(best_site)

    return chain[1:]


def _try_pivot_move(sequence, chain, where, contact_count, rng):
    """
    Try a pivot move: select a pivot point and rotate a contiguous segment.
    Returns (new_chain, new_where, new_contact_count) or None if move fails.
    """
    n = len(chain)
    if n < 5:
        return None
    
    # Choose pivot: a random interior point
    pivot_idx = rng.randint(1, n - 2)
    
    # Choose segment length to rotate
    max_len = min(n - pivot_idx - 1, pivot_idx - 1, 15)
    if max_len < 2:
        return None
    seg_len = rng.randint(2, max_len)
    
    # The segment is chain[pivot_idx : pivot_idx + seg_len + 1]
    # We rotate it 90 degrees around pivot_idx
    pivot_pos = chain[pivot_idx]
    segment = chain[pivot_idx + 1 : pivot_idx + seg_len + 1]
    
    # Check that all segment positions (except pivot) are not occupied by other parts
    segment_positions = set(segment)
    for pos in segment_positions:
        if pos in where and where[pos] != pivot_idx:
            # Check if this position is in our segment
            if pos not in set(chain[pivot_idx + 1 : pivot_idx + seg_len + 1]):
                return None
    
    # Rotate 90 degrees clockwise around pivot
    # (x, y) -> (y, -x) relative to pivot, then add pivot back
    px, py = pivot_pos
    new_segment = []
    for pos in segment:
        dx = pos[0] - px
        dy = pos[1] - py
        # 90 degree clockwise: (dx, dy) -> (dy, -dx)
        new_x = px + dy
        new_y = py - dx
        new_segment.append((new_x, new_y))
    
    # Check self-avoidance: new positions must not collide with rest of chain
    rest_positions = set()
    for i, pos in enumerate(chain):
        if i <= pivot_idx or i > pivot_idx + seg_len:
            rest_positions.add(pos)
    
    for pos in new_segment:
        if pos in rest_positions:
            return None
    
    # Check that consecutive residues in new segment are still unit steps
    for i in range(len(new_segment) - 1):
        dx = new_segment[i+1][0] - new_segment[i][0]
        dy = new_segment[i+1][1] - new_segment[i][1]
        if abs(dx) + abs(dy) != 1:
            return None
    
    # Check connection: last of new segment must be unit step from next residue after segment
    if pivot_idx + seg_len + 1 < n:
        next_pos = chain[pivot_idx + seg_len + 1]
        last_new = new_segment[-1]
        if abs(next_pos[0] - last_new[0]) + abs(next_pos[1] - last_new[1]) != 1:
            return None
    
    # Build new chain
    new_chain = chain[:pivot_idx + 1] + new_segment + chain[pivot_idx + seg_len + 1:]
    
    # Verify self-avoidance
    if len(set(new_chain)) != len(new_chain):
        return None
    
    # Build new where dict
    new_where = {p: i for i, p in enumerate(new_chain)}
    
    # Recompute contact count (expensive but necessary for correctness after large move)
    _, new_contact_count = _init_contact_tracker(sequence, new_chain)
    
    return new_chain, new_where, new_contact_count


def _local_improve(sequence, chain, rng, max_moves=3000, time_limit=None):
    """
    Improve conformation via local moves: kink flips, pivot moves, and short regrows.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)

    where, contact_count = _init_contact_tracker(sequence, chain)
    current_energy = -contact_count
    best_chain = list(chain)
    best_energy = current_energy
    no_improve_count = 0
    start_time = time.time()

    for _ in range(max_moves):
        if time.time() > _DEADLINE - 5:
            break
        if time_limit and time.time() > start_time + time_limit:
            break

        move_type = rng.random()
        improved = False

        if move_type < 0.5:
            # Kink flip
            i = rng.randint(1, n - 2)
            p = chain[i]
            prev, nxt = chain[i-1], chain[i+1]

            candidates = []
            for np in _neighbours(prev):
                if np == p:
                    continue
                if abs(np[0]-nxt[0]) + abs(np[1]-nxt[1]) != 1:
                    continue
                if np in where and where[np] != i:
                    continue
                candidates.append(np)

            if candidates:
                np = rng.choice(candidates)
                del where[p]
                old_contact_count = contact_count
                new_contact_count = _update_contact_on_move(sequence, chain, where, contact_count, p, np, i)
                chain[i] = np
                where[np] = i

                new_energy = -new_contact_count

                if new_energy < current_energy:
                    current_energy = new_energy
                    contact_count = new_contact_count
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                    improved = True
                else:
                    del where[np]
                    contact_count = old_contact_count
                    chain[i] = p
                    where[p] = i
                    no_improve_count += 1

        elif move_type < 0.75:
            # Pivot move
            result = _try_pivot_move(sequence, chain, where, contact_count, rng)
            if result is not None:
                new_chain, new_where, new_contact_count = result
                new_energy = -new_contact_count
                
                if new_energy < current_energy:
                    chain = new_chain
                    where = new_where
                    current_energy = new_energy
                    contact_count = new_contact_count
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                    improved = True
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

        else:
            # Short regrow
            length = rng.randint(3, min(8, n-3))
            split = n - length
            prefix = chain[:split]
            suffix_seq = sequence[split:]

            new_suffix = _greedy_grow_from_fixed(suffix_seq, prefix[-1], prefix, sequence, rng, temperature=0.5)
            if new_suffix is not None:
                new_chain = prefix + new_suffix
                _, new_contact_count = _init_contact_tracker(sequence, new_chain)
                new_energy = -new_contact_count

                if new_energy < current_energy:
                    where = {p: i for i, p in enumerate(new_chain)}
                    chain = new_chain
                    current_energy = new_energy
                    contact_count = new_contact_count
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                    improved = True
                else:
                    no_improve_count += 1

        if not improved:
            no_improve_count += 1

        if no_improve_count > 300:
            break

    return best_chain, best_energy


def _simulated_annealing_full(sequence, chain, rng, iterations=8000, start_temp=5.0, cool_rate=0.999):
    """
    Full simulated annealing with kink flips, pivot moves, and regrows.
    """
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)

    where, contact_count = _init_contact_tracker(sequence, chain)
    current_energy = -contact_count
    best_chain = list(chain)
    best_energy = current_energy
    temp = start_temp
    no_improve = 0

    for it in range(iterations):
        if time.time() > _DEADLINE - 10:
            break
        if temp < 0.01:
            break

        move_type = rng.random()
        accepted = False

        if move_type < 0.45:
            # Kink flip
            i = rng.randint(1, n - 2)
            p = chain[i]
            prev, nxt = chain[i-1], chain[i+1]

            candidates = []
            for np in _neighbours(prev):
                if np == p:
                    continue
                if abs(np[0]-nxt[0]) + abs(np[1]-nxt[1]) != 1:
                    continue
                if np in where and where[np] != i:
                    continue
                candidates.append(np)

            if candidates:
                np = rng.choice(candidates)
                del where[p]
                old_contact_count = contact_count
                new_contact_count = _update_contact_on_move(sequence, chain, where, contact_count, p, np, i)
                chain[i] = np
                where[np] = i

                new_energy = -new_contact_count
                delta = new_energy - current_energy
                if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
                    current_energy = new_energy
                    contact_count = new_contact_count
                    accepted = True
                else:
                    del where[np]
                    contact_count = old_contact_count
                    chain[i] = p
                    where[p] = i

        elif move_type < 0.70:
            # Pivot move
            result = _try_pivot_move(sequence, chain, where, contact_count, rng)
            if result is not None:
                new_chain, new_where, new_contact_count = result
                new_energy = -new_contact_count
                delta = new_energy - current_energy
                if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
                    chain = new_chain
                    where = new_where
                    contact_count = new_contact_count
                    current_energy = new_energy
                    accepted = True

        elif move_type < 0.85:
            # Short Regrow (3-8 residues)
            length = rng.randint(3, min(8, n-3))
            split = n - length
            prefix = chain[:split]
            suffix_seq = sequence[split:]

            new_suffix = _greedy_grow_from_fixed(suffix_seq, prefix[-1], prefix, sequence, rng, temperature=temp)
            if new_suffix is not None:
                new_chain = prefix + new_suffix
                _, new_contact_count = _init_contact_tracker(sequence, new_chain)
                new_energy = -new_contact_count

                delta = new_energy - current_energy
                if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
                    chain = new_chain
                    where = {p: i for i, p in enumerate(chain)}
                    contact_count = new_contact_count
                    current_energy = new_energy
                    accepted = True

        else:
            # Long Regrow (10-20% of chain)
            if n > 20:
                min_len = max(10, n // 10)
                max_len = min(n - 5, n // 5)
                if max_len > min_len:
                    length = rng.randint(min_len, max_len)
                    split = n - length
                    prefix = chain[:split]
                    suffix_seq = sequence[split:]

                    new_suffix = _greedy_grow_from_fixed(suffix_seq, prefix[-1], prefix, sequence, rng, temperature=temp)
                    if new_suffix is not None:
                        new_chain = prefix + new_suffix
                        _, new_contact_count = _init_contact_tracker(sequence, new_chain)
                        new_energy = -new_contact_count

                        delta = new_energy - current_energy
                        if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
                            chain = new_chain
                            where = {p: i for i, p in enumerate(chain)}
                            contact_count = new_contact_count
                            current_energy = new_energy
                            accepted = True

        if accepted:
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        temp *= cool_rate

        if no_improve > 200:
            chain = list(best_chain)
            where, contact_count = _init_contact_tracker(sequence, chain)
            current_energy = best_energy
            temp = max(temp, 2.0)
            no_improve = 0

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence using a portfolio of greedy growths with different strategies,
    plus SA with pivot moves and local improvement. Keep the best within the time budget.
    """
    rng = random.Random(42)
    best_chain = None
    best_energy = 0

    n = len(sequence)

    # Adaptive time allocation
    if n < 30:
        growth_time = 12.0
        sa_time = 40.0
        local_time = 10.0
    elif n < 60:
        growth_time = 10.0
        sa_time = 50.0
        local_time = 12.0
    else:
        growth_time = 8.0
        sa_time = 60.0
        local_time = 10.0

    # Phase 1: Multiple greedy restarts with different strategies
    phase1_end = time.time() + growth_time
    strategies = ['default', 'compact', 'cluster', 'random', 'default', 'compact', 'cluster', 'random']
    attempt = 0
    while time.time() < phase1_end:
        strategy = strategies[attempt % len(strategies)]
        temp = rng.uniform(0.0, 2.0)
        
        chain = _greedy_grow(sequence, dimension, rng, temperature=temp, strategy=strategy)
        
        if chain is None:
            attempt += 1
            continue

        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
            
        attempt += 1

    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
        return [list(p) for p in best_chain]

    # Phase 2: Simulated Annealing with pivot moves
    if time.time() < _DEADLINE - 15:
        phase2_end = time.time() + sa_time
        sa_chain, sa_energy = _simulated_annealing_full(
            sequence, best_chain, rng, iterations=8000, 
            start_temp=5.0, cool_rate=0.999)
        
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
            
        # Second SA pass with different parameters
        if time.time() < phase2_end - 5:
            sa_chain2, sa_energy2 = _simulated_annealing_full(
                sequence, best_chain, rng, iterations=5000, 
                start_temp=3.0, cool_rate=0.9995)
            if sa_energy2 < best_energy:
                best_chain = sa_chain2
                best_energy = sa_energy2

    # Phase 3: Local Improvement with pivot moves
    if time.time() < _DEADLINE - 3:
        improved_chain, improved_energy = _local_improve(
            sequence, best_chain, rng, max_moves=3000)
        
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

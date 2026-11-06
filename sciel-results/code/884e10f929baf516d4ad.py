# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain, pos_map=None):
    if pos_map is None:
        pos_map = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            # Check 4 neighbors
            j = pos_map.get((x + 1, y))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
            j = pos_map.get((x - 1, y))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
            j = pos_map.get((x, y + 1))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
            j = pos_map.get((x, y - 1))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


# Direction vectors: 0=East, 1=North, 2=West, 3=South
_DX = [1, 0, -1, 0]
_DY = [0, 1, 0, -1]


def _beam_search_init_fast(sequence, beam_width=30, compactness=0.2, rng=None):
    """Faster beam search: avoid recomputing contacts from scratch each step."""
    if rng is None:
        rng = random.Random()

    n = len(sequence)
    if n <= 1:
        return [(0, 0)] if n == 1 else []

    beam = []
    for d in range(4):
        beam.append({
            'dir': d,
            'x': 0,
            'y': 0,
            'occupied': {(0, 0): 0},  # point -> index
            'chain': [(0, 0)],
            'score': 0.0
        })

    for i in range(1, n):
        new_beam = []
        for state in beam:
            d = state['dir']
            x, y = state['x'], state['y']
            occ = state['occupied']
            chain = state['chain']
            score = state['score']
            seq_i = sequence[i]

            for turn in (0, 1, 2):
                new_d = (d + turn) % 4
                nx = x + _DX[new_d]
                ny = y + _DY[new_d]
                if (nx, ny) in occ:
                    continue

                new_occ = dict(occ)
                new_occ[(nx, ny)] = i
                new_chain = chain + [(nx, ny)]

                # Incremental contact count
                contact_gain = 0
                if seq_i == 'H':
                    for q in ((nx + 1, ny), (nx - 1, ny), (nx, ny + 1), (nx, ny - 1)):
                        j = new_occ.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            contact_gain += 1

                # Compactness: bounding box
                # Approximate with Manhattan distance from origin
                max_x = max(abs(p[0]) for p in new_chain)
                max_y = max(abs(p[1]) for p in new_chain)
                compact = max_x + max_y

                new_score = score + contact_gain - compactness * compact + rng.random() * 0.05

                new_beam.append({
                    'dir': new_d,
                    'x': nx,
                    'y': ny,
                    'occupied': new_occ,
                    'chain': new_chain,
                    'score': new_score
                })

        if not new_beam:
            break

        new_beam.sort(key=lambda s: s['score'], reverse=True)
        beam = new_beam[:beam_width]

    if not beam:
        return None
    return beam[0]['chain']


def _local_search(sequence, chain, max_iterations=20000, time_limit=5.0, rng=None):
    """Simulated annealing with fast energy computation."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))

    current_chain = list(chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain, pos_map)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 3.0
    cooling = 0.9995

    n = len(current_chain)
    # Precompute H indices for faster checks?
    # h_indices = [i for i, s in enumerate(sequence) if s == 'H']

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        r = rng.random()
        if r < 0.5:
            # Slither move (fast)
            if n < 4:
                continue
            last_idx = n - 1
            last = current_chain[last_idx]
            prev = current_chain[last_idx - 1]
            
            # Find a valid neighbor for prev that is not occupied
            # Occupied is everything except last
            # We can just check the 4 neighbors of prev
            candidate_sites = []
            for s in _neighbours(prev):
                if s == last:
                    continue
                if s not in pos_map:
                    candidate_sites.append(s)
            
            if not candidate_sites:
                continue
                
            new_site = rng.choice(candidate_sites)
            
            # Check energy delta for slither
            # Only last residue moves.
            # Remove last from pos_map conceptually for check?
            # Contacts involving last:
            # Old contacts: H at last with H in neighbors (not prev)
            # New contacts: H at new_site with H in neighbors (not prev)
            
            delta = 0
            if sequence[last_idx] == 'H':
                # Old
                for q in _neighbours(last):
                    if q == prev: continue
                    j = pos_map.get(q)
                    if j is not None and j < last_idx and sequence[j] == 'H':
                        delta -= 1 # Losing a contact (energy increases? No, energy is -contacts. Losing contact -> energy increases -> delta positive)
                        # Wait. Energy = -Contacts.
                        # If we lose a contact, Contacts decrease, Energy increases.
                        # So delta = NewE - OldE.
                        # NewE = -(C-1) = -C + 1. OldE = -C. Delta = 1.
                        # So delta += 1 for each lost contact.
                        # Let's stick to counting contacts gained/lost.
                        pass
                
                # Let's just compute full energy for slither? It's O(1) if we maintain count.
                # But we didn't maintain count.
                # Let's use the pos_map to compute new energy quickly?
                # No, let's just do the O(1) delta properly.
                
                # Reset delta logic
                delta = 0
                # Check neighbors of old last
                for q in _neighbours(last):
                    if q == prev: continue
                    j = pos_map.get(q)
                    if j is not None and j < last_idx and sequence[j] == 'H':
                        # This was a contact. Now it's gone.
                        # Energy increases by 1.
                        delta += 1
                
                # Check neighbors of new site
                for q in _neighbours(new_site):
                    if q == prev: continue
                    j = pos_map.get(q)
                    if j is not None and j < last_idx and sequence[j] == 'H':
                        # This is a new contact.
                        # Energy decreases by 1.
                        delta -= 1

            # Acceptance
            if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.1)):
                # Update state
                pos_map[last] = last_idx # Keep it? No, remove old, add new
                del pos_map[last]
                pos_map[new_site] = last_idx
                current_chain[last_idx] = new_site
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

        else:
            # Pivot move (slower, use fast delta)
            pivot_idx = rng.randint(1, n - 2)
            if pivot_idx >= n - 1: continue
            pivot = current_chain[pivot_idx]
            seg_start = pivot_idx + 1
            if seg_start >= n: continue
            
            angle = rng.choice((90, -90, 180))
            
            # Check validity and compute delta
            # We need to check if new positions collide with fixed part.
            # And compute energy delta.
            
            # Fixed part indices: 0 .. pivot_idx
            # Moving part indices: seg_start .. n-1
            
            # Build set of fixed positions
            fixed_positions = set(current_chain[:seg_start])
            
            # Identify H in moving part
            h_moving_indices = [i for i in range(seg_start, n) if sequence[i] == 'H']
            # Identify H in fixed part positions
            h_fixed_positions = set()
            for i in range(seg_start):
                if sequence[i] == 'H':
                    h_fixed_positions.add(current_chain[i])
            
            # Compute new positions for moving segment
            new_positions = []
            valid = True
            for i in range(seg_start, n):
                p = current_chain[i]
                dx, dy = p[0] - pivot[0], p[1] - pivot[1]
                if angle == 90:
                    ndx, ndy = -dy, dx
                elif angle == -90:
                    ndx, ndy = dy, -dx
                else:
                    ndx, ndy = -dx, -dy
                new_p = (pivot[0] + ndx, pivot[1] + ndy)
                if new_p in fixed_positions:
                    valid = False
                    break
                new_positions.append(new_p)
            
            if not valid:
                continue
                
            # Compute Energy Delta
            # Internal contacts in moving segment are invariant.
            # Only cross contacts (Fixed-Moving) change.
            
            delta = 0
            
            # 1. Lost Contacts:
            # H in moving (old pos) adjacent to H in fixed.
            for i in h_moving_indices:
                old_p = current_chain[i]
                for q in _neighbours(old_p):
                    if q in h_fixed_positions:
                        # Check if q is actually in fixed part (index < seg_start)
                        # h_fixed_positions only contains positions of H in fixed part.
                        # So yes.
                        delta += 1 # Losing contact -> Energy up
            
            # 2. Gained Contacts:
            # H in moving (new pos) adjacent to H in fixed.
            for k, i in enumerate(h_moving_indices):
                new_p = new_positions[k]
                for q in _neighbours(new_p):
                    if q in h_fixed_positions:
                        delta -= 1 # Gaining contact -> Energy down

            if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.1)):
                # Update state
                # Update pos_map
                for i in range(seg_start, n):
                    del pos_map[current_chain[i]]
                for k, i in enumerate(range(seg_start, n)):
                    pos_map[new_positions[k]] = i
                    current_chain[i] = new_positions[k]
                
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

        temperature *= cooling

    return best_chain


def _polish(sequence, chain, max_iterations=3000, time_limit=3.0, rng=None):
    """Deterministic hill-climbing polish."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(42)

    current_chain = list(chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain, pos_map)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    n = len(current_chain)

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 12:
            break

        best_move = None
        best_delta = 0
        best_new_chain = None
        best_new_map = None

        # Try slithers
        if n >= 4:
            last_idx = n - 1
            last = current_chain[last_idx]
            prev = current_chain[last_idx - 1]
            
            for site in _neighbours(prev):
                if site == last or site in pos_map:
                    continue
                
                # Compute delta
                delta = 0
                if sequence[last_idx] == 'H':
                    for q in _neighbours(last):
                        if q == prev: continue
                        j = pos_map.get(q)
                        if j is not None and j < last_idx and sequence[j] == 'H':
                            delta += 1
                    for q in _neighbours(site):
                        if q == prev: continue
                        j = pos_map.get(q)
                        if j is not None and j < last_idx and sequence[j] == 'H':
                            delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_move = 'slither'
                    best_new_chain = current_chain[:-1] + [site]
                    best_new_map = dict(pos_map)
                    del best_new_map[last]
                    best_new_map[site] = last_idx

        # Try pivots (limited set for speed)
        # Pick a few random pivots
        for _ in range(5):
            pivot_idx = rng.randint(1, n - 2)
            if pivot_idx >= n - 1: continue
            pivot = current_chain[pivot_idx]
            seg_start = pivot_idx + 1
            if seg_start >= n: continue
            
            angle = rng.choice((90, -90, 180))
            fixed_positions = set(current_chain[:seg_start])
            h_moving_indices = [i for i in range(seg_start, n) if sequence[i] == 'H']
            h_fixed_positions = set()
            for i in range(seg_start):
                if sequence[i] == 'H':
                    h_fixed_positions.add(current_chain[i])
            
            new_positions = []
            valid = True
            for i in range(seg_start, n):
                p = current_chain[i]
                dx, dy = p[0] - pivot[0], p[1] - pivot[1]
                if angle == 90:
                    ndx, ndy = -dy, dx
                elif angle == -90:
                    ndx, ndy = dy, -dx
                else:
                    ndx, ndy = -dx, -dy
                new_p = (pivot[0] + ndx, pivot[1] + ndy)
                if new_p in fixed_positions:
                    valid = False
                    break
                new_positions.append(new_p)
            
            if not valid:
                continue
                
            delta = 0
            for i in h_moving_indices:
                old_p = current_chain[i]
                for q in _neighbours(old_p):
                    if q in h_fixed_positions:
                        delta += 1
            for k, i in enumerate(h_moving_indices):
                new_p = new_positions[k]
                for q in _neighbours(new_p):
                    if q in h_fixed_positions:
                        delta -= 1

            if delta < best_delta:
                best_delta = delta
                best_move = 'pivot'
                best_new_chain = current_chain[:seg_start] + new_positions
                best_new_map = dict(pos_map)
                for i in range(seg_start, n):
                    del best_new_map[current_chain[i]]
                for k, i in enumerate(range(seg_start, n)):
                    best_new_map[new_positions[k]] = i

        if best_move is None:
            break

        current_chain = best_new_chain
        pos_map = best_new_map
        current_energy += best_delta
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)

    return best_chain


def _greedy_grow(sequence, rng, bias=1.0):
    """Greedy chain growth as fallback initialization."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = bias * gain - 0.5 * dist + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def fold(sequence, dimension):
    seq_len = len(sequence)

    # Time allocation based on sequence length
    if seq_len <= 30:
        num_beam = 10
        beam_width = 40
        local_time = 6.0
        polish_time = 3.0
        num_grow = 100
    elif seq_len <= 50:
        num_beam = 8
        beam_width = 30
        local_time = 5.0
        polish_time = 2.5
        num_grow = 80
    elif seq_len <= 70:
        num_beam = 6
        beam_width = 25
        local_time = 4.0
        polish_time = 2.0
        num_grow = 60
    else:
        num_beam = 5
        beam_width = 20
        local_time = 3.5
        polish_time = 1.5
        num_grow = 40

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Phase 1: Beam search initialization (diverse compactness)
    for i in range(num_beam):
        if _time_left() < 25:
            break
        compactness = 0.1 + rng.random() * 0.4
        bw = beam_width + rng.randint(-5, 5)
        chain = _beam_search_init_fast(sequence, beam_width=max(10, bw),
                                        compactness=compactness, rng=rng)
        if chain is None or len(chain) != seq_len:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Phase 2: Greedy growth initialization (diverse biases)
    for i in range(num_grow):
        if _time_left() < 20:
            break
        bias = 0.5 + rng.random() * 2.0
        chain = _greedy_grow(sequence, rng, bias=bias)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Phase 3: Local search on best chain
    if best_chain is not None and _time_left() > 20:
        improved = _local_search(sequence, best_chain, max_iterations=50000,
                                  time_limit=local_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Phase 4: Polish
        if _time_left() > 15:
            polished = _polish(sequence, best_chain, max_iterations=5000,
                               time_limit=polish_time, rng=rng)
            if polished is not None:
                polished_energy = _energy(sequence, polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_chain = polished

        # Phase 5: Second local search with different seed
        if _time_left() > 25:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search(sequence, best_chain, max_iterations=20000,
                                       time_limit=local_time * 0.5, rng=rng2)
            if improved2 is not None:
                e2 = _energy(sequence, improved2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = improved2

            if _time_left() > 15:
                final_polish = _polish(sequence, best_chain, max_iterations=3000,
                                       time_limit=polish_time * 0.5, rng=rng2)
                if final_polish is not None:
                    fp_energy = _energy(sequence, final_polish)
                    if fp_energy < best_energy:
                        best_energy = fp_energy
                        best_chain = final_polish

    # Fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(seq_len)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

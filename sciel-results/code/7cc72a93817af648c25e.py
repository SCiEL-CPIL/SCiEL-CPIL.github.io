"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation.
    Energy = - (number of H-H contacts)
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _build_contact_map(sequence, chain):
    """Build a set of (i,j) pairs that are H-H contacts, and return energy."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add((i, j))
    return -len(contacts), contacts


def _apply_pivot_fast(chain, pivot_idx, angle):
    """
    Apply a pivot move at pivot_idx with given angle (90, -90, 180).
    Returns None if invalid (collision), else new chain list.
    Optimized: uses list slicing and avoids redundant work.
    """
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return None

    pivot = chain[pivot_idx]
    px, py = pivot
    segment = chain[pivot_idx + 1:]
    if not segment:
        return chain

    fixed_set = set(chain[:pivot_idx + 1])

    rotated = []
    for p in segment:
        dx, dy = p[0] - px, p[1] - py
        if angle == 90:
            ndx, ndy = -dy, dx
        elif angle == -90:
            ndx, ndy = dy, -dx
        else:
            ndx, ndy = -dx, -dy
        new_p = (px + ndx, py + ndy)
        if new_p in fixed_set:
            return None
        rotated.append(new_p)

    return chain[:pivot_idx + 1] + rotated


def _apply_slither_fast(chain, rng):
    """
    Slither move: Move the last residue to a different valid neighbor of the second-to-last.
    """
    n = len(chain)
    if n < 2:
        return None

    prev = chain[-2]
    last = chain[-1]
    occupied = set(chain[:-1])

    candidates = []
    for site in _neighbours(prev):
        if site != last and site not in occupied:
            candidates.append(site)

    if not candidates:
        return None

    new_last = rng.choice(candidates)
    return chain[:-1] + [new_last]


def _grow_greedy(sequence, bias, rng):
    """
    Greedy growth with bias towards contacts and compactness.
    """
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, n):
        curr = chain[-1]
        options = []

        cx = sum(p[0] for p in chain) / len(chain)
        cy = sum(p[1] for p in chain) / len(chain)

        for site in _neighbours(curr):
            if site in occupied:
                continue

            gain = 0
            if sequence[i] == "H":
                sx, sy = site
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    q = (sx + dx, sy + dy)
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1

            dist = math.sqrt((site[0] - cx) ** 2 + (site[1] - cy) ** 2)
            free = sum(1 for q in _neighbours(site) if q not in occupied)

            score = bias * gain - 0.5 * dist + 0.1 * free + rng.random()
            options.append((score, site))

        if not options:
            return None

        best_site = max(options, key=lambda x: x[0])[1]
        chain.append(best_site)
        occupied[best_site] = i

    return chain


def _spiral_fold(sequence):
    """
    Generate a square spiral conformation. This provides a compact,
    self-avoiding initial conformation that is often close to optimal
    for HP sequences with regular H/P patterns.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]

    # Square spiral: go right 1, down 1, left 2, up 2, right 3, down 3, ...
    chain = [(0, 0)]
    x, y = 0, 0
    occupied = {(0, 0)}

    # Directions: right, down, left, up
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    step_count = 0
    steps_in_dir = 1
    steps_in_current = 0

    for i in range(1, n):
        # Move one step in current direction
        dx, dy = dirs[dir_idx]
        x += dx
        y += dy
        new_pos = (x, y)

        if new_pos in occupied:
            # Collision: try to turn
            # This shouldn't happen with a proper spiral, but handle gracefully
            break

        chain.append(new_pos)
        occupied.add(new_pos)
        steps_in_current += 1

        if steps_in_current >= steps_in_dir:
            steps_in_current = 0
            dir_idx = (dir_idx + 1) % 4
            if dir_idx % 2 == 0:
                # Every other turn (after completing a pair of directions), increase step length
                # Actually, the pattern is: 1,1,2,2,3,3,4,4,...
                # So we increase after every 2 direction changes
                pass
            # Increase step length every 2 direction changes
            if dir_idx == 0 or dir_idx == 2:
                # We've completed a "layer" of the spiral
                pass

        # Update steps_in_dir based on spiral pattern
        # The spiral pattern: R1, D1, L2, U2, R3, D3, L4, U4, ...
        # Step lengths: 1,1,2,2,3,3,4,4,...
        # So steps_in_dir increases by 1 every 2 direction changes
        # Let's track this properly:
        # After 2 direction changes, increment the step length
        # We'll use a counter for direction changes
        # Actually, let's just use a simpler approach:
        # Track the "ring" number. Ring 0: R1,D1. Ring 1: L2,U2. Ring 2: R3,D3. Ring 3: L4,U4.
        # Ring r has step length r+1 for the first two directions.
        # So after completing 2 directions in a ring, move to next ring.

    # The above spiral logic is getting complex. Let me use a cleaner implementation:
    # Reset and use a well-known spiral algorithm
    chain = [(0, 0)]
    x, y = 0, 0
    occupied = {(0, 0)}

    # Spiral pattern: R1, D1, L2, U2, R3, D3, L4, U4, ...
    # Step lengths: 1, 1, 2, 2, 3, 3, 4, 4, ...
    # Direction sequence: R, D, L, U, R, D, L, U, ...
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    steps_taken = 0
    current_step_len = 1
    dir_changes = 0

    for i in range(1, n):
        dx, dy = dirs[dir_idx]
        x += dx
        y += dy
        new_pos = (x, y)

        if new_pos in occupied:
            # Spiral collision: this means the spiral is closing in
            # Try to continue with a different approach or just break
            # For robustness, we'll try to find an alternative path
            # But for now, just break and use greedy fallback
            break

        chain.append(new_pos)
        occupied.add(new_pos)
        steps_taken += 1

        if steps_taken >= current_step_len:
            steps_taken = 0
            dir_idx = (dir_idx + 1) % 4
            dir_changes += 1
            # Increase step length every 2 direction changes
            if dir_changes % 2 == 0:
                current_step_len += 1

    # If spiral didn't complete, fill in with greedy
    if len(chain) < n:
        # Use greedy to extend
        for i in range(len(chain), n):
            curr = chain[-1]
            best_site = None
            best_score = float('-inf')
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)

            for site in _neighbours(curr):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    sx, sy = site
                    for dx2, dy2 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        q = (sx + dx2, sy + dy2)
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and sequence[j] == "H":
                                gain += 1
                dist = math.sqrt((site[0] - cx) ** 2 + (site[1] - cy) ** 2)
                score = 2.0 * gain - 0.5 * dist
                if score > best_score:
                    best_score = score
                    best_site = site

            if best_site is None:
                # Can't extend, return what we have
                break

            chain.append(best_site)
            occupied.add(best_site)

    return chain if len(chain) == n else None


def _local_search(sequence, chain, time_limit, rng):
    """
    Simulated Annealing local search with incremental energy updates.
    """
    current_chain = list(chain)
    current_energy, current_contacts = _build_contact_map(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    n = len(current_chain)
    T = 5.0
    T_min = 0.01

    start_time = time.time()
    if n > 80:
        cooling = 0.9999
    elif n > 50:
        cooling = 0.9997
    else:
        cooling = 0.9995

    max_iters = 10000

    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break

        r = rng.random()
        new_chain = None

        if r < 0.8:
            pivot_idx = rng.randint(1, n - 2)
            angle = rng.choice([90, -90, 180])
            new_chain = _apply_pivot_fast(current_chain, pivot_idx, angle)
        else:
            new_chain = _apply_slither_fast(current_chain, rng)

        if new_chain is None:
            continue

        # Compute new energy
        new_energy, new_contacts = _build_contact_map(sequence, new_chain)
        delta = new_energy - current_energy

        if delta < 0:
            current_chain = new_chain
            current_energy = new_energy
            current_contacts = new_contacts
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            if T > T_min:
                exp_val = math.exp(-delta / T)
                if rng.random() < exp_val:
                    current_chain = new_chain
                    current_energy = new_energy
                    current_contacts = new_contacts

        T *= cooling

    return best_chain


def _polish_hill_climb(sequence, chain, time_limit):
    """
    Final hill-climb polish: try all pivot moves and accept any improvement.
    """
    current = list(chain)
    curr_e = _energy(sequence, current)
    n = len(current)
    start_time = time.time()

    while time.time() - start_time < time_limit and _time_left() > 15:
        improved = False
        # Try pivots in random order to avoid bias
        pivot_indices = list(range(1, n - 1))
        random.shuffle(pivot_indices)

        for pivot_idx in pivot_indices:
            if time.time() - start_time > time_limit:
                break
            if _time_left() < 15:
                break
            for angle in (90, -90, 180):
                new_c = _apply_pivot_fast(current, pivot_idx, angle)
                if new_c:
                    new_e = _energy(sequence, new_c)
                    if new_e < curr_e:
                        current = new_c
                        curr_e = new_e
                        improved = True
                        break
            if improved:
                break

        if not improved:
            break

    return current, curr_e


def fold(sequence, dimension):
    """
    Main entry point.
    1. Generate multiple diverse initial conformations (including spiral).
    2. Run local search on the top-K best initial conformations.
    3. Polish the best result with hill climbing.
    4. Return the best result found.
    """
    seq_len = len(sequence)

    remaining = _time_left()

    if remaining < 3:
        # Emergency: return a simple self-avoiding walk
        return [[i, 0] for i in range(seq_len)]

    # Time allocation
    if remaining < 10:
        time_for_init = 0.5
        time_for_search = 3.0
    elif remaining < 30:
        time_for_init = 1.0
        time_for_search = 5.0
    else:
        time_for_init = 2.0
        time_for_search = 8.0

    rng = random.Random(12345 + seq_len)

    best_chain = None
    best_energy = 0

    # Phase 1: Initialization
    init_start = time.time()
    candidates = []

    # Try spiral first
    spiral = _spiral_fold(sequence)
    if spiral and len(spiral) == seq_len:
        e = _energy(sequence, spiral)
        candidates.append((e, spiral))

    # Generate greedy candidates
    num_greedy = 20
    for i in range(num_greedy):
        if time.time() - init_start > time_for_init:
            break
        bias = 0.5 + rng.random() * 2.0
        chain = _grow_greedy(sequence, bias, rng)
        if chain and len(chain) == seq_len:
            e = _energy(sequence, chain)
            candidates.append((e, chain))

    # Sort by energy
    candidates.sort(key=lambda x: x[0])

    # Keep top 3
    top_candidates = candidates[:3]

    if not top_candidates:
        # Fallback
        best_chain = [[i, 0] for i in range(seq_len)]
        best_energy = _energy(sequence, best_chain)
        return best_chain

    # Phase 2: Local Search on top candidates
    search_start = time.time()

    for e, chain in top_candidates:
        if time.time() - search_start > time_for_search:
            break
        if _time_left() < 15:
            break

        remaining_search = time_for_search - (time.time() - search_start)
        if remaining_search < 0.5:
            break

        improved = _local_search(sequence, chain, remaining_search * 0.9, rng)

        if improved:
            e_imp = _energy(sequence, improved)
            if e_imp < best_energy or best_chain is None:
                best_energy = e_imp
                best_chain = improved

    # Phase 3: Final Polish (Hill Climb)
    if best_chain and _time_left() > 15:
        polish_time = min(2.0, _time_left() - 15)
        polished, polished_e = _polish_hill_climb(sequence, best_chain, polish_time)
        if polished_e < best_energy:
            best_energy = polished_e
            best_chain = polished

    if best_chain is None:
        best_chain = [[i, 0] for i in range(seq_len)]

    return best_chain
# EVOLVE-BLOCK-END

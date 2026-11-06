"""HP lattice protein folding: fast greedy growth + simulated annealing with reversals."""
# EVOLVE-BLOCK-START
import random
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _build_contact_index(sequence, chain):
    """Map each lattice point to its residue index; return H-set for quick lookup."""
    where = {}
    for i, p in enumerate(chain):
        where[tuple(p)] = i
    h_set = set()
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            h_set.add(tuple(p))
    return where, h_set

def _compute_energy(sequence, chain):
    """Count non-consecutive H-H contacts; energy = -contacts."""
    where, h_set = _build_contact_index(sequence, chain)
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth with incremental H-contact gain."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    h_occupied = set() if sequence[0] == 'H' else set()
    for i in range(1, len(sequence)):
        best_gain = -1
        best_site = None
        best_free = -1
        best_rand = -1
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in h_occupied:
                        j = occupied[q]
                        if j < i - 1:
                            gain += 1
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            rand_val = rng.random()
            score = gain * 10.0 + free * 0.1 + rand_val
            if score > best_gain:
                best_gain = score
                best_site = site
                best_free = free
                best_rand = rand_val
        if best_site is None:
            return None
        chain.append(best_site)
        occupied[best_site] = i
        if sequence[i] == 'H':
            h_occupied.add(best_site)
    return chain

def _sa_fold(sequence, dimension, chain, time_budget, rng):
    """Simulated annealing with k-move reversals. Returns best chain found."""
    n = len(sequence)
    if n < 4:
        return chain

    best_chain = list(chain)
    best_energy = _compute_energy(sequence, chain)
    cur_chain = list(chain)
    cur_energy = best_energy

    # Build lattice count map: point -> count of residues at that point (should be 1 each)
    # For energy delta we need: for each site in the flipped segment, count H-neighbours outside segment
    h_flags = [1 if c == 'H' else 0 for c in sequence]

    # Precompute neighbours list for each residue for speed
    # We'll recompute neighbours on the fly but cache the list
    start_time = time.monotonic()
    deadline = start_time + time_budget

    # Temperature schedule
    t_start = 2.0
    t_end = 0.001
    # We'll do a fixed number of iterations based on time
    max_iters = 500000  # cap to avoid infinite loops; time guard handles it

    for it in range(max_iters):
        if time.monotonic() >= deadline:
            break

        # Choose k (reversal length)
        # Smaller k for finer moves, larger k for bigger jumps
        # Bias towards small k
        if rng.random() < 0.7:
            k = 2
        elif rng.random() < 0.5:
            k = 3
        elif rng.random() < 0.3:
            k = 4
        else:
            k = 5
        if k >= n:
            k = max(2, n - 1)

        # Choose start position i
        i = rng.randint(0, n - 1 - k)
        # The segment to reverse is [i, i+k-1]
        # Reversal: new_chain[j] = old_chain[i+k-1-j] for j in [i, i+k-1]
        # Check self-avoidance of the new positions
        # The new positions are the old positions of the segment in reverse order
        # Since the old chain was self-avoiding, the reversed segment occupies the same set of points
        # BUT the connectivity changes: residue i now connects to residue i-1 (if i>0) and residue i+k (if i+k<n)
        # We need to check that the new connections don't create self-avoidance violations
        # Specifically:
        #   - If i > 0: new position of residue i is old position of residue i+k-1.
        #     Check it's not adjacent to old position of residue i-1 (unless i-1 is in the segment, which it's not)
        #     Actually we need to check: new_pos[i] != pos[i-1] (already guaranteed since self-avoiding)
        #     And new_pos[i] not adjacent to any residue j < i-1? No, only adjacency matters for contacts,
        #     but self-avoidance only requires no two residues share the same point.
        #     Since we're just permuting the same set of points, self-avoidance is automatically preserved!
        #     Wait - no. The chain is a sequence of points. Reversing a segment keeps the same set of points,
        #     so no two points coincide. Self-avoidance is preserved.
        #     BUT we also need consecutive residues to be unit steps. After reversal:
        #       - Residue i-1 (if exists) connects to new residue i (which is old residue i+k-1).
        #         We need |pos[i-1] - pos[i+k-1]| = 1 (Manhattan). This is NOT guaranteed.
        #       - Residue i+k (if exists) connects to new residue i+k-1 (which is old residue i).
        #         We need |pos[i+k] - pos[i]| = 1. This is NOT guaranteed.
        #     So we must check these two boundary conditions.

        # Check boundary 1: if i > 0, new pos of residue i is old pos of residue i+k-1
        #   Must be unit step from pos[i-1]
        if i > 0:
            p_prev = cur_chain[i - 1]
            p_new_i = cur_chain[i + k - 1]
            if abs(p_prev[0] - p_new_i[0]) + abs(p_prev[1] - p_new_i[1]) != 1:
                continue

        # Check boundary 2: if i+k < n, new pos of residue i+k-1 is old pos of residue i
        #   Must be unit step from pos[i+k]
        if i + k < n:
            p_next = cur_chain[i + k]
            p_new_last = cur_chain[i]
            if abs(p_next[0] - p_new_last[0]) + abs(p_next[1] - p_new_last[1]) != 1:
                continue

        # Valid move! Now compute energy delta.
        # The set of occupied points is unchanged, but the mapping from residue index to point changes.
        # Energy = -(# of H-H pairs on neighbouring lattice sites, not consecutive along chain)
        # We need to recompute contacts for residues in the affected region.
        # The affected residues are: i-1, i, i+1, ..., i+k-1, i+k (within bounds)
        # Because contacts involve neighbouring lattice sites, and the positions of residues in [i, i+k-1] change.
        # Actually the positions of residues OUTSIDE the segment don't change, but the residue indices at those
        # positions do change for the segment residues.
        #
        # Simpler approach: build a new chain, compute energy. But that's O(n) per move.
        # For speed, let's do incremental:
        #
        # Let's define: for each residue r, its position is pos[r]. A contact is a pair (a, b) with
        # |pos[a]-pos[b]|=1, |a-b|>1, seq[a]='H', seq[b]='H'.
        #
        # When we reverse segment [i, i+k-1], the positions of residues in that range change.
        # Specifically, new_pos[i+t] = old_pos[i+k-1-t] for t=0..k-1.
        #
        # Contacts that can change are those involving at least one residue in [i-1, i+k] (with bounds).
        # Let's collect the set of residues whose position changed: S = {i, i+1, ..., i+k-1}
        # Contacts involving a residue in S and a residue outside S, or two residues in S, can change.
        #
        # To be safe and simple, let's just compute the energy of the new chain directly.
        # Given time budget, this should be fine for n up to 100.

        new_chain = cur_chain[:i] + cur_chain[i:i+k][::-1] + cur_chain[i+k:]
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - cur_energy

        # Temperature: linear cooling
        frac = it / max_iters
        temp = t_start + (t_end - t_start) * frac
        if temp < t_end:
            temp = t_end

        # Accept?
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / temp):
            cur_chain = new_chain
            cur_energy = new_energy
            if cur_energy < best_energy:
                best_energy = cur_energy
                best_chain = list(cur_chain)

    return best_chain

def fold(sequence, dimension):
    """Best conformation using greedy growth + simulated annealing, with deadline guard."""
    n = len(sequence)
    start_time = time.monotonic()
    total_budget = 200.0  # safety margin before 210s kill
    max_time = start_time + total_budget

    # Allocate time per sequence based on length
    # Short sequences: more restarts; long sequences: fewer
    if n <= 20:
        num_grow = 150
        sa_time_each = 1.5
    elif n <= 40:
        num_grow = 80
        sa_time_each = 2.5
    elif n <= 60:
        num_grow = 50
        sa_time_each = 3.5
    else:
        num_grow = 30
        sa_time_each = 5.0

    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 1

    for attempt in range(num_grow):
        # Check time
        if time.monotonic() >= max_time - 2.0:
            break

        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue

        # Run SA on this chain
        sa_budget = min(sa_time_each, max(0.1, max_time - time.monotonic() - 1.0))
        if sa_budget <= 0.05:
            break

        sa_chain = _sa_fold(sequence, dimension, chain, sa_budget, rng)
        sa_energy = _compute_energy(sequence, sa_chain)

        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = list(sa_chain)

        # Also check the original chain in case SA made it worse (unlikely but safe)
        orig_energy = _compute_energy(sequence, chain)
        if orig_energy < best_energy:
            best_energy = orig_energy
            best_chain = list(chain)

    # Fallback: straight line
    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]

    return [[p[0], p[1]] for p in best_chain]
# EVOLVE-BLOCK-END

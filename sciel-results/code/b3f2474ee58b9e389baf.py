"""HP lattice protein folding: optimized beam search + fast incremental SA + local polish."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 110.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    h_positions = set()
    for i, c in enumerate(sequence):
        if c == "H":
            h_positions.add(i)
    contacts = 0
    for i, p in enumerate(chain):
        if i in h_positions:
            x, y = p
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = where.get((x + dx, y + dy))
                if j is not None and abs(i - j) > 1 and j in h_positions:
                    contacts += 1
    return -contacts // 2


def _build_where(chain):
    """Build position -> index map."""
    return {p: i for i, p in enumerate(chain)}


def _fast_move_delta(sequence, chain, where, j, new_pos):
    """
    Compute energy delta for moving residue j from chain[j] to new_pos.
    Returns delta_e (new - old) or None if move is invalid.
    """
    n = len(chain)
    old_pos = chain[j]
    if old_pos == new_pos:
        return 0

    # Check collision
    if new_pos in where:
        return None

    # Check unit step with neighbors in chain
    if j > 0:
        prev = chain[j - 1]
        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
            return None
    if j < n - 1:
        nxt = chain[j + 1]
        if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
            return None

    is_h_j = (sequence[j] == "H")
    if not is_h_j:
        return 0

    delta = 0
    ox, oy = old_pos
    nx, ny = new_pos
    # Remove contacts at old position
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        k = where.get((ox + dx, oy + dy))
        if k is not None and abs(j - k) > 1 and sequence[k] == "H":
            delta += 1
    # Add contacts at new position
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        k = where.get((nx + dx, ny + dy))
        if k is not None and abs(j - k) > 1 and sequence[k] == "H":
            delta -= 1
    return delta


def _apply_move(chain, where, j, new_pos):
    """Apply a single-residue move, updating chain and where."""
    old_pos = chain[j]
    chain[j] = new_pos
    del where[old_pos]
    where[new_pos] = j


def _reversal_delta(sequence, chain, where, i, j):
    """
    Compute energy delta for reversing chain[i..j].
    Returns (new_chain, new_where, delta_e) or (None, None, None).
    """
    n = len(chain)
    if i >= j:
        return None, None, None

    # Check boundary unit steps
    if i > 0:
        p_prev = chain[i - 1]
        p_end = chain[j]
        if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
            return None, None, None
    if j < n - 1:
        p_start = chain[i]
        p_next = chain[j + 1]
        if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
            return None, None, None

    # Build new chain
    new_chain = chain[:i] + chain[i:j + 1][::-1] + chain[j + 1:]

    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None, None, None
        seen.add(p)

    # Compute energy delta
    # The set of positions is the same, but residue-to-position mapping changes for indices in [i,j]
    # Old: residue k at chain[k] for k in [i,j]
    # New: residue k at new_chain[k] = chain[i+j-k] for k in [i,j]

    # We need to recompute contacts involving any position in the segment.
    # Contacts are between H residues at adjacent lattice positions that are not chain-adjacent.

    # Build new where map
    new_where = {}
    for idx, p in enumerate(new_chain):
        new_where[p] = idx

    # Compute delta by comparing contacts
    # Old contacts: for each H residue k in [i,j], check its 4 neighbors in old chain
    # New contacts: for each H residue k in [i,j], check its 4 neighbors in new chain

    old_contacts = 0
    new_contacts = 0

    # We need to count contacts where at least one endpoint is in [i,j]
    # But to avoid double counting, we count each contact once (when the lower index endpoint is processed)

    # Actually, let's just compute the full energy difference for affected positions.
    # A contact is between positions p and q (adjacent lattice) where the residues at p and q are both H
    # and their chain indices differ by more than 1.

    # For efficiency, iterate over positions in the segment and their neighbors.
    segment_positions = set(chain[i:j+1])

    # Count old contacts involving segment positions
    for p in segment_positions:
        x, y = p
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (x + dx, y + dy)
            if q in where:
                idx_p = where[p]
                idx_q = where[q]
                if abs(idx_p - idx_q) > 1:
                    if sequence[idx_p] == 'H' and sequence[idx_q] == 'H':
                        # Count this contact once
                        if idx_p < idx_q:
                            old_contacts += 1

    # Count new contacts involving segment positions
    for p in segment_positions:
        x, y = p
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (x + dx, y + dy)
            if q in new_where:
                idx_p = new_where[p]
                idx_q = new_where[q]
                if abs(idx_p - idx_q) > 1:
                    if sequence[idx_p] == 'H' and sequence[idx_q] == 'H':
                        if idx_p < idx_q:
                            new_contacts += 1

    delta_e = -(new_contacts - old_contacts)  # energy = -contacts, so delta = -(new-old)
    # Wait: energy = -contacts. delta_energy = new_energy - old_energy = -new_contacts - (-old_contacts) = old_contacts - new_contacts
    delta_e = old_contacts - new_contacts

    return new_chain, new_where, delta_e


def _beam_search_init(sequence, rng, beam_width=30, time_limit=3.0):
    """
    Beam search over self-avoiding walks with contact-aware scoring.
    Uses incremental contact counting to avoid O(n) per state.
    """
    n = len(sequence)
    if n <= 1:
        return [[(0, 0)]] if n == 1 else [[]]

    deadline = time.time() + time_limit

    # Precompute H positions for quick lookup
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')

    # State: (score, chain, occupied_set, pos_to_idx)
    states = []
    for first_dir in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if n >= 2:
            chain = [(0, 0), (first_dir[0], first_dir[1])]
            occupied = {(0, 0), (first_dir[0], first_dir[1])}
            pos_to_idx = {(0, 0): 0, (first_dir[0], first_dir[1]): 1}
            score = 0.0
            states.append((score, chain, occupied, pos_to_idx))

    if not states:
        return [[(0, 0)]]

    for step in range(2, n):
        if time.time() > deadline:
            break
        new_states = []
        is_h_step = step in h_set
        for score, chain, occupied, pos_to_idx in states:
            last = chain[-1]
            lx, ly = last
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_pos = (lx + dx, ly + dy)
                if new_pos in occupied:
                    continue

                add_score = 0.0
                if is_h_step:
                    nx, ny = new_pos
                    for ndx, ndy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nb = (nx + ndx, ny + ndy)
                        if nb in pos_to_idx:
                            k = pos_to_idx[nb]
                            if abs(step - k) > 1 and k in h_set:
                                add_score += 1.5

                # Compactness: prefer positions with fewer free neighbors
                free = 0
                for ndx, ndy in ((1, 0), (-1, 0), (0, 1), (0, -4)):
                    if (new_pos[0] + ndx, new_pos[1] + ndy) not in occupied:
                        free += 1
                add_score -= 0.05 * free

                new_score = score + add_score + rng.random() * 0.15
                new_chain = chain + [new_pos]
                new_occupied = occupied | {new_pos}
                new_pos_to_idx = pos_to_idx.copy()
                new_pos_to_idx[new_pos] = step
                new_states.append((new_score, new_chain, new_occupied, new_pos_to_idx))

        if not new_states:
            break

        new_states.sort(key=lambda x: x[0], reverse=True)
        states = new_states[:beam_width]

    result = []
    for score, chain, occupied, pos_to_idx in states:
        if len(chain) == n:
            result.append(chain)
    if not result:
        # Fallback: straight line
        result = [[(i, 0) for i in range(n)]]
    return result


def _sa_optimized(sequence, initial_chain, time_limit, rng):
    """
    Simulated annealing with O(1) energy delta for moves, O(segment) for reversals.
    Maintains where dict incrementally.
    """
    n = len(initial_chain)
    if n < 3:
        return initial_chain[:], _energy(sequence, initial_chain)

    current_chain = list(initial_chain)
    current_where = _build_where(current_chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy

    T = 1.2
    cooling_rate = 0.9997
    deadline = time.time() + time_limit
    stall_counter = 0
    max_stall = 200
    iterations = 0

    while time.time() < deadline:
        if _remaining_time() < 1.0:
            break

        iterations += 1
        r = rng.random()

        if r < 0.45:
            # Single-residue move
            j = rng.randint(1, n - 2)
            cur = current_chain[j]
            cx, cy = cur
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_pos = (cx + dx, cy + dy)
                delta_e = _fast_move_delta(sequence, current_chain, current_where, j, new_pos)
                if delta_e is None:
                    continue
                if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
                    _apply_move(current_chain, current_where, j, new_pos)
                    current_energy += delta_e
                    stall_counter = 0
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
                    break
            else:
                stall_counter += 1

        elif r < 0.85:
            # Reversal
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain, new_where, delta_e = _reversal_delta(sequence, current_chain, current_where, i, j)
            if new_chain is None:
                stall_counter += 1
                continue
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
                current_chain = new_chain
                current_where = new_where
                current_energy += delta_e
                stall_counter = 0
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
            else:
                stall_counter += 1

        else:
            # Pivot
            i = rng.randint(1, n - 2)
            angle = rng.choice([1, 2, 3])
            pivot = current_chain[i]
            px, py = pivot
            tail = current_chain[i + 1:]
            prefix = current_chain[:i + 1]

            rotated_tail = []
            for p in tail:
                x, y = p[0] - px, p[1] - py
                for _ in range(angle):
                    x, y = -y, x
                rotated_tail.append((px + x, py + y))

            new_chain = prefix + rotated_tail

            # Check validity
            seen = set()
            valid = True
            for p in new_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            if valid:
                for k in range(1, len(new_chain)):
                    if abs(new_chain[k][0] - new_chain[k - 1][0]) + abs(new_chain[k][1] - new_chain[k - 1][1]) != 1:
                        valid = False
                        break

            if not valid:
                stall_counter += 1
                continue

            new_where = _build_where(new_chain)
            new_energy = _energy(sequence, new_chain)
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
                current_chain = new_chain
                current_where = new_where
                current_energy = new_energy
                stall_counter = 0
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
            else:
                stall_counter += 1

        T *= cooling_rate
        if T < 0.03:
            if rng.random() < 0.3:
                T = 1.0
            else:
                break

        if stall_counter > max_stall:
            # Perturbation
            if rng.random() < 0.6:
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                new_chain, new_where, delta_e = _reversal_delta(sequence, current_chain, current_where, i, j)
                if new_chain is not None:
                    current_chain = new_chain
                    current_where = new_where
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
            else:
                # Random single moves
                for _ in range(5):
                    jj = rng.randint(1, n - 2)
                    cur = current_chain[jj]
                    cx, cy = cur
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        new_pos = (cx + dx, cy + dy)
                        delta_e = _fast_move_delta(sequence, current_chain, current_where, jj, new_pos)
                        if delta_e is not None:
                            _apply_move(current_chain, current_where, jj, new_pos)
                            current_energy += delta_e
                            break
            stall_counter = 0

    return best_chain, best_energy


def _local_polish(sequence, chain, time_limit, rng):
    """
    Greedy local search: try all single moves and short reversals to find local optimum.
    """
    n = len(chain)
    if n < 3:
        return chain[:], _energy(sequence, chain)

    current_chain = list(chain)
    current_where = _build_where(current_chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy

    deadline = time.time() + time_limit

    while time.time() < deadline:
        if _remaining_time() < 1.0:
            break
        improved = False

        # Try all single-residue moves (greedy)
        for j in range(1, n - 1):
            if time.time() > deadline or _remaining_time() < 1.0:
                break
            cur = current_chain[j]
            cx, cy = cur
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_pos = (cx + dx, cy + dy)
                delta_e = _fast_move_delta(sequence, current_chain, current_where, j, new_pos)
                if delta_e is not None and delta_e < 0:
                    _apply_move(current_chain, current_where, j, new_pos)
                    current_energy += delta_e
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]

        # Try short reversals (length 2-6)
        for length in range(2, min(7, n)):
            if time.time() > deadline or _remaining_time() < 1.0:
                break
            for i in range(0, n - length):
                j = i + length - 1
                new_chain, new_where, delta_e = _reversal_delta(sequence, current_chain, current_where, i, j)
                if new_chain is not None and delta_e < 0:
                    current_chain = new_chain
                    current_where = new_where
                    current_energy += delta_e
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]

        if not improved:
            # Perturbation
            if rng.random() < 0.6:
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                new_chain, new_where, delta_e = _reversal_delta(sequence, current_chain, current_where, i, j)
                if new_chain is not None:
                    current_chain = new_chain
                    current_where = new_where
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
            else:
                break

    return best_chain, best_energy


def _is_valid(chain):
    """Check self-avoidance and unit steps."""
    if not chain:
        return True
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    for i in range(1, len(chain)):
        if abs(chain[i][0] - chain[i - 1][0]) + abs(chain[i][1] - chain[i - 1][1]) != 1:
            return False
    return True


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    rng = random.Random(42 + len(sequence) * 7)
    n = len(sequence)

    if n < 2:
        return [[0, 0]] * n

    best_chain = None
    best_energy = 0

    # Phase 1: Beam search initialization
    beam_width = 50 if n <= 30 else 25
    beam_time = min(5.0, _remaining_time() * 0.15)
    if beam_time > 0.5:
        initial_chains = _beam_search_init(sequence, rng, beam_width, beam_time)
    else:
        initial_chains = [[(i, 0) for i in range(n)]]

    # Evaluate and sort
    scored = []
    for chain in initial_chains:
        if len(chain) == n and _is_valid(chain):
            e = _energy(sequence, chain)
            scored.append((e, chain))

    scored.sort(key=lambda x: x[0])

    # Keep top candidates
    top_k = min(8, len(scored))
    candidates = [c[1] for c in scored[:top_k]]

    if not candidates:
        candidates = [[(i, 0) for i in range(n)]]

    # Phase 2: SA on each candidate
    total_time = _remaining_time() - 10.0  # reserve for polish
    if total_time < 2.0:
        total_time = max(1.0, _remaining_time() - 3.0)

    time_per = total_time / len(candidates)

    for idx, chain in enumerate(candidates):
        if _remaining_time() < 5.0:
            break
        alloc = min(time_per, _remaining_time() - 4.0)
        if alloc < 0.5:
            continue

        bc, be = _sa_optimized(sequence, chain, alloc, rng)
        if be < best_energy:
            best_energy = be
            best_chain = bc

    # Fallback
    if best_chain is None:
        best_chain = candidates[0]
        best_energy = _energy(sequence, best_chain)

    # Phase 3: Local polish
    if _remaining_time() > 4.0 and best_chain is not None:
        polish_time = min(_remaining_time() - 2.0, 15.0)
        if polish_time > 1.0:
            polish_rng = random.Random(123 + len(sequence))
            best_chain, best_energy = _local_polish(sequence, best_chain, polish_time, polish_rng)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

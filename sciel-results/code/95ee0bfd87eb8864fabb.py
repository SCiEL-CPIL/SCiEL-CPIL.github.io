"""HP lattice protein folding: contact-guided growth + local search."""
# EVOLVE-BLOCK-START
import random
import math
import time

_START_TIME = time.time()
_DEADLINE = 210.0
_MARGIN = 5.0  # safety margin seconds


def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START_TIME)


def _neighbours(p):
    x, y = p
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy_from_contacts(contacts):
    return -len(contacts)


def _compute_contacts(sequence, chain):
    """Return set of frozenset({i,j}) for H-H contacts (non-adjacent, neighbouring sites)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    n = len(chain)
    for i, p in enumerate(chain):
        if sequence[i] != "H":
            continue
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                if i < j:
                    contacts.add(frozenset((i, j)))
    return contacts


def _build_feasible_pairs(sequence):
    """Return list of (priority, i, j) for feasible H-H pairs, sorted by priority desc."""
    n = len(sequence)
    pairs = []
    for i in range(n):
        if sequence[i] != "H":
            continue
        for j in range(i + 2, n):
            if sequence[j] != "H":
                continue
            d = j - i  # min lattice distance is d (need |i-j| steps)
            # feasible if d <= some bound (we allow up to n)
            # priority: shorter sequence distance = more constrained = higher priority
            pairs.append((d, i, j))
    # sort by d ascending (shorter distance first = higher priority)
    pairs.sort(key=lambda x: x[0])
    return pairs


def _grow_guided(sequence, feasible_pairs, rng, max_contacts_to_target=0):
    """Grow chain guided by contact priorities. Returns chain or None."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    # Target contacts: top feasible pairs
    targets = feasible_pairs[:max_contacts_to_target] if max_contacts_to_target > 0 else feasible_pairs
    target_set = set()
    for (_, i, j) in targets:
        target_set.add((i, j))

    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    satisfied = set()

    for i in range(1, n):
        best_score = -1
        best_site = None
        candidates = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Compute new contacts this site would make
            new_contacts = []
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                        if i < j:
                            new_contacts.append((i, j))
                        else:
                            new_contacts.append((j, i))
            # Score: number of target contacts satisfied + free space + random
            target_hits = sum(1 for c in new_contacts if c in target_set)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = 10.0 * target_hits + 0.5 * len(new_contacts) + 0.1 * free + rng.random()
            candidates.append((score, site, new_contacts))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[0])
        # Pick from top few with some randomness
        k = min(3, len(candidates))
        chosen = candidates[rng.randrange(k)]
        best_site = chosen[1]
        new_contacts = chosen[2]

        occupied[best_site] = i
        chain.append(best_site)
        for c in new_contacts:
            satisfied.add(c)

    return chain


def _local_search(sequence, chain, rng, time_limit):
    """Local search with hill climbing and simulated annealing on single moves and 2-flips."""
    n = len(chain)
    if n < 3:
        return chain

    where = {p: i for i, p in enumerate(chain)}
    contacts = _compute_contacts(sequence, chain)
    best_chain = list(chain)
    best_energy = _energy_from_contacts(contacts)

    def try_move(chain, where, contacts, energy, move_type, i, new_pos):
        """Attempt to move residue i to new_pos. Returns (new_chain, new_where, new_contacts, new_energy) or None."""
        if move_type == "single":
            old_pos = chain[i]
            if new_pos == old_pos:
                return None
            if new_pos in where:
                j = where[new_pos]
                if j == i:
                    return None
                if abs(j - i) <= 1:
                    return None  # would be adjacent
            # Check new_pos doesn't conflict with neighbours of i
            for nb in (i - 1, i + 1):
                if 0 <= nb < n and chain[nb] == new_pos:
                    return None

            # Remove contacts involving i
            old_contacts = set()
            for c in contacts:
                pair = tuple(c)
                if i in pair:
                    old_contacts.add(c)
            new_contacts = contacts - old_contacts

            # Add new contacts involving i at new_pos
            if sequence[i] == "H":
                for q in _neighbours(new_pos):
                    j = where.get(q)
                    if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                        key = frozenset((i, j))
                        new_contacts.add(key)

            new_energy = _energy_from_contacts(new_contacts)
            new_chain = list(chain)
            new_chain[i] = new_pos
            new_where = dict(where)
            del new_where[old_pos]
            new_where[new_pos] = i
            return (new_chain, new_where, new_contacts, new_energy)
        elif move_type == "flip2":
            # Move residues i and i+1 to new positions (swap or shift)
            if i + 1 >= n:
                return None
            old_i = chain[i]
            old_j = chain[i + 1]
            # new_pos is the new position for i; i+1 goes to a neighbour of new_pos
            # Check feasibility
            if new_pos in where:
                j_idx = where[new_pos]
                if j_idx not in (i, i + 1):
                    return None
            # Check i's neighbour constraints
            if i > 0 and chain[i - 1] == new_pos:
                return None
            # Find position for i+1: must be neighbour of new_pos, not occupied (except by i), not adjacent to i+2
            candidates_for_j = []
            for q in _neighbours(new_pos):
                if q == old_j and old_j in where and where[old_j] == i + 1:
                    pass  # ok, it's its own old pos
                if q in where:
                    j_idx = where[q]
                    if j_idx not in (i, i + 1):
                        continue
                # Check not adjacent to i+2
                if i + 2 < n and q == chain[i + 2]:
                    continue
                # Check not adjacent to i-1 (already checked for new_pos)
                if i > 0 and q == chain[i - 1]:
                    continue
                candidates_for_j.append(q)
            if not candidates_for_j:
                return None
            new_j = rng.choice(candidates_for_j)
            if new_j == new_pos:
                return None

            # Remove all contacts involving i or i+1
            old_contacts = set()
            for c in contacts:
                pair = tuple(c)
                if i in pair or (i + 1) in pair:
                    old_contacts.add(c)
            new_contacts = contacts - old_contacts

            # Add new contacts
            for idx, pos in ((i, new_pos), (i + 1, new_j)):
                if sequence[idx] == "H":
                    for q in _neighbours(pos):
                        j = where.get(q)
                        if j is not None and abs(idx - j) > 1 and sequence[j] == "H":
                            key = frozenset((idx, j))
                            new_contacts.add(key)

            new_energy = _energy_from_contacts(new_contacts)
            new_chain = list(chain)
            new_chain[i] = new_pos
            new_chain[i + 1] = new_j
            new_where = dict(where)
            del new_where[old_i]
            del new_where[old_j]
            new_where[new_pos] = i
            new_where[new_j] = i + 1
            return (new_chain, new_where, new_contacts, new_energy)
        return None

    t = 1.0
    t_min = 0.01
    cooling = 0.995
    max_iter = 2000

    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if t < t_min:
            break

        # Choose move type
        if rng.random() < 0.7:
            move_type = "single"
            i = rng.randrange(n)
            old_pos = chain[i]
            new_pos = rng.choice(list(_neighbours(old_pos)))
        else:
            move_type = "flip2"
            i = rng.randrange(n - 1)
            old_pos = chain[i]
            new_pos = rng.choice(list(_neighbours(old_pos)))

        result = try_move(chain, where, contacts, _energy_from_contacts(contacts), move_type, i, new_pos)
        if result is None:
            continue
        new_chain, new_where, new_contacts, new_energy = result
        delta = new_energy - best_energy  # negative is better

        if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-9)):
            chain = new_chain
            where = new_where
            contacts = new_contacts
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
        t *= cooling

    return best_chain


def fold(sequence, dimension):
    """Best conformation via contact-guided growth + local search."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    if dimension != 2:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]

    feasible_pairs = _build_feasible_pairs(sequence)
    rng = random.Random(42 + len(sequence))

    best_chain = None
    best_energy = 1

    # Phase 1: Multiple guided growths
    n_growth = 50
    for g in range(n_growth):
        if _remaining() < 2.0:
            break
        # Vary target count
        target_count = rng.choice([0, 5, 10, 15, 20, len(feasible_pairs)])
        chain = _grow_guided(sequence, feasible_pairs, rng, max_contacts_to_target=target_count)
        if chain is None:
            continue
        energy = _energy_from_contacts(_compute_contacts(sequence, chain))
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)

    if best_chain is None:
        # Fallback: simple spiral
        best_chain = _simple_spiral(n)
        best_energy = _energy_from_contacts(_compute_contacts(sequence, best_chain))

    # Phase 2: Local search on best chain
    if _remaining() > 3.0:
        time_limit = time.time() + max(1.0, _remaining() - 2.0)
        improved = _local_search(sequence, best_chain, rng, time_limit)
        energy = _energy_from_contacts(_compute_contacts(sequence, improved))
        if energy < best_energy:
            best_energy = energy
            best_chain = improved

    # Phase 3: If time remains, do a few more growths + local search
    if _remaining() > 5.0 and best_chain is not None:
        for g in range(20):
            if _remaining() < 3.0:
                break
            chain = _grow_guided(sequence, feasible_pairs, rng, max_contacts_to_target=rng.choice([10, 15, 20]))
            if chain is None:
                continue
            if _remaining() > 3.0:
                time_limit = time.time() + max(0.5, _remaining() - 2.0)
                chain = _local_search(sequence, chain, rng, time_limit)
            energy = _energy_from_contacts(_compute_contacts(sequence, chain))
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)

    if best_chain is None:
        best_chain = _simple_spiral(n)

    return [list(p) for p in best_chain]


def _simple_spiral(n):
    """Generate a simple spiral conformation."""
    chain = [(0, 0)]
    occupied = {(0, 0)}
    # Spiral: right, up, left, down, right, up, ...
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    di = 0
    step = 1
    x, y = 0, 0
    for _ in range(n - 1):
        for _ in range(2):
            dx, dy = directions[di % 4]
            for _ in range(step):
                x += dx
                y += dy
                if len(chain) < n:
                    chain.append((x, y))
                    occupied.add((x, y))
                    if len(chain) == n:
                        break
            if len(chain) == n:
                break
            di += 1
        if len(chain) == n:
            break
        step += 1
    # Fallback if spiral failed
    while len(chain) < n:
        # Just extend in a line
        last = chain[-1]
        for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            np = (last[0] + d[0], last[1] + d[1])
            if np not in occupied:
                chain.append(np)
                occupied.add(np)
                break
        else:
            # Shouldn't happen, but add offset
            chain.append((last[0] + len(chain), last[1]))
    return chain


def _energy(sequence, chain):
    """Compute energy for validation."""
    return _energy_from_contacts(_compute_contacts(sequence, chain))
# EVOLVE-BLOCK-END

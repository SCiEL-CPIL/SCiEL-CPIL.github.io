"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

# Evaluator kills after 210s wall-clock; keep a safety margin.
_DEADLINE = time.time() + 175.0


def _neighbours(point):
    """Return the 4 unit-step neighbours of a 2D point (tuple)."""
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


class _EnergyState:
    """Maintains a position->index map and the contact count so that
    regrow moves can be scored incrementally without a full O(n) recount."""

    def __init__(self, sequence, chain):
        self.seq = sequence
        self.n = len(sequence)
        self.pos = [None] * self.n
        self.occ = {}
        for i, p in enumerate(chain):
            self.pos[i] = p
            self.occ[p] = i
        self.energy = _compute_energy(sequence, chain)

    def _contacts_at(self, i):
        """Count H-H contacts involving residue i (each contact counted once,
        for i being the lower index)."""
        if self.seq[i] != 'H':
            return 0
        c = 0
        for q in _neighbours(self.pos[i]):
            j = self.occ.get(q)
            if j is not None and j > i and self.seq[j] == 'H':
                c += 1
        return c

    def _remove(self, i):
        p = self.pos[i]
        c = self._contacts_at(i)
        for q in _neighbours(p):
            j = self.occ.get(q)
            if j is not None and j < i and self.seq[j] == 'H':
                c += 1
        self.occ.pop(p, None)
        self.pos[i] = None
        self.energy += c  # removing a contact increases (less negative) energy

    def _add(self, i):
        p = self.pos[i]
        c = 0
        if self.seq[i] == 'H':
            for q in _neighbours(p):
                j = self.occ.get(q)
                if j is not None and j > i and self.seq[j] == 'H':
                    c += 1
        self.occ[p] = i
        self.pos[i] = p
        self.energy -= c  # adding a contact decreases (more negative) energy

    def regrow(self, split, suffix):
        """Cut the chain at `split` and replace the suffix with the given
        positions. `suffix[k]` becomes residue (split + k). Returns new energy."""
        n = self.n
        # Remove old suffix residues (highest index first so contact counting
        # stays consistent as the map shrinks).
        for i in range(n - 1, split - 1, -1):
            self._remove(i)
        # Insert new suffix residues in index order.
        for k, p in enumerate(suffix):
            self._add(split + k)
        return self.energy


def _greedy_grow(sequence, dimension, rng, temperature=0.0):
    """Randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then sites with more free space."""
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


def _greedy_grow_from_fixed(suffix_seq, start_pos, prefix, full_sequence, rng, temperature=0.0):
    """Greedy growth for a suffix, with proper access to the full sequence for
    H checks. Returns the list of new positions (excluding the anchor)."""
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
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < global_idx - 1 and full_sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = global_idx
        chain.append(best_site)

    return chain[1:]


def _regrow_sa(sequence, chain, rng, max_iters=5000, start_temp=3.0, cool_rate=0.997):
    """Simulated annealing using regrow moves: pick a split point, regrow the
    suffix, accept with a Metropolis criterion. Energy is tracked incrementally."""
    n = len(sequence)
    if n < 4:
        return chain, _compute_energy(sequence, chain)

    st = _EnergyState(sequence, chain)
    best_chain = list(chain)
    best_energy = st.energy
    temp = start_temp
    no_improve = 0

    min_split = 3
    max_split = max(min_split, n - 2)
    if max_split <= min_split:
        return best_chain, best_energy

    for it in range(max_iters):
        if time.time() > _DEADLINE - 12:
            break
        if temp < 0.05:
            break

        split = rng.randint(min_split, max_split)
        prefix = st.pos[:split]
        suffix_seq = sequence[split:]

        new_suffix = _greedy_grow_from_fixed(
            suffix_seq, prefix[-1], prefix, sequence, rng, temperature=temp)
        if new_suffix is None:
            temp *= cool_rate
            no_improve += 1
            continue

        new_energy = st.regrow(split, new_suffix)
        delta = new_energy - st.energy

        if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(st.pos)
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Revert: restore the old suffix.
            old_suffix = st.pos[split:]
            st.regrow(split, old_suffix)
            no_improve += 1

        temp *= cool_rate

        if no_improve > 400:
            break

    return best_chain, best_energy


def _local_improve(sequence, chain, rng, max_moves=20000):
    """Improve via single-residue kink moves that preserve connectivity."""
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve_count = 0

    for _ in range(max_moves):
        if time.time() > _DEADLINE - 6:
            break

        i = rng.randint(1, n - 2)
        p = chain[i]
        prev, nxt = chain[i - 1], chain[i + 1]

        candidates = []
        for np in _neighbours(prev):
            if np == p:
                continue
            if abs(np[0] - nxt[0]) + abs(np[1] - nxt[1]) != 1:
                continue
            if np in occupied and occupied[np] != i:
                continue
            candidates.append(np)

        if not candidates:
            no_improve_count += 1
            if no_improve_count > 300:
                break
            continue

        np = rng.choice(candidates)
        del occupied[p]
        chain[i] = np
        occupied[np] = i
        new_energy = _compute_energy(sequence, chain)

        if new_energy < current_energy:
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
            no_improve_count = 0
        else:
            del occupied[np]
            chain[i] = p
            occupied[p] = i
            no_improve_count += 1

        if no_improve_count > 300:
            break

    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold using greedy restarts + regrow simulated annealing + local polish."""
    rng = random.Random(42)
    best_chain = None
    best_energy = 0
    n = len(sequence)

    # Phase 1: Greedy restarts to get a good starting conformation.
    for attempt in range(2000):
        if time.time() > _DEADLINE - 25:
            break
        temp = rng.uniform(0.0, 3.0)
        chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue
        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Regrow simulated annealing (the main refinement machinery).
    if best_chain is not None:
        improved_chain, improved_energy = _regrow_sa(
            sequence, best_chain, rng, max_iters=5000)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

    # Phase 3: Local kink polish.
    if best_chain is not None:
        improved_chain, improved_energy = _local_improve(
            sequence, best_chain, rng, max_moves=20000)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

    # Fallback (should not happen for valid sequences).
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

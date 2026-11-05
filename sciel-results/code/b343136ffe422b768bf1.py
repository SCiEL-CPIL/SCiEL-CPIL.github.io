"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random

_GLOBAL_START = time.time()
_DEADLINE = _GLOBAL_START + 320.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _energy(sequence, chain):
    """Compute the HP energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    is_h = [c == "H" for c in sequence]
    for i, p in enumerate(chain):
        if is_h[i]:
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    contacts += 1
    return -contacts


def _beam_grow(sequence, dimension, rng, beam_width=8, diversity_window=3):
    """
    Beam search growth: keep top-K partial chains by contact score + diversity.
    Returns the best full chain or None if stuck.
    """
    n = len(sequence)
    if n < 4:
        return _greedy_grow(sequence, dimension, rng)

    is_h = [c == "H" for c in sequence]
    start = (0, 0, 0)

    # Each beam: [score, chain_list, occupied_dict]
    beams = [[0.0, [start], {start: 0}]]

    for i in range(1, n):
        if _time_left() < 3.0:
            break
        new_beams = []
        seen_keys = set()

        for score, chain, occupied in beams:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Diversity key: last few positions
                tail = chain[-diversity_window:] if len(chain) >= diversity_window else chain
                key = tuple(tail + [site])
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                # Contact gain
                gain = 0
                if is_h[i]:
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and is_h[j]:
                            gain += 1

                # Free space heuristic
                free = sum(1 for q in _neighbours(site) if q not in occupied)

                new_score = score + gain * 3.0 + 0.03 * free + rng.random() * 0.3

                new_chain = chain + [site]
                new_occ = dict(occupied)
                new_occ[site] = i
                new_beams.append([new_score, new_chain, new_occ])

        if not new_beams:
            break

        # Keep top beam_width by score
        new_beams.sort(key=lambda x: -x[0])
        beams = new_beams[:beam_width]

    if not beams:
        return None

    best = max(beams, key=lambda x: x[0])
    return best[1]


def _greedy_grow(sequence, dimension, rng):
    """Simple randomized greedy growth as fallback."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    is_h = [c == "H" for c in sequence]

    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if is_h[i]:
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and is_h[j]:
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 3.0 + 0.05 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
    return chain


class _FoldingState:
    """Incremental local search state with optimized contact tracking."""

    def __init__(self, sequence, chain):
        self.seq = sequence
        self.n = len(chain)
        self.chain = list(chain)
        self.occupied = {p: i for i, p in enumerate(chain)}
        self.is_h = [c == "H" for c in sequence]
        self.contacts = set()
        self.contact_map = {i: set() for i in range(self.n)}
        self._compute_contacts()
        self.energy = -len(self.contacts)
        self.best_chain = list(chain)
        self.best_energy = self.energy
        self.best_contacts = set(self.contacts)
        self.stagnation = 0

    def _compute_contacts(self):
        self.contacts = set()
        self.contact_map = {i: set() for i in range(self.n)}
        where = {p: i for i, p in enumerate(self.chain)}
        for i, p in enumerate(self.chain):
            if not self.is_h[i]:
                continue
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and self.is_h[j]:
                    pair = (i, j)
                    self.contacts.add(pair)
                    self.contact_map[i].add(j)
                    self.contact_map[j].add(i)

    def _accept_if_better(self):
        if self.energy < self.best_energy:
            self.best_energy = self.energy
            self.best_chain = list(self.chain)
            self.best_contacts = set(self.contacts)
            self.stagnation = 0
        else:
            self.stagnation += 1

    def _remove_contacts_involving(self, indices):
        indices_set = set(indices)
        for i in indices_set:
            partners = list(self.contact_map[i])
            for j in partners:
                pair = (i, j) if i < j else (j, i)
                if pair in self.contacts:
                    self.contacts.discard(pair)
                    self.energy += 1
                    self.contact_map[i].discard(j)
                    self.contact_map[j].discard(i)

    def _add_new_contacts(self, indices, new_positions):
        indices_set = set(indices)
        for i in indices_set:
            if not self.is_h[i]:
                continue
            pos = new_positions[i]
            for q in _neighbours(pos):
                j = self.occupied.get(q)
                if j is None or j == i:
                    continue
                if abs(j - i) == 1:
                    continue
                if self.is_h[j]:
                    pair = (i, j) if i < j else (j, i)
                    if pair not in self.contacts:
                        self.contacts.add(pair)
                        self.energy -= 1
                        self.contact_map[i].add(j)
                        self.contact_map[j].add(i)

    def _validate_and_commit(self, moved_indices, new_positions):
        """Check self-avoidance and unit steps for moved segment, commit if valid."""
        n = self.n
        # Check self-avoidance: no two positions in new_positions collide
        pos_set = set()
        for i in moved_indices:
            pos = new_positions[i]
            if pos in pos_set:
                return False
            pos_set.add(pos)
        # Check collision with unmoved positions
        unmoved_positions = set()
        for i, p in enumerate(self.chain):
            if i not in moved_indices:
                unmoved_positions.add(p)
        for pos in pos_set:
            if pos in unmoved_positions:
                return False
        # Check unit steps at boundaries
        # For each consecutive pair where one is moved and one isn't, or both moved
        # We need to verify the full chain is valid
        # Build full new chain
        new_chain = list(self.chain)
        for i in moved_indices:
            new_chain[i] = new_positions[i]
        # Check unit steps
        for k in range(1, n):
            diff = abs(new_chain[k][0] - new_chain[k - 1][0]) + \
                   abs(new_chain[k][1] - new_chain[k - 1][1]) + \
                   abs(new_chain[k][2] - new_chain[k - 1][2])
            if diff != 1:
                return False
        # Commit
        self._remove_contacts_involving(moved_indices)
        for i in moved_indices:
            self.chain[i] = new_positions[i]
            self.occupied[new_positions[i]] = i
        self._add_new_contacts(moved_indices, new_positions)
        self._accept_if_better()
        return True

    def _pivot_move(self, rng):
        """
        Pivot move: select a pivot index, reverse the subchain from pivot to end
        (or a segment), and try to re-embed it.
        For simplicity, we do a segment reversal.
        """
        n = self.n
        if n < 5:
            return False
        # Choose a segment to reverse
        i = rng.randint(0, n - 4)
        j = rng.randint(i + 2, n - 1)
        # Reverse segment i..j
        moved = list(range(i, j + 1))
        old_positions = {k: self.chain[k] for k in moved}
        # New positions: reverse the order
        new_positions = {}
        for idx, k in enumerate(moved):
            new_positions[k] = old_positions[moved[-1 - idx]]
        # Remove old occupied entries for moved indices
        for k in moved:
            del self.occupied[self.chain[k]]
        # Try to commit
        if self._validate_and_commit(moved, new_positions):
            return True
        # Rollback
        for k in moved:
            self.chain[k] = old_positions[k]
            self.occupied[old_positions[k]] = k
        return False

    def _shift_move(self, rng):
        """
        Shift move: move a short segment (2-3 residues) to an adjacent empty position.
        """
        n = self.n
        if n < 5:
            return False
        seg_len = rng.choice([2, 3])
        i = rng.randint(0, n - seg_len - 1)
        moved = list(range(i, i + seg_len))
        old_positions = {k: self.chain[k] for k in moved}
        # Try shifting in each of 6 directions
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            new_positions = {}
            for k in moved:
                px, py, pz = old_positions[k]
                new_positions[k] = (px + dx, py + dy, pz + dz)
            # Remove old occupied
            for k in moved:
                del self.occupied[self.chain[k]]
            if self._validate_and_commit(moved, new_positions):
                return True
            # Rollback
            for k in moved:
                self.chain[k] = old_positions[k]
                self.occupied[old_positions[k]] = k
        return False

    def local_search(self, rng, max_iters=200):
        """Run local search with pivot and shift moves."""
        for _ in range(max_iters):
            if _time_left() < 5.0:
                break
            if self.stagnation > 50:
                break
            if rng.random() < 0.5:
                self._pivot_move(rng)
            else:
                self._shift_move(rng)
        return self.best_chain, self.best_energy


def fold(sequence, dimension):
    """
    Fold the sequence into a 3D lattice conformation.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    if n <= 3:
        return [[i, 0, 0] for i in range(n)]

    best_chain = None
    best_energy = 0

    rng = random.Random(42)

    # Number of restarts depends on chain length
    if n <= 50:
        max_restarts = 300
    elif n <= 100:
        max_restarts = 150
    else:
        max_restarts = 80

    for restart in range(max_restarts):
        if _time_left() < 10.0:
            break

        # Vary beam width
        if restart % 3 == 0:
            beam_width = 6
        elif restart % 3 == 1:
            beam_width = 10
        else:
            beam_width = 14

        chain = _beam_grow(sequence, dimension, rng, beam_width=beam_width)
        if chain is None:
            # Fallback to greedy
            chain = _greedy_grow(sequence, dimension, rng)
            if chain is None:
                continue

        # Local search
        state = _FoldingState(sequence, chain)
        if _time_left() > 15.0:
            max_ls_iters = 100 if n <= 50 else (50 if n <= 100 else 30)
            state.local_search(rng, max_iters=max_ls_iters)

        if state.best_energy < best_energy:
            best_energy = state.best_energy
            best_chain = state.best_chain

    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

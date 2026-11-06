"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 320  # safety margin before 330s kill

_AX = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _neighbours(point):
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


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


def _greedy_grow(sequence, dimension, rng, time_limit=None):
    """One randomized greedy chain growth with contact + free-space scoring."""
    start = (0, 0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        if time_limit is not None and time.time() > time_limit:
            return None
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


class _State:
    __slots__ = ("seq", "chain", "where", "contacts", "n", "H")

    def __init__(self, sequence, chain):
        self.seq = sequence
        self.chain = list(chain)
        self.n = len(chain)
        self.H = [c == "H" for c in sequence]
        self.where = {p: i for i, p in enumerate(self.chain)}
        self.contacts = 0
        for i, p in enumerate(self.chain):
            if self.H[i]:
                for q in _neighbours(p):
                    j = self.where.get(q)
                    if j is not None and j > i + 1 and self.H[j]:
                        self.contacts += 1

    def energy(self):
        return -self.contacts

    def _add_contact(self, i, p):
        """count new H-H contacts involving residue i at position p (i not adjacent)."""
        if not self.H[i]:
            return 0
        c = 0
        for q in _neighbours(p):
            j = self.where.get(q)
            if j is not None and abs(j - i) > 1 and self.H[j]:
                c += 1
        return c

    def _remove_contact(self, i, p):
        if not self.H[i]:
            return 0
        c = 0
        for q in _neighbours(p):
            j = self.where.get(q)
            if j is not None and abs(j - i) > 1 and self.H[j]:
                c += 1
        return c

    def try_move(self, j, new_pos):
        """Attempt to move residue j to new_pos (a neighbor of chain[i]).
        Returns True if accepted (state updated), False otherwise.
        Caller must ensure new_pos is free and unit-step validity holds."""
        old_pos = self.chain[j]
        # check unit-step validity after move
        if j > 0:
            d = sum(abs(a - b) for a, b in zip(new_pos, self.chain[j - 1]))
            if d != 1:
                return False
        if j < self.n - 1:
            d = sum(abs(a - b) for a, b in zip(new_pos, self.chain[j + 1]))
            if d != 1:
                return False
        # check self-avoidance: new_pos must not be occupied by other residue
        occ = self.where.get(new_pos)
        if occ is not None and occ != j:
            return False
        # check old_pos is not occupied by neighbor (would break) - it's j itself, fine
        # apply
        delta = self._add_contact(j, new_pos) - self._remove_contact(j, old_pos)
        self.chain[j] = new_pos
        del self.where[old_pos]
        self.where[new_pos] = j
        self.contacts += delta
        return True

    def undo_move(self, j, old_pos, delta):
        new_pos = self.chain[j]
        self.chain[j] = old_pos
        del self.where[new_pos]
        self.where[old_pos] = j
        self.contacts -= delta


def _local_refine(sequence, chain, max_iters=2000, rng=None, deadline=None):
    """Local search with kink moves and simulated annealing."""
    if len(chain) < 3:
        return chain
    if rng is None:
        rng = random.Random()

    st = _State(sequence, chain)
    best_energy = st.energy()
    best_chain = list(st.chain)
    n = st.n

    # Simulated annealing temperature schedule
    T = 3.0
    cool = 0.9995

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        improved = False
        # Try several random kink moves per iteration
        for _ in range(8):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if abs(i - j) < 3:
                continue

            old_pos = st.chain[j]
            # pick a random neighbor of chain[i]
            cand = list(_neighbours(st.chain[i]))
            rng.shuffle(cand)
            for new_pos in cand:
                if new_pos == old_pos:
                    continue
                # must be free
                if new_pos in st.where:
                    occ = st.where[new_pos]
                    if occ != j:
                        continue
                # attempt move
                if st.try_move(j, new_pos):
                    e = st.energy()
                    delta = e - st.energy() if False else 0
                    # compute delta properly: we already updated, so compare
                    if e < best_energy:
                        best_energy = e
                        best_chain = list(st.chain)
                        improved = True
                        break
                    # simulated annealing acceptance
                    # current energy is e, previous was best? no, previous was st.energy() before move
                    # we need previous energy; recompute: previous = -st.contacts before. Let's just accept by T
                    # Actually we lost previous energy. Accept with prob based on e vs current best? 
                    # Simpler: accept if e <= best_energy OR random < exp(-(e - best_energy)/T)
                    if e > best_energy and rng.random() > (1.0 / (1.0 + (1.0 / T))):
                        # revert
                        st.undo_move(j, old_pos, 0)
                    # keep the move (either improving or accepted)
                    break
            if improved:
                break

        T *= cool
        if not improved and it > 0 and it % 200 == 0:
            # occasionally reheat to escape
            T = max(T, 1.0)

    return best_chain


def fold(sequence, dimension):
    """Best conformation using greedy growth + local refinement."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0

    seq_len = len(sequence)

    if seq_len <= 48:
        n_restarts = 800
    elif seq_len <= 64:
        n_restarts = 400
    elif seq_len <= 96:
        n_restarts = 200
    else:
        n_restarts = 100

    seq_deadline = _DEADLINE - 5

    for r in range(n_restarts):
        if time.time() > seq_deadline:
            break

        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue

        refined = _local_refine(sequence, chain,
                                max_iters=min(3000, 60 * seq_len),
                                rng=rng, deadline=seq_deadline)
        value = _energy(sequence, refined)

        if value < best_energy:
            best = refined
            best_energy = value

    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(seq_len)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END

"""HP lattice protein folding: multi-start SA with tail-regeneration moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = 315.0
_START_TIME = None

def _get_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME + _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _lattice_adjacent(p1, p2):
    return sum(abs(a - b) for a, b in zip(p1, p2)) == 1

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

class _State:
    """Incremental state with contact tracking for fast energy updates."""
    def __init__(self, sequence, chain):
        self.seq = sequence
        self.n = len(chain)
        self.chain = list(chain)
        self.occupied = {p: i for i, p in enumerate(chain)}
        self.is_h = [c == 'H' for c in sequence]
        self.contacts = set()
        self.contact_map = {i: set() for i in range(self.n)}
        self._recompute_contacts()
        self.energy = -len(self.contacts)
        self.best_chain = list(chain)
        self.best_energy = self.energy

    def _recompute_contacts(self):
        self.contacts.clear()
        for i in range(self.n):
            self.contact_map[i].clear()
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

    def _remove_contacts_involving(self, indices):
        indices_set = set(indices)
        removed = 0
        for i in indices_set:
            partners = list(self.contact_map[i])
            for j in partners:
                if j in indices_set:
                    if i < j:
                        pair = (i, j)
                        if pair in self.contacts:
                            self.contacts.discard(pair)
                            self.contact_map[i].discard(j)
                            self.contact_map[j].discard(i)
                            removed += 1
                else:
                    pair = (i, j) if i < j else (j, i)
                    if pair in self.contacts:
                        self.contacts.discard(pair)
                        self.contact_map[i].discard(j)
                        self.contact_map[j].discard(i)
                        removed += 1
        self.energy += removed

    def _add_new_contacts(self, indices, new_positions):
        indices_set = set(indices)
        added = 0
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
                        self.contact_map[i].add(j)
                        self.contact_map[j].add(i)
                        added += 1
        self.energy -= added

    def _update_for_move(self, moved_indices, old_positions, new_positions):
        self._remove_contacts_involving(moved_indices)
        self._add_new_contacts(moved_indices, new_positions)

    def _pivot_move(self, rng):
        """Rotate the tail from pivot_idx+1 onward by a random non-trivial rotation."""
        n = self.n
        if n < 5:
            return None
        pivot_idx = rng.randint(1, n - 4)
        orig_dir = tuple(self.chain[pivot_idx + 1][k] - self.chain[pivot_idx][k] for k in range(3))

        # Generate 3 orthogonal directions excluding orig_dir and -orig_dir
        dirs = []
        for ax in range(3):
            for s in (-1, 1):
                d = [0, 0, 0]
                d[ax] = s
                d = tuple(d)
                if d != orig_dir and d != tuple(-x for x in orig_dir):
                    dirs.append(d)
        if not dirs:
            return None
        new_dir = dirs[rng.randint(0, len(dirs) - 1)]

        # Build rotation: find an orthonormal basis mapping orig_dir -> new_dir
        # Simple approach: just translate the tail with the new first step, keeping relative steps
        # This is a "translation pivot" - simpler and more effective
        subchain_len = n - pivot_idx - 1
        subchain_old = list(self.chain[pivot_idx + 1:])
        new_subchain = [None] * subchain_len
        new_subchain[0] = tuple(self.chain[pivot_idx][k] + new_dir[k] for k in range(3))
        if new_subchain[0] in self.occupied:
            return None

        valid = True
        for k in range(1, subchain_len):
            old_step = tuple(subchain_old[k][m] - subchain_old[k - 1][m] for m in range(3))
            new_pos = tuple(new_subchain[k - 1][m] + old_step[m] for m in range(3))
            if new_pos in self.occupied:
                valid = False
                break
            new_subchain[k] = new_pos
        if not valid:
            return None

        seen = set()
        for pos in new_subchain:
            if pos in seen:
                return None
            seen.add(pos)

        moved_indices = list(range(pivot_idx + 1, n))
        old_positions = {i: self.chain[i] for i in moved_indices}
        for pos in subchain_old:
            del self.occupied[pos]
        for k, pos in enumerate(new_subchain):
            self.chain[pivot_idx + 1 + k] = pos
            self.occupied[pos] = pivot_idx + 1 + k
        new_positions = {i: self.chain[i] for i in moved_indices}
        self._update_for_move(moved_indices, old_positions, new_positions)
        return {'type': 'pivot', 'pivot_idx': pivot_idx, 'old_positions': old_positions,
                'new_positions': new_positions, 'moved_indices': moved_indices}

    def _crank_move(self, rng):
        """Translate a subchain by one unit along a random axis."""
        n = self.n
        if n < 6:
            return None
        start_idx = rng.randint(1, n - 4)
        end_idx = rng.randint(start_idx + 1, n - 2)
        subchain = list(self.chain[start_idx:end_idx + 1])

        ax = rng.randint(0, 2)
        s = rng.choice((-1, 1))
        new_subchain = []
        for p in subchain:
            np = list(p)
            np[ax] += s
            new_subchain.append(tuple(np))

        seen = set()
        for np in new_subchain:
            if np in seen:
                return None
            seen.add(np)

        # Check occupancy
        for np in new_subchain:
            if np in self.occupied:
                return None

        # Check adjacency at boundaries
        if start_idx > 0:
            if not _lattice_adjacent(self.chain[start_idx - 1], new_subchain[0]):
                return None
        if end_idx < n - 1:
            if not _lattice_adjacent(new_subchain[-1], self.chain[end_idx + 1]):
                return None

        moved_indices = list(range(start_idx, end_idx + 1))
        old_positions = {i: self.chain[i] for i in moved_indices}
        for p in subchain:
            del self.occupied[p]
        for i, np in enumerate(new_subchain):
            self.chain[start_idx + i] = np
            self.occupied[np] = start_idx + i
        new_positions = {i: self.chain[i] for i in moved_indices}
        self._update_for_move(moved_indices, old_positions, new_positions)
        return {'type': 'crank', 'start_idx': start_idx, 'end_idx': end_idx,
                'old_positions': old_positions, 'new_positions': new_positions, 'moved_indices': moved_indices}

    def _tail_regrow(self, rng):
        """
        Quantum-tunneling-like move: detach the last L residues and regrow them
        from the attachment point, exploring a new conformation for the tail.
        This allows jumping between disconnected basins.
        """
        n = self.n
        if n < 8:
            return None
        # Choose how many tail residues to regrow
        max_tail = min(n - 4, rng.randint(4, min(12, n - 4)))
        tail_len = rng.randint(3, max_tail)
        attach_idx = n - tail_len  # index of last residue in the fixed prefix
        prefix_end = self.chain[attach_idx]

        # Remove tail from occupied
        tail_indices = list(range(attach_idx + 1, n))
        old_tail_positions = {i: self.chain[i] for i in tail_indices}
        for i in tail_indices:
            del self.occupied[self.chain[i]]

        # Regrow tail greedily with some randomness
        new_tail = [None] * tail_len
        last_pos = prefix_end
        for k in range(tail_len):
            idx = attach_idx + 1 + k
            options = []
            for site in _neighbours(last_pos):
                if site in self.occupied:
                    continue
                # Check it's not already in the new tail
                if site in new_tail[:k]:
                    continue
                # Score: prefer H-H contacts
                gain = 0
                if self.is_h[idx]:
                    for q in _neighbours(site):
                        j = self.occupied.get(q)
                        if j is not None and j <= attach_idx and self.is_h[j] and abs(j - idx) > 1:
                            gain += 1
                        # Also check against already-placed tail residues
                        for kk in range(k):
                            if new_tail[kk] == q and self.is_h[attach_idx + 1 + kk]:
                                gain += 1
                free = sum(1 for q in _neighbours(site) if q not in self.occupied and q not in new_tail[:k])
                score = gain * 2.0 + free * 0.3 + rng.random() * 1.0
                options.append((score, site))
            if not options:
                # Backtrack: restore old positions
                for i in tail_indices:
                    self.chain[i] = old_tail_positions[i]
                    self.occupied[old_tail_positions[i]] = i
                return None
            best_site = max(options, key=lambda x: x[0])[1]
            new_tail[k] = best_site
            self.occupied[best_site] = idx
            last_pos = best_site

        # Update chain
        for k in range(tail_len):
            self.chain[attach_idx + 1 + k] = new_tail[k]
        new_positions = {i: self.chain[i] for i in tail_indices}
        self._update_for_move(tail_indices, old_tail_positions, new_positions)
        return {'type': 'tail_regrow', 'attach_idx': attach_idx,
                'old_positions': old_tail_positions, 'new_positions': new_positions,
                'moved_indices': tail_indices}

    def _undo(self, move_info):
        if move_info is None:
            return
        moved_indices = move_info['moved_indices']
        old_positions = move_info['old_positions']
        new_positions = move_info['new_positions']
        self._remove_contacts_involving(moved_indices)
        for i in moved_indices:
            del self.occupied[self.chain[i]]
            self.chain[i] = old_positions[i]
            self.occupied[old_positions[i]] = i
        self._add_new_contacts(moved_indices, old_positions)

def _grow(sequence, dimension, rng):
    """Greedy chain growth with contact preference."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    is_h = [c == 'H' for c in sequence]

    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if is_h[i]:
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and is_h[idx]:
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 2.0 + free * 0.2 + rng.random() * 0.8
            options.append((score, site))
        if not options:
            return None
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _sim_anneal(sequence, chain, rng, max_iters):
    """Simulated annealing with pivot, crank, and tail-regrow moves."""
    if chain is None:
        return None
    n = len(chain)
    if n < 5:
        return chain

    state = _State(sequence, chain)
    T_start = 6.0
    T_end = 0.02
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    T = T_start

    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        T *= cooling
        old_e = state.energy

        r = rng.random()
        if r < 0.30:
            move = state._pivot_move(rng)
        elif r < 0.55:
            move = state._crank_move(rng)
        else:
            move = state._tail_regrow(rng)

        if move is None:
            continue

        delta = state.energy - old_e
        if delta < 0 or (T > 0.005 and rng.random() < math.exp(-delta / T)):
            state._accept_if_better()
        else:
            state._undo(move)

    return state.best_chain

def fold(sequence, dimension):
    """Fold a single HP sequence on the cubic lattice."""
    n = len(sequence)
    rng = random.Random(42 + n * 7)
    best_chain = None
    best_energy = 0

    # Adaptive restart budget
    if n <= 50:
        time_per_restart = 0.12
        sa_iters = 600
    elif n <= 80:
        time_per_restart = 0.25
        sa_iters = 350
    elif n <= 120:
        time_per_restart = 0.5
        sa_iters = 200
    else:
        time_per_restart = 1.0
        sa_iters = 100

    max_restarts = max(20, min(int(_time_left() / time_per_restart), 5000))

    for r in range(max_restarts):
        if _time_left() < 2.0:
            break

        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue

        # Refine with simulated annealing
        chain = _sim_anneal(sequence, chain, rng, sa_iters)
        if chain is None:
            continue

        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain

    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

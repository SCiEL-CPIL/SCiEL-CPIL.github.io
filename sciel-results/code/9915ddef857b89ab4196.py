import random
import time
import math
import numpy as np

_DEADLINE_MARGIN = 25.0

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

class _FoldingState:
    """Maintains chain conformation with incremental energy tracking."""
    
    def __init__(self, sequence, chain):
        self.sequence = sequence
        self.n = len(sequence)
        self.chain = chain[:]
        self.occupied = {p: i for i, p in enumerate(chain)}
        self.energy = self._compute_energy()
        self.best_energy = self.energy
        self.best_chain = chain[:]
        
    def _compute_energy(self):
        """Full energy computation (only used at init)."""
        contacts = 0
        for i, p in enumerate(self.chain):
            if self.sequence[i] == 'H':
                for q in _neighbours(p):
                    j = self.occupied.get(q)
                    if j is not None and j > i + 1 and self.sequence[j] == 'H':
                        contacts += 1
        return -contacts
    
    def _delta_energy(self, moved_indices, new_positions, old_positions):
        """
        Compute energy change for a move.
        moved_indices: list of indices that will move
        new_positions: {index: new_pos}
        old_positions: {index: old_pos}
        """
        moved_set = set(moved_indices)
        seq = self.sequence
        occ = self.occupied
        
        # Remove contacts involving moved residues (at old positions)
        remove_count = 0
        for i in moved_set:
            if seq[i] != 'H':
                continue
            p = old_positions[i]
            for q in _neighbours(p):
                j = occ.get(q)
                if j is not None and j > i and abs(j - i) > 1 and seq[j] == 'H':
                    remove_count += 1
        
        # Add contacts involving moved residues (at new positions)
        # Build temp occupied with new positions
        new_occ = dict(occ)
        for i, np_ in new_positions.items():
            new_occ[np_] = i
        
        add_count = 0
        for i in moved_set:
            if seq[i] != 'H':
                continue
            np_ = new_positions[i]
            for q in _neighbours(np_):
                j = new_occ.get(q)
                if j is not None and j > i and abs(j - i) > 1 and seq[j] == 'H':
                    add_count += 1
        
        return -(add_count - remove_count)
    
    def _apply_move(self, moved_indices, new_positions):
        """Apply move in-place."""
        old_positions = {i: self.chain[i] for i in moved_indices}
        for i in moved_indices:
            self.chain[i] = new_positions[i]
        for i in moved_indices:
            del self.occupied[old_positions[i]]
            self.occupied[new_positions[i]] = i
        self.energy += self._delta_energy(moved_indices, new_positions, old_positions)
        if self.energy < self.best_energy:
            self.best_energy = self.energy
            self.best_chain = self.chain[:]
    
    def _revert_move(self, moved_indices, old_positions):
        """Revert a move."""
        new_positions = {i: self.chain[i] for i in moved_indices}
        for i in moved_indices:
            self.chain[i] = old_positions[i]
        for i in moved_indices:
            del self.occupied[new_positions[i]]
            self.occupied[old_positions[i]] = i
        self.energy += self._delta_energy(moved_indices, old_positions, new_positions)

# Rotation matrices
_ROT_X = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=int)
_ROT_Y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=int)
_ROT_Z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
_ROT_MATRICES = [_ROT_X, _ROT_Y, _ROT_Z]
_ROT_POWERS = {1: {}, 2: {}, 3: {}}
for R in _ROT_MATRICES:
    for p in (1, 2, 3):
        _ROT_POWERS[p][id(R)] = np.linalg.matrix_power(R, p)

def _rotate_point(point, pivot, matrix):
    v = np.array(point, dtype=int) - np.array(pivot, dtype=int)
    v_rot = matrix @ v
    return tuple(np.array(pivot, dtype=int) + v_rot)

def _sa_fold(sequence, dimension, initial_chain, rng, deadline):
    """Simulated Annealing with diverse moves and incremental energy."""
    if initial_chain is None:
        return None, 0
    
    state = _FoldingState(sequence, initial_chain)
    t = 1.5
    final_temp = 0.003
    cooling = 0.9975
    no_improve = 0
    
    while time.time() < deadline:
        move_type = rng.random()
        moved = []
        new_pos = {}
        success = False
        delta_e = 0
        
        if move_type < 0.45:
            # PIVOT
            n = state.n
            if n >= 4:
                k = rng.randint(0, n - 2)
                pivot_pos = state.chain[k]
                axes = [0, 1, 2]
                rng.shuffle(axes)
                for axis in axes:
                    R = _ROT_MATRICES[axis]
                    for power in (1, 2, 3):
                        Rp = np.linalg.matrix_power(R, power)
                        new_tail = []
                        valid = True
                        for i in range(k + 1, n):
                            new_p = _rotate_point(state.chain[i], pivot_pos, Rp)
                            new_tail.append(new_p)
                        
                        fixed_set = set(state.chain[:k])
                        seen = set()
                        for new_p in new_tail:
                            if new_p in fixed_set or new_p in seen:
                                valid = False
                                break
                            seen.add(new_p)
                        
                        if valid and k + 1 < n:
                            first_new = new_tail[0]
                            if sum(abs(a - b) for a, b in zip(pivot_pos, first_new)) == 1:
                                moved = list(range(k + 1, n))
                                new_pos = {i: new_tail[i - k - 1] for i in moved}
                                success = True
                                break
                    if success:
                        break
        
        elif move_type < 0.70:
            # CRANKSHAFT / LOCAL REFOLD
            n = state.n
            if n >= 8:
                L = rng.randint(3, min(6, n - 2))
                start = rng.randint(1, n - L - 1)
                end = start + L - 1
                left_pos = state.chain[start - 1]
                right_pos = state.chain[end + 1]
                fixed_occ = set(state.chain[:start]) | set(state.chain[end+1:])
                
                def dfs_find(pos, path, visited):
                    if len(path) == L:
                        last = path[-1]
                        if sum(abs(a - b) for a, b in zip(last, right_pos)) == 1:
                            return path
                        return None
                    remaining = L - len(path)
                    for q in _neighbours(pos):
                        if q in visited or q in fixed_occ:
                            continue
                        dist = sum(abs(a - b) for a, b in zip(q, right_pos))
                        if dist > remaining or (dist + remaining) % 2 != 0:
                            continue
                        visited.add(q)
                        res = dfs_find(q, path + [q], visited)
                        if res is not None:
                            return res
                        visited.remove(q)
                    return None
                
                found = None
                for first in _neighbours(left_pos):
                    if first in fixed_occ or first == right_pos:
                        continue
                    if sum(abs(a - b) for a, b in zip(first, right_pos)) > L:
                        continue
                    visited = {left_pos, first}
                    res = dfs_find(first, [first], visited)
                    if res is not None:
                        found = res
                        break
                
                if found is not None:
                    moved = list(range(start, end + 1))
                    new_pos = {start + i: found[i] for i in range(L)}
                    success = True
        
        elif move_type < 0.85:
            # TAIL REGENERATION
            n = state.n
            if n >= 5:
                k = rng.randint(1, n - 3)
                fixed_part = state.chain[:k+1]
                local_occ = set(fixed_part)
                new_chain_tail = []
                valid = True
                
                for i in range(k + 1, n):
                    options = []
                    last = new_chain_tail[-1] if new_chain_tail else fixed_part[-1]
                    for site in _neighbours(last):
                        if site in local_occ:
                            continue
                        gain = 0
                        if state.sequence[i] == 'H':
                            for q in _neighbours(site):
                                j = None
                                for fi, fp in enumerate(fixed_part):
                                    if fp == q:
                                        j = fi
                                        break
                                if j is not None and state.sequence[j] == 'H':
                                    gain += 1
                                # Also check against already-placed tail
                                for ti, tp in enumerate(new_chain_tail):
                                    if tp == q and state.sequence[k+1+ti] == 'H':
                                        gain += 1
                        score = gain * 3.0 + rng.random() * 0.5
                        options.append((score, site))
                    
                    if not options:
                        valid = False
                        break
                    
                    options.sort(reverse=True)
                    if rng.random() < 0.7:
                        site = options[0][1]
                    else:
                        site = options[min(rng.randint(1, 3), len(options)-1)][1]
                    
                    local_occ.add(site)
                    new_chain_tail.append(site)
                
                if valid:
                    moved = list(range(k + 1, n))
                    new_pos = {i: new_chain_tail[i - k - 1] for i in moved}
                    success = True
        
        else:
            # SINGLE RESIDUE MOVE
            n = state.n
            if n >= 4:
                i = rng.randint(1, n - 2)
                left = state.chain[i - 1]
                right = state.chain[i + 1]
                common = set()
                for q in _neighbours(left):
                    if sum(abs(a - b) for a, b in zip(q, right)) == 1:
                        common.add(q)
                candidates = [q for q in common if q != state.chain[i] and q not in state.occupied]
                if candidates:
                    moved = [i]
                    new_pos = {i: rng.choice(candidates)}
                    success = True
        
        if not success:
            t = max(t * cooling, final_temp)
            continue
        
        # Compute delta
        old_pos = {i: state.chain[i] for i in moved}
        delta_e = state._delta_energy(moved, new_pos, old_pos)
        
        # Accept/reject
        accept = False
        if delta_e <= 0:
            accept = True
            no_improve = 0
        else:
            prob = math.exp(-delta_e / t)
            if rng.random() < prob:
                accept = True
            no_improve += 1
        
        if accept:
            state._apply_move(moved, new_pos)
        else:
            pass  # Don't apply
        
        t = max(t * cooling, final_temp)
        
        if no_improve > 8000:
            t = min(t * 1.5, 3.0)
            no_improve = 0
    
    return state.best_chain, state.best_energy

def _greedy_growth(sequence, dimension, rng):
    """Contact-weighted greedy chain growth."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            score = gain * 3.0 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(reverse=True)
        if rng.random() < 0.7:
            site = options[0][1]
        else:
            site = options[min(rng.randint(1, 3), len(options)-1)][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain

def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        occupied[site] = i
        chain.append(site)
        
    return chain

def fold(sequence, dimension):
    """
    Best conformation using multi-start simulated annealing with 
    true rigid-body pivot rotations, crankshaft moves, tail regeneration, 
    and single-residue moves, with incremental energy tracking.
    """
    n = len(sequence)
    rng = random.Random(42 + len(sequence))
    
    start_time = time.time()
    deadline = start_time + 275.0
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate diverse initial conformations
    init_chains = []
    
    for _ in range(10):
        c = _greedy_growth(sequence, dimension, rng)
        if c:
            init_chains.append(c)
    
    for _ in range(10):
        c = _random_walk(sequence, dimension, rng)
        if c:
            init_chains.append(c)
    
    if not init_chains:
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Evaluate and sort
    scored = []
    for c in init_chains:
        e = _FoldingState(sequence, c).energy
        scored.append((e, c))
    scored.sort(key=lambda x: x[0])
    
    best_energy = scored[0][0]
    best_chain = scored[0][1][:]
    
    # Phase 2: SA with time allocation
    remaining = deadline - time.time()
    if remaining <= 0:
        return [list(p) for p in best_chain]
    
    top = scored[:4]
    mid = scored[4:10]
    rest = scored[10:]
    
    time_top = remaining * 0.5 / max(1, len(top))
    time_mid = remaining * 0.3 / max(1, len(mid))
    time_rest = remaining * 0.2 / max(1, len(rest))
    
    for e, c in top:
        if time.time() > deadline:
            break
        chain, energy = _sa_fold(sequence, dimension, c, rng, min(deadline, time.time() + time_top))
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    for e, c in mid:
        if time.time() > deadline:
            break
        chain, energy = _sa_fold(sequence, dimension, c, rng, min(deadline, time.time() + time_mid))
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    for e, c in rest:
        if time.time() > deadline:
            break
        chain, energy = _sa_fold(sequence, dimension, c, rng, min(deadline, time.time() + time_rest))
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        best_chain = init_chains[0]
    
    return [list(p) for p in best_chain]

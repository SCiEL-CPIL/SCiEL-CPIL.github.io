# EVOLVE-BLOCK-START
import time
import random
import sys

_DEADLINE = time.time() + 320  # safety margin before 330s kill

def _time_left():
    return _DEADLINE - time.time()

def _neighbors(p):
    """Yield 6 neighbors of point p in 3D."""
    x, y, z = p
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _is_neighbor(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2]) == 1

def _energy(seq, pos):
    """Compute HP energy: -1 per H-H contact (non-consecutive)."""
    n = len(seq)
    where = {}
    for i, p in enumerate(pos):
        where[p] = i
    contacts = 0
    for i in range(n):
        if seq[i] != 'H':
            continue
        for nb in _neighbors(pos[i]):
            j = where.get(nb)
            if j is not None and abs(i - j) > 1 and seq[j] == 'H':
                contacts += 1
    return -contacts // 2  # each contact counted twice

def _grow(seq, dim, rng):
    """Randomized greedy growth with deadlock detection."""
    n = len(seq)
    start = (0, 0, 0)
    pos = [start]
    occ = {start: 0}
    for i in range(1, n):
        last = pos[-1]
        cands = []
        for nb in _neighbors(last):
            if nb in occ:
                continue
            # Score: prefer H-H contacts, then more free space
            score = rng.random()
            if seq[i] == 'H':
                for nb2 in _neighbors(nb):
                    j = occ.get(nb2)
                    if j is not None and j < i - 1 and seq[j] == 'H':
                        score += 10.0
            # Free space bonus
            free = sum(1 for nb2 in _neighbors(nb) if nb2 not in occ)
            score += 0.1 * free
            cands.append((score, nb))
        if not cands:
            return None
        cands.sort(key=lambda x: -x[0])
        # Pick from top candidates with some randomness
        if len(cands) > 1 and rng.random() < 0.3:
            idx = min(rng.randrange(len(cands)), len(cands)-1)
            chosen = cands[idx][1]
        else:
            chosen = cands[0][1]
        pos.append(chosen)
        occ[chosen] = i
    return pos

def _local_search(seq, pos, rng, max_iter=200):
    """Refine conformation via moves, pivots, and flips."""
    n = len(seq)
    current_energy = _energy(seq, pos)
    best_energy = current_energy
    best_pos = list(pos)
    
    def _valid(pos):
        """Check self-avoidance."""
        s = set()
        for p in pos:
            if p in s:
                return False
            s.add(p)
        return True
    
    def _try_move(pos, i):
        """Try moving residue i to a random free neighbor."""
        if i == 0 or i == n-1:
            return None
        old = pos[i]
        cands = [nb for nb in _neighbors(old) if nb not in set(pos)]
        if not cands:
            return None
        new_pos = list(pos)
        new_pos[i] = rng.choice(cands)
        # Check no collision with adjacent residues
        if new_pos[i] == new_pos[i-1] or new_pos[i] == new_pos[i+1]:
            return None
        if not _valid(new_pos):
            return None
        return new_pos
    
    def _try_pivot(pos, i):
        """Try rotating residue i around its neighbor (i-1 or i+1)."""
        if i <= 0 or i >= n-1:
            return None
        anchor = pos[i-1]
        old = pos[i]
        # Find all positions at distance 1 from anchor except old
        cands = [nb for nb in _neighbors(anchor) if nb != old and nb != pos[i-2] if i-2 >= 0]
        if i+1 < n:
            cands = [c for c in cands if c != pos[i+1]]
        if not cands:
            return None
        new_pos = list(pos)
        new_pos[i] = rng.choice(cands)
        if not _valid(new_pos):
            return None
        return new_pos
    
    def _try_flip(pos, i, j):
        """Try swapping two non-adjacent residues."""
        if abs(i-j) <= 1:
            return None
        new_pos = list(pos)
        new_pos[i], new_pos[j] = pos[j], pos[i]
        # Check adjacency constraints
        for k in range(n):
            if k+1 < n and not _is_neighbor(new_pos[k], new_pos[k+1]):
                return None
        if not _valid(new_pos):
            return None
        return new_pos
    
    for _ in range(max_iter):
        if _time_left() < 1.0:
            break
        new_pos = None
        op = rng.random()
        i = rng.randrange(n)
        if op < 0.5:
            new_pos = _try_move(pos, i)
        elif op < 0.8:
            new_pos = _try_pivot(pos, i)
        else:
            j = rng.randrange(n)
            new_pos = _try_flip(pos, i, j)
        
        if new_pos is None:
            continue
        
        new_energy = _energy(seq, new_pos)
        if new_energy < current_energy:
            current_energy = new_energy
            pos = new_pos
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = list(pos)
        elif rng.random() < 0.1:  # occasional acceptance of equal/worse for diversity
            pos = new_pos
            current_energy = new_energy
    
    return best_pos, best_energy

def fold(sequence, dimension):
    """Fold an HP sequence on the 3D cubic lattice."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Determine budget based on sequence length
    if n <= 50:
        n_grow = 500
        ls_iter = 300
    elif n <= 80:
        n_grow = 300
        ls_iter = 200
    elif n <= 120:
        n_grow = 150
        ls_iter = 100
    else:
        n_grow = 80
        ls_iter = 50
    
    best_pos = None
    best_energy = 0  # 0 means no valid conformation yet
    
    for attempt in range(n_grow):
        if _time_left() < 2.0:
            break
        
        pos = _grow(sequence, dimension, rng)
        if pos is None:
            continue
        
        e = _energy(sequence, pos)
        if e < best_energy:
            best_energy = e
            best_pos = list(pos)
        
        # Local search refinement
        refined, refined_e = _local_search(sequence, pos, rng, ls_iter)
        if refined_e < best_energy:
            best_energy = refined_e
            best_pos = refined
    
    if best_pos is None:
        # Fallback: straight line
        best_pos = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_pos]
# EVOLVE-BLOCK-END

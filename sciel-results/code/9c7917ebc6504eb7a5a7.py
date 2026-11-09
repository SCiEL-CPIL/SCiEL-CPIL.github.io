"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                q = (x + dx, y + dy)
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _is_valid(chain):
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    return True


def _apply_reversal(chain, i, j):
    if i > j:
        i, j = j, i
    
    # Verify unit steps are preserved at the cut points
    # Left cut: chain[i-1] must be adjacent to chain[j]
    if i > 0:
        p_prev = chain[i - 1]
        p_new_next = chain[j]
        if abs(p_prev[0] - p_new_next[0]) + abs(p_prev[1] - p_new_next[1]) != 1:
            return None
            
    # Right cut: chain[i] must be adjacent to chain[j+1]
    if j < len(chain) - 1:
        p_new_prev = chain[i]
        p_next = chain[j + 1]
        if abs(p_new_prev[0] - p_next[0]) + abs(p_new_prev[1] - p_next[1]) != 1:
            return None
            
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    if not _is_valid(new_chain):
        return None
    return new_chain


def _anneal(sequence, dimension, rng, max_iter=3000):
    n = len(sequence)
    chain = [(0, 0)]
    occupied = {(0, 0)}
    for i in range(1, n):
        last = chain[-1]
        neighbors = [(last[0]+1, last[1]), (last[0]-1, last[1]), (last[0], last[1]+1), (last[0], last[1]-1)]
        rng.shuffle(neighbors)
        placed = False
        for nb in neighbors:
            if nb not in occupied:
                chain.append(nb)
                occupied.add(nb)
                placed = True
                break
        if not placed:
            return None

    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    T = 1.0
    cooling_rate = 0.999
    
    for _ in range(max_iter):
        if _remaining_time() < 0.5:
            break
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        
        new_chain = _apply_reversal(chain, i, j)
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                
        T *= cooling_rate
        if T < 0.01:
            break
            
    return best_chain, best_energy


def fold(sequence, dimension):
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 1
    
    while _remaining_time() > 1.0:
        result = _anneal(sequence, dimension, rng, max_iter=3000)
        if result is None:
            continue
        chain, energy = result
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    if best_chain is None:
        best_chain = [(i, 0) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

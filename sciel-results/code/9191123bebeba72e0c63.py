"""Improved HP lattice folding: fast growth + efficient local search + time guard."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_fast(seq, chain):
    """Fast energy using dict."""
    n = len(chain)
    pos = {}
    for i, p in enumerate(chain):
        pos[(p[0], p[1])] = i
    h = [c == 'H' for c in seq]
    total = 0
    for i in range(n):
        if not h[i]:
            continue
        x, y = chain[i]
        for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            j = pos.get((nx, ny), -1)
            if j > i + 1 and h[j]:
                total += 1
    return -total


def _grow(seq, rng, bias):
    """Randomized greedy chain growth. Returns list of (x,y) or None."""
    n = len(seq)
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    h = [c == 'H' for c in seq]
    for i in range(1, n):
        x, y = chain[-1]
        cands = []
        for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            if (nx, ny) in occupied:
                continue
            gain = 0
            if h[i]:
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    q = (nx+dx, ny+dy)
                    j = occupied.get(q, -1)
                    if j != -1 and j < i - 1 and h[j]:
                        gain += 1
            free = 0
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                if (nx+dx, ny+dy) not in occupied:
                    free += 1
            score = gain * bias + free * 0.1 + rng.random()
            cands.append((score, (nx, ny)))
        if not cands:
            return None
        cands.sort(reverse=True)
        k = min(len(cands), 3)
        chosen = cands[rng.randrange(k)][1]
        occupied[chosen] = i
        chain.append(chosen)
    return chain


def _local_search(seq, chain, rng, max_iters=2000):
    """Efficient local search: try moving last residue to adjacent free sites."""
    n = len(chain)
    if n < 3:
        return list(chain), _energy_fast(seq, chain)
    h = [c == 'H' for c in seq]
    current = list(chain)
    cur_energy = _energy_fast(seq, current)
    pos = {}
    for i, p in enumerate(current):
        pos[(p[0], p[1])] = i

    for it in range(max_iters):
        if _remaining() < 3.0:
            break
        found = False
        # Try moving the last residue to each adjacent free site
        last_idx = n - 1
        lx, ly = current[last_idx]
        del pos[(lx, ly)]
        cands = []
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = lx + dx, ly + dy
            if (nx, ny) in pos:
                continue
            # Check if this move is valid (unit step from new neighbor)
            if last_idx > 0:
                px, py = current[last_idx - 1]
                if abs(px - nx) + abs(py - ny) != 1:
                    continue
            # Compute energy delta: only last residue's contacts change
            delta = 0
            # Remove old contacts of last residue
            if h[last_idx]:
                for dx2, dy2 in ((1,0),(-1,0),(0,1),(0,-1)):
                    ox, oy = lx + dx2, ly + dy2
                    j = pos.get((ox, oy), -1)
                    if j != -1 and j < last_idx - 1 and h[j]:
                        delta += 1  # losing a contact = +1 energy (worse)
            # Add new contacts
            if h[last_idx]:
                for dx2, dy2 in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx2, ny2 = nx + dx2, ny + dy2
                    j = pos.get((nx2, ny2), -1)
                    if j != -1 and j < last_idx - 1 and h[j]:
                        delta -= 1  # gaining a contact = -1 energy (better)
            cands.append((delta, (nx, ny)))
        # Restore last position in pos for now
        pos[(lx, ly)] = last_idx

        if cands:
            cands.sort(key=lambda x: x[0])
            # Try best candidate first
            for delta, (nx, ny) in cands:
                if delta >= 0:
                    break
                # Accept the move
                current[last_idx] = (nx, ny)
                del pos[(lx, ly)]
                pos[(nx, ny)] = last_idx
                cur_energy += delta
                found = True
                break

        if not found and it % 100 == 0:
            # Try a random perturbation: swap two non-adjacent positions
            # This is expensive, so do it rarely
            if rng.random() < 0.1 and _remaining() > 5.0:
                i = rng.randrange(n - 2)
                j = rng.randrange(i + 2, n)
                # Only try if the swap maintains connectivity
                # This is complex; skip for now and rely on growth restarts
                pass

        if not found and it > 500:
            # No improvement in 500 iterations, stop
            break

    return current, cur_energy


def fold(sequence, dimension):
    """Best conformation found within time budget."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    best_chain = None
    best_energy = 0  # energy is negative or zero

    # Fallback: straight line
    fallback = [[i, 0] for i in range(n)]

    # Multiple growth strategies with different biases
    biases = [1.0, 0.5, 0.3, 0.1, 2.0, 1.5]
    attempts = 0
    max_attempts = 200

    while _remaining() > 5.0 and attempts < max_attempts:
        bias = biases[attempts % len(biases)]
        chain = _grow(sequence, rng, bias)
        if chain is None:
            attempts += 1
            continue
        energy = _energy_fast(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

        # Local search on this chain
        if _remaining() > 10.0:
            improved, imp_energy = _local_search(sequence, chain, rng, max_iters=300)
            if imp_energy < best_energy:
                best_chain = improved
                best_energy = imp_energy

        attempts += 1

    if best_chain is None:
        return fallback
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

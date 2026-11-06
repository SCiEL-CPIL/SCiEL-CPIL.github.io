"""HP lattice protein folding: fragment re-insertion + reversal moves with fast energy."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import math

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_fast(seq, chain):
    """Fast energy using a position->index map and precomputed H indices."""
    where = {p: i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(seq) if c == "H")
    contacts = 0
    for i in h_set:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return -contacts


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            dx = abs(chain[i][0] - chain[i - 1][0])
            dy = abs(chain[i][1] - chain[i - 1][1])
            if dx + dy != 1:
                return False
    return True


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    h_set = set(i for i, c in enumerate(sequence) if c == "H")
    for i in range(1, n):
        options = []
        x, y = chain[-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            gain = 0
            if i in h_set:
                sx, sy = site
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (sx + adx, sy + ady)
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and idx in h_set:
                            gain += 1
            free = 0
            sx, sy = site
            for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (sx + adx, sy + ady) not in occupied:
                    free += 1
            # Weight H-H contacts much more heavily
            score = 2.0 * gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _try_reinsert_fragments(chain, sequence, rng, max_frag_len=8):
    """
    Attempt to re-insert a fragment: cut out chain[a:b], then try to regrow it
    from chain[a-1] (if a>0) connecting to chain[b] (if b<n).
    Returns (new_chain, energy_delta) or (None, 0).
    Uses DFS with pruning for speed.
    """
    n = len(chain)
    if n < 5:
        return None, 0

    # Pick fragment length and position
    L = rng.randint(2, min(max_frag_len, n - 3))
    a = rng.randint(1, n - L - 1)  # a >= 1 so there's a left neighbor
    b = a + L  # b <= n-1 so there's a right neighbor

    # The fragment to remove: chain[a:b]
    # Left neighbor: chain[a-1], Right neighbor: chain[b]
    left_neighbor = chain[a - 1]
    right_neighbor = chain[b]

    # The rest of the chain without the fragment
    rest_occupied = set(chain[:a] + chain[b:])

    # Check if right_neighbor is at Manhattan distance L from left_neighbor
    lx, ly = left_neighbor
    rx, ry = right_neighbor
    manhattan = abs(rx - lx) + abs(ry - ly)
    if manhattan > L or (L - manhattan) % 2 != 0:
        return None, 0

    # DFS to find a self-avoiding path of exactly L steps
    def _find_path_dfs(current, remaining, path, occupied_set, depth=0):
        """Find a self-avoiding path of exactly `remaining` steps from current to end."""
        if depth > 50:  # prevent deep recursion
            return None
        if remaining == 0:
            if current == right_neighbor:
                return path
            return None

        x, y = current
        # Prune: can we still reach right_neighbor in remaining steps?
        dist_to_end = abs(rx - x) + abs(ry - y)
        if dist_to_end > remaining or (remaining - dist_to_end) % 2 != 0:
            return None

        # Try directions in order of preference (towards target)
        dirs = []
        if rx > x:
            dirs.append((1, 0))
        if rx < x:
            dirs.append((-1, 0))
        if ry > y:
            dirs.append((0, 1))
        if ry < y:
            dirs.append((0, -1))
        # Add remaining directions
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if d not in dirs:
                dirs.append(d)

        for dx, dy in dirs:
            site = (x + dx, y + dy)
            if site in occupied_set:
                continue
            if site in path:
                continue
            # Don't go back to start unless it's the end
            if remaining == 1 and site != right_neighbor:
                continue
            if remaining > 1 and site == left_neighbor:
                continue

            path.append(site)
            occupied_set.add(site)
            result = _find_path_dfs(site, remaining - 1, path, occupied_set, depth + 1)
            if result is not None:
                return result
            path.pop()
            occupied_set.discard(site)
        return None

    path = [left_neighbor]
    path_result = _find_path_dfs(left_neighbor, L, path, rest_occupied)
    if path_result is None:
        return None, 0

    # path_result[0] is left_neighbor, path_result[-1] should be right_neighbor
    if path_result[-1] != right_neighbor:
        return None, 0

    # The new fragment positions are path_result[1:] (excluding the start which is already in chain)
    new_frag = path_result[1:]  # L elements
    if len(new_frag) != L:
        return None, 0

    new_chain = chain[:a] + new_frag + chain[b:]

    # Validate
    if not _is_valid_chain(new_chain):
        return None, 0

    return new_chain, 0


def _reversal_move(chain, rng, max_len=None):
    """Random reversal move."""
    n = len(chain)
    if n < 4:
        return None
    if max_len is None:
        max_len = min(8, n - 2)
    L = rng.randint(2, max_len)
    a = rng.randint(0, n - L - 1)
    b = a + L
    new_chain = chain[:a] + chain[a:b + 1][::-1] + chain[b + 1:]
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _kink_move(chain, rng):
    """
    Kink move: pick a residue i (1..n-2) and try to move it to an adjacent
    free position, if the result is a valid self-avoiding walk.
    """
    n = len(chain)
    if n < 3:
        return None
    i = rng.randint(1, n - 2)
    x, y = chain[i]
    options = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        site = (x + dx, y + dy)
        # Must not be occupied by any other residue
        if site in chain and site != chain[i]:
            continue
        # Must maintain unit step with neighbors
        # Check with chain[i-1]
        px, py = chain[i - 1]
        if abs(site[0] - px) + abs(site[1] - py) != 1:
            continue
        # Check with chain[i+1]
        nx, ny = chain[i + 1]
        if abs(site[0] - nx) + abs(site[1] - ny) != 1:
            continue
        options.append(site)

    if not options:
        return None
    site = options[rng.randint(0, len(options) - 1)]
    new_chain = chain[:]
    new_chain[i] = site
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _segment_shift_move(chain, rng, max_seg_len=6):
    """
    Segment shift move: cut out a segment chain[a:b] and shift it by one unit
    in a random direction, checking validity.
    """
    n = len(chain)
    if n < 5:
        return None
    
    L = rng.randint(2, min(max_seg_len, n - 4))
    a = rng.randint(1, n - L - 1)
    b = a + L  # segment is chain[a:b]
    
    # The segment to move
    segment = chain[a:b]
    # Left neighbor: chain[a-1], Right neighbor: chain[b]
    left_neighbor = chain[a - 1]
    right_neighbor = chain[b]
    
    # Try shifting in 4 directions
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        new_segment = [(x + dx, y + dy) for x, y in segment]
        
        # Check connectivity: left_neighbor to new_segment[0]
        if abs(new_segment[0][0] - left_neighbor[0]) + abs(new_segment[0][1] - left_neighbor[1]) != 1:
            continue
        # Check connectivity: new_segment[-1] to right_neighbor
        if abs(right_neighbor[0] - new_segment[-1][0]) + abs(right_neighbor[1] - new_segment[-1][1]) != 1:
            continue
        
        # Build new chain
        new_chain = chain[:a] + new_segment + chain[b:]
        
        # Check self-avoidance
        if _is_valid_chain(new_chain):
            return new_chain
    
    return None


def _local_search(sequence, chain, rng, max_iters=500):
    """Local search with fragment re-insertion, reversals, kinks, and segment shifts."""
    best_chain = list(chain)
    best_energy = _energy_fast(sequence, best_chain)
    n = len(best_chain)
    no_improve_count = 0
    max_no_improve = 30
    temperature = 0.05

    for it in range(max_iters):
        if _remaining() < 5.0:
            break

        # Adaptive: if stuck, perturb to escape local minimum
        if no_improve_count >= max_no_improve:
            # Try multiple perturbation strategies
            improved = False
            for _attempt in range(5):
                # Try fragment re-insertion for perturbation
                if rng.random() < 0.4:
                    perturbed, _ = _try_reinsert_fragments(best_chain, sequence, rng)
                elif rng.random() < 0.7:
                    perturbed = _reversal_move(best_chain, rng, max_len=min(15, n - 2))
                else:
                    perturbed = _segment_shift_move(best_chain, rng)
                if perturbed is not None:
                    e_pert = _energy_fast(sequence, perturbed)
                    if e_pert < best_energy:
                        best_chain = perturbed
                        best_energy = e_pert
                        improved = True
                        break
                    # Simulated annealing style acceptance
                    delta = e_pert - best_energy
                    if delta < 0 or rng.random() < (1.0 / (1.0 + math.exp(delta / temperature))):
                        best_chain = perturbed
                        best_energy = e_pert
                        break
            no_improve_count = 0
            max_no_improve = min(60, max_no_improve + 10)
            temperature = max(0.01, temperature * 0.8)

        # Choose move type - weight based on what's working
        r = rng.random()
        if r < 0.30:
            new_chain, _ = _try_reinsert_fragments(best_chain, sequence, rng)
        elif r < 0.55:
            new_chain = _reversal_move(best_chain, rng)
        elif r < 0.75:
            new_chain = _segment_shift_move(best_chain, rng)
        else:
            new_chain = _kink_move(best_chain, rng)

        if new_chain is None:
            no_improve_count += 1
            continue

        e = _energy_fast(sequence, new_chain)
        if e < best_energy:
            best_chain = new_chain
            best_energy = e
            no_improve_count = 0
        else:
            # Simulated annealing style acceptance
            delta = e - best_energy
            if delta < 0 or rng.random() < (1.0 / (1.0 + math.exp(delta / temperature))):
                best_chain = new_chain
                best_energy = e
            no_improve_count += 1

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using multiple restarts and local search with time guard."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0  # start at 0 (unfolded)
    n = len(sequence)

    # Phase 1: Quick growth to get initial candidates
    num_quick = 200
    for r in range(num_quick):
        if _remaining() < 10.0:
            break

        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue

        e = _energy_fast(sequence, chain)
        if e < best_energy:
            best_energy = e
            best = chain

    # Phase 2: Local search on best candidates
    if best is not None and _remaining() > 8.0:
        iters = 400 if _remaining() > 30 else 200
        improved, imp_e = _local_search(sequence, best, rng, max_iters=iters)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved

    # Phase 3: More restarts with local search
    num_restarts = 400
    for r in range(num_restarts):
        if _remaining() < 5.0:
            break

        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue

        e = _energy_fast(sequence, chain)
        if e < best_energy:
            best_energy = e
            best = chain

        # Apply local search on the best candidate periodically
        if r % 3 == 0 and best is not None and _remaining() > 10.0:
            iters = 300 if _remaining() > 40 else 150
            improved, imp_e = _local_search(sequence, best, rng, max_iters=iters)
            if imp_e < best_energy:
                best_energy = imp_e
                best = improved

    # Final polish: high-iteration local search on the best
    if best is not None and _remaining() > 5.0:
        iters = 800 if _remaining() > 25 else 400
        improved, imp_e = _local_search(sequence, best, rng, max_iters=iters)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved

    # Second polish if time allows
    if best is not None and _remaining() > 3.0:
        improved, imp_e = _local_search(sequence, best, rng, max_iters=500)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved

    # Third polish if time allows
    if best is not None and _remaining() > 2.0:
        improved, imp_e = _local_search(sequence, best, rng, max_iters=300)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved

    if best is None:
        # Fallback: straight line
        best = [(i, 0) for i in range(n)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END

"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
import math

# Deadline guard: 330s wall-clock limit, use 320s to be safe
_DEADLINE = time.time() + 318.0


def _time_left():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H non-consecutive contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2 if False else -contacts  # each contact counted once per pair


def _energy_full(sequence, chain):
    """Proper energy: count each H-H pair once."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_fold(sequence, dimension, rng, temperature=0.0):
    """
    Greedy chain growth with stochastic selection.
    Prefers sites that create H-H contacts, then free space.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}

    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Count new H-H contacts this placement would create
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Free space heuristic: more open neighbours = less likely to trap
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            # Score: heavy weight on gain, small on free, plus noise
            noise = rng.random() * (1.0 + temperature)
            score = gain * 5.0 + free * 0.05 + noise
            options.append((score, site))

        if not options:
            return None

        # Stochastic selection: pick from top options with softmax-like probability
        if temperature > 0 and len(options) > 1:
            # Sort by score descending
            options.sort(key=lambda x: -x[0])
            # Take top k and pick with probability proportional to score
            top_k = min(3, len(options))
            top = options[:top_k]
            # Use softmax
            max_s = top[0][0]
            exps = [math.exp((s - max_s) / max(temperature, 0.1)) for s, _ in top]
            total = sum(exps)
            r = rng.random() * total
            cum = 0
            chosen = top[-1][1]
            for idx, (s, site) in enumerate(top):
                cum += exps[idx]
                if r <= cum:
                    chosen = site
                    break
            site = chosen
        else:
            site = max(options)[1]

        occupied[site] = i
        chain.append(site)

    return chain


def _local_search(sequence, chain, rng, max_moves=2000):
    """
    Simple local search: try to move individual residues to improve energy.
    Uses delta-energy evaluation.
    """
    if chain is None:
        return None

    n = len(chain)
    current_energy = _energy_full(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    occupied = {p: i for i, p in enumerate(chain)}

    for _ in range(max_moves):
        if _time_left() < 2.0:
            break

        # Pick a random residue to try moving (not endpoints)
        i = rng.randint(1, n - 2)
        old_pos = chain[i]

        # Try each neighbour of old_pos (excluding the two chain neighbours)
        prev_pos = chain[i - 1]
        next_pos = chain[i + 1]

        candidates = []
        for site in _neighbours(old_pos):
            if site == prev_pos or site == next_pos:
                continue
            if site in occupied and occupied[site] != i:
                continue
            candidates.append(site)

        if not candidates:
            continue

        # Evaluate delta for each candidate
        best_candidate = None
        best_delta = 0

        for site in candidates:
            # Compute delta energy
            delta = 0
            if sequence[i] == 'H':
                # Remove old contacts at old_pos
                for q in _neighbours(old_pos):
                    j = occupied.get(q)
                    if j is not None and j != i and j != i - 1 and j != i + 1:
                        if sequence[j] == 'H':
                            delta += 1  # losing a contact (energy increases)
                # Add new contacts at site
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j != i and j != i - 1 and j != i + 1:
                        if sequence[j] == 'H':
                            delta -= 1  # gaining a contact (energy decreases)

            if delta < best_delta:
                best_delta = delta
                best_candidate = site

        if best_candidate is not None and best_delta < 0:
            # Apply the move
            del occupied[old_pos]
            occupied[best_candidate] = i
            chain[i] = best_candidate
            current_energy += best_delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)

    return best_chain


def _simulated_annealing(sequence, chain, rng, max_iter=5000):
    """
    Simulated annealing with single-residue moves.
    """
    if chain is None:
        return None

    n = len(chain)
    current_energy = _energy_full(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    occupied = {p: i for i, p in enumerate(chain)}

    T_start = 1.0
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max(max_iter, 1))

    for it in range(max_iter):
        if _time_left() < 2.0:
            break

        T = T_start * (cooling ** it)

        # Pick random interior residue
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        prev_pos = chain[i - 1]
        next_pos = chain[i + 1]

        # Find valid candidate sites
        candidates = []
        for site in _neighbours(old_pos):
            if site == prev_pos or site == next_pos:
                continue
            if site in occupied and occupied[site] != i:
                continue
            candidates.append(site)

        if not candidates:
            continue

        site = rng.choice(candidates)

        # Compute delta
        delta = 0
        if sequence[i] == 'H':
            for q in _neighbours(old_pos):
                j = occupied.get(q)
                if j is not None and j != i and j != i - 1 and j != i + 1:
                    if sequence[j] == 'H':
                        delta += 1
            for q in _neighbours(site):
                j = occupied.get(q)
                if j is not None and j != i and j != i - 1 and j != i + 1:
                    if sequence[j] == 'H':
                        delta -= 1

        # Accept?
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-6)):
            del occupied[old_pos]
            occupied[site] = i
            chain[i] = site
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)

    return best_chain


def fold(sequence, dimension):
    """
    Fold a sequence on the cubic lattice.
    Returns a list of len(sequence) lattice points (lists of length dimension).
    """
    if dimension != 3:
        dimension = 3

    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]

    rng = random.Random(42 + len(sequence) * 7)

    best_chain = None
    best_energy = float('inf')

    # Phase 1: Multiple greedy folds with varying temperatures
    num_greedy = 150
    for k in range(num_greedy):
        if _time_left() < 5.0:
            break
        temp = 0.0 if k < 50 else 0.5 if k < 100 else 1.0
        chain = _greedy_fold(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue
        e = _energy_full(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain

    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy_full(sequence, best_chain)

    # Phase 2: Local search / simulated annealing on the best chain
    if _time_left() > 10.0:
        # Run local search
        ls_chain = _local_search(sequence, list(best_chain), rng, max_moves=3000)
        if ls_chain is not None:
            e = _energy_full(sequence, ls_chain)
            if e < best_energy:
                best_energy = e
                best_chain = ls_chain

    if _time_left() > 10.0:
        # Run simulated annealing
        sa_chain = _simulated_annealing(sequence, list(best_chain), rng, max_iter=8000)
        if sa_chain is not None:
            e = _energy_full(sequence, sa_chain)
            if e < best_energy:
                best_energy = e
                best_chain = sa_chain

    # Phase 3: If time remains, do more greedy + refinement cycles
    if _time_left() > 15.0:
        for k in range(100):
            if _time_left() < 5.0:
                break
            temp = rng.random() * 2.0
            chain = _greedy_fold(sequence, dimension, rng, temperature=temp)
            if chain is None:
                continue
            # Quick local search
            chain = _local_search(sequence, chain, rng, max_moves=500)
            if chain is None:
                continue
            e = _energy_full(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain

    # Final safety check: ensure self-avoiding
    seen = set()
    for p in best_chain:
        tp = tuple(p)
        if tp in seen:
            # Should not happen, but fallback
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
            break
        seen.add(tp)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

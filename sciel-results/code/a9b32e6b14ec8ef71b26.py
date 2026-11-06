"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict

# Global deadline: 320 seconds (safety margin before 330s wall-clock kill)
_DEADLINE = time.time() + 320.0
_DEADLINE_PER_SEQ = 320.0 / 26.0  # rough per-sequence budget

def _neighbours(point):
    """Yield all 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP energy: -1 per H-H non-bond contact."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow_greedy(sequence, dimension, rng):
    """
    Greedy chain growth with lookahead: at each step, score candidate sites
    by (new H-H contacts formed, free-space heuristic, small random tiebreak).
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}

    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                # Count new H-H contacts this site would form with earlier H residues
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            # Free-space heuristic: prefer sites with more open neighbours
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = gain * 10.0 + free * 0.5 + rng.random() * 2.0
            options.append((score, site))
        if not options:
            return None
        options.sort(key=lambda x: -x[0])
        # Pick from top candidates with some stochasticity
        top_k = min(3, len(options))
        if rng.random() < 0.7:
            site = options[0][1]
        else:
            site = options[rng.randrange(top_k)][1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, dimension, chain, max_iters=500, rng=None):
    """
    Local search via single-residue moves (move one residue to an adjacent empty site
    and re-check self-avoidance). Accept moves that improve energy.
    """
    if rng is None:
        rng = random.Random()
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    energy = _energy(sequence, chain)

    for _ in range(max_iters):
        # Pick a random interior residue to move
        i = rng.randrange(1, n - 1)
        old_pos = chain[i]
        # Try all 6 neighbours
        for new_pos in _neighbours(old_pos):
            if new_pos in where:
                continue
            # Check that moving residue i to new_pos doesn't break self-avoidance
            # (it's a single move, so we just need new_pos not occupied)
            # Update chain
            chain[i] = new_pos
            where[new_pos] = i
            del where[old_pos]

            # Recompute energy (expensive but simple)
            new_energy = _energy(sequence, chain)
            if new_energy < energy:
                energy = new_energy
            else:
                # Revert
                chain[i] = old_pos
                where[old_pos] = i
                del where[new_pos]
        # Early exit if no improvement possible in this iteration is hard to check,
        # so just run all iterations
    return chain, energy


def _simulated_annealing(sequence, dimension, chain, max_iters=2000, rng=None):
    """
    Simulated annealing with single-residue moves.
    """
    if rng is None:
        rng = random.Random()
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    energy = _energy(sequence, chain)
    T_start = 5.0
    T_end = 0.1
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))

    T = T_start
    for it in range(max_iters):
        T *= cooling
        i = rng.randrange(1, n - 1)
        old_pos = chain[i]
        new_pos = list(_neighbours(old_pos))
        rng.shuffle(new_pos)
        for cand in new_pos:
            if cand in where:
                continue
            # Try the move
            chain[i] = cand
            where[cand] = i
            del where[old_pos]
            new_energy = _energy(sequence, chain)
            delta = new_energy - energy
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                energy = new_energy
                old_pos = cand  # keep the move
                break
            else:
                # Revert
                chain[i] = old_pos
                where[old_pos] = i
                del where[cand]
        else:
            # No valid move found, revert (already reverted in loop)
            pass
    return chain, energy


def _hybrid_optimize(sequence, dimension, chain, time_budget, rng):
    """
    Hybrid: local search then simulated annealing, with time guard.
    """
    start_time = time.time()
    n = len(chain)

    # Phase 1: Local search
    ls_iters = min(300, max(50, n * 3))
    chain, energy = _local_search(sequence, dimension, chain, ls_iters, rng)

    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.4:
        # Phase 2: Simulated annealing
        sa_iters = min(2000, max(200, n * 10))
        chain, energy = _simulated_annealing(sequence, dimension, chain, sa_iters, rng)

    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.7:
        # Phase 3: Another round of local search
        ls_iters2 = min(200, max(30, n * 2))
        chain, energy = _local_search(sequence, dimension, chain, ls_iters2, rng)

    return chain, energy


def fold(sequence, dimension):
    """
    Fold an HP sequence on the cubic lattice.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    rng = random.Random(42 + n)

    # Per-sequence time budget
    seq_budget = min(_DEADLINE_PER_SEQ, 12.0)  # cap at 12s per sequence
    start_time = time.time()

    # Generate initial conformations via randomized greedy growth
    num_grows = min(80, max(20, n * 2))
    candidates = []

    for _ in range(num_grows):
        if time.time() - start_time > seq_budget * 0.5:
            break
        chain = _grow_greedy(sequence, dimension, rng)
        if chain is not None:
            energy = _energy(sequence, chain)
            candidates.append((energy, chain))

    if not candidates:
        # Fallback: straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]

    # Sort by energy
    candidates.sort(key=lambda x: x[0])

    best_chain = candidates[0][1]
    best_energy = candidates[0][0]

    # Optimize the top candidates
    num_opt = min(5, len(candidates))
    per_opt_budget = (seq_budget * 0.5) / max(1, num_opt)

    for idx in range(num_opt):
        if time.time() - start_time > seq_budget * 0.9:
            break
        chain_copy = list(candidates[idx][1])
        opt_budget = min(per_opt_budget, seq_budget * 0.3)
        opt_chain, opt_energy = _hybrid_optimize(sequence, dimension, chain_copy, opt_budget, rng)
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain

    # Also try optimizing the best initial candidate more thoroughly
    if time.time() - start_time < seq_budget * 0.9:
        chain_copy = list(best_chain)
        remaining = seq_budget * 0.9 - (time.time() - start_time)
        if remaining > 0.5:
            opt_chain, opt_energy = _hybrid_optimize(sequence, dimension, chain_copy, remaining, rng)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

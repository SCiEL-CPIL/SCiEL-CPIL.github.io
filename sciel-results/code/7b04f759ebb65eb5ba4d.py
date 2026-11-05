"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Precompute neighbor offsets for 3D cubic lattice
_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _energy_correct(sequence, chain):
    """Compute energy: -number of H-H contacts on neighboring non-consecutive sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width=200):
    """
    Beam search for HP protein folding.
    Maintains a beam of the best partial conformations at each step.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    # Beam: list of (energy_so_far, chain_tuple, occupied_dict)
    # We use tuples for chains to make them hashable for deduplication
    beam = [(0, ((0, 0, 0),), {(0, 0, 0): 0})]

    for i in range(1, n):
        if _time_left() < 10.0:
            break

        new_beam = []
        seen = set()

        for energy_so_far, chain_tuple, occupied in beam:
            last = chain_tuple[-1]
            seq_i = sequence[i]

            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                site = (last[0] + dx, last[1] + dy, last[2] + dz)
                if site in occupied:
                    continue

                # Compute new contacts gained
                gain = 0
                if seq_i == 'H':
                    for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                        q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1

                # New energy is more negative when gain is higher
                new_energy = energy_so_far - gain

                # Deduplicate: use the last position and a hash of the chain shape
                # For efficiency, we'll use a tuple of the last few positions as a proxy
                key = (site, chain_tuple[-3:] if len(chain_tuple) >= 3 else chain_tuple)
                if key in seen:
                    continue
                seen.add(key)

                new_chain = chain_tuple + (site,)
                new_occupied = dict(occupied)
                new_occupied[site] = i

                new_beam.append((new_energy, new_chain, new_occupied))

        if not new_beam:
            break

        # Sort by energy (most negative first) and keep top beam_width
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

    if not beam:
        return None

    # Return the best complete chain
    best_energy, best_chain_tuple, _ = beam[0]
    return list(best_chain_tuple)


def _check_segment_reversal_valid(chain, start_idx, end_idx):
    """Check if reversing segment [start_idx, end_idx] maintains self-avoidance and connectivity."""
    n = len(chain)
    
    # Build set of positions occupied by residues OUTSIDE the segment
    outside_positions = set()
    for i in range(n):
        if i < start_idx or i > end_idx:
            outside_positions.add(tuple(chain[i]))
    
    # Check that reversed segment positions don't collide with outside
    for i in range(start_idx, end_idx + 1):
        # After reversal, position i gets the value that was at (start_idx + end_idx - i)
        new_pos = tuple(chain[start_idx + end_idx - i])
        if new_pos in outside_positions:
            return False
    
    # Check connectivity at boundaries
    if start_idx > 0:
        p1 = tuple(chain[start_idx - 1])
        p2 = tuple(chain[end_idx])  # After reversal, start_idx gets old end_idx value
        if sum(abs(a - b) for a, b in zip(p1, p2)) != 1:
            return False
    
    if end_idx < n - 1:
        p1 = tuple(chain[start_idx])  # After reversal, end_idx gets old start_idx value
        p2 = tuple(chain[end_idx + 1])
        if sum(abs(a - b) for a, b in zip(p1, p2)) != 1:
            return False
    
    return True


def _apply_segment_reversal(chain, start_idx, end_idx):
    """Apply segment reversal and return new chain."""
    new_chain = chain[:]
    new_chain[start_idx:end_idx + 1] = new_chain[start_idx:end_idx + 1][::-1]
    return new_chain


def _local_search(sequence, dimension, chain, rng, max_iterations=5000):
    """
    Local search using simulated annealing with segment reversals of varying lengths.
    """
    n = len(chain)
    if n < 5:
        return chain

    current_chain = list(chain)
    current_energy = _energy_correct(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    # Simulated Annealing parameters
    T = 3.0
    T_min = 0.01
    alpha = 0.9997

    for iteration in range(max_iterations):
        if _time_left() < 5.0:
            break

        # Choose segment length with a distribution favoring smaller segments
        # but allowing larger ones for better exploration
        max_seg_len = min(n - 2, 20)
        if max_seg_len < 2:
            continue

        # Use a power-law distribution: mostly small, occasionally large
        r = rng.random()
        if r < 0.4:
            seg_len = rng.randint(2, min(4, max_seg_len))
        elif r < 0.7:
            seg_len = rng.randint(5, min(10, max_seg_len))
        else:
            seg_len = rng.randint(11, max_seg_len)

        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1

        # Check validity before applying
        if not _check_segment_reversal_valid(current_chain, start_idx, end_idx):
            continue

        new_chain = _apply_segment_reversal(current_chain, start_idx, end_idx)
        new_energy = _energy_correct(sequence, new_chain)

        # Metropolis Acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-9)):
            current_chain = new_chain
            current_energy = new_energy

            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy

        T *= alpha
        if T < T_min:
            T = T_min

    return best_chain


def _quantum_tunneling(sequence, dimension, chain, rng, max_perturbations=50):
    """
    Large-scale conformational changes to escape local minima.
    Performs random segment reversals of large segments, accepting only
    if the result is valid, then runs local search from the new state.
    """
    n = len(chain)
    if n < 10:
        return chain

    best_chain = list(chain)
    best_energy = _energy_correct(sequence, best_chain)

    for _ in range(max_perturbations):
        if _time_left() < 10.0:
            break

        # Choose a large segment to reverse
        min_seg = max(5, n // 5)
        max_seg = min(n - 2, n // 2)
        if min_seg >= max_seg:
            continue

        seg_len = rng.randint(min_seg, max_seg)
        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1

        # Check validity
        if not _check_segment_reversal_valid(best_chain, start_idx, end_idx):
            continue

        perturbed_chain = _apply_segment_reversal(best_chain, start_idx, end_idx)
        perturbed_energy = _energy_correct(sequence, perturbed_chain)

        # Run local search from the perturbed state
        improved = _local_search(sequence, dimension, perturbed_chain, rng, max_iterations=2000)
        improved_energy = _energy_correct(sequence, improved)

        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy

    return best_chain


def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]

    best_chain = None
    best_energy = 0

    # Time budget for this sequence
    seq_start = time.time()
    time_budget = min(120.0, max(15.0, n * 0.8))

    # Phase 1: Beam search with different widths
    # Use multiple beam searches with varying widths for diversity
    beam_configs = [
        (100, 12345),
        (200, 54321),
        (300, 98765),
        (150, 11111),
        (250, 22222),
    ]

    for beam_width, seed in beam_configs:
        if time.time() - seq_start > time_budget * 0.6:
            break
        if _time_left() < 15.0:
            break

        chain = _beam_search(sequence, dimension, beam_width=beam_width)
        if chain is None or len(chain) != n:
            continue

        energy = _energy_correct(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Local search on the best chain
    if best_chain is not None and _time_left() > 15.0:
        # Run local search with different seeds
        rng = random.Random(42)
        improved = _local_search(sequence, dimension, best_chain, rng, max_iterations=5000)
        improved_energy = _energy_correct(sequence, improved)
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy

        # Second pass with different seed if time remains
        if _time_left() > 10.0:
            rng2 = random.Random(123)
            improved2 = _local_search(sequence, dimension, best_chain, rng2, max_iterations=5000)
            improved2_energy = _energy_correct(sequence, improved2)
            if improved2_energy < best_energy:
                best_chain = improved2
                best_energy = improved2_energy

        # Phase 3: Quantum tunneling for large-scale exploration
        if _time_left() > 20.0:
            rng3 = random.Random(777)
            tunneled = _quantum_tunneling(sequence, dimension, best_chain, rng3, max_perturbations=30)
            tunneled_energy = _energy_correct(sequence, tunneled)
            if tunneled_energy < best_energy:
                best_chain = tunneled
                best_energy = tunneled_energy

            # Final local search polish
            if _time_left() > 10.0:
                rng4 = random.Random(999)
                final = _local_search(sequence, dimension, best_chain, rng4, max_iterations=3000)
                final_energy = _energy_correct(sequence, final)
                if final_energy < best_energy:
                    best_chain = final
                    best_energy = final_energy

    # Fallback: straight line if no valid chain found
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _energy_correct(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

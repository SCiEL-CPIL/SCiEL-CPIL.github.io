"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

# Precompute neighbor offsets for 3D cubic lattice
_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 315.0  # safety margin before 330s kill


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

                gain = 0
                if seq_i == 'H':
                    for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                        q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1

                new_energy = energy_so_far - gain

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

        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

    if not beam:
        return None

    best_energy, best_chain_tuple, _ = beam[0]
    return list(best_chain_tuple)


def _fast_local_search(sequence, dimension, chain, rng, max_iterations=10000):
    """
    Optimized local search using incremental energy updates for segment reversals.
    """
    n = len(chain)
    if n < 5:
        return chain

    current_chain = [tuple(p) for p in chain]
    current_energy = _energy_correct(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    # Precompute positions for fast lookup
    # We will maintain a dictionary pos -> index for the current chain
    # But since we only reverse segments, we can update it incrementally or just rebuild for the move check?
    # Rebuilding is O(N). Move check is O(1) for connectivity + O(L_seg) for energy delta.
    # Let's stick to the O(L_seg) energy delta calculation.
    
    T = 2.0
    T_min = 0.005
    alpha = 0.9999 # Slower cooling for more exploration

    for iteration in range(max_iterations):
        if _time_left() < 5.0:
            break

        # Choose segment length
        # Bias towards shorter segments for efficiency, but allow longer ones for basin jumping
        r = rng.random()
        if r < 0.6:
            seg_len = rng.randint(2, 5)
        elif r < 0.9:
            seg_len = rng.randint(5, 10)
        else:
            seg_len = rng.randint(10, min(20, n - 2))
            
        seg_len = min(seg_len, n - 2)
        if seg_len < 2:
            continue

        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1

        # 1. Check Connectivity (O(1))
        # After reversal:
        # C[start_idx-1] connects to C[end_idx] (old)
        # C[start_idx] (old) connects to C[end_idx+1]
        
        p_left = current_chain[start_idx - 1]
        p_right = current_chain[end_idx + 1]
        p_new_start = current_chain[end_idx]
        p_new_end = current_chain[start_idx]

        if sum(abs(a - b) for a, b in zip(p_left, p_new_start)) != 1:
            continue
        if sum(abs(a - b) for a, b in zip(p_new_end, p_right)) != 1:
            continue

        # 2. Calculate Energy Delta
        # We need to check contacts involving indices in [start_idx, end_idx]
        # and their neighbors in the chain (indices start_idx-1, end_idx+1)
        # AND spatial neighbors.
        
        # To do this efficiently, we can iterate over the residues in the segment
        # and check their spatial neighbors.
        
        # Old Energy Contribution:
        old_delta = 0
        # New Energy Contribution:
        new_delta = 0
        
        # We need a map from position to index for the CURRENT chain to find spatial neighbors quickly?
        # Yes, we need it.
        pos_to_idx = {p: i for i, p in enumerate(current_chain)}
        
        # Helper to check if a pair (i, j) is a valid contact (non-consecutive, H-H)
        def is_contact(i, j):
            if abs(i - j) <= 1:
                return False
            return sequence[i] == 'H' and sequence[j] == 'H'

        # Iterate over indices in the segment
        for k in range(start_idx, end_idx + 1):
            pos_k = current_chain[k]
            # Find spatial neighbors of pos_k
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                neighbor_pos = (pos_k[0] + dx, pos_k[1] + dy, pos_k[2] + dz)
                j = pos_to_idx.get(neighbor_pos)
                if j is None:
                    continue
                # Only count if j is "outside" the segment or if j > k to avoid double counting?
                # Actually, we are calculating the total change.
                # Let's just sum all contacts involving k that are affected.
                # A contact is affected if one endpoint is in [start_idx, end_idx].
                # We must avoid double counting pairs where BOTH are in the segment.
                
                # Strategy:
                # For each k in segment, for each spatial neighbor j:
                # If j is in segment:
                #   Only count if j > k (to count each pair once).
                #   Old: is_contact(k, j)
                #   New: is_contact(new_idx_k, new_idx_j)
                # If j is NOT in segment:
                #   Count it.
                #   Old: is_contact(k, j)
                #   New: is_contact(new_idx_k, j)
                
                if start_idx <= j <= end_idx:
                    if j <= k:
                        continue # Skip, will be counted when k' = j
                    # Both in segment
                    # Old residues: k, j
                    if is_contact(k, j):
                        old_delta += 1
                    # New residues: 
                    # k -> new_k = start_idx + end_idx - k
                    # j -> new_j = start_idx + end_idx - j
                    new_k = start_idx + end_idx - k
                    new_j = start_idx + end_idx - j
                    if is_contact(new_k, new_j):
                        new_delta += 1
                else:
                    # j outside segment
                    if is_contact(k, j):
                        old_delta += 1
                    # New residue at pos_k is new_k
                    new_k = start_idx + end_idx - k
                    if is_contact(new_k, j):
                        new_delta += 1

        delta_e = new_delta - old_delta
        new_energy = current_energy - delta_e # Energy is negative contacts, so gain in contacts = decrease in energy
        
        # Wait, _energy_correct returns -contacts.
        # So if new_delta > old_delta, new_energy is lower (more negative).
        # new_energy = current_energy + (old_delta - new_delta)
        new_energy = current_energy - (new_delta - old_delta)

        if delta_e < 0 or rng.random() < math.exp(-(new_energy - current_energy) / max(T, 1e-9)):
            # Accept move
            # Apply reversal
            current_chain[start_idx:end_idx + 1] = current_chain[start_idx:end_idx + 1][::-1]
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy

        T = max(T * alpha, T_min)

    return best_chain


def _quantum_tunneling(sequence, dimension, chain, rng, max_jumps=50):
    """
    Quantum-inspired tunneling: probabilistically jump to distant conformations
    by performing multiple large-scale perturbations, then local search to recover.
    """
    n = len(chain)
    if n < 10:
        return chain

    best_chain = list(chain)
    best_energy = _energy_correct(sequence, chain)

    for jump in range(max_jumps):
        if _time_left() < 10.0:
            break

        # Create a perturbed version
        candidate = [tuple(p) for p in chain]
        
        # Large-scale perturbation: reverse a large random segment
        if n > 12:
            max_seg = min(n - 2, max(10, n // 3))
            seg_len = rng.randint(max(5, n // 4), max_seg)
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1
            
            # Check validity quickly
            p_left = candidate[start_idx - 1]
            p_right = candidate[end_idx + 1]
            p_new_start = candidate[end_idx]
            p_new_end = candidate[start_idx]
            
            if sum(abs(a - b) for a, b in zip(p_left, p_new_start)) == 1 and \
               sum(abs(a - b) for a, b in zip(p_new_end, p_right)) == 1:
                candidate[start_idx:end_idx + 1] = candidate[start_idx:end_idx + 1][::-1]
            else:
                # Try a different segment
                for _ in range(5):
                    seg_len = rng.randint(max(5, n // 4), max_seg)
                    start_idx = rng.randint(1, n - seg_len - 1)
                    end_idx = start_idx + seg_len - 1
                    p_left = candidate[start_idx - 1]
                    p_right = candidate[end_idx + 1]
                    p_new_start = candidate[end_idx]
                    p_new_end = candidate[start_idx]
                    if sum(abs(a - b) for a, b in zip(p_left, p_new_start)) == 1 and \
                       sum(abs(a - b) for a, b in zip(p_new_end, p_right)) == 1:
                        candidate[start_idx:end_idx + 1] = candidate[start_idx:end_idx + 1][::-1]
                        break
        
        # Additional smaller perturbations
        for _ in range(rng.randint(2, 5)):
            seg_len = rng.randint(2, min(8, n - 2))
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1
            p_left = candidate[start_idx - 1]
            p_right = candidate[end_idx + 1]
            p_new_start = candidate[end_idx]
            p_new_end = candidate[start_idx]
            if sum(abs(a - b) for a, b in zip(p_left, p_new_start)) == 1 and \
               sum(abs(a - b) for a, b in zip(p_new_end, p_right)) == 1:
                candidate[start_idx:end_idx + 1] = candidate[start_idx:end_idx + 1][::-1]

        # Local search to recover
        local_rng = random.Random(rng.randint(0, 2**31))
        improved = _fast_local_search(sequence, dimension, candidate, local_rng, max_iterations=500)
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

    seq_start = time.time()
    # Time budget: shorter for small chains, longer for large ones
    time_budget = min(110.0, max(12.0, n * 0.7))

    # Phase 1: Beam search with multiple configurations for diversity
    beam_configs = [
        (150, 12345),
        (250, 54321),
        (350, 98765),
        (200, 11111),
        (300, 22222),
        (400, 33333),
    ]

    for beam_width, seed in beam_configs:
        if time.time() - seq_start > time_budget * 0.4:
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

    # Phase 2: Local search refinement
    if best_chain is not None and _time_left() > 15.0:
        # Multiple local search passes with different seeds
        for ls_seed in [42, 123, 456, 789]:
            if _time_left() < 10.0:
                break
            rng = random.Random(ls_seed)
            improved = _fast_local_search(sequence, dimension, best_chain, rng, max_iterations=10000)
            improved_energy = _energy_correct(sequence, improved)
            if improved_energy < best_energy:
                best_chain = improved
                best_energy = improved_energy

    # Phase 3: Quantum-inspired tunneling for basin jumping
    if best_chain is not None and _time_left() > 20.0:
        tunnel_rng = random.Random(999)
        tunneled = _quantum_tunneling(sequence, dimension, best_chain, tunnel_rng, max_jumps=30)
        tunneled_energy = _energy_correct(sequence, tunneled)
        if tunneled_energy < best_energy:
            best_chain = tunneled
            best_energy = tunneled_energy

    # Phase 4: Final polish if time remains
    if best_chain is not None and _time_left() > 15.0:
        polish_rng = random.Random(7777)
        polished = _fast_local_search(sequence, dimension, best_chain, polish_rng, max_iterations=20000)
        polished_energy = _energy_correct(sequence, polished)
        if polished_energy < best_energy:
            best_chain = polished
            best_energy = polished_energy

    # Fallback: straight line if no valid chain found
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _energy_correct(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

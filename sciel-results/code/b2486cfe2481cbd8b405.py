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


def _check_segment_reversal_valid(chain, start_idx, end_idx):
    """Check if reversing segment [start_idx, end_idx] produces a valid conformation."""
    n = len(chain)
    if start_idx < 0 or end_idx >= n or start_idx >= end_idx:
        return False
    
    # Build set of positions outside the segment
    outside = set()
    for i in range(n):
        if i < start_idx or i > end_idx:
            outside.add(tuple(chain[i]))
    
    # Check reversed segment for collisions with outside
    for i in range(start_idx, end_idx + 1):
        new_pos = tuple(chain[end_idx + start_idx - i])
        if new_pos in outside:
            return False
    
    # Check connectivity at boundaries
    if start_idx > 0:
        p1 = tuple(chain[start_idx - 1])
        p2 = tuple(chain[end_idx])  # chain[start_idx] after reversal = old chain[end_idx]
        if sum(abs(a - b) for a, b in zip(p1, p2)) != 1:
            return False
    
    if end_idx < n - 1:
        p1 = tuple(chain[end_idx])  # after reversal = old chain[end_idx]
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

    T = 2.0
    T_min = 0.005
    alpha = 0.9995

    for iteration in range(max_iterations):
        if _time_left() < 5.0:
            break

        # Choose segment length with bias toward shorter segments
        max_seg_len = min(n - 2, 16)
        if max_seg_len < 2:
            continue

        # Use a distribution that favors shorter segments but allows longer ones
        r = rng.random()
        if r < 0.5:
            seg_len = rng.randint(2, 4)
        elif r < 0.8:
            seg_len = rng.randint(4, 8)
        else:
            seg_len = rng.randint(8, max_seg_len)
        
        seg_len = min(seg_len, max_seg_len)
        
        if seg_len < 2:
            continue

        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1

        if not _check_segment_reversal_valid(current_chain, start_idx, end_idx):
            continue

        new_chain = _apply_segment_reversal(current_chain, start_idx, end_idx)
        new_energy = _energy_correct(sequence, new_chain)

        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-9)):
            current_chain = new_chain
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
    This helps escape deep local minima that trap classical local search.
    """
    n = len(chain)
    if n < 10:
        return chain

    best_chain = list(chain)
    best_energy = _energy_correct(sequence, chain)

    for jump in range(max_jumps):
        if _time_left() < 10.0:
            break

        # Create a perturbed version by reversing a large segment
        # and/or performing multiple smaller reversals
        candidate = list(chain)
        
        # Large-scale perturbation: reverse a large random segment
        if n > 12:
            max_seg = min(n - 2, max(10, n // 3))
            seg_len = rng.randint(max(5, n // 4), max_seg)
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1
            
            if _check_segment_reversal_valid(candidate, start_idx, end_idx):
                candidate = _apply_segment_reversal(candidate, start_idx, end_idx)
            else:
                # Try a different segment
                for _ in range(5):
                    seg_len = rng.randint(max(5, n // 4), max_seg)
                    start_idx = rng.randint(1, n - seg_len - 1)
                    end_idx = start_idx + seg_len - 1
                    if _check_segment_reversal_valid(candidate, start_idx, end_idx):
                        candidate = _apply_segment_reversal(candidate, start_idx, end_idx)
                        break
        
        # Additional smaller perturbations to further diversify
        for _ in range(rng.randint(2, 5)):
            seg_len = rng.randint(2, min(8, n - 2))
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1
            if _check_segment_reversal_valid(candidate, start_idx, end_idx):
                candidate = _apply_segment_reversal(candidate, start_idx, end_idx)

        # Local search to recover from the perturbation
        local_rng = random.Random(rng.randint(0, 2**31))
        improved = _local_search(sequence, dimension, candidate, local_rng, max_iterations=200)
        improved_energy = _energy_correct(sequence, improved)

        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy

    return best_chain


def _contact_map_prediction(sequence):
    """
    Predict a likely contact map using a greedy approach based on chain distance.
    Returns a set of (i, j) pairs representing predicted H-H contacts.
    """
    n = len(sequence)
    h_positions = [i for i in range(n) if sequence[i] == 'H']
    
    # For each pair of H positions, compute a "desirability" score
    # based on chain distance (closer in chain = more likely to be in contact
    # in a compact fold, but not too close due to self-avoidance)
    contacts = set()
    h_count = len(h_positions)
    
    if h_count < 2:
        return contacts
    
    # Score pairs by chain distance
    pair_scores = []
    for idx_i in range(h_count):
        for idx_j in range(idx_i + 1, h_count):
            i = h_positions[idx_i]
            j = h_positions[idx_j]
            dist = j - i
            
            # Skip adjacent residues (bonded)
            if dist <= 1:
                continue
            
            # Skip very close residues (hard to contact without self-intersection)
            if dist <= 3:
                continue
            
            # Score: prefer medium distances (around 5-15) for compact folds
            # Use a Gaussian-like peak around optimal distance
            optimal = min(10, n // 5)
            score = -((dist - optimal) ** 2) / (2 * (optimal * 0.5) ** 2)
            
            # Bonus for closer H residues in sequence (more likely to fold back)
            score += 0.1 * (1.0 / (1.0 + dist))
            
            pair_scores.append((score, i, j))
    
    # Sort by score descending and greedily select non-conflicting contacts
    pair_scores.sort(key=lambda x: -x[0])
    
    # Track which positions are already in a contact
    in_contact = [0] * n
    
    for score, i, j in pair_scores:
        if in_contact[i] == 0 and in_contact[j] == 0:
            contacts.add((i, j))
            in_contact[i] = 1
            in_contact[j] = 1
            # Limit number of contacts to avoid over-constraining
            if len(contacts) >= min(h_count // 2, 15):
                break
    
    return contacts


def _constraint_satisfaction_fold(sequence, dimension, predicted_contacts, rng, max_iterations=20000):
    """
    Try to find a self-avoiding walk that realizes the predicted contact map
    using a randomized backtracking / constraint satisfaction approach.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Build contact constraints
    contact_dict = {}
    for i, j in predicted_contacts:
        contact_dict.setdefault(i, []).append(j)
        contact_dict.setdefault(j, []).append(i)
    
    best_chain = None
    best_score = -float('inf')
    
    for attempt in range(20):
        if _time_left() < 15.0:
            break
        
        chain = [(0, 0, 0)]
        occupied = {(0, 0, 0): 0}
        valid = True
        
        for i in range(1, n):
            if _time_left() < 10.0:
                valid = False
                break
            
            last = chain[-1]
            candidates = []
            
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                site = (last[0] + dx, last[1] + dy, last[2] + dz)
                if site in occupied:
                    continue
                
                # Score this position based on potential contacts
                score = 0
                if sequence[i] == 'H':
                    for j in contact_dict.get(i, []):
                        if j < i:
                            # Check if j's position is adjacent to site
                            p_j = chain[j]
                            if sum(abs(a - b) for a, b in zip(site, p_j)) == 1:
                                score += 10
                        # Also check future H positions that might be near
                    # Check proximity to other H positions already placed
                    for j in range(i - 1):
                        if sequence[j] == 'H' and j not in contact_dict.get(i, []):
                            p_j = chain[j]
                            if sum(abs(a - b) for a, b in zip(site, p_j)) == 1:
                                score += 5
                
                candidates.append((score, site))
            
            if not candidates:
                valid = False
                break
            
            # Sort by score descending and pick with some randomness
            candidates.sort(key=lambda x: -x[0])
            
            # Pick from top candidates with probability bias
            top_k = min(3, len(candidates))
            weights = [1.0 / (k + 1) for k in range(top_k)]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen = candidates[0][1]
            for k in range(top_k):
                cumulative += weights[k]
                if r <= cumulative:
                    chosen = candidates[k][1]
                    break
            
            chain.append(chosen)
            occupied[chosen] = i
        
        if not valid or len(chain) != n:
            continue
        
        # Evaluate this chain
        energy = _energy_correct(sequence, chain)
        
        # Score: how many predicted contacts are realized
        realized = 0
        where = {p: i for i, p in enumerate(chain)}
        for i, j in predicted_contacts:
            p_i = chain[i]
            p_j = chain[j]
            if sum(abs(a - b) for a, b in zip(p_i, p_j)) == 1:
                realized += 1
        
        # Combined score: prioritize energy, then contact realization
        score = energy * 10 + realized
        
        if score > best_score:
            best_score = score
            best_chain = chain
    
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

    # Phase 1: Contact map prediction
    predicted_contacts = _contact_map_prediction(sequence)
    
    # Phase 2: Try constraint satisfaction with predicted contacts
    if predicted_contacts and _time_left() > 30.0:
        cs_rng = random.Random(12345)
        cs_chain = _constraint_satisfaction_fold(sequence, dimension, predicted_contacts, cs_rng, max_iterations=10000)
        if cs_chain is not None and len(cs_chain) == n:
            energy = _energy_correct(sequence, cs_chain)
            if best_chain is None or energy < best_energy:
                best_chain = cs_chain
                best_energy = energy
    
    # Phase 3: Beam search with multiple configurations for diversity
    beam_configs = [
        (150, 12345),
        (250, 54321),
        (350, 98765),
        (200, 11111),
        (300, 22222),
        (400, 33333),
    ]

    for beam_width, seed in beam_configs:
        if time.time() - seq_start > time_budget * 0.5:
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

    # Phase 4: Local search refinement
    if best_chain is not None and _time_left() > 15.0:
        # Multiple local search passes with different seeds
        for ls_seed in [42, 123, 456, 789]:
            if _time_left() < 10.0:
                break
            rng = random.Random(ls_seed)
            improved = _local_search(sequence, dimension, best_chain, rng, max_iterations=3000)
            improved_energy = _energy_correct(sequence, improved)
            if improved_energy < best_energy:
                best_chain = improved
                best_energy = improved_energy

    # Phase 5: Quantum-inspired tunneling for basin jumping
    if best_chain is not None and _time_left() > 20.0:
        tunnel_rng = random.Random(999)
        tunneled = _quantum_tunneling(sequence, dimension, best_chain, tunnel_rng, max_jumps=30)
        tunneled_energy = _energy_correct(sequence, tunneled)
        if tunneled_energy < best_energy:
            best_chain = tunneled
            best_energy = tunneled_energy

    # Phase 6: Final polish if time remains
    if best_chain is not None and _time_left() > 15.0:
        polish_rng = random.Random(7777)
        polished = _local_search(sequence, dimension, best_chain, polish_rng, max_iterations=5000)
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

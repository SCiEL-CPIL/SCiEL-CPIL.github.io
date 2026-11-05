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


def _predict_contact_map(sequence):
    """
    Predict a soft contact map using chain-distance-based heuristic.
    Returns a dict mapping (i, j) -> weight for i < j, where both are H
    and j - i > 1. Higher weight means more likely to be a contact.
    """
    n = len(sequence)
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    contact_weights = {}
    
    # Heuristic: closer H-H pairs in sequence are more likely to be spatially close
    # but not too close (need j - i > 1 for non-bonded contact)
    for idx_a, i in enumerate(h_indices):
        for j in h_indices[idx_a + 1:]:
            dist = j - i
            if dist <= 1:
                continue
            # Weight decreases with chain distance, but has a sweet spot
            # Contacts are more likely at moderate distances (3-15)
            if dist <= 3:
                w = 0.3
            elif dist <= 8:
                w = 1.0
            elif dist <= 15:
                w = 0.8
            elif dist <= 25:
                w = 0.5
            else:
                w = 0.2
            contact_weights[(i, j)] = w
    
    return contact_weights


def _guided_energy(sequence, chain, contact_map, beta=0.3):
    """
    Compute a guided energy that combines actual contacts with predicted contacts.
    """
    actual_energy = _energy_correct(sequence, chain)
    
    where = {p: i for i, p in enumerate(chain)}
    predicted_reward = 0.0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                key = (i, j) if i < j else (j, i)
                predicted_reward += contact_map.get(key, 0.0)
    
    return actual_energy - beta * predicted_reward


def _beam_search(sequence, dimension, beam_width=200, contact_map=None):
    """
    Beam search for HP protein folding, optionally guided by a contact map.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    use_guidance = contact_map is not None

    # Beam: list of (score, chain_tuple, occupied_dict)
    beam = [(0, ((0, 0, 0),), {(0, 0, 0): 0})]

    for i in range(1, n):
        if _time_left() < 10.0:
            break

        new_beam = []
        seen = set()

        for score_so_far, chain_tuple, occupied in beam:
            last = chain_tuple[-1]
            seq_i = sequence[i]

            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                site = (last[0] + dx, last[1] + dy, last[2] + dz)
                if site in occupied:
                    continue

                # Compute new contacts gained
                gain = 0
                guided_gain = 0.0
                if seq_i == 'H':
                    for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                        q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                            if use_guidance:
                                key = (j, i) if j < i else (i, j)
                                guided_gain += contact_map.get(key, 0.0)

                # New score: lower is better (more negative = more contacts)
                if use_guidance:
                    new_score = score_so_far - gain - 0.3 * guided_gain
                else:
                    new_score = score_so_far - gain

                # Deduplicate
                key = (site, chain_tuple[-3:] if len(chain_tuple) >= 3 else chain_tuple)
                if key in seen:
                    continue
                seen.add(key)

                new_chain = chain_tuple + (site,)
                new_occupied = dict(occupied)
                new_occupied[site] = i

                new_beam.append((new_score, new_chain, new_occupied))

        if not new_beam:
            break

        # Sort by score (most negative first) and keep top beam_width
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

    if not beam:
        return None

    # Return the best complete chain
    _, best_chain_tuple, _ = beam[0]
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
        new_pos = tuple(chain[start_idx + end_idx - i])
        if new_pos in outside_positions:
            return False
    
    # Check connectivity at boundaries
    if start_idx > 0:
        p1 = tuple(chain[start_idx - 1])
        p2 = tuple(chain[end_idx])
        if sum(abs(a - b) for a, b in zip(p1, p2)) != 1:
            return False
    
    if end_idx < n - 1:
        p1 = tuple(chain[start_idx])
        p2 = tuple(chain[end_idx + 1])
        if sum(abs(a - b) for a, b in zip(p1, p2)) != 1:
            return False
    
    return True


def _apply_segment_reversal(chain, start_idx, end_idx):
    """Apply segment reversal and return new chain."""
    new_chain = chain[:]
    new_chain[start_idx:end_idx + 1] = new_chain[start_idx:end_idx + 1][::-1]
    return new_chain


def _fast_local_search(sequence, dimension, chain, rng, max_iterations=5000, contact_map=None):
    """
    Fast local search using incremental energy updates.
    """
    n = len(chain)
    if n < 5:
        return chain

    current_chain = [tuple(p) for p in chain]
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    # Initial energy calculation
    current_energy = _energy_correct(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    use_guidance = contact_map is not None
    if use_guidance:
        current_guided = _guided_energy(sequence, current_chain, contact_map)
        best_guided = current_guided

    T = 3.0
    T_min = 0.01
    alpha = 0.9997

    for iteration in range(max_iterations):
        if _time_left() < 5.0:
            break

        max_seg_len = min(n - 2, 20)
        if max_seg_len < 2:
            continue

        r = rng.random()
        if r < 0.4:
            seg_len = rng.randint(2, min(4, max_seg_len))
        elif r < 0.7:
            seg_len = rng.randint(5, min(10, max_seg_len))
        else:
            seg_len = rng.randint(11, max_seg_len)

        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1

        # Check validity and calculate delta
        # 1. Check connectivity at boundaries
        valid = True
        if start_idx > 0:
            p1 = current_chain[start_idx - 1]
            p2 = current_chain[end_idx]
            if sum(abs(a - b) for a, b in zip(p1, p2)) != 1:
                valid = False
        if valid and end_idx < n - 1:
            p1 = current_chain[start_idx]
            p2 = current_chain[end_idx + 1]
            if sum(abs(a - b) for a, b in zip(p1, p2)) != 1:
                valid = False
        
        if not valid:
            continue

        # 2. Check collisions with outside
        # Get new positions for segment
        new_positions = []
        for i in range(start_idx, end_idx + 1):
            new_pos = current_chain[start_idx + end_idx - i]
            new_positions.append(new_pos)
        
        # Check if any new position collides with outside
        # We need to check against all positions except those in the segment
        # Since we are reversing, the set of positions is the same, but assigned to different indices.
        # Wait, the positions themselves are the same set!
        # Reversing a segment of a self-avoiding walk:
        # The set of lattice points occupied by the segment is unchanged.
        # So, if the original was self-avoiding, the reversed segment is also self-avoiding internally.
        # The only potential collision is with the OUTSIDE.
        # But the positions are the same! So if it didn't collide before, it won't collide now.
        # EXCEPT: The connectivity at the boundaries changes.
        # So, if the boundary connectivity is valid, the move is ALWAYS valid?
        # Let's verify.
        # Original: ... A - B - ... - C - D ...
        # Segment B...C is reversed.
        # New: ... A - C' - ... - B' - D ...
        # Positions of B...C are the same set.
        # So no new collisions with outside.
        # So, we only need to check boundary connectivity!
        
        # This is a huge optimization. We don't need to check collisions at all for segment reversal.
        
        # Calculate Delta Energy
        # Remove segment from occupied
        for i in range(start_idx, end_idx + 1):
            del occupied[current_chain[i]]
            
        old_boundary_contacts = 0
        if use_guidance:
            old_guided_reward = 0.0
            
        for i in range(start_idx, end_idx + 1):
            if sequence[i] != 'H':
                continue
            pos = current_chain[i]
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                j = occupied.get(q)
                if j is not None:
                    # j is outside segment
                    if sequence[j] == 'H':
                        old_boundary_contacts += 1
                        if use_guidance:
                            key = (i, j) if i < j else (j, i)
                            old_guided_reward += contact_map.get(key, 0.0)

        # Apply reversal to chain
        for i in range(start_idx, end_idx + 1):
            current_chain[i] = new_positions[i - start_idx]
            occupied[current_chain[i]] = i

        new_boundary_contacts = 0
        if use_guidance:
            new_guided_reward = 0.0
            
        for i in range(start_idx, end_idx + 1):
            if sequence[i] != 'H':
                continue
            pos = current_chain[i]
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                j = occupied.get(q)
                if j is not None:
                    # j is outside segment (since we re-added segment to occupied, we must exclude segment indices)
                    # Wait, occupied now contains the segment.
                    # We need to check if j is OUTSIDE the segment.
                    if (j < start_idx or j > end_idx):
                        if sequence[j] == 'H':
                            new_boundary_contacts += 1
                            if use_guidance:
                                key = (i, j) if i < j else (j, i)
                                new_guided_reward += contact_map.get(key, 0.0)

        delta_e = -(new_boundary_contacts - old_boundary_contacts)
        
        if use_guidance:
            delta_g = -(new_guided_reward - old_guided_reward)
            total_delta = delta_e - 0.3 * delta_g
            if total_delta < 0 or rng.random() < math.exp(-total_delta / max(T, 1e-9)):
                current_energy += delta_e
                current_guided += total_delta
                if current_energy < best_energy:
                    best_chain = list(current_chain)
                    best_energy = current_energy
                    best_guided = current_guided
        else:
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-9)):
                current_energy += delta_e
                if current_energy < best_energy:
                    best_chain = list(current_chain)
                    best_energy = current_energy

        T *= alpha
        if T < T_min:
            T = T_min

    return best_chain


def _local_search(sequence, dimension, chain, rng, max_iterations=5000, contact_map=None):
    """
    Wrapper to keep interface compatibility.
    """
    return _fast_local_search(sequence, dimension, chain, rng, max_iterations, contact_map)


def _quantum_tunneling(sequence, dimension, chain, rng, max_perturbations=50, contact_map=None):
    """
    Large-scale conformational changes to escape local minima.
    """
    n = len(chain)
    if n < 10:
        return chain

    best_chain = list(chain)
    best_energy = _energy_correct(sequence, best_chain)

    for _ in range(max_perturbations):
        if _time_left() < 10.0:
            break

        min_seg = max(5, n // 5)
        max_seg = min(n - 2, n // 2)
        if min_seg >= max_seg:
            continue

        seg_len = rng.randint(min_seg, max_seg)
        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1

        # Check boundary validity for tunneling move
        valid = True
        if start_idx > 0:
            p1 = tuple(best_chain[start_idx - 1])
            p2 = tuple(best_chain[end_idx])
            if sum(abs(a - b) for a, b in zip(p1, p2)) != 1:
                valid = False
        if valid and end_idx < n - 1:
            p1 = tuple(best_chain[start_idx])
            p2 = tuple(best_chain[end_idx + 1])
            if sum(abs(a - b) for a, b in zip(p1, p2)) != 1:
                valid = False
        
        if not valid:
            continue

        perturbed_chain = _apply_segment_reversal(best_chain, start_idx, end_idx)

        improved = _fast_local_search(sequence, dimension, perturbed_chain, rng, max_iterations=2000, contact_map=contact_map)
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
    time_budget = min(120.0, max(15.0, n * 0.8))

    # Phase 0: Predict contact map
    contact_map = _predict_contact_map(sequence)

    # Phase 1: Beam search with different widths, some guided
    # Reduced to 3 runs to save time for local search
    beam_configs = [
        (150, 12345, True),
        (200, 54321, True),
        (250, 98765, True),
    ]

    for beam_width, seed, use_guidance in beam_configs:
        if time.time() - seq_start > time_budget * 0.4:
            break
        if _time_left() < 15.0:
            break

        cm = contact_map if use_guidance else None
        chain = _beam_search(sequence, dimension, beam_width=beam_width, contact_map=cm)
        if chain is None or len(chain) != n:
            continue

        energy = _energy_correct(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Local search on the best chain, guided
    if best_chain is not None and _time_left() > 15.0:
        rng = random.Random(42)
        improved = _fast_local_search(sequence, dimension, best_chain, rng, max_iterations=5000, contact_map=contact_map)
        improved_energy = _energy_correct(sequence, improved)
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy

        # Second pass with different seed
        if _time_left() > 10.0:
            rng2 = random.Random(123)
            improved2 = _fast_local_search(sequence, dimension, best_chain, rng2, max_iterations=5000, contact_map=contact_map)
            improved2_energy = _energy_correct(sequence, improved2)
            if improved2_energy < best_energy:
                best_chain = improved2
                best_energy = improved2_energy

        # Phase 3: Quantum tunneling for large-scale exploration
        if _time_left() > 20.0:
            rng3 = random.Random(777)
            tunneled = _quantum_tunneling(sequence, dimension, best_chain, rng3, max_perturbations=30, contact_map=contact_map)
            tunneled_energy = _energy_correct(sequence, tunneled)
            if tunneled_energy < best_energy:
                best_chain = tunneled
                best_energy = tunneled_energy

            # Final local search polish - unguided for precision
            if _time_left() > 10.0:
                rng4 = random.Random(999)
                final = _fast_local_search(sequence, dimension, best_chain, rng4, max_iterations=3000, contact_map=None)
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

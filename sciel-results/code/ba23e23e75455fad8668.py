# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 195.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, temperature=0.0):
    """
    Randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then sites with more free space.
    Avoids dead ends for the next step.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Avoid dead ends if there are more residues to place
            if i < n - 1:
                has_exit = False
                for q in _neighbours(site):
                    if q not in occupied and q != last:
                        has_exit = True
                        break
                if not has_exit:
                    continue

            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.5 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _greedy_grow_biased(sequence, dimension, rng, temperature=0.0, bias_center=False):
    """
    Greedy growth with optional bias toward the center of the bounding box.
    This helps create more compact conformations.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            center_bias = 0.0
            if bias_center and i > n * 0.3:
                # Bias toward center of current bounding box
                xs = [p[0] for p in chain]
                ys = [p[1] for p in chain]
                cx = (min(xs) + max(xs)) / 2.0
                cy = (min(ys) + max(ys)) / 2.0
                dist_to_center = abs(site[0] - cx) + abs(site[1] - cy)
                center_bias = -dist_to_center * 0.5

            score = gain * 10.0 + free * 0.1 + center_bias + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _greedy_grow_from_fixed(suffix_seq, start_pos, prefix, full_sequence, rng, temperature=0.0):
    """
    Greedy growth for suffix, with proper access to full sequence for H checks.
    """
    if not suffix_seq:
        return []

    # Build occupied set from prefix with their global indices
    occupied = {p: i for i, p in enumerate(prefix)}
    chain = [start_pos]

    for i, res in enumerate(suffix_seq):
        global_idx = len(prefix) + i
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < global_idx - 1 and full_sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = global_idx
        chain.append(best_site)

    return chain[1:]


def _pivot_move(sequence, chain, occupied, rng, pivot_start=2, pivot_end=None):
    """
    Try a pivot move: rotate a subchain around a pivot point.
    Returns (new_chain, new_occupied, new_energy) if successful, else (None, None, None).
    """
    n = len(chain)
    if pivot_end is None:
        pivot_end = n - 2
    
    # Pick a pivot point
    pivot = rng.randint(pivot_start, pivot_end)
    if pivot >= n - 1:
        return None, None, None
    
    # The subchain to rotate is from pivot to some end
    # Try different rotation lengths
    for length in range(2, min(10, n - pivot)):
        end = pivot + length - 1
        if end >= n:
            break
        
        # The pivot point is chain[pivot]
        # The subchain is chain[pivot+1:end+1]
        # We rotate the subchain around chain[pivot]
        
        subchain = chain[pivot+1:end+1]
        sub_len = len(subchain)
        
        # Try rotating in different directions
        for direction in [1, -1]:
            # Generate the rotated subchain
            new_subchain = [subchain[0]]
            valid = True
            occupied_check = dict(occupied)
            # Remove the subchain positions from occupied check
            for pos in subchain:
                if pos in occupied_check:
                    del occupied_check[pos]
            
            for i in range(1, sub_len):
                last = new_subchain[-1]
                # Try all 4 neighbors
                found = False
                for np in _neighbours(last):
                    if np in occupied_check:
                        continue
                    new_subchain.append(np)
                    occupied_check[np] = pivot + 1 + i
                    found = True
                    break
                if not found:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check that the end of the rotated subchain connects to the rest
            if end + 1 < n:
                if abs(new_subchain[-1][0] - chain[end+1][0]) + abs(new_subchain[-1][1] - chain[end+1][1]) != 1:
                    continue
            
            # Check that the start of the rotated subchain connects to the pivot
            if abs(new_subchain[0][0] - chain[pivot][0]) + abs(new_subchain[0][1] - chain[pivot][1]) != 1:
                continue
            
            # Build new chain
            new_chain = chain[:pivot+1] + new_subchain + chain[end+1:]
            
            # Check self-avoidance
            if len(set(new_chain)) != len(new_chain):
                continue
            
            # Compute new energy
            new_occupied = {p: i for i, p in enumerate(new_chain)}
            new_energy = _compute_energy(sequence, new_chain)
            
            return new_chain, new_occupied, new_energy
    
    return None, None, None


def _local_improve(sequence, chain, rng, max_moves=5000):
    """
    Improve conformation via local moves: kink flips, 2-residue moves, short regrows, and pivot moves.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve_count = 0

    for _ in range(max_moves):
        if time.time() > _DEADLINE - 5:
            break

        move_type = rng.random()
        improved = False

        if move_type < 0.3:
            # Kink flip: try to move residue i to a neighbor that maintains connectivity
            i = rng.randint(1, n - 2)
            p = chain[i]
            prev, nxt = chain[i - 1], chain[i + 1]

            candidates = []
            for np in _neighbours(prev):
                if np == p:
                    continue
                if abs(np[0] - nxt[0]) + abs(np[1] - nxt[1]) != 1:
                    continue
                if np in occupied and occupied[np] != i:
                    continue
                candidates.append(np)

            if candidates:
                np = rng.choice(candidates)
                # Perform move
                del occupied[p]
                chain[i] = np
                occupied[np] = i
                new_energy = _compute_energy(sequence, chain)

                if new_energy < current_energy:
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                    improved = True
                else:
                    # Revert
                    del occupied[np]
                    chain[i] = p
                    occupied[p] = i
                    no_improve_count += 1
        elif move_type < 0.45:
            # 2-residue move: try to move two consecutive residues
            if n >= 4:
                i = rng.randint(1, n - 3)
                p1, p2 = chain[i], chain[i + 1]
                prev, nxt = chain[i - 1], chain[i + 2]
                
                candidates = []
                for np1 in _neighbours(prev):
                    if np1 == p1:
                        continue
                    if np1 in occupied and occupied[np1] != i:
                        continue
                    for np2 in _neighbours(np1):
                        if np2 == p2:
                            continue
                        if abs(np2[0] - nxt[0]) + abs(np2[1] - nxt[1]) != 1:
                            continue
                        if np2 in occupied and occupied[np2] != i + 1:
                            continue
                        # Check no collision with other occupied sites
                        if np1 in occupied and occupied[np1] != i:
                            continue
                        if np2 in occupied and occupied[np2] != i + 1:
                            continue
                        candidates.append((np1, np2))
                
                if candidates:
                    np1, np2 = rng.choice(candidates)
                    # Perform move
                    del occupied[p1]
                    del occupied[p2]
                    chain[i] = np1
                    chain[i + 1] = np2
                    occupied[np1] = i
                    occupied[np2] = i + 1
                    new_energy = _compute_energy(sequence, chain)

                    if new_energy < current_energy:
                        current_energy = new_energy
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_chain = list(chain)
                        no_improve_count = 0
                        improved = True
                    else:
                        # Revert
                        del occupied[np1]
                        del occupied[np2]
                        chain[i] = p1
                        chain[i + 1] = p2
                        occupied[p1] = i
                        occupied[p2] = i + 1
                        no_improve_count += 1
        elif move_type < 0.55:
            # 3-residue move: rotate a 3-residue segment
            if n >= 5:
                i = rng.randint(1, n - 4)
                p1, p2, p3 = chain[i], chain[i + 1], chain[i + 2]
                prev, nxt = chain[i - 1], chain[i + 3]
                
                candidates = []
                for np1 in _neighbours(prev):
                    if np1 == p1:
                        continue
                    if np1 in occupied and occupied[np1] != i:
                        continue
                    for np2 in _neighbours(np1):
                        if np2 == p2:
                            continue
                        if np2 in occupied and occupied[np2] != i + 1:
                            continue
                        for np3 in _neighbours(np2):
                            if np3 == p3:
                                continue
                            if abs(np3[0] - nxt[0]) + abs(np3[1] - nxt[1]) != 1:
                                continue
                            if np3 in occupied and occupied[np3] != i + 2:
                                continue
                            if np1 in occupied and occupied[np1] != i:
                                continue
                            if np2 in occupied and occupied[np2] != i + 1:
                                continue
                            if np3 in occupied and occupied[np3] != i + 2:
                                continue
                            if np1 == np2 or np2 == np3 or np1 == np3:
                                continue
                            candidates.append((np1, np2, np3))
                
                if candidates:
                    np1, np2, np3 = rng.choice(candidates)
                    del occupied[p1]
                    del occupied[p2]
                    del occupied[p3]
                    chain[i] = np1
                    chain[i + 1] = np2
                    chain[i + 2] = np3
                    occupied[np1] = i
                    occupied[np2] = i + 1
                    occupied[np3] = i + 2
                    new_energy = _compute_energy(sequence, chain)

                    if new_energy < current_energy:
                        current_energy = new_energy
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_chain = list(chain)
                        no_improve_count = 0
                        improved = True
                    else:
                        del occupied[np1]
                        del occupied[np2]
                        del occupied[np3]
                        chain[i] = p1
                        chain[i + 1] = p2
                        chain[i + 2] = p3
                        occupied[p1] = i
                        occupied[p2] = i + 1
                        occupied[p3] = i + 2
                        no_improve_count += 1
        elif move_type < 0.75:
            # Short regrow: regrow last 3-6 residues
            length = rng.randint(3, min(6, n - 3))
            split = n - length
            prefix = chain[:split]
            suffix_seq = sequence[split:]

            new_suffix = _greedy_grow_from_fixed(suffix_seq, prefix[-1], prefix, sequence, rng, temperature=0.5)
            if new_suffix is not None:
                new_chain = prefix + new_suffix
                new_energy = _compute_energy(sequence, new_chain)

                if new_energy < current_energy:
                    # Update occupied set
                    occupied = {p: i for i, p in enumerate(new_chain)}
                    chain = new_chain
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                    improved = True
                else:
                    no_improve_count += 1
        else:
            # Pivot move
            new_chain, new_occupied, new_energy = _pivot_move(sequence, chain, occupied, rng)
            if new_chain is not None:
                if new_energy < current_energy:
                    chain = new_chain
                    occupied = new_occupied
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                    improved = True
                else:
                    no_improve_count += 1

        if not improved:
            no_improve_count += 1
            if no_improve_count > 300:
                break

    return best_chain, best_energy


def _regrow_sa(sequence, chain, rng, max_iters=5000, start_temp=5.0, cool_rate=0.997, tail_bias=0.0):
    """
    Simulated annealing using regrow moves: pick a split point, regrow the suffix.
    Uses adaptive cooling and biased split selection.
    tail_bias > 0 favors longer tails (smaller split indices).
    """
    n = len(sequence)
    if n < 4:
        return chain, _compute_energy(sequence, chain)

    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = start_temp
    no_improve = 0
    stuck_count = 0

    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        if temp < 0.05:
            break

        # Pick a split point
        min_split = 3
        max_split = max(min_split, n - 3)
        if max_split <= min_split:
            break
        
        # Distribution of split points
        if tail_bias > 0.5:
            # Heavily biased towards long tails (small split indices)
            # Sample from a distribution that decays exponentially with split index
            # P(split) ~ exp(-split / scale)
            scale = n * 0.2
            while True:
                split = min_split + int(rng.expovariate(1.0 / scale))
                if split <= max_split:
                    break
                split = max_split
        elif tail_bias > 0.2:
            # Moderate bias towards long tails
            # Linear ramp from min_split to max_split, weighted towards min_split
            u = rng.random()
            split = min_split + int(u * u * (max_split - min_split))
        else:
            # Original behavior: biased toward middle
            mid = (min_split + max_split) // 2
            if rng.random() < 0.7:
                offset = int(rng.gauss(0, (max_split - min_split) / 4))
                split = min(max(min_split, mid + offset), max_split)
            else:
                split = rng.randint(min_split, max_split)

        prefix = chain[:split]
        suffix_seq = sequence[split:]

        # Regrow suffix from the end of prefix
        new_suffix = _greedy_grow_from_fixed(suffix_seq, prefix[-1], prefix, sequence, rng, temperature=temp)
        if new_suffix is None:
            temp *= cool_rate
            no_improve += 1
            stuck_count += 1
            continue

        new_chain = prefix + new_suffix
        new_energy = _compute_energy(sequence, new_chain)

        delta = new_energy - current_energy
        if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve = 0
                stuck_count = 0
            else:
                no_improve += 1
                stuck_count += 1
        else:
            no_improve += 1
            stuck_count += 1

        # Adaptive cooling: cool faster when stuck
        if stuck_count > 100:
            temp *= cool_rate ** 1.5
            stuck_count = 0
        else:
            temp *= cool_rate

        if no_improve > 600:
            break

    return best_chain, best_energy


def _exhaustive_local_search(sequence, chain):
    """
    Exhaustively try all single-residue, 2-residue, and 3-residue moves until no improving move exists.
    This is a deterministic final polish.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    improved = True

    while improved and time.time() < _DEADLINE - 1:
        improved = False
        
        # Try all single-residue moves
        for i in range(1, n - 1):
            if time.time() > _DEADLINE - 1:
                break
            p = chain[i]
            prev, nxt = chain[i - 1], chain[i + 1]
            for np in _neighbours(prev):
                if np == p:
                    continue
                if abs(np[0] - nxt[0]) + abs(np[1] - nxt[1]) != 1:
                    continue
                if np in occupied and occupied[np] != i:
                    continue
                # Perform move
                del occupied[p]
                chain[i] = np
                occupied[np] = i
                new_energy = _compute_energy(sequence, chain)
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                else:
                    # Revert
                    del occupied[np]
                    chain[i] = p
                    occupied[p] = i
        
        # Try all 2-residue moves
        if n >= 4:
            for i in range(1, n - 2):
                if time.time() > _DEADLINE - 1:
                    break
                p1, p2 = chain[i], chain[i + 1]
                prev, nxt = chain[i - 1], chain[i + 2]
                
                for np1 in _neighbours(prev):
                    if np1 == p1:
                        continue
                    if np1 in occupied and occupied[np1] != i:
                        continue
                    for np2 in _neighbours(np1):
                        if np2 == p2:
                            continue
                        if abs(np2[0] - nxt[0]) + abs(np2[1] - nxt[1]) != 1:
                            continue
                        if np2 in occupied and occupied[np2] != i + 1:
                            continue
                        # Check no collision
                        if np1 in occupied and occupied[np1] != i:
                            continue
                        if np2 in occupied and occupied[np2] != i + 1:
                            continue
                        
                        # Perform move
                        del occupied[p1]
                        del occupied[p2]
                        chain[i] = np1
                        chain[i + 1] = np2
                        occupied[np1] = i
                        occupied[np2] = i + 1
                        new_energy = _compute_energy(sequence, chain)
                        if new_energy < current_energy:
                            current_energy = new_energy
                            improved = True
                        else:
                            # Revert
                            del occupied[np1]
                            del occupied[np2]
                            chain[i] = p1
                            chain[i + 1] = p2
                            occupied[p1] = i
                            occupied[p2] = i + 1

        # Try all 3-residue moves (kink rotation of 3-residue segment)
        if n >= 5:
            for i in range(1, n - 3):
                if time.time() > _DEADLINE - 1:
                    break
                p1, p2, p3 = chain[i], chain[i + 1], chain[i + 2]
                prev, nxt = chain[i - 1], chain[i + 3]
                
                # Try all valid 3-step walks from prev to nxt avoiding occupied
                for np1 in _neighbours(prev):
                    if np1 == p1:
                        continue
                    if np1 in occupied and occupied[np1] != i:
                        continue
                    for np2 in _neighbours(np1):
                        if np2 == p2:
                            continue
                        if np2 in occupied and occupied[np2] != i + 1:
                            continue
                        if abs(np2[0] - np1[0]) + abs(np2[1] - np1[1]) != 1:
                            continue
                        for np3 in _neighbours(np2):
                            if np3 == p3:
                                continue
                            if abs(np3[0] - nxt[0]) + abs(np3[1] - nxt[1]) != 1:
                                continue
                            if np3 in occupied and occupied[np3] != i + 2:
                                continue
                            # Check no collision with other occupied sites
                            if np1 in occupied and occupied[np1] != i:
                                continue
                            if np2 in occupied and occupied[np2] != i + 1:
                                continue
                            if np3 in occupied and occupied[np3] != i + 2:
                                continue
                            # Also ensure the 3 new positions don't collide with each other
                            if np1 == np2 or np2 == np3 or np1 == np3:
                                continue
                            
                            # Perform move
                            del occupied[p1]
                            del occupied[p2]
                            del occupied[p3]
                            chain[i] = np1
                            chain[i + 1] = np2
                            chain[i + 2] = np3
                            occupied[np1] = i
                            occupied[np2] = i + 1
                            occupied[np3] = i + 2
                            new_energy = _compute_energy(sequence, chain)
                            if new_energy < current_energy:
                                current_energy = new_energy
                                improved = True
                            else:
                                # Revert
                                del occupied[np1]
                                del occupied[np2]
                                del occupied[np3]
                                chain[i] = p1
                                chain[i + 1] = p2
                                chain[i + 2] = p3
                                occupied[p1] = i
                                occupied[p2] = i + 1
                                occupied[p3] = i + 2

    return chain, current_energy


def fold(sequence, dimension):
    """
    Fold a sequence into a low-energy conformation.
    """
    if dimension != 2:
        raise ValueError("Only dimension 2 is supported")

    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(12345 + len(sequence))

    best_chain = None
    best_energy = float('inf')

    # Phase 1: Generate diverse initial solutions
    initial_chains = []
    
    # Standard greedy with various temperatures
    for temp in [0.0, 0.3, 0.8, 1.5, 2.5]:
        for _ in range(10):
            if time.time() > _DEADLINE - 30:
                break
            chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
            if chain is not None:
                initial_chains.append(chain)
        
        if time.time() > _DEADLINE - 30:
            break
    
    # Center-biased greedy for more compact conformations
    for temp in [0.0, 0.5, 1.0]:
        for _ in range(5):
            if time.time() > _DEADLINE - 25:
                break
            chain = _greedy_grow_biased(sequence, dimension, rng, temperature=temp, bias_center=True)
            if chain is not None:
                initial_chains.append(chain)
        
        if time.time() > _DEADLINE - 25:
            break

    # Evaluate all initial chains and keep the top K
    evaluated_chains = []
    for chain in initial_chains:
        energy = _compute_energy(sequence, chain)
        evaluated_chains.append((energy, chain))
    
    # Sort by energy
    evaluated_chains.sort(key=lambda x: x[0])
    
    if not evaluated_chains:
        # Fallback: straight line
        best_chain = [[i, 0] for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    else:
        best_energy, best_chain = evaluated_chains[0]

    # Phase 2: Local improvement on the best chain
    if time.time() < _DEADLINE - 20:
        improved_chain, improved_energy = _local_improve(sequence, best_chain, rng, max_moves=10000)
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain

    # Phase 3: SA with regrow moves (Standard)
    if time.time() < _DEADLINE - 15:
        sa_chain, sa_energy = _regrow_sa(sequence, best_chain, rng, max_iters=12000, start_temp=6.0, cool_rate=0.996)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain

    # Phase 4: E2T SA (Long tail bias) - Structural change
    if time.time() < _DEADLINE - 10:
        # Use a bias that favors longer tails to allow global reconfiguration
        e2t_chain, e2t_energy = _regrow_sa(sequence, best_chain, rng, max_iters=12000, start_temp=10.0, cool_rate=0.994, tail_bias=0.8)
        if e2t_energy < best_energy:
            best_energy = e2t_energy
            best_chain = e2t_chain

    # Phase 5: More diverse restarts with local search + SA
    if time.time() < _DEADLINE - 5:
        restarts_done = 0
        max_restarts = 60
        while restarts_done < max_restarts:
            if time.time() > _DEADLINE - 5:
                break
            
            # Alternate between different growth strategies
            if restarts_done % 4 == 0:
                chain = _greedy_grow(sequence, dimension, rng, temperature=0.3)
            elif restarts_done % 4 == 1:
                chain = _greedy_grow_biased(sequence, dimension, rng, temperature=0.5, bias_center=True)
            elif restarts_done % 4 == 2:
                chain = _greedy_grow(sequence, dimension, rng, temperature=1.5)
            else:
                chain = _greedy_grow_biased(sequence, dimension, rng, temperature=1.0, bias_center=True)
            
            restarts_done += 1
            
            if chain is None:
                continue
            
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
            
            # Local improvement
            if time.time() < _DEADLINE - 5:
                improved = _local_improve(sequence, chain, rng, max_moves=4000)
                if improved[1] < best_energy:
                    best_energy = improved[1]
                    best_chain = improved[0]
            
            # Short SA on the improved chain
            if time.time() < _DEADLINE - 4:
                sa_c, sa_e = _regrow_sa(sequence, best_chain, rng, max_iters=3000, start_temp=4.0, cool_rate=0.997, tail_bias=0.5)
                if sa_e < best_energy:
                    best_energy = sa_e
                    best_chain = sa_c

    # Final local polish
    if time.time() < _DEADLINE - 3:
        final_chain, final_energy = _local_improve(sequence, best_chain, rng, max_moves=10000)
        if final_energy < best_energy:
            best_energy = final_energy
            best_chain = final_chain

    # Exhaustive local-search polish: try every single, 2- and 3-residue move until no improvement
    if time.time() < _DEADLINE - 1:
        best_chain, best_energy = _exhaustive_local_search(sequence, best_chain)

    # Convert to list of lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

# Set a global deadline to ensure we return well before the 120s/180s limit.
# The task says 120s CPU budget for all sequences, but evaluator kills at 210s wall clock.
# We assume fold is called once per sequence, but the contract says "fold is called once per benchmark sequence in a sandboxed subprocess with a 120 s CPU budget for all sequences together."
# This implies the total time for the script is 120s.
# However, the prompt says "Each evaluation must finish within 180 s." and "evaluator KILLS the program after 210s wall-clock".
# To be safe, I will use a deadline relative to the start of the script.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    # Iterate only over H residues to save time
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            # Check 4 neighbors
            for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _greedy_grow(sequence, dimension, rng, temperature=0.0, bias_center=False, bias_cluster=False):
    """
    Randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then sites with more free space.
    Optional biases for compactness and clustering.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    # Track H positions for cluster bias
    h_positions = []

    for i in range(1, n):
        last = chain[-1]
        lx, ly = last
        options = []
        
        # Calculate center of mass for bias if enabled and i is large enough
        cx, cy = 0, 0
        if bias_center and i > n * 0.1:
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)

        for site in _neighbours(last):
            if site in occupied:
                continue
            sx, sy = site
            gain = 0
            if sequence[i] == 'H':
                # Immediate contact gain
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
                        
                # Cluster bias: distance to nearest H
                if bias_cluster and h_positions:
                    min_dist = float('inf')
                    for hx, hy in h_positions:
                        d = abs(sx - hx) + abs(sy - hy)
                        if d < min_dist:
                            min_dist = d
                    # Reward being close to H, but not too close (already counted in gain)
                    # If min_dist is 2 or 3, it's a good spot for future contact
                    if 2 <= min_dist <= 3:
                        gain += 0.5  # Small bonus for proximity
            else:
                # For P residues, we want to stay away from H if possible? Or just minimize free space?
                # Actually, P residues should help fold the chain.
                pass

            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            center_bias = 0.0
            if bias_center:
                dist = abs(sx - cx) + abs(sy - cy)
                center_bias = -dist * 0.2

            # Score: Gain is most important, then free space (to avoid trapping), then center bias
            # Temperature adds randomness
            score = gain * 10.0 + free * 0.1 + center_bias + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        chain.append(best_site)
        
        if sequence[i] == 'H':
            h_positions.append(best_site)

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
    
    # Pre-compute H positions in prefix for faster lookup? 
    # The _greedy_grow logic is similar, but we need to be careful with indices.
    
    h_positions_prefix = [p for i, p in enumerate(prefix) if full_sequence[i] == 'H']

    for i, res in enumerate(suffix_seq):
        global_idx = len(prefix) + i
        last = chain[-1]
        options = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            sx, sy = site
            gain = 0
            if res == 'H':
                # Check neighbors in occupied
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < global_idx - 1 and full_sequence[j] == 'H':
                        gain += 1
                # Check proximity to existing Hs (simple heuristic)
                # This is O(N) per step, which might be slow for long regrows.
                # Let's skip cluster bias in regrow for speed, or make it very light.
                
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = global_idx
        chain.append(best_site)

    return chain[1:]


def _local_improve(sequence, chain, rng, max_moves=5000, time_limit=None):
    """
    Improve conformation via local moves: kink flips and short regrows.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve_count = 0
    start_time = time.time()

    for _ in range(max_moves):
        if time.time() > _DEADLINE - 2:
            break
        if time_limit and time.time() > start_time + time_limit:
            break

        move_type = rng.random()
        improved = False

        if move_type < 0.7:
            # Kink flip: try to move residue i to a neighbor that maintains connectivity
            i = rng.randint(1, n - 2)
            p = chain[i]
            prev, nxt = chain[i-1], chain[i+1]
            px, py = prev
            nx, ny = nxt

            candidates = []
            for np in _neighbours(prev):
                if np == p:
                    continue
                if abs(np[0]-nx) + abs(np[1]-ny) != 1:
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

        else:
            # Short regrow: regrow last 3-6 residues
            length = rng.randint(3, min(6, n-3))
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

        if not improved:
            no_improve_count += 1

        if no_improve_count > 500:
            break

    return best_chain, best_energy


def _simulated_annealing_full(sequence, chain, rng, iterations=10000, start_temp=5.0, cool_rate=0.9995):
    """
    Full simulated annealing with various move types.
    """
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = start_temp
    no_improve = 0

    for it in range(iterations):
        if time.time() > _DEADLINE - 10:
            break
        if temp < 0.01:
            break

        move_type = rng.random()
        accepted = False

        if move_type < 0.5:
            # Kink flip
            i = rng.randint(1, n - 2)
            p = chain[i]
            prev, nxt = chain[i-1], chain[i+1]
            px, py = prev
            nx, ny = nxt

            candidates = []
            for np in _neighbours(prev):
                if np == p:
                    continue
                if abs(np[0]-nx) + abs(np[1]-ny) != 1:
                    continue
                if np in occupied and occupied[np] != i:
                    continue
                candidates.append(np)

            if candidates:
                np = rng.choice(candidates)
                del occupied[p]
                chain[i] = np
                occupied[np] = i
                new_energy = _compute_energy(sequence, chain)

                delta = new_energy - current_energy
                if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
                    current_energy = new_energy
                    accepted = True
                else:
                    del occupied[np]
                    chain[i] = p
                    occupied[p] = i

        elif move_type < 0.8:
            # Short Regrow (3-8 residues)
            length = rng.randint(3, min(8, n-3))
            split = n - length
            prefix = chain[:split]
            suffix_seq = sequence[split:]

            new_suffix = _greedy_grow_from_fixed(suffix_seq, prefix[-1], prefix, sequence, rng, temperature=temp)
            if new_suffix is not None:
                new_chain = prefix + new_suffix
                new_energy = _compute_energy(sequence, new_chain)

                delta = new_energy - current_energy
                if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
                    chain = new_chain
                    occupied = {p: i for i, p in enumerate(chain)}
                    current_energy = new_energy
                    accepted = True

        else:
            # Long Regrow (10-20% of chain)
            # Only do this if chain is long enough
            if n > 20:
                min_len = max(10, n // 10)
                max_len = min(n - 5, n // 5)
                if max_len > min_len:
                    length = rng.randint(min_len, max_len)
                    split = n - length
                    prefix = chain[:split]
                    suffix_seq = sequence[split:]

                    new_suffix = _greedy_grow_from_fixed(suffix_seq, prefix[-1], prefix, sequence, rng, temperature=temp)
                    if new_suffix is not None:
                        new_chain = prefix + new_suffix
                        new_energy = _compute_energy(sequence, new_chain)

                        delta = new_energy - current_energy
                        if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
                            chain = new_chain
                            occupied = {p: i for i, p in enumerate(chain)}
                            current_energy = new_energy
                            accepted = True

        if accepted:
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        temp *= cool_rate

        if no_improve > 300:
            # Restart from best
            chain = list(best_chain)
            occupied = {p: i for i, p in enumerate(chain)}
            current_energy = best_energy
            temp = max(temp, 2.0)
            no_improve = 0

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence using a portfolio of greedy growths with different random seeds
    and temperatures, plus regrow-SA and local improvement. Keep the best within the time budget.
    """
    rng = random.Random(42)
    best_chain = None
    best_energy = 0  # We want to minimize energy (more negative is better)

    n = len(sequence)

    # Adaptive time allocation
    # Total budget is ~115s. 
    # We split it: 30% for initial growth, 50% for SA, 20% for local improvement.
    # But for short sequences, SA is less effective, so we can do more growths.
    
    if n < 30:
        growth_time = 40.0
        sa_time = 50.0
        local_time = 20.0
    elif n < 60:
        growth_time = 30.0
        sa_time = 60.0
        local_time = 25.0
    else:
        growth_time = 20.0
        sa_time = 70.0
        local_time = 15.0

    # Phase 1: Multiple greedy restarts
    phase1_end = time.time() + growth_time
    attempt = 0
    while time.time() < phase1_end:
        # Vary parameters
        temp = rng.uniform(0.0, 2.0)
        bias_center = attempt % 2 == 0
        bias_cluster = attempt % 3 == 0
        
        chain = _greedy_grow(sequence, dimension, rng, temperature=temp, 
                             bias_center=bias_center, bias_cluster=bias_cluster)
        
        if chain is None:
            attempt += 1
            continue

        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
            
        attempt += 1

    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
        return [list(p) for p in best_chain]

    # Phase 2: Simulated Annealing
    if time.time() < _DEADLINE - 20:
        phase2_end = time.time() + sa_time
        # Run SA on the best chain found
        # We might run multiple SA passes with different starting temps if time permits
        sa_chain, sa_energy = _simulated_annealing_full(
            sequence, best_chain, rng, iterations=20000, 
            start_temp=5.0, cool_rate=0.9998)
        
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
            
        # If we have time left, do a second pass with lower start temp to polish
        if time.time() < phase2_end - 10:
            sa_chain2, sa_energy2 = _simulated_annealing_full(
                sequence, best_chain, rng, iterations=10000, 
                start_temp=2.0, cool_rate=0.9999)
            if sa_energy2 < best_energy:
                best_chain = sa_chain2
                best_energy = sa_energy2

    # Phase 3: Local Improvement
    if time.time() < _DEADLINE - 5:
        improved_chain, improved_energy = _local_improve(
            sequence, best_chain, rng, max_moves=10000)
        
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

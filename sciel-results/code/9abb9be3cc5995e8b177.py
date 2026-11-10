"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import sys

_DEADLINE = time.time() + 175.0
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for dx, dy in _DIR4:
                j = where.get((x + dx, y + dy))
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _energy_delta_for_move(sequence, chain, i, new_pos, pos_to_idx=None):
    """
    Compute the change in contact count when moving residue i from chain[i] to new_pos.
    Returns delta_contacts (positive means more contacts, which means lower energy).
    Only valid if new_pos is not occupied by another residue in the chain.
    """
    n = len(chain)
    if i < 1 or i > n - 2:
        return 0
    
    old_pos = chain[i]
    old_x, old_y = old_pos
    new_x, new_y = new_pos
    
    if pos_to_idx is None:
        pos_to_idx = {pos: idx for idx, pos in enumerate(chain)}
    
    # Count old contacts involving residue i
    old_contacts = 0
    if sequence[i] == 'H':
        for dx, dy in _DIR4:
            neighbor_pos = (old_x + dx, old_y + dy)
            j = pos_to_idx.get(neighbor_pos)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                old_contacts += 1
    
    # Count new contacts involving residue i at new_pos
    new_contacts = 0
    if sequence[i] == 'H':
        for dx, dy in _DIR4:
            neighbor_pos = (new_x + dx, new_y + dy)
            j = pos_to_idx.get(neighbor_pos)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                new_contacts += 1
    
    return new_contacts - old_contacts


def _fast_grow(sequence, rng, temperature=0.0):
    """
    Fast randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then sites with more free space.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    res_type = [1 if c == 'H' else 0 for c in sequence]
    
    for i in range(1, n):
        last = chain[-1]
        x, y = last
        options = []
        
        for dx, dy in _DIR4:
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            
            gain = 0
            if res_type[i]:
                sx, sy = site
                for ddx, ddy in _DIR4:
                    j = occupied.get((sx + ddx, sy + ddy))
                    if j is not None and j < i - 1 and res_type[j]:
                        gain += 1
            
            free = 0
            for ddx, ddy in _DIR4:
                if (site[0] + ddx, site[1] + ddy) not in occupied:
                    free += 1
            
            # Tuned weights: Slightly higher weight on free space to avoid trapping
            score = gain * 10.0 + free * 0.2 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        best_score = -1e18
        best_site = None
        for s, st in options:
            if s > best_score:
                best_score = s
                best_site = st
        
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _local_improve_fast(sequence, chain, rng, max_moves=2000):
    """
    Improve conformation via kink flips and segment reversals with incremental energy.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve_count = 0
    
    res_type = [1 if c == 'H' else 0 for c in sequence]
    
    for _ in range(max_moves):
        if time.time() > _DEADLINE - 5:
            break

        move_type = rng.random()
        
        if move_type < 0.8:
            # Kink flip: rotate residue i around residue i-1
            i = rng.randint(1, n - 2)
            p = chain[i]
            prev = chain[i - 1]
            nxt = chain[i + 1]
            px, py = prev
            
            candidates = []
            for dx, dy in _DIR4:
                np = (px + dx, py + dy)
                if np == p:
                    continue
                if abs(np[0] - nxt[0]) + abs(np[1] - nxt[1]) != 1:
                    continue
                if np in occupied and occupied[np] != i:
                    continue
                candidates.append(np)

            if candidates:
                np = candidates[rng.randint(0, len(candidates) - 1)]
                del occupied[p]
                chain[i] = np
                occupied[np] = i
                
                # Use incremental energy delta
                delta = _energy_delta_for_move(sequence, chain, i, np, occupied)
                new_energy = current_energy - delta // 2
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                else:
                    del occupied[np]
                    chain[i] = p
                    occupied[p] = i
                    no_improve_count += 1
                    if no_improve_count > 300:
                        break
        else:
            # Segment Reversal: Reverse a segment of the chain if it fits
            # This is a powerful move for HP folding
            if n < 5:
                continue
            
            seg_len = min(rng.randint(2, max(2, n // 4)), n - 2)
            start_i = rng.randint(1, n - seg_len - 1)
            
            segment = chain[start_i:start_i + seg_len]
            pivot_point = chain[start_i - 1]
            after_seg = chain[start_i + seg_len] if start_i + seg_len < n else None
            
            # Remove segment from occupied
            for p in segment:
                del occupied[p]
            
            # Reverse the segment
            reversed_segment = list(reversed(segment))
            
            # Check if reversed segment is valid
            valid = True
            # Check connection to pivot
            if abs(reversed_segment[0][0] - pivot_point[0]) + abs(reversed_segment[0][1] - pivot_point[1]) != 1:
                valid = False
            
            if valid:
                # Check self-avoidance and connection to after_seg
                for k, np in enumerate(reversed_segment):
                    if np in occupied:
                        valid = False
                        break
                    if k < len(reversed_segment) - 1:
                        if abs(np[0] - reversed_segment[k+1][0]) + abs(np[1] - reversed_segment[k+1][1]) != 1:
                            valid = False
                            break
                if valid and after_seg is not None:
                    if abs(reversed_segment[-1][0] - after_seg[0]) + abs(reversed_segment[-1][1] - after_seg[1]) != 1:
                        valid = False
            
            if valid:
                # Apply reversal
                for k, np in enumerate(reversed_segment):
                    chain[start_i + k] = np
                    occupied[np] = start_i + k
                
                # Full energy recalculation for this less frequent move
                new_energy = _compute_energy(sequence, chain)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                else:
                    # Revert
                    for k, p in enumerate(segment):
                        chain[start_i + k] = p
                        occupied[p] = start_i + k
                    no_improve_count += 1
            else:
                # Revert
                for k, p in enumerate(segment):
                    chain[start_i + k] = p
                    occupied[p] = start_i + k
                no_improve_count += 1
                if no_improve_count > 300:
                    break

    return best_chain, best_energy


def _regrow_sa_fast(sequence, chain, rng, max_iters=3000, start_temp=5.0, cool_rate=0.997):
    """
    Simulated annealing using regrow moves: pick a split point, regrow the suffix.
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

        min_split = 3
        max_split = max(min_split, n - 3)
        if max_split <= min_split:
            break
        
        mid = (min_split + max_split) // 2
        if rng.random() < 0.7:
            offset = int(rng.gauss(0, (max_split - min_split) / 4))
            split = min(max(min_split, mid + offset), max_split)
        else:
            split = rng.randint(min_split, max_split)

        prefix = chain[:split]
        
        new_suffix = []
        last = prefix[-1]
        occupied = {p: i for i, p in enumerate(prefix)}
        success = True
        
        for i in range(split, n):
            x, y = last
            options = []
            for dx, dy in _DIR4:
                site = (x + dx, y + dy)
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == 'H':
                    sx, sy = site
                    for ddx, ddy in _DIR4:
                        j = occupied.get((sx + ddx, sy + ddy))
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                free = 0
                for ddx, ddy in _DIR4:
                    if (site[0] + ddx, site[1] + ddy) not in occupied:
                        free += 1
                score = gain * 10.0 + free * 0.2 + rng.random() * temp
                options.append((score, site))
            
            if not options:
                success = False
                break
            
            best_score = -1e18
            best_site = None
            for s, st in options:
                if s > best_score:
                    best_score = s
                    best_site = st
            
            occupied[best_site] = i
            last = best_site
            new_suffix.append(best_site)
        
        if not success:
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

        if stuck_count > 100:
            temp *= cool_rate ** 1.5
            stuck_count = 0
        else:
            temp *= cool_rate

        if no_improve > 600:
            break

    return best_chain, best_energy


def _min_conflicts_search(sequence, chain, rng, max_iters=2000):
    """
    Min-conflicts style global search: randomly select residues and try to relocate them
    to reduce the number of "conflicts" (unfavorable positions).
    This avoids sequential growth bias.
    """
    n = len(chain)
    if n < 5:
        return chain, _compute_energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve = 0
    
    res_type = [1 if c == 'H' else 0 for c in sequence]
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 8:
            break
        if no_improve > 400:
            break
        
        # Select a random residue to relocate (not endpoints)
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        prev_pos = chain[i - 1]
        next_pos = chain[i + 1]
        px, py = prev_pos
        
        # Find all valid positions for residue i
        candidates = []
        for dx, dy in _DIR4:
            np = (px + dx, py + dy)
            if np == old_pos:
                continue
            if abs(np[0] - next_pos[0]) + abs(np[1] - next_pos[1]) != 1:
                continue
            if np in occupied and occupied[np] != i:
                continue
            candidates.append(np)
        
        if not candidates:
            no_improve += 1
            continue
        
        # Evaluate all candidates and pick the best
        best_delta = -1000
        best_np = None
        for np in candidates:
            delta = _energy_delta_for_move(sequence, chain, i, np, occupied)
            if delta > best_delta:
                best_delta = delta
                best_np = np
        
        if best_np is None or best_delta <= 0:
            no_improve += 1
            continue
        
        # Apply the move
        del occupied[old_pos]
        chain[i] = best_np
        occupied[best_np] = i
        
        current_energy = current_energy - best_delta // 2
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(chain)
            no_improve = 0
        else:
            no_improve += 1
    
    return best_chain, best_energy


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

    # Phase 1: Generate diverse initial solutions with fast growth
    initial_chains = []
    
    temps = [0.0, 0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    for temp in temps:
        for _ in range(12):
            if time.time() > _DEADLINE - 35:
                break
            chain = _fast_grow(sequence, rng, temperature=temp)
            if chain is not None:
                initial_chains.append(chain)
        
        if time.time() > _DEADLINE - 35:
            break

    # Evaluate all initial chains and keep the top K
    evaluated_chains = []
    for chain in initial_chains:
        energy = _compute_energy(sequence, chain)
        evaluated_chains.append((energy, chain))
    
    evaluated_chains.sort(key=lambda x: x[0])
    
    if not evaluated_chains:
        best_chain = [[i, 0] for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    else:
        top_k = 8
        for energy, chain in evaluated_chains[:top_k]:
            if time.time() > _DEADLINE - 30:
                break
            
            # Local improvement on this candidate
            if time.time() < _DEADLINE - 25:
                improved_chain, improved_energy = _local_improve_fast(sequence, chain, rng, max_moves=5000)
                if improved_energy < energy:
                    energy = improved_energy
                    chain = improved_chain
            
            # Min-conflicts search
            if time.time() < _DEADLINE - 22:
                mc_chain, mc_energy = _min_conflicts_search(sequence, chain, rng, max_iters=2000)
                if mc_energy < energy:
                    energy = mc_energy
                    chain = mc_chain
            
            # SA on the improved candidate
            if time.time() < _DEADLINE - 18:
                sa_chain, sa_energy = _regrow_sa_fast(sequence, chain, rng, max_iters=5000, start_temp=6.0, cool_rate=0.995)
                if sa_energy < energy:
                    energy = sa_energy
                    chain = sa_chain
            
            # Second local improvement pass
            if time.time() < _DEADLINE - 12:
                improved_chain2, improved_energy2 = _local_improve_fast(sequence, chain, rng, max_moves=3000)
                if improved_energy2 < energy:
                    energy = improved_energy2
                    chain = improved_chain2
            
            if energy < best_energy:
                best_energy = energy
                best_chain = chain

    # Phase 2: More restarts with local search + min-conflicts + SA
    if time.time() < _DEADLINE - 15:
        restarts_done = 0
        max_restarts = 100
        while restarts_done < max_restarts:
            if time.time() > _DEADLINE - 15:
                break
            
            temp = [0.0, 0.3, 0.5, 1.0, 1.5, 2.0][restarts_done % 6]
            chain = _fast_grow(sequence, rng, temperature=temp)
            
            restarts_done += 1
            
            if chain is None:
                continue
            
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
            
            # Local improvement
            if time.time() < _DEADLINE - 12:
                improved = _local_improve_fast(sequence, chain, rng, max_moves=3500)
                if improved[1] < best_energy:
                    best_energy = improved[1]
                    best_chain = improved[0]
            
            # Min-conflicts search
            if time.time() < _DEADLINE - 10:
                mc_c, mc_e = _min_conflicts_search(sequence, best_chain, rng, max_iters=1500)
                if mc_e < best_energy:
                    best_energy = mc_e
                    best_chain = mc_c
            
            # SA on the improved chain
            if time.time() < _DEADLINE - 8:
                sa_c, sa_e = _regrow_sa_fast(sequence, best_chain, rng, max_iters=3500, start_temp=5.0, cool_rate=0.996)
                if sa_e < best_energy:
                    best_energy = sa_e
                    best_chain = sa_c

    # Final local polish - multiple passes
    if time.time() < _DEADLINE - 5:
        for _ in range(4):
            if time.time() > _DEADLINE - 5:
                break
            final_chain, final_energy = _local_improve_fast(sequence, best_chain, rng, max_moves=4000)
            if final_energy < best_energy:
                best_energy = final_energy
                best_chain = final_chain
            else:
                break
    
    # Final min-conflicts polish
    if time.time() < _DEADLINE - 2:
        mc_chain, mc_energy = _min_conflicts_search(sequence, best_chain, rng, max_iters=2000)
        if mc_energy < best_energy:
            best_energy = mc_energy
            best_chain = mc_chain

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import sys

_DEADLINE = None
_DEADLINE_MARGIN = 5.0  # seconds to keep before hard limit


def _setup_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 235.0  # 235s budget, leaving 5s margin


def _time_left():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield 6 neighbors on cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, rng, beam_width=50, max_time=10.0):
    """Beam search with dynamic width and improved heuristic."""
    start = (0,) * dimension
    # Beam entry: (chain_list, occupied_dict, current_energy)
    beams = [([start], {start: 0}, 0)]
    n = len(sequence)
    
    t0 = time.time()
    for i in range(1, n):
        if time.time() - t0 > max_time:
            break
        
        new_beams = []
        for chain, occupied, energy in beams:
            last = chain[-1]
            options = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            idx = occupied[q]
                            if idx < i - 1 and sequence[idx] == "H":
                                gain += 1
                # Count potential future H contacts (look-ahead)
                potential = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q not in occupied:
                            # Check if future residues are H
                            for j in range(i+2, min(i+6, n)):
                                if sequence[j] == "H":
                                    potential += 1
                                    break
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                # Heuristic: prefer high gain, then potential, then free space
                score = gain * 100 + potential * 5 + free * 10 + rng.random() * 2
                options.append((score, site, gain))
            
            if not options:
                continue
            
            options.sort(key=lambda x: -x[0])
            # Keep top candidates per beam
            keep = min(beam_width, max(1, beam_width // 2))
            for score, site, gain in options[:keep]:
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_energy = energy - gain
                new_beams.append((new_chain, new_occupied, new_energy))
        
        if not new_beams:
            break
        
        # Keep top beam_width beams
        new_beams.sort(key=lambda x: x[2])
        beams = new_beams[:beam_width]
    
    if not beams:
        return None
    
    best = min(beams, key=lambda x: x[2])
    return best[0]


def _local_search(sequence, chain, max_iters=2000, seed=42):
    """Improve a conformation via 180-degree turns, 3-residue shifts with simulated annealing."""
    n = len(chain)
    if n < 4:
        return chain
    
    rng = random.Random(seed)
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    h_positions = set(i for i in range(n) if sequence[i] == 'H')
    
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])
    
    def compute_delta_180(i):
        """Compute energy delta for 180-turn at position i. Returns (delta, new_a, new_c) or None."""
        if i < 1 or i > n - 2:
            return None
        a, b, c = best_chain[i-1], best_chain[i], best_chain[i+1]
        new_a = (2*b[0] - c[0], 2*b[1] - c[1], 2*b[2] - c[2])
        new_c = (2*b[0] - a[0], 2*b[1] - a[1], 2*b[2] - a[2])
        
        if new_a in occupied and occupied[new_a] != i-1:
            return None
        if new_c in occupied and occupied[new_c] != i+1:
            return None
        if new_a == new_c or new_a == b or new_c == b:
            return None
        
        # Count contacts lost and gained
        lost = 0
        gained = 0
        # Contacts from i-1 at old_a
        if i-1 in h_positions:
            for q in _neighbours(a):
                j = occupied.get(q)
                if j is not None and j != i-1 and j != i and j != i+1 and j in h_positions:
                    lost += 1
        # Contacts from i+1 at old_c
        if i+1 in h_positions:
            for q in _neighbours(c):
                j = occupied.get(q)
                if j is not None and j != i-1 and j != i and j != i+1 and j in h_positions:
                    lost += 1
        # Contacts between i-1 and i+1 (they were 2 apart, now check if they become neighbors)
        # After move, i-1 is at new_a, i+1 is at new_c
        if manhattan(new_a, new_c) == 1 and (i-1) in h_positions and (i+1) in h_positions:
            # They are now neighbors but not chain-adjacent (chain-adjacent means |i-1 - (i+1)| = 2 > 1)
            gained += 1
        
        # Contacts from i-1 at new_a
        if i-1 in h_positions:
            for q in _neighbours(new_a):
                j = occupied.get(q)
                if j is not None and j != i-1 and j != i and j != i+1 and j in h_positions:
                    gained += 1
        # Contacts from i+1 at new_c
        if i+1 in h_positions:
            for q in _neighbours(new_c):
                j = occupied.get(q)
                if j is not None and j != i-1 and j != i and j != i+1 and j in h_positions:
                    gained += 1
        
        delta = lost - gained
        return (delta, new_a, new_c)
    
    def compute_delta_3shift(i):
        """Compute energy delta for 3-shift at position i. Returns (delta, new_a, new_b, new_c) or None."""
        if i < 1 or i > n - 2:
            return None
        a, b, c = best_chain[i-1], best_chain[i], best_chain[i+1]
        for axis in range(3):
            for step in (-1, 1):
                na = list(a); nb = list(b); nc = list(c)
                na[axis] += step; nb[axis] += step; nc[axis] += step
                na = tuple(na); nb = tuple(nb); nc = tuple(nc)
                
                if na in occupied and occupied[na] != i-1: continue
                if nb in occupied and occupied[nb] != i: continue
                if nc in occupied and occupied[nc] != i+1: continue
                if manhattan(na, nb) != 1 or manhattan(nb, nc) != 1: continue
                
                # Count lost contacts (from old positions, excluding the 3 moving residues)
                lost = 0
                for old_pos, idx in [(a, i-1), (b, i), (c, i+1)]:
                    if idx in h_positions:
                        for q in _neighbours(old_pos):
                            j = occupied.get(q)
                            if j is not None and j not in (i-1, i, i+1) and j in h_positions:
                                lost += 1
                # Count gained contacts (at new positions)
                gained = 0
                for new_pos, idx in [(na, i-1), (nb, i), (nc, i+1)]:
                    if idx in h_positions:
                        for q in _neighbours(new_pos):
                            j = occupied.get(q)
                            if j is not None and j not in (i-1, i, i+1) and j in h_positions:
                                gained += 1
                # Check new internal contacts among the 3
                if manhattan(na, nc) == 1 and (i-1) in h_positions and (i+1) in h_positions:
                    gained += 1
                
                delta = lost - gained
                return (delta, na, nb, nc)
        return None
    
    # Simulated annealing parameters
    t_start = 1.0
    t_end = 0.01
    current_temp = t_start
    
    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        
        # Anneal temperature
        if it < max_iters:
            current_temp = t_start * (t_end / t_start) ** (it / max_iters)
        
        improved = False
        no_improve_count = 0
        
        for _attempt in range(40):
            if _time_left() < 1.0:
                break
            
            move_type = rng.random()
            i = rng.randint(1, n - 2)
            
            if move_type < 0.55:
                # 180-degree turn
                result = compute_delta_180(i)
                if result is None:
                    continue
                delta, new_a, new_c = result
                old_a, old_c = best_chain[i-1], best_chain[i+1]
                
                # Accept if improvement or with probability exp(-delta/temp) for worsening
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    import math
                    accept = rng.random() < math.exp(-delta / max(current_temp, 0.001))
                
                if accept:
                    occupied.pop(old_a)
                    occupied.pop(old_c)
                    occupied[new_a] = i-1
                    occupied[new_c] = i+1
                    best_chain[i-1] = new_a
                    best_chain[i+1] = new_c
                    best_energy -= delta
                    improved = True
                else:
                    no_improve_count += 1
                    
            else:
                # 3-residue shift
                result = compute_delta_3shift(i)
                if result is None:
                    continue
                delta, na, nb, nc = result
                old_a, old_b, old_c = best_chain[i-1], best_chain[i], best_chain[i+1]
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    import math
                    accept = rng.random() < math.exp(-delta / max(current_temp, 0.001))
                
                if accept:
                    occupied.pop(old_a)
                    occupied.pop(old_b)
                    occupied.pop(old_c)
                    occupied[na] = i-1
                    occupied[nb] = i
                    occupied[nc] = i+1
                    best_chain[i-1] = na
                    best_chain[i] = nb
                    best_chain[i+1] = nc
                    best_energy -= delta
                    improved = True
                else:
                    no_improve_count += 1
        
        if not improved and no_improve_count > 200:
            break
    
    return best_chain


def _spiral_fold(sequence, dimension, seed=0):
    """Create a spiral-like initial conformation."""
    n = len(sequence)
    rng = random.Random(seed)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    dir_idx = rng.randint(0, 5)
    
    for i in range(1, n):
        placed = False
        # Try preferred direction first, then others
        order = [dir_idx % 6] + [d for d in range(6) if d != dir_idx % 6]
        rng.shuffle(order[1:])
        for d_idx in order:
            d = directions[d_idx % 6]
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                placed = True
                # Occasionally turn
                if rng.random() < 0.3:
                    dir_idx = (d_idx + 1) % 6
                break
        
        if not placed:
            # Stuck, try all 6 directions
            for d in directions:
                new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    break
            else:
                break
    
    return chain


def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice using multiple strategies."""
    _setup_deadline()
    n = len(sequence)
    
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Multiple diverse initializations
    for seed in range(8):
        if _time_left() < 3.0:
            break
        chain1 = _spiral_fold(sequence, dimension, seed=seed)
        if len(chain1) == n:
            e1 = _compute_energy(sequence, chain1)
            if best_chain is None or e1 < best_energy:
                best_chain = chain1
                best_energy = e1
    
    # Strategy 2: Beam search with different widths
    if _time_left() > 5.0:
        for bw in [50, 120, 250]:
            if _time_left() < 3.0:
                break
            max_t = min(12.0, _time_left() * 0.2)
            chain2 = _beam_search(sequence, dimension, rng, beam_width=bw, max_time=max_t)
            if chain2 and len(chain2) == n:
                e2 = _compute_energy(sequence, chain2)
                if best_chain is None or e2 < best_energy:
                    best_chain = chain2
                    best_energy = e2
    
    # Strategy 3: Local search refinement with simulated annealing
    if best_chain is not None and _time_left() > 2.0:
        # Run local search multiple times with different seeds
        for ls_seed in range(12):
            if _time_left() < 2.0:
                break
            max_iters = min(8000, int(_time_left() * 200))
            refined = _local_search(sequence, best_chain, max_iters=max_iters, seed=ls_seed + n)
            if refined and len(refined) == n:
                er = _compute_energy(sequence, refined)
                if er < best_energy:
                    best_chain = refined
                    best_energy = er
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

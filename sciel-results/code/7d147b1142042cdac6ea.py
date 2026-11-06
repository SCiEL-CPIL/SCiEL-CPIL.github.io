"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0  # Safety margin for 210s wall-clock kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _contact_map(sequence):
    """Identify high-priority H-H pairs for constraint-guided search.
    
    Returns list of (i, j, priority) where i < j, both are H residues,
    and priority indicates how important this contact is to satisfy.
    Pairs with sequence distance 3-15 are most likely to be real contacts.
    """
    n = len(sequence)
    h_positions = [i for i in range(n) if sequence[i] == 'H']
    pairs = []
    for a, pos1 in enumerate(h_positions):
        for pos2 in h_positions[a+1:]:
            dist = pos2 - pos1
            # Contact probability heuristic: pairs 3-15 apart in sequence
            # are most likely to form contacts in optimal folds
            if 3 <= dist <= 15:
                # Higher priority for distances in sweet spot 5-10
                if 5 <= dist <= 10:
                    priority = 3.0
                elif dist in (3, 4, 11, 12):
                    priority = 2.0
                else:  # 13, 14, 15
                    priority = 1.0
                pairs.append((pos1, pos2, priority))
    # Sort by priority descending
    pairs.sort(key=lambda x: -x[2])
    return pairs


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _backtrack_fold(sequence, dimension=2, contact_pairs=None):
    """Constraint-guided backtracking search for HP fold.
    
    Uses contact map to prioritize moves that satisfy high-priority H-H pairs.
    Works well for n <= 45.
    """
    n = len(sequence)
    if n < 2:
        return [(0, 0)]
    
    if contact_pairs is None:
        contact_pairs = _contact_map(sequence)
    
    # Directions: right, up, left, down
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    best_energy = [0]  # mutable for closure
    best_chain = [None]
    nodes_visited = [0]
    start_time = time.time()
    
    # Track which contact pairs are satisfied
    # contact_satisfied[pair_index] = True when both residues are placed and adjacent
    
    def _get_contact_gain(i, pos, occupied, sequence):
        """Count how many high-priority contacts are satisfied by placing residue i at pos."""
        if sequence[i] != 'H':
            return 0
        gain = 0
        for ox, oy in _neighbours(pos):
            j = occupied.get((ox, oy))
            if j is not None and j < i - 1 and sequence[j] == 'H':
                # Check if this is a high-priority contact pair
                dist = i - j
                if 3 <= dist <= 15:
                    gain += 2  # Weighted gain for priority contacts
                else:
                    gain += 1
        return gain
    
    def _search(chain, occupied, energy, i):
        if time.time() - start_time > _time_left() - 8:
            return False  # timeout signal
        
        if i == n:
            if energy < best_energy[0]:
                best_energy[0] = energy
                best_chain[0] = chain[:]
            return True
        
        # Pruning: optimistic bound on remaining energy
        remaining_h = sum(1 for j in range(i, n) if sequence[j] == 'H')
        max_additional = remaining_h * 2
        if energy + max_additional >= best_energy[0]:
            return True  # prune
        
        cx, cy = chain[-1]
        
        # Generate and score all valid moves
        moves = []
        for dx, dy in dirs:
            # Don't reverse direction
            if i >= 2:
                px, py = chain[-1]
                qx, qy = chain[-2]
                if (dx, dy) == (qx - px, qy - py):
                    continue
            
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in occupied:
                continue
            
            new_energy = energy
            if sequence[i] == 'H':
                # Count new contacts with existing H residues
                for ox, oy in _neighbours((nx, ny)):
                    j = occupied.get((ox, oy))
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        new_energy -= 1
            
            # Score this move: prioritize satisfying high-priority contacts
            contact_gain = _get_contact_gain(i, (nx, ny), occupied, sequence)
            
            # Also check: does this move block any future high-priority contacts?
            # Simple heuristic: count free neighbors
            free = sum(1 for q in _neighbours((nx, ny)) if q not in occupied)
            
            moves.append((contact_gain * 3 + free * 0.5, dx, dy, nx, ny, new_energy))
        
        # Sort moves by score descending (try best moves first)
        moves.sort(key=lambda x: -x[0])
        
        for _, dx, dy, nx, ny, new_energy in moves:
            chain.append((nx, ny))
            occupied[(nx, ny)] = i
            nodes_visited[0] += 1
            
            if not _search(chain, occupied, new_energy, i + 1):
                # timeout
                chain.pop()
                del occupied[(nx, ny)]
                return False
            
            chain.pop()
            del occupied[(nx, ny)]
        
        return True
    
    # Try all 4 initial directions
    for dx, dy in dirs:
        if time.time() - start_time > _time_left() - 8:
            break
        nx, ny = dx, dy
        chain = [(0, 0), (nx, ny)]
        occupied = {(0, 0): 0, (nx, ny): 1}
        
        new_energy = 0
        # Contact with residue 0 if both are H
        if sequence[0] == 'H' and sequence[1] == 'H':
            for ox, oy in _neighbours((nx, ny)):
                if (ox, oy) == (0, 0):
                    new_energy -= 1
        
        if not _search(chain, occupied, new_energy, 2):
            break
    
    if best_chain[0] is not None:
        return best_chain[0]
    
    # Fallback: straight line
    return [(i, 0) for i in range(n)]


def _grow_constrained(sequence, dimension, rng, bias=1.0, contact_pairs=None):
    """Greedy growth with contact-map-guided awareness."""
    n = len(sequence)
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    
    if contact_pairs is None:
        contact_pairs = _contact_map(sequence)
    
    # Build lookup: for each residue, which other residues form high-priority contacts
    residue_contacts = {}
    for i, j, priority in contact_pairs:
        if i not in residue_contacts:
            residue_contacts[i] = set()
        if j not in residue_contacts:
            residue_contacts[j] = set()
        residue_contacts[i].add(j)
        residue_contacts[j].add(i)
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == "H":
                # Count all H-H contacts
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
                # Bonus for high-priority contacts
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        if j in residue_contacts.get(i, set()):
                            gain += 2  # Extra weight for priority contacts
            
            # Look-ahead: check if placing here creates a dead end for future
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Penalize moves that reduce free space too much
            if free == 0 and i < n - 1:
                continue  # dead end
            
            # Look-ahead: how many future high-priority contacts could this placement enable?
            future_gain = 0
            if sequence[i] == "H":
                for j in residue_contacts.get(i, set()):
                    if j >= i + 1:  # Future residue
                        future_gain += 1
            
            score = bias * gain + 0.15 * free + 0.3 * future_gain + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _pivot_move(chain, rng):
    """Pivot move: rotate a subchain by 90 degrees."""
    n = len(chain)
    if n < 4:
        return None
    pivot = rng.randint(2, n - 2)
    head = chain[:pivot]
    tail = chain[pivot:]
    
    # Rotate 90 degrees clockwise: (dx,dy) -> (dy,-dx)
    new_tail = [chain[pivot]]
    for i in range(1, len(tail)):
        dx = tail[i][0] - tail[i-1][0]
        dy = tail[i][1] - tail[i-1][1]
        new_dx = dy
        new_dy = -dx
        new_tail.append((new_tail[-1][0] + new_dx, new_tail[-1][1] + new_dy))
    
    new_chain = head + new_tail
    
    if len(set(new_chain)) != n:
        return None
    
    dx = new_tail[0][0] - head[-1][0]
    dy = new_tail[0][1] - head[-1][1]
    if abs(dx) + abs(dy) != 1:
        return None
    
    return new_chain


def _crankshaft_move(chain, rng):
    """Crankshaft move: shift a short segment perpendicular to its direction."""
    n = len(chain)
    if n < 5:
        return None
    
    seg_len = rng.randint(2, min(4, n - 4))
    start = rng.randint(1, n - seg_len - 1)
    end = start + seg_len
    
    segment = chain[start:end]
    head = chain[:start]
    tail = chain[end:]
    
    dx = segment[-1][0] - segment[0][0]
    dy = segment[-1][1] - segment[0][1]
    
    perp_dirs = [(-dy, dx), (dy, -dx)]
    if dx == 0:
        perp_dirs = [(1, 0), (-1, 0)]
    elif dy == 0:
        perp_dirs = [(0, 1), (0, -1)]
    
    for pdx, pdy in perp_dirs:
        new_segment = [(x + pdx, y + pdy) for x, y in segment]
        new_chain = head + new_segment + tail
        
        if len(set(new_chain)) != n:
            continue
        
        if head:
            dx1 = new_segment[0][0] - head[-1][0]
            dy1 = new_segment[0][1] - head[-1][1]
            if abs(dx1) + abs(dy1) != 1:
                continue
        if tail:
            dx2 = tail[0][0] - new_segment[-1][0]
            dy2 = tail[0][1] - new_segment[-1][1]
            if abs(dx2) + abs(dy2) != 1:
                continue
        
        return new_chain
    
    return None


def _local_search(sequence, chain, rng, max_steps=2000, contact_pairs=None):
    """Simulated annealing local search with contact-map awareness."""
    n = len(chain)
    current = chain[:]
    current_energy = _energy(sequence, current)
    best = current[:]
    best_energy = current_energy
    
    if contact_pairs is None:
        contact_pairs = _contact_map(sequence)
    
    # Build lookup for fast contact checking
    h_set = set(i for i in range(n) if sequence[i] == 'H')
    
    T_start = 3.0
    T_end = 0.05
    cooling = (T_end / T_start) ** (1.0 / max(1, max_steps))
    T = T_start
    
    for step in range(max_steps):
        if _time_left() < 5.0:
            break
        
        if rng.random() < 0.6:
            new_chain = _pivot_move(current, rng)
        else:
            new_chain = _crankshaft_move(current, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Accept if better or with Boltzmann probability
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = current[:]
                best_energy = current_energy
        
        T *= cooling
    
    return best, best_energy


def fold(sequence, dimension):
    """Best conformation using contact-map-guided search."""
    n = len(sequence)
    rng = random.Random(42 + n)
    
    # Pre-compute contact map for guidance
    contact_pairs = _contact_map(sequence)
    
    best = None
    best_energy = 0
    
    # For short sequences, try constraint-guided backtracking first
    if n <= 45 and _time_left() > 25:
        try:
            bt_chain = _backtrack_fold(sequence, dimension, contact_pairs)
            if bt_chain is not None and len(bt_chain) == n:
                bt_energy = _energy(sequence, bt_chain)
                best = bt_chain
                best_energy = bt_energy
        except Exception:
            pass
    
    # Generate initial solutions via contact-guided greedy growth
    num_growths = 60 if n <= 40 else 120
    for i in range(num_growths):
        if _time_left() < 8.0:
            break
        
        bias = 0.5 + 0.8 * (i % 10) / 10.0
        chain = _grow_constrained(sequence, dimension, rng, bias=bias, contact_pairs=contact_pairs)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best = chain[:]
            best_energy = energy
    
    # If no valid fold found, use straight line
    if best is None:
        best = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best)
    
    # Local search from best
    if _time_left() > 12.0:
        improved, improved_energy = _local_search(sequence, best, rng, max_steps=3000, contact_pairs=contact_pairs)
        if improved_energy < best_energy:
            best = improved
            best_energy = improved_energy
    
    # Additional growths with local search if time permits
    if _time_left() > 20.0:
        for i in range(100):
            if _time_left() < 12.0:
                break
            
            bias = 0.3 + 0.7 * rng.random()
            chain = _grow_constrained(sequence, dimension, rng, bias=bias, contact_pairs=contact_pairs)
            if chain is None:
                continue
            
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best = chain[:]
                best_energy = energy
                
                if _time_left() > 12.0:
                    improved, improved_energy = _local_search(sequence, best, rng, max_steps=1500, contact_pairs=contact_pairs)
                    if improved_energy < best_energy:
                        best = improved
                        best_energy = improved_energy
    
    # Final polish: intensive local search on best solution
    if _time_left() > 10.0:
        improved, improved_energy = _local_search(sequence, best, rng, max_steps=2000, contact_pairs=contact_pairs)
        if improved_energy < best_energy:
            best = improved
            best_energy = improved_energy
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

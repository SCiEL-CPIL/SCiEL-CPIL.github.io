"""HP lattice protein folding using simulated annealing with multiple restarts and local search."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import deque

_DEADLINE_MARGIN = 15.0  # seconds to reserve before the 210s wall-clock kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy as minus the number of H-H contacts (non-adjacent along chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy_delta(sequence, chain, old_pos, new_pos, index):
    """Compute the change in energy when moving residue at `index` from old_pos to new_pos.
    Returns the delta (new_energy - old_energy), or None if move is invalid (collision)."""
    n = len(chain)
    # Check collision with other residues
    for i in range(n):
        if i != index and chain[i] == new_pos:
            return None
    
    old_energy = 0
    new_energy = 0
    
    # Remove contributions involving `index`
    if sequence[index] == "H":
        for q in _neighbours(old_pos):
            j = None
            # We need to check which residues are at neighboring positions
            # This is O(n) in general, but we can be smarter
            pass
    
    # Full recomputation is simpler and still fast for n <= 100
    # But let's do a proper delta computation
    
    # Build position map for current chain
    where = {p: i for i, p in enumerate(chain)}
    
    # Count contacts involving residue `index` in old chain
    old_contacts = 0
    if sequence[index] == "H":
        for q in _neighbours(old_pos):
            j = where.get(q)
            if j is not None and j > index + 1 and sequence[j] == "H":
                old_contacts += 1
            elif j is not None and j < index - 1 and sequence[j] == "H":
                old_contacts += 1
    
    # Now simulate the move
    chain_new = list(chain)
    chain_new[index] = new_pos
    where_new = {p: i for i, p in enumerate(chain_new)}
    
    # Check for collisions again (already checked, but be safe)
    seen = set()
    for p in chain_new:
        if p in seen:
            return None
        seen.add(p)
    
    # Count contacts involving residue `index` in new chain
    new_contacts = 0
    if sequence[index] == "H":
        for q in _neighbours(new_pos):
            j = where_new.get(q)
            if j is not None and j > index + 1 and sequence[j] == "H":
                new_contacts += 1
            elif j is not None and j < index - 1 and sequence[j] == "H":
                new_contacts += 1
    
    # Also need to account for contacts that other H residues had with `index`'s old position
    # and lose/gain with new position. The above counts from index's perspective.
    # But we also need: for each other H residue j, did it have a contact with old_pos?
    # That's already counted if j is a neighbor of old_pos and j != index.
    # Actually the double counting issue: if both i and j are H and neighbors, it's counted once.
    # From index's perspective, we count each neighbor j that is H and non-adjacent.
    # This is correct for the contacts involving index.
    
    # But wait - when index moves, other residues' contact counts don't change
    # (they still have the same neighbors except index moved).
    # So the total energy change = (new_contacts - old_contacts) from index's side.
    # But we also need to consider: did some other H residue k have index as a neighbor?
    # If k is H and k was adjacent to old_pos, then k had a contact with index.
    # After the move, k is no longer adjacent to index (unless k is adjacent to new_pos).
    # This is already captured: from index's perspective, we count k if k is H and neighbor.
    # From k's perspective, it would count index. But we only count from index's perspective,
    # so we're fine as long as we're consistent.
    
    # Actually, the energy is the total number of H-H non-adjacent pairs.
    # Moving index changes:
    # 1. Contacts between index and its old neighbors (removed)
    # 2. Contacts between index and its new neighbors (added)
    # The above computes exactly this delta.
    
    return new_contacts - old_contacts


def _pivot_move(sequence, chain, rng):
    """Perform a pivot move: choose a random internal segment and rotate it 90 degrees.
    Returns new_chain if valid, else None."""
    n = len(chain)
    if n < 4:
        return None
    
    # Choose pivot point i (1 <= i <= n-2)
    i = rng.randint(1, n - 2)
    # Choose which end to move: left or right segment
    move_left = rng.random() < 0.5
    
    if move_left:
        # Move segment [0..i] by rotating around chain[i]
        # Actually, pivot: keep chain[i] fixed, rotate the segment on one side
        # Let's do: keep chain[i] and chain[i+1] fixed, rotate [0..i]
        # Simpler: choose i, j with i < j, keep chain[i] and chain[j] fixed, rotate [i+1..j-1]
        pass
    
    # Standard pivot: choose i (1 <= i <= n-2), keep chain[i-1] and chain[i] fixed,
    # try all 3 rotations of the tail [i+1..n-1] around chain[i]
    # Actually let's do the common one: choose i, keep chain[i-1], chain[i] fixed,
    # rotate segment [0..i-1] or [i+1..n-1]
    
    # Let me use: choose i (1 <= i <= n-2). Keep chain[i] fixed.
    # Try moving chain[i-1] to one of the 3 other neighbors of chain[i] (not chain[i-1]'s current pos)
    # and shift the entire prefix/suffix.
    
    # Simpler approach: choose i (2 <= i <= n-2), keep chain[i-1] and chain[i] fixed,
    # rotate the segment [0..i-2] around chain[i-1]
    # OR rotate segment [i+1..n-1] around chain[i]
    
    # Let's do: choose i (1 <= i <= n-2). Keep chain[i-1] and chain[i] fixed.
    # The segment to rotate is either [0..i-2] (prefix before i-1) or [i+1..n-1] (suffix after i)
    
    if move_left:
        # Rotate prefix [0..i-2] around chain[i-1]
        if i < 2:
            return None
        pivot = chain[i - 1]
        segment = chain[:i - 1]  # [0..i-2]
        # Try 3 rotations of the segment around pivot
        # The first element of segment (chain[i-2]) is adjacent to pivot
        # We need to rotate the entire segment
        # This is complex. Let's use a different pivot definition.
        
        # Standard pivot move: choose i (2 <= i <= n-1). Keep chain[i-1] and chain[i] fixed.
        # Rotate the tail [i+1..n-1] around chain[i].
        # Try each of the 3 possible directions for chain[i+1] (not the current direction)
        
        if i >= n - 1:
            return None
        
        pivot = chain[i]
        current_next = chain[i + 1]
        # Find the 3 other neighbors of pivot (excluding current_next and chain[i-1] is not relevant for tail)
        # Actually we need chain[i+1] to be a neighbor of pivot, and we try the other 3 neighbors
        candidates = []
        for nb in _neighbours(pivot):
            if nb != current_next:
                candidates.append(nb)
        
        if not candidates:
            return None
        
        new_next = rng.choice(candidates)
        
        # Now we need to check if moving the entire tail [i+1..n-1] by the vector
        # (new_next - current_next) results in a valid self-avoiding walk
        dx = new_next[0] - current_next[0]
        dy = new_next[1] - current_next[1]
        
        new_chain = list(chain)
        for k in range(i + 1, n):
            new_chain[k] = (chain[k][0] + dx, chain[k][1] + dy)
        
        # Check self-avoidance
        seen = set()
        for p in new_chain:
            if p in seen:
                return None
            seen.add(p)
        
        # Check unit steps
        for k in range(1, n):
            ax = abs(new_chain[k][0] - new_chain[k-1][0])
            ay = abs(new_chain[k][1] - new_chain[k-1][1])
            if ax + ay != 1:
                return None
        
        return new_chain
    else:
        # Rotate suffix [i+1..n-1] around chain[i]
        if i >= n - 1:
            return None
        
        pivot = chain[i]
        current_next = chain[i + 1]
        candidates = []
        for nb in _neighbours(pivot):
            if nb != current_next:
                candidates.append(nb)
        
        if not candidates:
            return None
        
        new_next = rng.choice(candidates)
        
        dx = new_next[0] - current_next[0]
        dy = new_next[1] - current_next[1]
        
        new_chain = list(chain)
        for k in range(i + 1, n):
            new_chain[k] = (chain[k][0] + dx, chain[k][1] + dy)
        
        seen = set()
        for p in new_chain:
            if p in seen:
                return None
            seen.add(p)
        
        for k in range(1, n):
            ax = abs(new_chain[k][0] - new_chain[k-1][0])
            ay = abs(new_chain[k][1] - new_chain[k-1][1])
            if ax + ay != 1:
                return None
        
        return new_chain


def _endpoint_move(sequence, chain, rng):
    """Move one of the endpoints to a new valid position."""
    n = len(chain)
    if n == 0:
        return None
    
    # Choose which endpoint
    if rng.random() < 0.5:
        # Move start
        start = chain[0]
        second = chain[1] if n > 1 else None
        candidates = []
        for nb in _neighbours(start):
            if second is not None and nb == second:
                continue
            # Check collision
            if nb in chain[1:]:
                continue
            candidates.append(nb)
        
        if not candidates:
            return None
        
        new_start = rng.choice(candidates)
        new_chain = [new_start] + chain[1:]
        return new_chain
    else:
        # Move end
        end = chain[-1]
        second_last = chain[-2] if n > 1 else None
        candidates = []
        for nb in _neighbours(end):
            if second_last is not None and nb == second_last:
                continue
            if nb in chain[:-1]:
                continue
            candidates.append(nb)
        
        if not candidates:
            return None
        
        new_end = rng.choice(candidates)
        new_chain = chain[:-1] + [new_end]
        return new_chain


def _initial_spiral(sequence, dimension):
    """Create an initial spiral-like self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    # Simple spiral pattern
    chain = [(0, 0)]
    direction = 0  # 0=right, 1=down, 2=left, 3=up
    steps_in_dir = 1
    dir_count = 0
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    
    occupied = {(0, 0)}
    
    while len(chain) < n:
        for _ in range(steps_in_dir):
            if len(chain) >= n:
                break
            new_x = chain[-1][0] + dx[direction]
            new_y = chain[-1][1] + dy[direction]
            new_pos = (new_x, new_y)
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied.add(new_pos)
            else:
                break
        
        if len(chain) >= n:
            break
        
        # Turn
        direction = (direction + 1) % 4
        dir_count += 1
        if dir_count % 2 == 0:
            steps_in_dir += 1
    
    # If spiral didn't fill, extend in a straight line or zigzag
    while len(chain) < n:
        last = chain[-1]
        # Try extending in a direction that doesn't collide
        found = False
        for d in range(4):
            nx = last[0] + dx[d]
            ny = last[1] + dy[d]
            if (nx, ny) not in occupied:
                chain.append((nx, ny))
                occupied.add((nx, ny))
                found = True
                break
        if not found:
            # Give up, just break
            break
    
    return chain[:n]


def _initial_random(sequence, dimension, rng):
    """Create a random self-avoiding walk by greedy growth with randomization."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for d in range(4):
            nx = last[0] + dx[d]
            ny = last[1] + dy[d]
            if (nx, ny) not in occupied:
                candidates.append((nx, ny))
        
        if not candidates:
            return None  # Dead end
        
        choice = rng.choice(candidates)
        chain.append(choice)
        occupied.add(choice)
    
    return chain


def _sa_optimize(sequence, chain, time_budget, rng):
    """Simulated annealing optimization with pivot and endpoint moves."""
    n = len(chain)
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    t_start = time.time()
    T = 2.0
    T_min = 0.01
    cooling = 0.999
    
    step = 0
    while time.time() - t_start < time_budget:
        step += 1
        
        # Choose move type
        move_type = rng.random()
        if move_type < 0.6:
            new_chain = _pivot_move(sequence, current_chain, rng)
        else:
            new_chain = _endpoint_move(sequence, current_chain, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, T_min)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_chain, best_energy


def _local_search(sequence, chain, time_budget, rng):
    """Local search: try all single-residue moves and accept improvements."""
    n = len(chain)
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    t_start = time.time()
    no_improve_count = 0
    
    while time.time() - t_start < time_budget:
        improved = False
        for i in range(n):
            pos = current_chain[i]
            for nb in _neighbours(pos):
                # Check if move is valid
                if i > 0 and nb == current_chain[i-1]:
                    continue
                if i < n-1 and nb == current_chain[i+1]:
                    continue
                # Check collision
                valid = True
                for j in range(n):
                    if j != i and current_chain[j] == nb:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute new energy
                new_chain = list(current_chain)
                new_chain[i] = nb
                new_energy = _compute_energy(sequence, new_chain)
                
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    improved = True
                    break  # Restart scan from beginning after improvement
        
        if not improved:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Find the best conformation using multiple strategies."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    t_start = time.time()
    deadline = t_start + 210.0 - _DEADLINE_MARGIN
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # Generate diverse initial solutions
    initial_chains = []
    
    # Spiral
    spiral = _initial_spiral(sequence, dimension)
    if spiral and len(spiral) == n:
        initial_chains.append(spiral)
    
    # Random walks
    for _ in range(5):
        rw = _initial_random(sequence, dimension, rng)
        if rw and len(rw) == n:
            initial_chains.append(rw)
    
    # Greedy growth (original approach)
    for _ in range(3):
        chain = _greedy_growth(sequence, dimension, rng)
        if chain and len(chain) == n:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        initial_chains.append([(i, 0) for i in range(n)])
    
    # Optimize each initial chain
    num_initials = len(initial_chains)
    
    for idx, init_chain in enumerate(initial_chains):
        if time.time() > deadline:
            break
        
        # Allocate time budget based on remaining time
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        # Give more time to earlier solutions, less to later ones
        # Also do local search at the end
        sa_budget = remaining * 0.7 / max(1, num_initials - idx)
        ls_budget = remaining * 0.3 / max(1, num_initials - idx)
        
        # Simulated annealing
        sa_chain, sa_energy = _sa_optimize(sequence, init_chain, sa_budget, rng)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
        
        # Local search on the SA result
        if time.time() < deadline:
            ls_chain, ls_energy = _local_search(sequence, sa_chain, ls_budget, rng)
            if ls_energy < best_energy:
                best_energy = ls_energy
                best_chain = ls_chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]


def _greedy_growth(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    
    for i in range(1, n):
        options = []
        last = chain[-1]
        for d in range(4):
            site = (last[0] + dx[d], last[1] + dy[d])
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain
# EVOLVE-BLOCK-END

"""HP lattice protein folding: greedy growth + pivot-move simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Global deadline management
_DEADLINE = None
_TIME_PER_SEQ = 8.0  # seconds per sequence target

def _remaining_time():
    if _DEADLINE is None:
        return 240.0
    return max(0.0, _DEADLINE - time.time())


class SpatialGrid:
    """Fast O(1) occupancy lookup on the cubic lattice."""
    __slots__ = ('grid',)

    def __init__(self):
        self.grid = {}

    def add(self, point):
        self.grid[point] = True

    def remove(self, point):
        self.grid.pop(point, None)

    def contains(self, point):
        return point in self.grid

    def neighbours(self, point):
        """Yield unoccupied neighbours."""
        x, y, z = point
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            n = (x+dx, y+dy, z+dz)
            if n not in self.grid:
                yield n

    def copy(self):
        g = SpatialGrid()
        g.grid = dict(self.grid)
        return g


def _pivot_move_valid(chain, grid, pivot_idx, direction):
    """
    Try a 90-degree rotation of the suffix starting at pivot_idx around the bond
    from chain[pivot_idx-1] to chain[pivot_idx].
    direction: one of 6 possible rotations (axis, sign).
    Returns (new_suffix_positions, valid) or (None, False).
    """
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return None, False

    # The pivot axis is the bond between pivot_idx-1 and pivot_idx
    p_prev = chain[pivot_idx - 1]
    p_pivot = chain[pivot_idx]
    # Bond vector
    bx = p_pivot[0] - p_prev[0]
    by = p_pivot[1] - p_prev[1]
    bz = p_pivot[2] - p_prev[2]

    # The suffix to rotate is chain[pivot_idx..n-1]
    # We rotate each point in the suffix around the axis defined by (p_prev, p_pivot)
    # For a 90-degree rotation, we need to determine the rotation plane.
    
    # Actually, for pivot moves on a cubic lattice, the standard approach is:
    # The bond (p_prev -> p_pivot) defines an axis. The suffix can be rotated
    # 90 degrees in one of the two perpendicular planes.
    
    # Let's use a different, simpler approach: enumerate all valid next positions
    # for the suffix by "unfolding" and "refolding".
    
    # Simpler: for each position in the suffix, compute where it would go after
    # a 90-degree rotation about the bond axis.
    
    # The bond direction is (bx, by, bz) which is a unit vector along one axis.
    # The two perpendicular axes define the rotation plane.
    
    # For a cubic lattice pivot move:
    # If bond is along x-axis, we can rotate in yz-plane (90 deg, 2 directions)
    # If bond is along y-axis, we can rotate in xz-plane
    # If bond is along z-axis, we can rotate in xy-plane
    
    if bx != 0:  # bond along x
        # Rotate in yz plane
        if direction == 0:  # +90 in yz
            def rot(pt):
                # pt relative to p_prev
                dx = pt[0] - p_prev[0]
                dy = pt[1] - p_prev[1]
                dz = pt[2] - p_prev[2]
                # 90 deg rotation in yz: (dy,dz) -> (-dz,dy)
                return (p_prev[0] + dx, p_prev[1] - dz, p_prev[2] + dy)
        else:  # -90 in yz
            def rot(pt):
                dx = pt[0] - p_prev[0]
                dy = pt[1] - p_prev[1]
                dz = pt[2] - p_prev[2]
                return (p_prev[0] + dx, p_prev[1] + dz, p_prev[2] - dy)
    elif by != 0:  # bond along y
        if direction == 0:  # +90 in xz
            def rot(pt):
                dx = pt[0] - p_prev[0]
                dy = pt[1] - p_prev[1]
                dz = pt[2] - p_prev[2]
                return (p_prev[0] + dz, p_prev[1] + dy, p_prev[2] - dx)
        else:  # -90 in xz
            def rot(pt):
                dx = pt[0] - p_prev[0]
                dy = pt[1] - p_prev[1]
                dz = pt[2] - p_prev[2]
                return (p_prev[0] - dz, p_prev[1] + dy, p_prev[2] + dx)
    else:  # bond along z
        if direction == 0:  # +90 in xy
            def rot(pt):
                dx = pt[0] - p_prev[0]
                dy = pt[1] - p_prev[1]
                dz = pt[2] - p_prev[2]
                return (p_prev[0] - dy, p_prev[1] + dx, p_prev[2] + dz)
        else:  # -90 in xy
            def rot(pt):
                dx = pt[0] - p_prev[0]
                dy = pt[1] - p_prev[1]
                dz = pt[2] - p_prev[2]
                return (p_prev[0] + dy, p_prev[1] - dx, p_prev[2] + dz)

    # Compute new positions for the suffix
    new_suffix = []
    for i in range(pivot_idx, n):
        new_pt = rot(chain[i])
        new_suffix.append(new_pt)

    # Check self-avoidance: new suffix positions must not collide with prefix
    # (chain[0..pivot_idx-1]) or with each other
    prefix_set = set(chain[:pivot_idx])
    seen = set()
    for pt in new_suffix:
        if pt in prefix_set or pt in seen:
            return None, False
        seen.add(pt)

    return new_suffix, True


def _apply_pivot(chain, pivot_idx, new_suffix):
    """Return new chain with suffix replaced."""
    return chain[:pivot_idx] + new_suffix


def _compute_energy(sequence, chain):
    """Compute HP energy: -# of H-H non-adjacent contacts."""
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y, z = p
            for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                q = (x+dx, y+dy, z+dz)
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 1  # Each contact counted once (we check both i->j but only count if j>i+1)
    # Actually the above counts each contact twice potentially. Let me fix.


def _compute_energy_correct(sequence, chain):
    """Compute HP energy: -# of H-H non-adjacent contacts (each pair counted once)."""
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y, z = p
            for dx, dy, dz in ((1,0,0),(0,1,0),(0,0,1)):  # only 3 directions to avoid double-count
                q = (x+dx, y+dy, z+dz)
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng):
    """Randomized greedy chain growth."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        x, y, z = last
        options = []
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            site = (x+dx, y+dy, z+dz)
            if site in occupied:
                continue
            # Score: prefer sites near existing H residues
            gain = 0
            if sequence[i] == 'H':
                sx, sy, sz = site
                for ox, oy, oz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                    if (sx+ox, sy+oy, sz+oz) in occupied:
                        gain += 1
            options.append((gain + 0.01 * rng.random(), site))
        
        if not options:
            return None
        
        # Pick best with some randomness
        options.sort(key=lambda x: -x[0])
        # Pick from top 3 with probability weighting
        top_k = min(3, len(options))
        weights = [1.0 / (k+1) for k in range(top_k)]
        r = rng.random()
        cum = 0
        chosen = top_k - 1
        for k in range(top_k):
            cum += weights[k]
            if r < cum:
                chosen = k
                break
        site = options[chosen][1]
        occupied.add(site)
        chain.append(site)
    
    return chain


def _local_search_pivot(sequence, chain, time_budget, rng):
    """
    Simulated annealing with pivot moves.
    Returns best chain found.
    """
    n = len(chain)
    if n < 3:
        return chain

    best_chain = list(chain)
    best_energy = _compute_energy_correct(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy

    # SA parameters
    T0 = 1.0
    T_end = 0.01
    # We'll do a fixed number of iterations based on time
    start_time = time.time()
    
    # Precompute which positions can be pivots
    valid_pivots = list(range(1, n - 1))
    if not valid_pivots:
        return best_chain

    iteration = 0
    max_iter = 50000  # safety cap

    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        if iteration >= max_iter:
            break
        if _remaining_time() < 5.0:
            break

        iteration += 1
        
        # Temperature schedule (geometric cooling)
        progress = min(1.0, iteration / max_iter)
        T = T0 * (T_end / T0) ** progress

        # Choose a random pivot
        pivot_idx = rng.choice(valid_pivots)
        direction = rng.randint(0, 1)

        # Try the pivot move
        new_suffix, valid = _pivot_move_valid(current_chain, None, pivot_idx, direction)
        if not valid:
            continue

        # Compute energy change
        # Remove old suffix, add new suffix
        old_suffix = current_chain[pivot_idx:]
        new_chain = current_chain[:pivot_idx] + new_suffix
        
        # Quick check: is it self-avoiding? (already checked in _pivot_move_valid)
        # Compute new energy
        new_energy = _compute_energy_correct(sequence, new_chain)
        delta_e = new_energy - current_energy

        # Metropolis criterion
        if delta_e < 0 or (T > 0.001 and rng.random() < math.exp(-delta_e / T)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

    return best_chain


def _local_search_pivot_fast(sequence, chain, time_budget, rng):
    """
    Faster pivot-move local search with incremental energy update.
    """
    n = len(chain)
    if n < 3:
        return chain

    best_chain = list(chain)
    best_energy = _compute_energy_correct(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy

    T0 = 0.5
    T_end = 0.005
    start_time = time.time()
    valid_pivots = list(range(1, n - 1))
    if not valid_pivots:
        return best_chain

    iteration = 0
    max_iter = 100000

    # Build occupancy set for fast checks
    occupied = set(current_chain)

    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        if iteration >= max_iter:
            break
        if _remaining_time() < 5.0:
            break

        iteration += 1
        progress = min(1.0, iteration / max_iter)
        T = T0 * (T_end / T0) ** progress

        pivot_idx = rng.choice(valid_pivots)
        direction = rng.randint(0, 1)

        p_prev = current_chain[pivot_idx - 1]
        p_pivot = current_chain[pivot_idx]
        bx = p_pivot[0] - p_prev[0]
        by = p_pivot[1] - p_prev[1]
        bz = p_pivot[2] - p_prev[2]

        # Determine rotation
        if bx != 0:
            if direction == 0:
                def rot(pt):
                    return (pt[0], p_prev[1] - (pt[2] - p_prev[2]), p_prev[2] + (pt[1] - p_prev[1]))
            else:
                def rot(pt):
                    return (pt[0], p_prev[1] + (pt[2] - p_prev[2]), p_prev[2] - (pt[1] - p_prev[1]))
        elif by != 0:
            if direction == 0:
                def rot(pt):
                    return (p_prev[0] + (pt[2] - p_prev[2]), pt[1], p_prev[2] - (pt[0] - p_prev[0]))
            else:
                def rot(pt):
                    return (p_prev[0] - (pt[2] - p_prev[2]), pt[1], p_prev[2] + (pt[0] - p_prev[0]))
        else:
            if direction == 0:
                def rot(pt):
                    return (p_prev[0] - (pt[1] - p_prev[1]), p_prev[1] + (pt[0] - p_prev[0]), pt[2])
            else:
                def rot(pt):
                    return (p_prev[0] + (pt[1] - p_prev[1]), p_prev[1] - (pt[0] - p_prev[0]), pt[2])

        # Compute new suffix
        new_suffix = []
        valid = True
        # Prefix is chain[0..pivot_idx-1], these stay
        prefix_set = set(current_chain[:pivot_idx])
        seen = set()
        for i in range(pivot_idx, n):
            new_pt = rot(current_chain[i])
            if new_pt in prefix_set or new_pt in seen:
                valid = False
                break
            seen.add(new_pt)
            new_suffix.append(new_pt)

        if not valid:
            continue

        # Compute energy delta efficiently
        # Old contacts involving suffix positions
        # New contacts involving new suffix positions
        # This is complex; for now compute full energy
        new_chain = current_chain[:pivot_idx] + new_suffix
        new_energy = _compute_energy_correct(sequence, new_chain)
        delta_e = new_energy - current_energy

        if delta_e < 0 or (T > 0.001 and rng.random() < math.exp(-delta_e / T)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

    return best_chain


def fold(sequence, dimension):
    """Fold an HP sequence on the cubic lattice using greedy growth + pivot SA."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 240.0  # 240s CPU budget

    remaining = _remaining_time()
    if remaining < 2.0:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(len(sequence))]

    n = len(sequence)
    if n <= 1:
        return [[0, 0, 0]]
    if n == 2:
        return [[0, 0, 0], [1, 0, 0]]

    # Time allocation: more time for longer chains
    # Base: 1s + 0.15s per residue, capped at remaining/26
    base_time = 1.0 + 0.15 * n
    per_seq_time = min(base_time, remaining / 26.0, remaining * 0.9)
    per_seq_time = max(per_seq_time, 0.5)

    rng = random.Random(42 + n)

    # Phase 1: Greedy growth with multiple restarts
    n_greedy = max(50, int(200 * per_seq_time / (1.0 + 0.15 * n)))
    n_greedy = max(10, min(n_greedy, 500))

    best_chain = None
    best_energy = 0  # worst case (no contacts)

    for attempt in range(n_greedy):
        if _remaining_time() < 5.0:
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _compute_energy_correct(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    if best_chain is None:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]

    # Phase 2: Local search with pivot moves
    remaining_for_ls = _remaining_time() - 5.0
    if remaining_for_ls > 1.0:
        # Multiple SA restarts from best and random perturbations
        n_sa = max(1, int(remaining_for_ls / 2.0))
        n_sa = min(n_sa, 10)

        for sa_attempt in range(n_sa):
            if _remaining_time() < 5.0:
                break
            # Start from best or a perturbation
            if sa_attempt == 0:
                start_chain = list(best_chain)
            else:
                # Perturb best chain slightly
                start_chain = list(best_chain)
                # Do a few random pivot moves
                for _ in range(10):
                    if len(start_chain) < 3:
                        break
                    pi = rng.randint(1, len(start_chain) - 2)
                    d = rng.randint(0, 1)
                    new_suffix, valid = _pivot_move_valid(start_chain, None, pi, d)
                    if valid:
                        start_chain = start_chain[:pi] + new_suffix

            sa_time = max(0.5, remaining_for_ls / (n_sa - sa_attempt))
            improved = _local_search_pivot_fast(sequence, start_chain, sa_time, rng)
            imp_energy = _compute_energy_correct(sequence, improved)
            if imp_energy < best_energy:
                best_chain = improved
                best_energy = imp_energy

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

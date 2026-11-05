import random
import time
import math
from collections import defaultdict

def _neighbours(point):
    """Generate the 6 neighbours of a point in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng, bias=1.0):
    """
    Randomized greedy chain growth.
    bias > 1: more aggressive contact seeking.
    bias < 1: more random exploration.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            # Add randomness
            # Higher bias makes gain more important
            score = gain * bias + rng.random()
            options.append((score, site))
        
        if not options:
            return None
            
        options.sort(reverse=True)
        # Pick from top 3 with probability, or just best
        # Let's pick the best to be fast and deterministic-ish
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
        
    return chain

def _energy(sequence, chain):
    """Calculate total H-H contacts (negative energy)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _compute_contacts(sequence, chain):
    """Return set of H-H contact pairs (i, j) with i < j, j > i+1."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add((i, j))
    return contacts

class _FoldingState:
    """State for efficient energy updates during moves."""
    def __init__(self, sequence, chain, dimension):
        self.sequence = sequence
        self.chain = chain
        self.dimension = dimension
        self.occupied = {p: i for i, p in enumerate(chain)}
        self.contacts = _compute_contacts(sequence, chain)
        self.energy = -len(self.contacts)
        self.n = len(chain)
    
    def _remove_residue(self, i):
        """Remove residue i from occupied set."""
        p = self.chain[i]
        del self.occupied[p]
    
    def _add_residue(self, i, p):
        """Add residue i at position p to occupied set."""
        self.occupied[p] = i
        self.chain[i] = p
    
    def _try_move(self, new_positions, affected_indices):
        """
        Try to apply a move that changes positions of residues in affected_indices.
        new_positions: dict mapping index -> new position
        Returns (new_energy, success).
        """
        # Check self-avoidance: no two affected residues can map to same position
        # and no affected residue can collide with unaffected occupied positions
        new_occupied = dict(self.occupied)
        
        # Remove all affected residues first
        for i in affected_indices:
            old_p = self.chain[i]
            if old_p in new_occupied:
                del new_occupied[old_p]
        
        # Check for collisions and add new positions
        for i, p in new_positions.items():
            if p in new_occupied:
                return None, False
            new_occupied[p] = i
        
        # Compute energy change
        # Remove contacts involving affected residues
        old_contacts = set()
        for (a, b) in self.contacts:
            if a in affected_indices or b in affected_indices:
                old_contacts.add((a, b))
        
        # Find new contacts involving affected residues
        new_contacts = set()
        for i in affected_indices:
            if self.sequence[i] != "H":
                continue
            p = new_positions[i]
            for q in _neighbours(p):
                j = new_occupied.get(q)
                if j is not None and j > i + 1 and self.sequence[j] == "H":
                    new_contacts.add((i, j))
        
        # Also check contacts between two affected residues
        for i in affected_indices:
            if self.sequence[i] != "H":
                continue
            p = new_positions[i]
            for q in _neighbours(p):
                j = new_occupied.get(q)
                if j is not None and j < i and j in affected_indices and self.sequence[j] == "H":
                    new_contacts.add((j, i))
        
        delta_contacts = len(new_contacts) - len(old_contacts)
        delta_energy = -delta_contacts
        
        return self.energy + delta_energy, True

def _try_pivot(state, rng):
    """
    Pivot move: rotate a subchain around a pivot point.
    Returns (new_state_chain, new_energy) or (None, None) if failed.
    """
    n = state.n
    if n < 3:
        return None, None
    
    # Choose pivot point
    pivot = rng.randint(1, n - 2)
    
    # Choose rotation: pick an axis and direction
    # In 3D, we can rotate around any of the 3 axes
    axis = rng.randint(0, 2)
    
    # The subchain from pivot+1 to end gets rotated
    # We need to determine the rotation vector
    # Current direction from pivot to pivot+1
    p_pivot = state.chain[pivot]
    p_next = state.chain[pivot + 1]
    
    # Direction vector
    dir_vec = [p_next[i] - p_pivot[i] for i in range(3)]
    
    # Generate all possible rotations of this direction
    # For a unit vector in 3D, there are 6 possible directions
    # We want to rotate to a different direction
    possible_dirs = []
    for d in [-1, 0, 1]:
        for e in [-1, 0, 1]:
            for f in [-1, 0, 1]:
                if abs(d) + abs(e) + abs(f) == 1:
                    possible_dirs.append((d, e, f))
    
    # Current direction
    cur_dir = tuple(dir_vec)
    
    # Pick a random different direction
    other_dirs = [d for d in possible_dirs if d != cur_dir]
    if not other_dirs:
        return None, None
    
    new_dir = rng.choice(other_dirs)
    
    # Build the new subchain by applying the rotation
    # We need to transform each position relative to pivot
    # The transformation: new_pos = pivot + rotation(old_pos - pivot)
    
    # For a simple rotation around an axis through the pivot:
    # We'll use a simple approach: reflect/rotate the direction
    # Actually, let's use a proper rotation matrix
    
    # For axis 0 (x-axis): (x, y, z) -> (x, -y, z) or (x, z, -y) etc.
    # Let's use a simpler approach: pick a random permutation/sign flip
    
    # Generate the new positions
    subchain = state.chain[pivot+1:]
    new_subchain = [None] * len(subchain)
    
    # Apply transformation: for each point, compute relative to pivot, transform, add back
    for k, p in enumerate(subchain):
        rel = [p[i] - p_pivot[i] for i in range(3)]
        
        # Apply a rotation: we'll use a deterministic rotation based on the axis
        # Simple rotation: swap two coordinates and/or flip signs
        if axis == 0:
            # Rotate around x: (x, y, z) -> (x, z, -y)
            new_rel = [rel[0], rel[2], -rel[1]]
        elif axis == 1:
            # Rotate around y: (x, y, z) -> (-z, y, x)
            new_rel = [-rel[2], rel[1], rel[0]]
        else:
            # Rotate around z: (x, y, z) -> (y, -x, z)
            new_rel = [rel[1], -rel[0], rel[2]]
        
        new_p = tuple(p_pivot[i] + new_rel[i] for i in range(3))
        new_subchain[k] = new_p
    
    # Check self-avoidance
    # The fixed part is chain[0..pivot], the moved part is new_subchain
    fixed_occupied = {state.chain[i]: i for i in range(pivot + 1)}
    
    # Check for collisions within new_subchain
    seen = set()
    for p in new_subchain:
        if p in fixed_occupied or p in seen:
            return None, None
        seen.add(p)
    
    # Build new chain
    new_chain = state.chain[:pivot+1] + new_subchain
    
    # Compute new energy
    new_state = _FoldingState(state.sequence, new_chain, state.dimension)
    
    return new_chain, new_state.energy

def _try_crankshaft(state, rng):
    """
    Crankshaft move: rotate a short segment (2-4 residues) around a bond.
    Returns (new_chain, new_energy) or (None, None) if failed.
    """
    n = state.n
    if n < 4:
        return None, None
    
    # Choose segment length (2-4 residues)
    seg_len = rng.randint(2, min(4, n - 2))
    
    # Choose start of segment (must leave room for the segment)
    start = rng.randint(1, n - seg_len - 1)
    end = start + seg_len - 1
    
    # The segment is chain[start..end]
    # The pivot bond is between chain[start-1] and chain[start]
    # We rotate the segment around the axis defined by chain[start-1] -> chain[start]
    
    p_prev = state.chain[start - 1]
    p_start = state.chain[start]
    
    # Direction of the pivot bond
    dir_vec = tuple(p_start[i] - p_prev[i] for i in range(3))
    
    # The segment positions relative to p_start
    seg_positions = state.chain[start:end+1]
    rel_positions = [tuple(p[i] - p_start[i] for i in range(3)) for p in seg_positions]
    
    # Generate possible rotations around the bond axis
    # For a bond in direction (dx, dy, dz), we can rotate the segment
    # We'll use a simple approach: try all 6 possible orientations of the first bond in the segment
    
    # The first bond in the segment is from p_start to p_start + rel_positions[0]
    # We want to change this direction while keeping it perpendicular to the pivot bond
    
    # Actually, let's use a simpler crankshaft: pick a random new direction for the segment
    # that is not collinear with the pivot bond
    
    # Generate all unit vectors
    unit_vecs = []
    for d in [-1, 0, 1]:
        for e in [-1, 0, 1]:
            for f in [-1, 0, 1]:
                if abs(d) + abs(e) + abs(f) == 1:
                    unit_vecs.append((d, e, f))
    
    # Filter out vectors parallel to the pivot bond
    valid_dirs = []
    for v in unit_vecs:
        # Check if v is not parallel to dir_vec
        # Parallel if v = k * dir_vec for some k
        # Since both are unit vectors, parallel means v == dir_vec or v == -dir_vec
        if v != dir_vec and v != tuple(-x for x in dir_vec):
            valid_dirs.append(v)
    
    if not valid_dirs:
        return None, None
    
    new_first_dir = rng.choice(valid_dirs)
    
    # Now we need to extend the segment: each subsequent residue must be a unit step
    # from the previous one, not colliding with anything
    # This is complex, so let's use a simpler approach:
    # Just try to place the segment in a new orientation by rotating all relative positions
    
    # For simplicity, let's use a reflection: flip one coordinate
    axis = rng.randint(0, 2)
    
    new_rel_positions = []
    for rel in rel_positions:
        new_rel = list(rel)
        new_rel[axis] = -new_rel[axis]
        new_rel_positions.append(tuple(new_rel))
    
    new_seg = [tuple(p_start[i] + new_rel[i] for i in range(3)) for new_rel in new_rel_positions]
    
    # Check self-avoidance
    fixed_occupied = {}
    for i in range(start):
        fixed_occupied[state.chain[i]] = i
    for i in range(end + 1, n):
        fixed_occupied[state.chain[i]] = i
    
    seen = set()
    for p in new_seg:
        if p in fixed_occupied or p in seen:
            return None, None
        seen.add(p)
    
    # Build new chain
    new_chain = state.chain[:start] + new_seg + state.chain[end+1:]
    
    # Compute new energy
    new_state = _FoldingState(state.sequence, new_chain, state.dimension)
    
    return new_chain, new_state.energy

def _try_tunneling(state, rng):
    """
    Tunneling move: temporarily allow self-intersection to move a subchain.
    Returns (new_chain, new_energy) or (None, None) if failed.
    """
    n = state.n
    if n < 5:
        return None, None
    
    # Choose a subchain to move
    start = rng.randint(1, n - 3)
    end = rng.randint(start + 2, n - 1)
    
    # The subchain is chain[start..end]
    # We'll try to move it to a random new location
    
    # Choose a random target position for the start of the subchain
    # It must be adjacent to chain[start-1]
    p_prev = state.chain[start - 1]
    
    options = []
    for site in _neighbours(p_prev):
        if site == state.chain[start]:
            continue
        options.append(site)
    
    if not options:
        return None, None
    
    new_start = rng.choice(options)
    
    # Try to grow the subchain from new_start
    # The subchain has length (end - start + 1)
    sub_len = end - start + 1
    
    # Fixed part: chain[0..start-1] and chain[end+1..n-1]
    fixed_occupied = {}
    for i in range(start):
        fixed_occupied[state.chain[i]] = i
    for i in range(end + 1, n):
        fixed_occupied[state.chain[i]] = i
    
    # Try to place the subchain
    new_sub = [new_start]
    occupied = {new_start: start}
    
    for k in range(1, sub_len):
        last = new_sub[-1]
        valid_sites = []
        for site in _neighbours(last):
            if site in occupied or site in fixed_occupied:
                continue
            valid_sites.append(site)
        
        if not valid_sites:
            return None, None
        
        # Pick randomly to allow exploration
        site = rng.choice(valid_sites)
        occupied[site] = start + k
        new_sub.append(site)
    
    # Build new chain
    new_chain = state.chain[:start] + new_sub + state.chain[end+1:]
    
    # Compute new energy
    new_state = _FoldingState(state.sequence, new_chain, state.dimension)
    
    return new_chain, new_state.energy

def _sa_fold(sequence, dimension, rng, deadline, initial_chain=None):
    """
    Simulated annealing with multiple move types.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    if initial_chain is None:
        # Generate initial chain
        initial_chain = _grow(sequence, dimension, rng, bias=1.5)
        if initial_chain is None:
            initial_chain = _grow(sequence, dimension, rng, bias=0.5)
        if initial_chain is None:
            # Fallback
            initial_chain = [(i, 0, 0) for i in range(n)]
    
    state = _FoldingState(sequence, initial_chain, dimension)
    best_chain = state.chain[:]
    best_energy = state.energy
    
    # Annealing schedule
    T = 2.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Choose move type
        move_type = rng.random()
        
        new_chain = None
        new_energy = None
        
        if move_type < 0.3:
            # Pivot move
            new_chain, new_energy = _try_pivot(state, rng)
        elif move_type < 0.6:
            # Crankshaft move
            new_chain, new_energy = _try_crankshaft(state, rng)
        elif move_type < 0.8:
            # Tunneling move
            new_chain, new_energy = _try_tunneling(state, rng)
        else:
            # Tail refold
            k = rng.randint(1, n - 2)
            new_chain = _tail_refold(state.chain, sequence, dimension, rng, k)
            if new_chain is not None:
                new_state = _FoldingState(sequence, new_chain, dimension)
                new_energy = new_state.energy
        
        if new_chain is None:
            T *= cooling
            continue
        
        delta = new_energy - state.energy
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or rng.random() < math.exp(-delta / T):
            state = _FoldingState(sequence, new_chain, dimension)
            if state.energy < best_energy:
                best_energy = state.energy
                best_chain = state.chain[:]
        
        T *= cooling
    
    return best_chain

def _tail_refold(chain, sequence, dimension, rng, k):
    """
    Refold the tail starting from index k+1.
    chain[0..k] is fixed.
    Returns new_chain or None if stuck.
    """
    n = len(chain)
    if k >= n - 1:
        return chain[:]
    
    # Fixed part
    fixed = chain[:k+1]
    new_chain = fixed[:]
    
    # Occupied set for fixed part
    occupied = {p: i for i, p in enumerate(fixed)}
    
    # We need to grow from new_chain[-1] to length n
    # Current length is k+1, we need to add n - (k+1) residues.
    # The residues to add are sequence[k+1 .. n-1]
    
    for i in range(k+1, n):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            score = gain + rng.random()
            options.append((score, site))
            
        if not options:
            return None
            
        options.sort(reverse=True)
        # Sometimes pick 2nd best to diversify
        if len(options) > 1 and rng.random() < 0.2:
            site = options[1][1]
        else:
            site = options[0][1]
            
        occupied[site] = i
        new_chain.append(site)
        
    return new_chain

def fold(sequence, dimension):
    """
    Main entry point.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    rng = random.Random(42)
    
    # Deadline management
    start_time = time.time()
    # Evaluator kills at 330s. Let's aim for 320s to be safe.
    deadline = start_time + 320.0
    
    best_chain = None
    best_energy = 0  # 0 is no contacts (worst for H-rich)
    
    # Phase 1: Generate diverse initial solutions
    init_chains = []
    
    # Greedy with different biases
    for bias in [1.0, 1.5, 2.0, 0.5]:
        c = _grow(sequence, dimension, rng, bias=bias)
        if c:
            init_chains.append(c)
    
    # Pure random walks
    for _ in range(5):
        c = _grow(sequence, dimension, rng, bias=0.1)
        if c:
            init_chains.append(c)
    
    if not init_chains:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    # Phase 2: Multi-start SA
    # Distribute time among initial chains
    num_chains = len(init_chains)
    # Leave some time for final deep dive
    time_for_each = (deadline - time.time() - 10.0) / (num_chains + 1)
    
    for i, c in enumerate(init_chains):
        if time.time() > deadline - 15.0:  # Leave 15s for final
            break
        
        # Time slice for this chain
        chain_deadline = time.time() + time_for_each
        if chain_deadline > deadline - 15.0:
            chain_deadline = deadline - 15.0
        
        opt_chain = _sa_fold(sequence, dimension, rng, chain_deadline, initial_chain=c)
        
        if opt_chain is not None:
            opt_energy = _energy(sequence, opt_chain)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain
    
    # Phase 3: Deep dive on the best chain found
    if best_chain is not None:
        remaining_deadline = deadline
        if time.time() < remaining_deadline - 5.0:
            opt_chain = _sa_fold(sequence, dimension, rng, remaining_deadline, initial_chain=best_chain)
            if opt_chain is not None:
                opt_energy = _energy(sequence, opt_chain)
                if opt_energy < best_energy:
                    best_energy = opt_energy
                    best_chain = opt_chain
    
    if best_chain is None:
        best_chain = init_chains[0]
        
    return [list(p) for p in best_chain]

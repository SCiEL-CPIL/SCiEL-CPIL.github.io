"""HP lattice protein folding on the cubic lattice with contact-based energy and local search."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _lattice_adjacent(p1, p2):
    """Check if two points are adjacent on the lattice."""
    return sum(abs(a - b) for a, b in zip(p1, p2)) == 1


def _energy(sequence, chain):
    """Compute energy as negative of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _count_contacts(sequence, chain):
    """Count H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return contacts


class _FoldingState:
    """State for incremental folding with contact tracking."""
    
    def __init__(self, sequence, dimension):
        self.sequence = sequence
        self.dimension = dimension
        self.chain = []
        self.occupied = {}  # point -> index
        self.contacts = 0
        self._hh_positions = set()  # indices of H residues
        for i, s in enumerate(sequence):
            if s == 'H':
                self._hh_positions.add(i)
    
    def _add_contact_gain(self, pos, idx):
        """Return number of new H-H contacts if we place residue idx at pos."""
        if self.sequence[idx] != 'H':
            return 0
        gain = 0
        for q in _neighbours(pos):
            if q in self.occupied:
                j = self.occupied[q]
                if j < idx - 1 and self.sequence[j] == 'H':
                    gain += 1
        return gain
    
    def _remove_contact_loss(self, pos, idx):
        """Return number of H-H contacts lost if we remove residue idx from pos."""
        if self.sequence[idx] != 'H':
            return 0
        loss = 0
        for q in _neighbours(pos):
            if q in self.occupied:
                j = self.occupied[q]
                if j < idx - 1 and self.sequence[j] == 'H':
                    loss += 1
        return loss
    
    def grow(self, rng, temperature=0.5):
        """Grow the chain using contact-based scoring."""
        start = (0,) * self.dimension
        self.chain = [start]
        self.occupied = {start: 0}
        self.contacts = 0
        
        n = len(self.sequence)
        for i in range(1, n):
            options = []
            for site in _neighbours(self.chain[-1]):
                if site in self.occupied:
                    continue
                
                gain = self._add_contact_gain(site, i)
                free = sum(1 for q in _neighbours(site) if q not in self.occupied)
                
                score = gain * 10.0 + free * 0.1 + rng.random() * temperature
                options.append((score, site))
            
            if not options:
                return False
            
            options.sort(key=lambda x: -x[0])
            if len(options) == 1:
                site = options[0][1]
            else:
                scores = [o[0] for o in options]
                max_score = max(scores)
                weights = [math.exp((s - max_score) / max(temperature, 0.1)) for s in scores]
                total = sum(weights)
                r = rng.random() * total
                cum = 0
                site = options[-1][1]
                for idx, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        site = options[idx][1]
                        break
            
            self.occupied[site] = i
            self.chain.append(site)
            self.contacts += self._add_contact_gain(site, i)
        
        return True
    
    def pivot_move(self, i, rng):
        """
        Perform a pivot move at position i.
        Rotates the segment from i to end around the bond (i-1, i).
        Returns (success, energy_delta).
        """
        if i < 2 or i >= len(self.chain) - 1:
            return False, 0
        
        # Get the bond direction
        bond = tuple(self.chain[i] - self.chain[i-1] for _ in [0])
        
        # Find possible rotations: the segment can be rotated to 4 other orientations
        # around the bond axis
        axis = None
        for a in range(self.dimension):
            if bond[a] != 0:
                axis = a
                break
        
        if axis is None:
            return False, 0
        
        # The perpendicular axes
        perp_axes = [a for a in range(self.dimension) if a != axis]
        
        # Current direction of the segment from i
        if i + 1 < len(self.chain):
            curr_dir = tuple(self.chain[i+1] - self.chain[i] for _ in [0])
        else:
            curr_dir = None
        
        # Try rotating the segment
        # For a 90-degree rotation around the bond axis, we need to find
        # where the segment would go
        
        # Simpler approach: try all 5 possible directions for the bond (i-1, i)
        # that are different from the current one
        new_chain = list(self.chain[:i])
        
        # The pivot point is chain[i-1]
        pivot_point = self.chain[i-1]
        
        # Possible new directions for the bond from pivot to new position i
        # Must be different from current bond direction and not going back
        current_bond = bond
        possible_dirs = []
        for axis_idx in range(self.dimension):
            for step in (-1, 1):
                d = [0] * self.dimension
                d[axis_idx] = step
                d = tuple(d)
                if d != current_bond:
                    possible_dirs.append(d)
        
        # Also include the reverse of current bond (180 degree turn)
        reverse_bond = tuple(-x for x in current_bond)
        if reverse_bond not in possible_dirs:
            possible_dirs.append(reverse_bond)
        
        rng.shuffle(possible_dirs)
        
        for new_bond in possible_dirs:
            new_pos_i = tuple(pivot_point[a] + new_bond[a] for a in range(self.dimension))
            
            # Check if new_pos_i is valid
            if new_pos_i in self.occupied:
                # Skip if occupied by a residue that's not in the moving segment
                if new_pos_i not in self.chain[i:]:
                    continue
            
            # Try to place the rest of the segment
            new_chain = list(self.chain[:i])
            new_chain.append(new_pos_i)
            
            # Place the rest of the chain
            valid = True
            for j in range(i + 1, len(self.chain)):
                # The direction from j-1 to j should be the same as before
                old_dir = tuple(self.chain[j] - self.chain[j-1] for _ in [0])
                new_pos_j = tuple(new_chain[-1][a] + old_dir[a] for a in range(self.dimension))
                
                if new_pos_j in self.occupied and new_pos_j not in self.chain[i:]:
                    valid = False
                    break
                if new_pos_j in new_chain:
                    valid = False
                    break
                
                new_chain.append(new_pos_j)
            
            if not valid:
                continue
            
            # Check self-avoidance
            if len(set(new_chain)) != len(new_chain):
                continue
            
            # Compute energy change
            old_contacts = self.contacts
            new_energy = _energy(self.sequence, new_chain)
            old_energy = -self.contacts
            delta = new_energy - old_energy
            
            # Accept if better or with some probability
            if delta <= 0 or rng.random() < math.exp(-delta / 0.5):
                # Update state
                self.chain = new_chain
                self.occupied = {p: idx for idx, p in enumerate(new_chain)}
                self.contacts = -new_energy
                return True, delta
        
        return False, 0
    
    def crankshaft_move(self, i, rng):
        """
        Crankshaft move: rotate a subchain segment while keeping endpoints fixed.
        """
        if i < 3 or i >= len(self.chain) - 3:
            return False, 0
        
        # The segment from i to i+2 is rotated around the bond (i-1, i)
        # while keeping the bond (i+2, i+3) fixed
        
        old_energy = -self.contacts
        
        # Try rotating the middle part
        # This is complex, so let's try a simpler version:
        # Move residue i+1 to a new position
        
        target_idx = i + 1
        old_pos = self.chain[target_idx]
        
        # Try all neighbours of old_pos that are valid
        for new_pos in _neighbours(old_pos):
            # Check if new_pos is valid
            if new_pos == self.chain[target_idx - 1]:
                continue
            if new_pos == self.chain[target_idx + 1]:
                continue
            if new_pos in self.occupied:
                continue
            
            # Check if the move is valid (unit steps)
            if not _lattice_adjacent(self.chain[target_idx - 1], new_pos):
                continue
            if not _lattice_adjacent(new_pos, self.chain[target_idx + 1]):
                continue
            
            # Make the move
            new_chain = list(self.chain)
            new_chain[target_idx] = new_pos
            
            # Check self-avoidance
            if len(set(new_chain)) != len(new_chain):
                continue
            
            # Compute energy
            new_energy = _energy(self.sequence, new_chain)
            delta = new_energy - old_energy
            
            if delta <= 0 or rng.random() < math.exp(-delta / 0.5):
                self.chain = new_chain
                self.occupied = {p: idx for idx, p in enumerate(new_chain)}
                self.contacts = -new_energy
                return True, delta
        
        return False, 0
    
    def local_search(self, max_steps=100, rng=None):
        """Local search using pivot and crankshaft moves."""
        if rng is None:
            rng = random.Random()
        
        for _ in range(max_steps):
            # Randomly choose a move type
            if rng.random() < 0.5:
                i = rng.randint(2, len(self.chain) - 2)
                success, delta = self.pivot_move(i, rng)
            else:
                i = rng.randint(3, len(self.chain) - 4)
                success, delta = self.crankshaft_move(i, rng)
            
            if not success:
                continue
        
        return self.chain, -self.contacts


def _grow_with_contacts(sequence, dimension, rng, temperature=0.5):
    """Grow the chain using a contact-based scoring function."""
    state = _FoldingState(sequence, dimension)
    if state.grow(rng, temperature):
        return state.chain
    return None


def _local_search(sequence, chain, max_steps=100, rng=None):
    """Local search using pivot-like moves."""
    if rng is None:
        rng = random.Random()
    
    state = _FoldingState(sequence, len(chain[0]))
    state.chain = list(chain)
    state.occupied = {p: i for i, p in enumerate(chain)}
    state.contacts = _count_contacts(sequence, chain)
    
    state.local_search(max_steps, rng)
    return state.chain


def _straight_line(sequence, dimension):
    """Return a straight line conformation (always valid)."""
    return [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]


def fold(sequence, dimension):
    """
    Fold the sequence into a 3D conformation on the cubic lattice.
    Returns a list of lattice points (lists of length dimension).
    """
    start_time = time.time()
    deadline = start_time + 250  # Leave 80s safety margin
    
    rng = random.Random(42 + len(sequence))
    
    n = len(sequence)
    
    # For very short sequences, just do a simple search
    if n <= 10:
        best_chain = _straight_line(sequence, dimension)
        best_energy = _energy(sequence, best_chain)
        for _ in range(50):
            chain = _grow_with_contacts(sequence, dimension, rng, temperature=1.0)
            if chain is None:
                continue
            e = _energy(sequence, chain)
            if e < best_energy:
                best_chain = chain
                best_energy = e
        return [list(p) for p in best_chain]
    
    best_chain = None
    best_energy = 1  # Start with a high (bad) energy
    
    # Phase 1: Multiple growth attempts with different temperatures
    num_growths = 50
    for i in range(num_growths):
        if time.time() > deadline:
            break
        
        temp = 0.3 + 0.7 * (i / max(1, num_growths - 1))
        chain = _grow_with_contacts(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue
        
        e = _energy(sequence, chain)
        if e < best_energy:
            best_chain = chain
            best_energy = e
    
    if best_chain is None:
        best_chain = _straight_line(sequence, dimension)
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search on the best chain
    if time.time() < deadline - 30:
        improved = _local_search(sequence, best_chain, max_steps=200, rng=rng)
        if improved is not None:
            e = _energy(sequence, improved)
            if e < best_energy:
                best_chain = improved
                best_energy = e
    
    # Phase 3: More growth attempts with local search
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        num_additional = min(30, int(remaining_time / 2))
        for i in range(num_additional):
            if time.time() > deadline:
                break
            
            temp = 0.5 + rng.random() * 0.5
            chain = _grow_with_contacts(sequence, dimension, rng, temperature=temp)
            if chain is None:
                continue
            
            # Apply local search
            chain = _local_search(sequence, chain, max_steps=50, rng=rng)
            
            e = _energy(sequence, chain)
            if e < best_energy:
                best_chain = chain
                best_energy = e
    
    # Ensure we return a valid conformation
    if best_chain is None or len(best_chain) != n:
        best_chain = _straight_line(sequence, dimension)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

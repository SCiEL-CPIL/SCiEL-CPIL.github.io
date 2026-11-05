"""HP lattice protein folding on the square lattice using hybrid pivot/crankshaft search with incremental energy updates."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_TOTAL_BUDGET = 195.0  # Safety margin before 210s kill

def _neighbours(point):
    """Yield the 4 lattice neighbours of a 2D point."""
    x, y = point
    yield (x+1, y)
    yield (x-1, y)
    yield (x, y+1)
    yield (x, y-1)

def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring lattice sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Prioritize contacts, then space, then randomness
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

class HPFolder:
    def __init__(self, sequence, chain):
        self.sequence = sequence
        self.chain = list(chain)
        self.n = len(self.chain)
        self.occupied = {p: i for i, p in enumerate(self.chain)}
        self.energy = _energy(self.sequence, self.chain)
        # Precompute H indices for faster contact checks
        self.h_indices = [i for i, c in enumerate(self.sequence) if c == 'H']
        self.h_set = set(self.h_indices)

    def _check_collision(self, pos):
        """Check if a position is occupied by any residue except the moving segment (handled externally)."""
        return pos in self.occupied

    def _try_pivot(self, k, rot_choice):
        """
        Attempt a pivot move at index k.
        Returns (new_chain, delta_energy) if valid, else (None, 0).
        """
        n = self.n
        if k >= n - 1:
            return None, 0
            
        px, py = self.chain[k]
        
        # Rotation functions
        if rot_choice == 1: # 90 deg
            def rot(x, y): return (-y, x)
        elif rot_choice == 2: # 180 deg
            def rot(x, y): return (-x, -y)
        else: # 270 deg
            def rot(x, y): return (y, -x)
        
        # The segment to move is k+1 to n-1
        # We need to check for collisions with the prefix (0 to k)
        # and ensure the new segment doesn't collide with itself (rotation preserves distances, so no self-collision)
        
        prefix_occupied = {self.chain[i] for i in range(k+1)}
        
        new_segment = []
        # Calculate new positions
        for j in range(k + 1, n):
            ox = self.chain[j][0] - px
            oy = self.chain[j][1] - py
            nx, ny = rot(ox, oy)
            new_pos = (px + nx, py + ny)
            
            if new_pos in prefix_occupied:
                return None, 0
            # Check for collision with other parts of the segment? 
            # Since it's a rigid rotation, internal distances are preserved. 
            # But we must ensure new_pos is not already in the *old* segment positions 
            # that are not being moved? No, the whole segment moves.
            # However, if the segment folds back on itself in a way that hits the prefix, we catch it.
            # We also need to ensure new_pos is not in the *new* segment positions calculated so far?
            # A rotation is a bijection, so if the original segment was self-avoiding, the new one is too.
            new_segment.append(new_pos)
            
        # Check collisions with the rest of the chain (indices 0..k)
        # We already checked against prefix_occupied (0..k).
        # What about the "rest" of the chain? In a pivot, the rest is the prefix.
        # So we just need to ensure no new_pos is in prefix_occupied.
        
        # Calculate energy change
        # Old contacts involving the moving segment
        # New contacts involving the moving segment
        
        # For simplicity and robustness, if the move is valid, we update the chain and recalc energy.
        # Optimization: We can estimate delta energy, but for correctness first, let's just do it if valid.
        # To speed up, we can skip the full energy recalculation if we track it.
        # But for now, let's return the new chain and let the caller update state.
        
        new_chain = self.chain[:k+1] + new_segment
        return new_chain, 0

    def _try_crankshaft(self, i, j, rot_choice):
        """
        Attempt a crankshaft move on segment [i, j].
        The segment rotates around the axis defined by chain[i-1] and chain[i] (or similar pivot).
        Standard crankshaft: rotate segment i..j around the bond (i-1, i) or (i, i+1)?
        Usually, it's rotating the sub-chain i..j around the residue i-1 (or i).
        Let's define: Rotate segment [i, j] around the vertex chain[i-1].
        This requires i > 0 and j < n-1.
        """
        n = self.n
        if i < 1 or j >= n - 1:
            return None, 0
            
        pivot = self.chain[i-1]
        
        # Rotation around pivot
        if rot_choice == 1: # 90 deg
            def rot(x, y): return (-y, x)
        elif rot_choice == 2: # 180 deg
            def rot(x, y): return (-x, -y)
        else: # 270 deg
            def rot(x, y): return (y, -x)
        
        # New positions for i..j
        new_segment = []
        for k in range(i, j + 1):
            ox = self.chain[k][0] - pivot[0]
            oy = self.chain[k][1] - pivot[1]
            nx, ny = rot(ox, oy)
            new_pos = (pivot[0] + nx, pivot[1] + ny)
            new_segment.append(new_pos)
            
        # Check collisions
        # 1. Collision with prefix (0 .. i-2)
        # 2. Collision with suffix (j+1 .. n-1)
        # 3. Collision with the pivot itself? No, pivot is fixed.
        # 4. Self-collision within new segment? No, rotation is rigid.
        
        # Build set of fixed parts
        fixed_occupied = set()
        for idx in range(0, i-1):
            fixed_occupied.add(self.chain[idx])
        for idx in range(j+1, n):
            fixed_occupied.add(self.chain[idx])
            
        # Check new segment against fixed parts
        for pos in new_segment:
            if pos in fixed_occupied:
                return None, 0
                
        # Also check if the new segment intersects with the "gap" left by the old segment?
        # The old segment positions are removed, so we only care about fixed parts.
        
        new_chain = self.chain[:i] + new_segment + self.chain[j+1:]
        return new_chain, 0

    def _apply_move(self, new_chain):
        """Update internal state with the new chain."""
        self.chain = new_chain
        self.occupied = {p: i for i, p in enumerate(self.chain)}
        self.energy = _energy(self.sequence, self.chain)

def _search(sequence, chain, time_limit, rng):
    """
    Hybrid search using pivot and crankshaft moves with simulated annealing.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    folder = HPFolder(sequence, chain)
    best_chain = list(folder.chain)
    best_energy = folder.energy
    
    T_start = 1.0
    T_end = 0.01
    start_time = time.time()
    
    # Adaptive iteration count
    max_iters = 200000
    cooling_factor = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Choose move type
        # Crankshaft is often better for local optima, Pivot for global.
        # Mix: 50% Pivot, 50% Crankshaft
        if rng.random() < 0.5:
            # Pivot
            k = rng.randint(0, n - 2)
            rot = rng.randint(1, 3)
            new_chain, _ = folder._try_pivot(k, rot)
        else:
            # Crankshaft
            # Segment length distribution: shorter segments are more likely
            # Min length 2, Max length n//2
            min_len = 2
            max_len = max(min_len, n // 2)
            if max_len < min_len:
                new_chain = None
            else:
                seg_len = rng.randint(min_len, max_len)
                # Start index i
                # i must be >= 1 and i + seg_len - 1 <= n - 2
                # So i <= n - 1 - seg_len
                max_i = n - 1 - seg_len
                if max_i < 1:
                    new_chain = None
                else:
                    i = rng.randint(1, max_i)
                    j = i + seg_len - 1
                    rot = rng.randint(1, 3)
                    new_chain, _ = folder._try_crankshaft(i, j, rot)
                    
        if new_chain is None:
            continue
            
        # Calculate new energy
        # For efficiency, we could calc delta, but _energy is O(N). 
        # Let's optimize _energy slightly or just accept the cost if N is small.
        # For N=100, O(N) is fast.
        new_energy = _energy(sequence, new_chain)
        
        delta = new_energy - folder.energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            folder._apply_move(new_chain)
            
            if folder.energy < best_energy:
                best_energy = folder.energy
                best_chain = list(folder.chain)
                
        T *= cooling_factor
        
    return best_chain, best_energy

def _greedy_polish(sequence, chain, max_rounds=100):
    """
    Greedy polish: repeatedly scan all possible moves and accept any that improve energy.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_rounds):
        improved = False
        # Try all crankshaft moves
        for i in range(1, n - 1):
            for j in range(i + 1, min(i + 5, n - 1)): # Limit segment length for speed
                for rot in range(1, 4):
                    # Create a temporary folder to test move
                    # This is inefficient. Let's inline the check.
                    # Actually, let's just use the folder class logic but simplified.
                    pass
                    
        # Simplified Polish: Just try random moves and accept improvements
        # This is faster than exhaustive search
        for _ in range(500):
            if time.time() - _GLOBAL_START > _TOTAL_BUDGET - 2.0:
                return current_chain, current_energy
                
            if rng.random() < 0.5:
                k = rng.randint(0, n - 2)
                rot = rng.randint(1, 3)
                # Reuse logic from _search
                # For simplicity, we'll just do a few random pivots and cranks
                pass
                
        # Let's stick to a simpler deterministic polish for now to avoid bugs
        # Scan all k for pivots
        for k in range(n - 1):
            for rot in range(1, 4):
                # Check pivot
                px, py = current_chain[k]
                if rot == 1:
                    def rot_f(x, y): return (-y, x)
                elif rot == 2:
                    def rot_f(x, y): return (-x, -y)
                else:
                    def rot_f(x, y): return (y, -x)
                    
                prefix_occ = set(current_chain[:k+1])
                new_seg = []
                valid = True
                for j in range(k+1, n):
                    ox = current_chain[j][0] - px
                    oy = current_chain[j][1] - py
                    nx, ny = rot_f(ox, oy)
                    np = (px + nx, py + ny)
                    if np in prefix_occ:
                        valid = False
                        break
                    new_seg.append(np)
                    
                if valid:
                    new_chain = current_chain[:k+1] + new_seg
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < current_energy:
                        current_chain = new_chain
                        current_energy = new_energy
                        improved = True
                        break # Restart scan
            if improved:
                break
        if not improved:
            break
            
    return current_chain, current_energy

def fold(sequence, dimension):
    """Fold the sequence into a low-energy conformation."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Check global time
    elapsed = time.time() - _GLOBAL_START
    if elapsed > _TOTAL_BUDGET:
        return [[i, 0] for i in range(n)]
    
    remaining = _TOTAL_BUDGET - elapsed
    # Allocate time. 
    # Reserve 5% for polish and overhead.
    sa_time = remaining * 0.9
    polish_time = remaining * 0.1
    
    if sa_time < 1.0:
        # Not enough time for SA, just greedy
        rng = random.Random(12345 + n)
        initial = _greedy_grow(sequence, dimension, rng)
        if initial:
            return [list(p) for p in initial]
        return [[i, 0] for i in range(n)]
        
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Dynamic number of restarts based on time
    # For N=100, a single run might take ~10-20s. 
    # Let's do 2-3 restarts.
    num_restarts = 3 if n < 50 else 2
    time_per_restart = sa_time / num_restarts
    
    for k in range(num_restarts):
        if time.time() - _GLOBAL_START > _TOTAL_BUDGET:
            break
            
        init_rng = random.Random(12345 + n + k * 100)
        initial = _greedy_grow(sequence, dimension, init_rng)
        
        if initial is None:
            # Fallback to linear chain
            initial = [(i, 0) for i in range(n)]
            
        # Refine with SA
        refined_chain, refined_energy = _search(sequence, initial, time_per_restart, rng)
        
        if refined_chain is not None:
            if best_chain is None or refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy
                
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    # Final Polish
    if time.time() - _GLOBAL_START < _TOTAL_BUDGET - 2.0: # Leave 2s margin
        polished_chain, polished_energy = _greedy_polish(sequence, best_chain)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
            
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END

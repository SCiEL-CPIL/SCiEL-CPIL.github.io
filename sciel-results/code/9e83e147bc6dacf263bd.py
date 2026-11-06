"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 320  # safety margin before 330s kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _initial_energy(sequence, chain):
    """Calculate initial energy for a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth with trap penalty."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            trap_penalty = -0.5 if free == 1 else 0.0
            options.append((gain + 0.1 * free + trap_penalty + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_refine(sequence, chain, dimension, rng, max_iters=2000):
    """
    Local search with incremental energy updates.
    Moves: 1-move, Kink, Jump.
    """
    if len(chain) < 4:
        return chain

    n = len(chain)
    best_chain = list(chain)
    current_chain = list(chain)
    
    # Precompute H indices for quick access? Or just check sequence[i]=='H'
    # We need to track energy incrementally.
    
    # Helper to calculate energy delta for a single residue move
    # Moving residue i from pos_old to pos_new
    # Contacts lost: H-H pairs (i, j) where j is neighbor of pos_old, j != i-1, i+1, and seq[j]=='H', seq[i]=='H'
    # Contacts gained: H-H pairs (i, j) where j is neighbor of pos_new, j != i-1, i+1, and seq[j]=='H', seq[i]=='H'
    
    def get_energy_delta(seq, chain, i, pos_old, pos_new):
        if seq[i] != 'H':
            return 0
        
        delta = 0
        # Remove old contacts
        for q in _neighbours(pos_old):
            # Check if q is occupied by a residue j
            # We need a fast lookup. We maintain occupied dict outside.
            # But this function needs access to occupied.
            pass
        
        # This inline function is tricky without passing occupied.
        # Better to implement the logic inside the loop where occupied is available.
        return 0 # Placeholder

    # We will maintain:
    # occupied: dict pos -> index
    # current_energy: int
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _initial_energy(sequence, current_chain)
    best_energy = current_energy
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
            
        improved = False
        move_type = rng.random()
        
        if move_type < 0.4:
            # 1-residue move
            j = rng.randint(1, n - 2)
            old_pos = current_chain[j]
            # Try neighbors of old_pos
            # To be faster, we can pick a random neighbor
            # Or iterate all 6
            for new_pos in _neighbours(old_pos):
                if new_pos in occupied:
                    continue
                # Check connectivity: dist(new_pos, chain[j-1]) == 1 and dist(new_pos, chain[j+1]) == 1
                prev_pos = current_chain[j-1]
                next_pos = current_chain[j+1]
                
                # Manhattan distance check
                if abs(new_pos[0]-prev_pos[0]) + abs(new_pos[1]-prev_pos[1]) + abs(new_pos[2]-prev_pos[2]) != 1:
                    continue
                if abs(new_pos[0]-next_pos[0]) + abs(new_pos[1]-next_pos[1]) + abs(new_pos[2]-next_pos[2]) != 1:
                    continue
                
                # Calculate energy delta
                delta_e = 0
                if sequence[j] == 'H':
                    # Lose contacts with neighbors of old_pos
                    for q in _neighbours(old_pos):
                        idx_q = occupied.get(q)
                        if idx_q is not None and idx_q != j and abs(idx_q - j) > 1 and sequence[idx_q] == 'H':
                            delta_e -= 1
                    # Gain contacts with neighbors of new_pos
                    for q in _neighbours(new_pos):
                        idx_q = occupied.get(q)
                        if idx_q is not None and idx_q != j and abs(idx_q - j) > 1 and sequence[idx_q] == 'H':
                            delta_e += 1
                
                new_energy = current_energy + delta_e
                
                # Acceptance criteria
                if new_energy < best_energy:
                    # Apply move
                    current_chain[j] = new_pos
                    del occupied[old_pos]
                    occupied[new_pos] = j
                    current_energy = new_energy
                    best_energy = new_energy
                    best_chain = current_chain[:]
                    improved = True
                    break
                elif new_energy == current_energy and rng.random() < 0.3:
                    # Apply move
                    current_chain[j] = new_pos
                    del occupied[old_pos]
                    occupied[new_pos] = j
                    current_energy = new_energy
                    improved = True
                    break
                # else reject, try next neighbor

        elif move_type < 0.65:
            # Kink move: move residues i, i+1
            if n >= 4:
                i = rng.randint(1, n - 3)
                p1_old = current_chain[i]
                p2_old = current_chain[i+1]
                
                # Try moving i to a neighbor of p1_old? No, kink usually means moving the pair.
                # A simpler kink: pick i, try to move i to a new spot p1_new such that p1_new is neighbor of chain[i-1]
                # and chain[i+1] is neighbor of p1_new? No, that's just a 1-move for i if i+1 stays.
                # Standard kink: Move residue i to p1_new and residue i+1 to p2_new.
                # p1_new is neighbor of chain[i-1].
                # p2_new is neighbor of p1_new.
                # chain[i+2] is neighbor of p2_new (if i+2 exists).
                
                # This is complex to do incrementally for 2 residues.
                # Given time constraints, 1-moves and Jumps are often sufficient if fast enough.
                # Let's stick to 1-moves and Jumps for simplicity and speed, maybe add a simple kink later if time permits.
                # For now, treat as 1-move on i+1? No.
                # Let's skip kinks in this version to ensure speed and reliability of incremental energy.
                # The "Jump" move is more powerful for escaping.
                pass

        else:
            # Jump move: move residue j to a neighbor of residue i, |i-j| >= 3
            j = rng.randint(1, n - 2)
            i = rng.randint(0, n - 1)
            if abs(i - j) < 3:
                continue
            
            old_pos = current_chain[j]
            anchor = current_chain[i]
            
            for new_pos in _neighbours(anchor):
                if new_pos in occupied:
                    continue
                
                # Check connectivity for j
                prev_pos = current_chain[j-1]
                next_pos = current_chain[j+1]
                
                if abs(new_pos[0]-prev_pos[0]) + abs(new_pos[1]-prev_pos[1]) + abs(new_pos[2]-prev_pos[2]) != 1:
                    continue
                if abs(new_pos[0]-next_pos[0]) + abs(new_pos[1]-next_pos[1]) + abs(new_pos[2]-next_pos[2]) != 1:
                    continue
                
                # Calculate energy delta
                delta_e = 0
                if sequence[j] == 'H':
                    # Lose contacts with neighbors of old_pos
                    for q in _neighbours(old_pos):
                        idx_q = occupied.get(q)
                        if idx_q is not None and idx_q != j and abs(idx_q - j) > 1 and sequence[idx_q] == 'H':
                            delta_e -= 1
                    # Gain contacts with neighbors of new_pos
                    for q in _neighbours(new_pos):
                        idx_q = occupied.get(q)
                        if idx_q is not None and idx_q != j and abs(idx_q - j) > 1 and sequence[idx_q] == 'H':
                            delta_e += 1
                
                new_energy = current_energy + delta_e
                
                if new_energy < best_energy:
                    # Apply move
                    current_chain[j] = new_pos
                    del occupied[old_pos]
                    occupied[new_pos] = j
                    current_energy = new_energy
                    best_energy = new_energy
                    best_chain = current_chain[:]
                    improved = True
                    break
                elif new_energy <= current_energy + 1 and rng.random() < 0.2:
                    # Accept slightly worse to escape basins
                    current_chain[j] = new_pos
                    del occupied[old_pos]
                    occupied[new_pos] = j
                    current_energy = new_energy
                    improved = True
                    break

        if not improved:
            # Try SA-style escape: accept a worse 1-move
            escaped = False
            for _ in range(5):
                j = rng.randint(1, n - 2)
                old_pos = current_chain[j]
                for new_pos in _neighbours(old_pos):
                    if new_pos in occupied:
                        continue
                    prev_pos = current_chain[j-1]
                    next_pos = current_chain[j+1]
                    
                    if abs(new_pos[0]-prev_pos[0]) + abs(new_pos[1]-prev_pos[1]) + abs(new_pos[2]-prev_pos[2]) != 1:
                        continue
                    if abs(new_pos[0]-next_pos[0]) + abs(new_pos[1]-next_pos[1]) + abs(new_pos[2]-next_pos[2]) != 1:
                        continue
                    
                    delta_e = 0
                    if sequence[j] == 'H':
                        for q in _neighbours(old_pos):
                            idx_q = occupied.get(q)
                            if idx_q is not None and idx_q != j and abs(idx_q - j) > 1 and sequence[idx_q] == 'H':
                                delta_e -= 1
                        for q in _neighbours(new_pos):
                            idx_q = occupied.get(q)
                            if idx_q is not None and idx_q != j and abs(idx_q - j) > 1 and sequence[idx_q] == 'H':
                                delta_e += 1
                    
                    new_energy = current_energy + delta_e
                    if new_energy <= best_energy + 3:
                        current_chain[j] = new_pos
                        del occupied[old_pos]
                        occupied[new_pos] = j
                        current_energy = new_energy
                        escaped = True
                        break
                    else:
                        continue # try next neighbor
                if escaped:
                    break
            
            if not escaped:
                if it > 100:
                    break
                else:
                    continue

    return best_chain


def fold(sequence, dimension):
    """Best conformation using many randomized greedy growths with hybrid local refinement."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))

    best_chain = None
    best_energy = 0  # energy is negative, 0 means no contacts

    # Scale number of attempts based on chain length
    # Longer chains need fewer restarts to fit in time, but each refinement is more expensive.
    if n <= 48:
        num_attempts = 1000
    elif n <= 80:
        num_attempts = 500
    elif n <= 100:
        num_attempts = 250
    else:
        num_attempts = 100

    for attempt in range(num_attempts):
        if time.time() > _DEADLINE - 5:
            break

        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue

        # Refine all successful growths
        # Adjust max_iters based on time remaining?
        # For now, fixed reasonable limit.
        max_iter = min(2000, 40 * n)
        refined = _local_refine(sequence, chain, dimension, rng, max_iters=max_iter)
        value = _initial_energy(sequence, refined)

        if value < best_energy:
            best_chain = refined
            best_energy = value

    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

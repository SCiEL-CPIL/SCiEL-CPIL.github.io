"""HP lattice protein folding on the cubic lattice using beam search and local refinement."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

_DEADLINE = 320.0  # safety margin before 330s kill
_START_TIME = time.time()


def _check_deadline():
    return time.time() - _START_TIME > _DEADLINE


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute negative H-H contact count for a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_score(sequence, chain, occupied, current_index):
    """Score a partial chain: negative contacts so far + potential."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    # Estimate potential: count H residues that could form contacts
    potential = 0
    if current_index < len(sequence) and sequence[current_index] == "H":
        for q in _neighbours(chain[-1]):
            if q not in occupied:
                potential += 1
    return -contacts - 0.1 * potential


def _beam_search(sequence, dimension, beam_width=50, max_depth=None):
    """Beam search over partial self-avoiding walks."""
    n = len(sequence)
    if max_depth is None:
        max_depth = n
    
    # State: (score, chain, occupied_set)
    start = (0,) * dimension
    initial_chain = [start]
    initial_occupied = {start}
    
    # Priority queue: (score, chain_length, chain, occupied)
    # Lower score is better (more negative = more contacts)
    heap = []
    heapq.heappush(heap, (0.0, 0, initial_chain, initial_occupied))
    
    best_chain = None
    best_score = float('inf')
    
    # Track best complete chains found
    complete_chains = []
    
    for depth in range(max_depth):
        if _check_deadline():
            break
        if not heap:
            break
        
        # Expand all states in current beam
        next_states = []
        for score, length, chain, occupied in heap:
            if _check_deadline():
                break
            if len(chain) >= n:
                # Complete chain
                energy = _compute_energy(sequence, chain)
                complete_chains.append((energy, chain))
                if energy < best_score:
                    best_score = energy
                    best_chain = chain
                continue
            
            current_idx = len(chain)
            last_pos = chain[-1]
            
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                new_chain = chain + [site]
                new_occupied = occupied | {site}
                
                # Compute incremental score
                new_score = score
                if sequence[current_idx] == "H":
                    # Count new H-H contacts formed
                    for q in _neighbours(site):
                        if q in occupied:
                            # Check if the occupant is H and not adjacent in chain
                            # We need to know which index is at q
                            # For simplicity, recompute contacts for this partial chain
                            pass
                
                # Recompute score for this partial chain
                where = {p: i for i, p in enumerate(new_chain)}
                contacts = 0
                for i, p in enumerate(new_chain):
                    if sequence[i] == "H":
                        for q in _neighbours(p):
                            j = where.get(q)
                            if j is not None and j > i + 1 and sequence[j] == "H":
                                contacts += 1
                new_score = -contacts
                
                next_states.append((new_score, len(new_chain), new_chain, new_occupied))
        
        if not next_states:
            break
        
        # Keep only top beam_width states
        next_states.sort(key=lambda x: x[0])
        heap = next_states[:beam_width]
    
    # If we found complete chains, return the best
    if complete_chains:
        complete_chains.sort(key=lambda x: x[0])
        return complete_chains[0][1]
    
    return best_chain


def _local_refine(sequence, chain, max_iterations=500):
    """Local search: try single-residue moves to improve energy."""
    if chain is None:
        return None
    
    n = len(sequence)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for iteration in range(max_iterations):
        if _check_deadline():
            break
        
        improved = False
        for i in range(1, n - 1):
            if _check_deadline():
                break
            
            # Try moving residue i to a neighbouring site
            current_pos = best_chain[i]
            for new_pos in _neighbours(current_pos):
                # Check if new_pos is unoccupied (except by residue i itself)
                occupied = set(best_chain)
                occupied.discard(current_pos)
                if new_pos in occupied:
                    continue
                
                # Check that the move doesn't create a self-intersection
                # The new position must not collide with any other residue
                valid = True
                for j, p in enumerate(best_chain):
                    if j != i and p == new_pos:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Also need to check that the chain remains connected
                # This is tricky for single-residue moves; skip for now
                # Instead, try pivot moves
                
                # Try pivot: rotate the tail starting at i around residue i-1
                break
            
            # Try pivot move: rotate segment from i onwards
            if i == 0:
                continue
                
            pivot_pos = best_chain[i - 1]
            current_pos = best_chain[i]
            
            # Try rotating the tail by 90 degrees around pivot
            # This is complex in 3D; instead, try moving residue i+1
            if i + 1 >= n:
                continue
                
            next_pos = best_chain[i + 1]
            # Try moving next_pos to a different neighbour of current_pos
            for new_next in _neighbours(current_pos):
                if new_next == pivot_pos or new_next == next_pos:
                    continue
                
                occupied = set(best_chain)
                occupied.discard(next_pos)
                if new_next in occupied:
                    continue
                
                # Check connectivity: new_next must be adjacent to current_pos (already true)
                # and the rest of the chain from i+2 onwards must still connect
                # This requires checking that best_chain[i+2] is adjacent to new_next
                if i + 2 < n:
                    if abs(best_chain[i+2][0] - new_next[0]) + abs(best_chain[i+2][1] - new_next[1]) + \
                       abs(best_chain[i+2][2] - new_next[2]) != 1:
                        continue
                
                # Check no self-intersection
                valid = True
                for j, p in enumerate(best_chain):
                    if j != i + 1 and p == new_next:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Apply move
                candidate = list(best_chain)
                candidate[i + 1] = new_next
                
                # Verify self-avoidance
                if len(set(candidate)) != len(candidate):
                    continue
                
                # Verify connectivity
                connected = True
                for k in range(len(candidate) - 1):
                    diff = sum(abs(candidate[k][a] - candidate[k+1][a]) for a in range(len(candidate[0])))
                    if diff != 1:
                        connected = False
                        break
                if not connected:
                    continue
                
                new_energy = _compute_energy(sequence, candidate)
                if new_energy < best_energy:
                    best_chain = candidate
                    best_energy = new_energy
                    improved = True
        
        if not improved:
            break
    
    return best_chain


def fold(sequence, dimension):
    """Find best conformation using beam search followed by local refinement."""
    n = len(sequence)
    
    # Use beam search to find good initial conformations
    best_chain = None
    best_energy = 0  # Start with 0 (unfolded) as baseline
    
    # Run beam search with moderate beam width
    beam_width = min(30, n)
    chain = _beam_search(sequence, dimension, beam_width=beam_width)
    
    if chain is not None:
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # Try multiple beam widths
    for bw in [10, 20, 50]:
        if _check_deadline():
            break
        chain = _beam_search(sequence, dimension, beam_width=bw)
        if chain is not None:
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    # Local refinement
    if best_chain is not None and not _check_deadline():
        refined = _local_refine(sequence, best_chain)
        if refined is not None:
            energy = _compute_energy(sequence, refined)
            if energy < best_energy:
                best_energy = energy
                best_chain = refined
    
    # Fallback: straight line if nothing found
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

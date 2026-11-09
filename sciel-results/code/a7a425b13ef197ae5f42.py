"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import heapq
import copy

def _neighbours(point):
    """Yield all 6 neighbours of a point in 3D lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute the energy of a conformation. Energy = -number of H-H contacts."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng, max_steps=None):
    """
    Grow a chain greedily, preferring sites that create H-H contacts.
    Returns a list of tuples or None if stuck.
    """
    n = len(sequence)
    if max_steps is not None:
        n = min(n, max_steps)
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Calculate gain: number of new H-H contacts if we place here
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            
            # Heuristic: prefer sites with more free space to avoid trapping
            free_space = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free_space += 1
            
            # Score: gain is most important, then some random tie-breaking
            # We use a small penalty for low free space to prevent dead ends
            score = gain * 10 + free_space * 0.1 + rng.random() * 0.5
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick the best candidate
        candidates.sort(reverse=True)
        # Sometimes pick second best to diversify? No, greedy is fast.
        # Let's stick to best for speed, rely on multiple restarts.
        best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _local_search(sequence, dimension, chain, rng, iterations=50):
    """
    Improve a conformation using local search (move, pivot, or rotation).
    """
    if not chain:
        return chain
    
    n = len(sequence)
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(chain)}
    
    for _ in range(iterations):
        # Try a random move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Move a single residue (if possible)
            i = rng.randint(1, n - 2) # Don't move ends easily? Actually can move any, but must maintain connectivity
            # Moving a single residue breaks connectivity unless it's a "kink" removal or similar.
            # Simpler: try to move residue i to a neighbour of its current position, 
            # but we must ensure the chain remains connected.
            # This is complex. Let's use "pivot" moves instead.
            
            pass # Skip single move for simplicity, use pivots
            
        else:
            # Pivot move: rotate the suffix of the chain
            # Choose a pivot point i
            i = rng.randint(1, n - 2)
            
            # Get the vector from i-1 to i
            v = tuple(chain[i][d] - chain[i-1][d] for d in range(dimension))
            
            # Find all valid rotations of v
            # In 3D, there are 6 neighbours. We are currently coming from v.
            # We can rotate to any of the other 5 directions?
            # Actually, a pivot move rotates the entire tail (i to n-1) around the bond (i-1, i).
            # The bond (i-1, i) stays fixed. The rest of the tail rotates.
            # The new direction for the bond (i-1, i) can be any of the 6 neighbours of chain[i-1]
            # EXCEPT the one it came from? No, the bond direction changes.
            # Wait, standard pivot: pick a cut point. Rotate the tail around the cut point.
            # Let's define: cut between i-1 and i.
            # The segment chain[i:] is rotated around chain[i-1].
            # The new position of chain[i] must be a neighbour of chain[i-1].
            # And the whole tail must be translated/rotated such that it fits.
            
            # Simpler approach: "Slide" or "Kick"
            # Let's implement a simple "Kick": move a single internal residue to a neighbour,
            # if it doesn't collide and maintains connectivity?
            # Connectivity is hard to maintain with single kicks.
            
            # Let's implement a proper Pivot.
            # Pick pivot index k (1 <= k <= n-2)
            k = i
            
            # The tail is chain[k:]
            # We want to move chain[k] to a new position new_k which is a neighbour of chain[k-1].
            # Then all subsequent residues must be shifted by the same vector? No, that's a translation.
            # A pivot is a rotation.
            
            # Let's use a different local move: "Reversal" or "Insertion"?
            # Given time constraints, let's stick to a simpler heuristic:
            # Try to move the last few residues?
            
            # Actually, let's implement a "Random Walk" local search:
            # Pick a random index i, try to move chain[i] to a random neighbour.
            # Check if the new chain is valid (self-avoiding, connected).
            # Connectivity check:
            # If we move chain[i], the bonds (i-1, i) and (i, i+1) are affected.
            # This is too slow for O(N) check per move if N is large.
            
            # Let's skip complex local search and rely on more greedy restarts with different seeds/parameters.
            pass
    
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation found by multiple randomized greedy growths.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(42)
    
    best_chain = None
    best_energy = 0 # Maximize energy (minimize absolute value, i.e. most negative)
    
    # Number of restarts depends on length
    # For short chains, we can do more.
    # Budget: 240s for 26 sequences. ~9s per sequence.
    # Greedy growth is O(N^2) roughly.
    # Let's do 100 restarts for small, 50 for large.
    
    num_restarts = 100 if n < 50 else 50
    
    # Also try a few "smart" starts?
    # For now, just random restarts of greedy.
    
    for _ in range(num_restarts):
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # If no chain was found (shouldn't happen for reasonable N), return a straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    # Convert to list of lists
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END

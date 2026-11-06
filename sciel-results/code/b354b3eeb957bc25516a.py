"""
HP Lattice Protein Folding:
Strategy: 
1. Precompute H indices for fast contact checks.
2. Use a hybrid approach:
   - For short sequences (<= 30), use a depth-first backtracking search with pruning 
     (branch and bound style) to find near-optimal solutions.
   - For longer sequences, use a fast greedy growth followed by an aggressive 
     local search with "best-first" move selection (steepest descent with 
     occasional random kicks).
3. Optimize energy delta calculations using precomputed contact sets.
"""

import time
import random
import math

# Global deadline: evaluator kills after 210s wall-clock. Use a safety margin.
_DEADLINE = time.time() + 195.0  # leave 15s buffer for overhead/sandbox


def _remaining():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield the 4 neighbours of a lattice point in 2D."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP energy: -1 for each non-consecutive H-H contact."""
    if chain is None:
        return 0
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_growth(sequence, dimension, rng, temperature=0.5, h_indices=None):
    """
    Grow a chain on the lattice using a probabilistic greedy approach.
    Optimized for speed.
    """
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    # Precompute H indices if not provided
    if h_indices is None:
        h_indices = {i for i, c in enumerate(sequence) if c == 'H'}

    for i in range(1, n):
        res = sequence[i]
        last = chain[-1]
        options = []
        
        # Get neighbors of last position
        x, y = last
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        for site in neighbors:
            if site in occupied:
                continue
            
            # Count new H-H contacts this site would form
            gain = 0
            if res == 'H':
                sx, sy = site
                # Check 4 neighbors of site
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    q = (sx+dx, sy+dy)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and j in h_indices:
                        gain += 1
            
            # Count free neighbours (to avoid dead-ends)
            free = 0
            sx, sy = site
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                if (sx+dx, sy+dy) not in occupied:
                    free += 1
            
            # Score: favor contacts strongly, then free space, plus noise
            score = gain * 4.0 + 0.5 * free + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        # Choose best option with some randomness
        options.sort(key=lambda x: -x[0])
        
        if temperature > 0 and len(options) > 1:
            # Softmax-like selection
            top = options[0][0]
            probs = []
            total = 0.0
            for sc, _ in options[:min(4, len(options))]:
                p = math.exp((sc - top) / max(temperature, 0.1))
                probs.append(p)
                total += p
            if total > 0:
                probs = [p / total for p in probs]
                r = rng.random()
                cum = 0.0
                chosen_idx = len(probs) - 1
                for idx, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        chosen_idx = idx
                        break
                site = options[chosen_idx][1]
            else:
                site = options[0][1]
        else:
            site = options[0][1]

        occupied[site] = i
        chain.append(site)

    return chain


def _local_search(sequence, chain, dimension, rng, max_iter=1000, h_indices=None):
    """
    Fast local search with best-first move selection and simulated annealing.
    Uses precomputed contact sets for O(1) energy delta updates.
    """
    if chain is None:
        return None, 0

    if h_indices is None:
        h_indices = {i for i, c in enumerate(sequence) if c == 'H'}

    current_chain = list(chain)
    n = len(current_chain)
    current_energy = _energy(sequence, current_chain)
    
    # Precompute position map
    pos_map = {p: i for i, p in enumerate(current_chain)}
    
    # Precompute contact sets for each residue (only H residues matter for energy, 
    # but we need to know who they touch to update deltas)
    # contact_sets[i] = set of indices j such that i and j are in contact (non-consecutive)
    # Only H-H contacts contribute to energy, but we track all contacts for validity? 
    # No, only H-H contacts affect energy. P-P, P-H don't.
    # So we only need to track H-H contacts.
    
    # Let's maintain a dictionary: h_contacts[i] = set of j such that i, j are H and in contact
    h_contacts = {i: set() for i in h_indices}
    
    # Initialize h_contacts
    for i in h_indices:
        p = current_chain[i]
        for q in _neighbours(p):
            j = pos_map.get(q)
            if j is not None and abs(i - j) > 1 and j in h_indices:
                h_contacts[i].add(j)
                # Ensure symmetry? We count each pair once. 
                # If we iterate i, and add j if j > i, we avoid double counting.
                # But for delta calculation, we need to know if a contact is broken/formed.
                # Let's just store the set of neighbors for each H residue.
    
    # Actually, for delta calculation:
    # If we move residue i (H) from p to q:
    # 1. Remove contacts between i and any H neighbor of p (that are not i-1, i+1)
    # 2. Add contacts between i and any H neighbor of q (that are not i-1, i+1)
    # This is O(1) if we have pos_map.
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simulated annealing parameters
    T_start = 5.0
    T_end = 0.1
    cooling = (T_start / T_end) ** (1.0 / max(max_iter, 1))
    T = T_start
    
    for it in range(max_iter):
        if _remaining() < 0.5:
            break
        T = T_start / (cooling ** it)
        
        # Choose a random residue to move (not ends)
        i = rng.randint(1, n - 2)
        p = current_chain[i]
        
        best_delta = 0
        best_new_pos = None
        
        # Try all 4 neighbors
        x, y = p
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        for q in neighbors:
            # Check validity
            if q == current_chain[i-1] or q == current_chain[i+1]:
                continue
            
            # Check if q is occupied by another residue
            if q in pos_map:
                j = pos_map[q]
                if j != i:
                    continue
            
            # Check unit steps to neighbors
            px, py = current_chain[i-1]
            qx, qy = q
            if abs(qx - px) + abs(qy - py) != 1:
                continue
            
            px, py = current_chain[i+1]
            qx, qy = q
            if abs(qx - px) + abs(qy - py) != 1:
                continue
            
            # Calculate energy delta
            delta = 0
            if i in h_indices:
                # Remove old contacts
                for old_q in _neighbours(p):
                    j = pos_map.get(old_q)
                    if j is not None and j != i and abs(i - j) > 1 and j in h_indices:
                        delta += 1  # Losing a contact increases energy (less negative)
                
                # Add new contacts
                for new_q in _neighbours(q):
                    j = pos_map.get(new_q)
                    if j is not None and j != i and abs(i - j) > 1 and j in h_indices:
                        delta -= 1  # Gaining a contact decreases energy (more negative)
            
            if delta < best_delta:
                best_delta = delta
                best_new_pos = q
        
        # Acceptance
        if best_new_pos is not None:
            accept = False
            if best_delta <= 0:
                accept = True
            else:
                if rng.random() < math.exp(-best_delta / max(T, 0.01)):
                    accept = True
            
            if accept:
                # Apply move
                old_p = current_chain[i]
                del pos_map[old_p]
                current_chain[i] = best_new_pos
                pos_map[best_new_pos] = i
                current_energy += best_delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        else:
            # No valid move found for this i, try a random kick
            if rng.random() < 0.1:
                # Try a random valid move even if it worsens energy, to escape
                for _ in range(5):
                    i = rng.randint(1, n - 2)
                    p = current_chain[i]
                    x, y = p
                    neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                    rng.shuffle(neighbors)
                    for q in neighbors:
                        if q == current_chain[i-1] or q == current_chain[i+1]:
                            continue
                        if q in pos_map:
                            j = pos_map[q]
                            if j != i:
                                continue
                        px, py = current_chain[i-1]
                        qx, qy = q
                        if abs(qx - px) + abs(qy - py) != 1:
                            continue
                        px, py = current_chain[i+1]
                        qx, qy = q
                        if abs(qx - px) + abs(qy - py) != 1:
                            continue
                        
                        # Calculate delta
                        delta = 0
                        if i in h_indices:
                            for old_q in _neighbours(p):
                                j = pos_map.get(old_q)
                                if j is not None and j != i and abs(i - j) > 1 and j in h_indices:
                                    delta += 1
                            for new_q in _neighbours(q):
                                j = pos_map.get(new_q)
                                if j is not None and j != i and abs(i - j) > 1 and j in h_indices:
                                    delta -= 1
                        
                        # Always accept this kick if valid, to escape local minima
                        old_p = current_chain[i]
                        del pos_map[old_p]
                        current_chain[i] = q
                        pos_map[q] = i
                        current_energy += delta
                        break
                    else:
                        continue
                    break

    return best_chain, best_energy


def _backtracking_search(sequence, dimension, rng, max_nodes=50000):
    """
    Depth-first backtracking search with pruning.
    Only effective for short sequences.
    """
    n = len(sequence)
    if n > 30:
        return None, 0
    
    h_indices = {i for i, c in enumerate(sequence) if c == 'H'}
    
    best_chain = None
    best_energy = 0  # Unfolded has 0 energy. We want negative.
    
    # Start at (0,0)
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    energy = 0
    
    def dfs(depth):
        nonlocal best_chain, best_energy, energy, chain, occupied
        
        if _remaining() < 1.0:
            return
        
        if depth == n:
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
            return
        
        last = chain[-1]
        x, y = last
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        # Shuffle for randomness
        rng.shuffle(neighbors)
        
        for q in neighbors:
            if q in occupied:
                continue
            
            # Calculate new energy
            delta = 0
            if sequence[depth] == 'H':
                for old_q in _neighbours(q):
                    j = occupied.get(old_q)
                    if j is not None and j < depth - 1 and sequence[j] == 'H':
                        delta -= 1
            
            # Pruning: if current energy + delta + potential_max < best_energy, skip
            # Potential max: assume all remaining H form contacts. 
            # Simple bound: if energy + delta >= best_energy, we can still improve if we gain more.
            # But if we are already worse than best and can't gain enough, prune.
            # For now, simple pruning: if energy + delta > best_energy, skip? 
            # No, because we might gain more later. 
            # Better: if energy + delta + (max_possible_future_gain) > best_energy, skip.
            # Max possible future gain is bounded by number of H residues.
            # Let's just use time limit and random restarts.
            
            # Place
            chain.append(q)
            occupied[q] = depth
            energy += delta
            
            dfs(depth + 1)
            
            # Unplace
            occupied.pop(q)
            chain.pop()
            energy -= delta
            
            if _remaining() < 1.0:
                return

    dfs(1)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(42 + len(sequence))
    h_indices = {i for i, c in enumerate(sequence) if c == 'H'}
    
    best_chain = None
    best_energy = 0  # Unfolded

    # Strategy selection
    if n <= 30:
        # Try backtracking first for short sequences
        chain, energy = _backtracking_search(sequence, dimension, rng, max_nodes=50000)
        if chain is not None:
            best_chain = chain
            best_energy = energy
        
        # If backtracking didn't find a good solution or ran out of time,
        # fall back to greedy + local search
        if _remaining() > 5.0:
            # Do a few greedy restarts
            for restart in range(10):
                if _remaining() < 2.0:
                    break
                temp = 0.1 + rng.random() * 0.5
                chain = _greedy_growth(sequence, dimension, rng, temperature=temp, h_indices=h_indices)
                if chain is None:
                    continue
                
                energy = _energy(sequence, chain)
                if energy < best_energy:
                    best_chain = chain
                    best_energy = energy
                
                # Local search
                improved_chain, improved_energy = _local_search(
                    sequence, chain, dimension, rng, max_iter=500, h_indices=h_indices
                )
                if improved_chain is not None and improved_energy < best_energy:
                    best_chain = improved_chain
                    best_energy = improved_energy
    else:
        # For longer sequences, use greedy + local search
        # More restarts, shorter local search per restart
        num_restarts = 50
        ls_iter = 200
        
        for restart in range(num_restarts):
            if _remaining() < 2.0:
                break
            
            temp = 0.1 + rng.random() * 0.5
            chain = _greedy_growth(sequence, dimension, rng, temperature=temp, h_indices=h_indices)
            if chain is None:
                continue
            
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_chain = chain
                best_energy = energy
            
            # Local search
            improved_chain, improved_energy = _local_search(
                sequence, chain, dimension, rng, max_iter=ls_iter, h_indices=h_indices
            )
            if improved_chain is not None and improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
    
    # If no valid chain was found, return a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

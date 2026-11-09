# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict
from itertools import combinations

def _get_neighbors(p):
    """Yield 6 neighbors in 3D."""
    x, y, z = p
    return [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1)]

def _compute_energy(sequence, chain):
    """Compute energy = -number of H-H contacts."""
    n = len(chain)
    if n == 0:
        return 0
    pos_to_idx = {chain[i]: i for i in range(n)}
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            for nb in _get_neighbors(chain[i]):
                j = pos_to_idx.get(nb)
                if j is not None and j > i and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _estimate_ground_state_energy(sequence):
    """
    Estimate the ground state energy for a given sequence.
    Uses heuristics based on sequence length and H count.
    """
    n = len(sequence)
    h_count = sequence.count('H')
    
    # Empirical formula: energy scales roughly with h_count^1.5 for compact folds
    # For 48-residue chains, typical energies are -20 to -38
    # For longer chains, scale accordingly
    
    if n <= 48:
        # Linear fit: energy ≈ -0.45 * h_count - 5
        return int(-0.45 * h_count - 5)
    elif n <= 80:
        return int(-0.42 * h_count - 8)
    elif n <= 120:
        return int(-0.38 * h_count - 12)
    else:
        return int(-0.35 * h_count - 15)

def _generate_target_contacts(sequence, target_energy):
    """
    Generate a target set of H-H contacts that would achieve the target energy.
    target_energy is negative, so number of contacts = -target_energy.
    
    Strategy: Select H-H pairs that are:
    1. Not adjacent in sequence (|i-j| > 1)
    2. Prioritize pairs with smaller sequence distance (easier to realize)
    3. Ensure no two contacts share the same residue in a way that makes it infeasible
    """
    target_contacts_count = -target_energy
    h_indices = [i for i, s in enumerate(sequence) if s == 'H']
    
    if len(h_indices) < 2 or target_contacts_count <= 0:
        return set()
    
    # Generate all valid H-H pairs (i, j) with i < j and |i-j| > 1
    candidates = []
    for idx_i, i in enumerate(h_indices):
        for j in h_indices[idx_i+1:]:
            if abs(i - j) > 1:
                seq_dist = abs(i - j)
                # Prefer smaller sequence distances (easier to fold)
                # But not too small - need at least some separation
                score = seq_dist
                candidates.append((score, (i, j)))
    
    # Sort by sequence distance (prefer closer pairs)
    candidates.sort(key=lambda x: x[0])
    
    # Select target_contacts_count pairs, but ensure feasibility
    selected = set()
    used_residues = defaultdict(int)  # Count how many contacts each residue is in
    
    for _, pair in candidates:
        if len(selected) >= target_contacts_count:
            break
        i, j = pair
        # Limit each residue to at most 4 contacts (lattice constraint: max 6 neighbors,
        # but some are chain neighbors)
        if used_residues[i] < 4 and used_residues[j] < 4:
            selected.add(pair)
            used_residues[i] += 1
            used_residues[j] += 1
    
    return selected

def _build_constraint_graph(sequence, target_contacts):
    """
    Build a constraint graph for the contact map.
    Returns adjacency info for constraint propagation.
    """
    n = len(sequence)
    # For each residue, track which other residues it must be in contact with
    contact_adj = defaultdict(set)
    for i, j in target_contacts:
        contact_adj[i].add(j)
        contact_adj[j].add(i)
    return contact_adj

def _check_contact_feasibility(chain, target_contacts):
    """
    Check how many target contacts are realized in the chain.
    Returns (realized_count, missing_contacts, extra_contacts).
    """
    n = len(chain)
    pos_to_idx = {chain[i]: i for i in range(n)}
    
    realized = set()
    for i, j in target_contacts:
        # Check if residues i and j are adjacent in the lattice
        pi = chain[i]
        pj = chain[j]
        dist = abs(pi[0]-pj[0]) + abs(pi[1]-pj[1]) + abs(pi[2]-pj[2])
        if dist == 1:
            realized.add((i, j))
    
    missing = target_contacts - realized
    return len(realized), missing

def _csp_fold(sequence, dimension, target_contacts, rng, time_limit):
    """
    Constraint satisfaction approach to fold the chain.
    Uses backtracking with constraint propagation.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    # Precompute which residues must be in contact with which
    contact_adj = defaultdict(set)
    for i, j in target_contacts:
        contact_adj[i].add(j)
        contact_adj[j].add(i)
    
    start_time = time.time()
    
    def get_heuristic_score(nb, i):
        """Score a candidate position based on contact constraints."""
        score = 0.0
        
        if sequence[i] == 'H':
            # Check if placing H at nb creates any required contacts
            for j in contact_adj[i]:
                if j < i:  # Only check already-placed residues
                    pi = chain[j]
                    dist = abs(nb[0]-pi[0]) + abs(nb[1]-pi[1]) + abs(nb[2]-pi[2])
                    if dist == 1:
                        score += 100  # Strong preference for required contacts
                    elif dist == 2:
                        score += 10   # Some preference for near-required contacts
        
        # Compactness: prefer positions closer to the center of mass
        if len(chain) > 1:
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            cz = sum(p[2] for p in chain) / len(chain)
            dist_c = (nb[0]-cx)**2 + (nb[1]-cy)**2 + (nb[2]-cz)**2
            score -= dist_c * 0.05
        
        return score
    
    def backtrack(i):
        if i == n:
            return True
        
        if time.time() - start_time > time_limit:
            return False
        
        last = chain[-1]
        neighbors = _get_neighbors(last)
        rng.shuffle(neighbors)
        
        # Sort by heuristic score
        neighbors.sort(key=lambda nb: get_heuristic_score(nb, i), reverse=True)
        
        for nb in neighbors:
            if nb not in occupied:
                chain.append(nb)
                occupied[nb] = i
                
                if backtrack(i + 1):
                    return True
                
                chain.pop()
                del occupied[nb]
        
        return False
    
    if not backtrack(1):
        # Fallback: greedy fold
        return _greedy_fold(sequence, dimension, rng)
    
    return chain

def _greedy_fold(sequence, dimension, rng):
    """Fast greedy fold for fallback."""
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for nb in _get_neighbors(last):
            if nb not in occupied:
                gain = 0
                if sequence[i] == 'H':
                    for q in _get_neighbors(nb):
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and sequence[j] == 'H':
                                gain += 1
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                cz = sum(p[2] for p in chain) / len(chain)
                dist_c = (nb[0]-cx)**2 + (nb[1]-cy)**2 + (nb[2]-cz)**2
                score = gain * 10 - dist_c * 0.5 + rng.random()
                candidates.append((score, nb))
        
        if not candidates:
            # Extend in a straight line if stuck
            while len(chain) < n:
                last = chain[-1]
                moved = False
                for axis in range(3):
                    for step in (-1, 1):
                        nb = list(last)
                        nb[axis] += step
                        nb = tuple(nb)
                        if nb not in occupied:
                            chain.append(nb)
                            occupied[nb] = len(chain)-1
                            moved = True
                            break
                    if moved:
                        break
                if not moved:
                    break
            return chain
            
        candidates.sort(reverse=True)
        best_nb = candidates[0][1]
        chain.append(best_nb)
        occupied[best_nb] = i
        
    return chain

def _refine_with_sa(sequence, dimension, chain, rng, time_limit):
    """
    Refine the chain using simulated annealing.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    temperature = 2.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        # Chain Reversal move
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        
        new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
        
        # Check validity
        valid = True
        occupied = set()
        for p in new_chain[:i]:
            occupied.add(p)
        
        for idx in range(i, j+1):
            p = new_chain[idx]
            if p in occupied:
                valid = False
                break
            occupied.add(p)
        
        if valid:
            if i > 0:
                prev = new_chain[i-1]
                curr = new_chain[i]
                if abs(prev[0]-curr[0]) + abs(prev[1]-curr[1]) + abs(prev[2]-curr[2]) != 1:
                    valid = False
            if valid and j < n-1:
                prev = new_chain[j]
                curr = new_chain[j+1]
                if abs(prev[0]-curr[0]) + abs(prev[1]-curr[1]) + abs(prev[2]-curr[2]) != 1:
                    valid = False
        
        if valid:
            new_energy = _compute_energy(sequence, new_chain)
            delta_energy = new_energy - current_energy
            
            if delta_energy < 0 or rng.random() < math.exp(-delta_energy / temperature):
                chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
        
        temperature *= cooling_rate
        if temperature < 0.1:
            temperature = 0.1
    
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence using contact-map first approach.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(42)
    
    # Time budget per sequence: 240s / 26 sequences ~ 9s.
    time_limit = 8.5
    
    # Step 1: Estimate ground state energy
    target_energy = _estimate_ground_state_energy(sequence)
    
    # Step 2: Generate target contact topology
    target_contacts = _generate_target_contacts(sequence, target_energy)
    
    # Step 3: CSP-based folding with constraint awareness
    chain = _csp_fold(sequence, dimension, target_contacts, rng, time_limit * 0.4)
    
    # Ensure chain is valid and full length
    if len(chain) != n:
        chain = [(i, 0, 0) for i in range(n)]
    
    # Step 4: Refine with SA
    chain = _refine_with_sa(sequence, dimension, chain, rng, time_limit * 0.6)
    
    return [list(p) for p in chain]

# EVOLVE-BLOCK-END

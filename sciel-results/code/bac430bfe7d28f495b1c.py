"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import heapq
from collections import defaultdict

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _build_contact_map(sequence):
    """
    Identify high-priority H-H pairs that are non-adjacent in the sequence.
    We prioritize pairs that are far apart in sequence (large |i-j|) because
    satisfying these long-range contacts is what distinguishes good folds from
    poor ones.
    """
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    pairs = []
    for i in range(len(h_indices)):
        for j in range(i + 1, len(h_indices)):
            idx_i = h_indices[i]
            idx_j = h_indices[j]
            # Only consider non-adjacent in chain
            if abs(idx_i - idx_j) > 1:
                # Prioritize pairs with larger sequence distance
                dist = abs(idx_i - idx_j)
                pairs.append((-dist, idx_i, idx_j))
    
    # Sort by priority (most negative dist first = largest distance first)
    pairs.sort(key=lambda x: x[0])
    return [(idx_i, idx_j) for _, idx_i, idx_j in pairs]

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _backtrack_fold(sequence, dimension, time_budget=100.0):
    """
    Constraint-satisfaction backtracking search that prioritizes satisfying
    high-priority long-range H-H contacts.
    """
    import time
    start_time = time.time()
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    # Pre-compute contact priorities
    contact_pairs = _build_contact_map(sequence)
    
    # For each residue index, track which high-priority contacts involve it
    # and the other residue index
    contacts_by_residue = defaultdict(list)
    for idx_i, idx_j in contact_pairs:
        contacts_by_residue[idx_i].append(idx_j)
        contacts_by_residue[idx_j].append(idx_i)
    
    # We'll use a priority queue for the search frontier
    # State: (negative_priority_score, chain, occupied)
    # We want to maximize satisfied high-priority contacts
    
    best_chain = None
    best_energy = 0  # We want most negative energy (most contacts)
    
    # Start with a few different initial moves to diversify
    initial_positions = [(0, 0)]
    
    # Try multiple randomized backtracking runs with time management
    rng = random.Random(42 + n)
    
    max_depth_searches = 50  # Limit number of full backtracking attempts
    
    for attempt in range(max_depth_searches):
        if time.time() - start_time > time_budget:
            break
        
        # Randomize the order of contact pairs slightly for diversity
        shuffled_pairs = contact_pairs[:]
        rng.shuffle(shuffled_pairs)
        
        # Backtracking search
        # We grow the chain one residue at a time
        # At each step, we prefer moves that bring the current residue close
        # to residues it has high-priority contacts with
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        
        # Track which high-priority contacts have been "satisfied" (made adjacent)
        # We'll use a heuristic score during search
        
        def backtrack(pos_idx, current_score):
            nonlocal best_chain, best_energy
            nonlocal chain, occupied
            
            if time.time() - start_time > time_budget:
                return False  # Timeout signal
            
            if pos_idx == n:
                # Complete chain found
                energy = _energy(sequence, chain)
                if energy < best_energy:
                    best_energy = energy
                    best_chain = chain[:]
                return True
            
            # Generate candidate positions for residue pos_idx
            candidates = []
            last_pos = chain[-1]
            
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                # Calculate heuristic score for placing residue pos_idx at site
                # Prefer sites that are adjacent to residues involved in
                # high-priority contacts with pos_idx
                heuristic = 0
                if sequence[pos_idx] == 'H':
                    for other_idx in contacts_by_residue[pos_idx]:
                        if other_idx < pos_idx:  # Only consider already placed
                            other_pos = chain[other_idx]
                            # Check if site is adjacent to other_pos
                            for neighbor in _neighbours(site):
                                if neighbor == other_pos:
                                    # This move would satisfy a high-priority contact
                                    # Weight by priority (longer distance = higher priority)
                                    dist = abs(pos_idx - other_idx)
                                    heuristic += dist * 10
                                    break
                
                # Also add a small random tie-breaker
                heuristic += rng.random() * 0.1
                
                candidates.append((heuristic, site))
            
            if not candidates:
                return False
            
            # Sort by heuristic (descending) and try them in order
            candidates.sort(key=lambda x: -x[0])
            
            for _, site in candidates:
                # Place the residue
                chain.append(site)
                occupied[site] = pos_idx
                
                if backtrack(pos_idx + 1, current_score + 1):
                    # Success or timeout
                    chain.pop()
                    del occupied[site]
                    return True
                
                # Undo
                chain.pop()
                del occupied[site]
                
                if time.time() - start_time > time_budget:
                    return False
            
            return False
        
        # Run backtracking
        result = backtrack(1, 0)
        
        if best_chain is not None:
            # We found at least one valid chain
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

def fold(sequence, dimension):
    """
    Fold a sequence into a low-energy conformation using constraint-satisfaction
    backtracking with contact-map heuristics.
    """
    if len(sequence) <= 2:
        return [[0] * dimension for _ in range(len(sequence))]
    
    # Use backtracking with contact map heuristics
    try:
        result = _backtrack_fold(sequence, dimension, time_budget=110.0)
        return result
    except Exception:
        # Fallback to simple greedy if backtracking fails
        rng = random.Random(12345 + len(sequence))
        best = None
        best_energy = 0
        
        for _ in range(100):
            # Simple greedy growth
            chain = [(0, 0)]
            occupied = {(0, 0): 0}
            success = True
            
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
                    options.append((gain + 0.1 * free + rng.random(), site))
                
                if not options:
                    success = False
                    break
                
                site = max(options)[1]
                occupied[site] = i
                chain.append(site)
            
            if success:
                energy = _energy(sequence, chain)
                if energy < best_energy:
                    best_energy = energy
                    best = chain
        
        if best is None:
            best = [(i, 0) for i in range(len(sequence))]
        
        return [list(p) for p in best]

# EVOLVE-BLOCK-END

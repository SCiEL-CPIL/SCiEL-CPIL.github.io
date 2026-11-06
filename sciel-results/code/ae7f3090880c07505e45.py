# EVOLVE-BLOCK-START
import random
import time
import heapq
from collections import defaultdict

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

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

def _prepare_sequence(sequence):
    n = len(sequence)
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    # Pre-compute contact pairs (i, j) where i < j, abs(i-j) > 1
    # Weight by sequence distance - longer range pairs are more valuable
    pairs = []
    for i in range(len(h_indices)):
        for j in range(i + 1, len(h_indices)):
            idx_i = h_indices[i]
            idx_j = h_indices[j]
            if abs(idx_i - idx_j) > 1:
                # Weight: longer range = higher priority
                weight = abs(idx_j - idx_i)
                pairs.append((idx_i, idx_j, weight))
    
    # Map residue index to list of (partner, weight)
    partners = defaultdict(list)
    for i, j, w in pairs:
        partners[i].append((j, w))
        partners[j].append((i, w))
        
    return pairs, partners

def _backtrack_fold(sequence, dimension, time_budget=100.0):
    start_time = time.time()
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    pairs, partners = _prepare_sequence(sequence)
    
    # We will run multiple independent search attempts
    # Each attempt uses a different random seed and heuristic weighting
    best_chain = None
    best_energy = 0
    
    # Time per attempt - use more attempts with shorter time each
    num_attempts = 15
    time_per_attempt = time_budget / num_attempts
    
    for attempt in range(num_attempts):
        if time.time() - start_time > time_budget:
            break
            
        attempt_start = time.time()
        rng = random.Random(12345 + attempt * 100 + n)
        
        # Heuristic weights - emphasize contact formation
        w_contact = 15.0 + rng.random() * 5.0
        w_priority_contact = 20.0 + rng.random() * 10.0  # Extra weight for high-priority pairs
        w_potential = 2.0 + rng.random() * 3.0
        w_openness = 0.2 + rng.random() * 0.2
        w_random = 0.01
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        current_energy = 0
        
        # Track which contact pairs have been satisfied
        satisfied_pairs = set()
        
        def backtrack(pos_idx):
            nonlocal best_chain, best_energy
            nonlocal chain, occupied, current_energy
            nonlocal satisfied_pairs
            
            if time.time() - attempt_start > time_per_attempt:
                return False  # Timeout for this attempt
            
            if pos_idx == n:
                # Complete chain
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                return True
            
            last_pos = chain[-1]
            candidates = []
            
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                # Calculate heuristic
                score = 0
                immediate_gain = 0
                priority_gain = 0
                
                # 1. Immediate contacts
                if sequence[pos_idx] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < pos_idx - 1 and sequence[j] == 'H':
                            immediate_gain += 1
                            # Check if this is a high-priority pair
                            if (pos_idx, j) in satisfied_pairs or (j, pos_idx) in satisfied_pairs:
                                pass
                            # Check partner list for weight
                            for partner_idx, weight in partners.get(pos_idx, []):
                                if partner_idx == j:
                                    priority_gain += weight
                                    break
                
                score += immediate_gain * w_contact
                score += priority_gain * w_priority_contact
                
                # 2. Potential contacts: for unplaced H residues that could contact this site
                if sequence[pos_idx] == 'H':
                    for q in _neighbours(site):
                        if q not in occupied:
                            # This site is free, but we can't check future placement
                            pass
                elif sequence[pos_idx] == 'P':
                    # P residues don't form contacts, but placing them well helps H residues
                    pass
                
                # 3. Openness: prefer sites with more free neighbors to avoid dead ends
                free_neighbors = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free_neighbors += 1
                score += free_neighbors * w_openness
                
                # 4. Randomness
                score += rng.random() * w_random
                
                candidates.append((score, site))
            
            if not candidates:
                return False
            
            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])
            
            for _, site in candidates:
                # Place
                chain.append(site)
                occupied[site] = pos_idx
                
                # Update energy and satisfied pairs if this move creates a contact
                delta_energy = 0
                new_satisfied = set()
                if sequence[pos_idx] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < pos_idx - 1 and sequence[j] == 'H':
                            delta_energy -= 1 # Energy decreases (becomes more negative)
                            new_satisfied.add((min(pos_idx, j), max(pos_idx, j)))
                            
                current_energy += delta_energy
                satisfied_pairs.update(new_satisfied)
                
                if backtrack(pos_idx + 1):
                    # Undo before returning
                    for pair in new_satisfied:
                        satisfied_pairs.discard(pair)
                    chain.pop()
                    del occupied[site]
                    current_energy -= delta_energy
                    return True
                
                # Undo
                for pair in new_satisfied:
                    satisfied_pairs.discard(pair)
                chain.pop()
                del occupied[site]
                current_energy -= delta_energy
                
                if time.time() - attempt_start > time_per_attempt:
                    return False
            
            return False
        
        # Start search
        backtrack(1)
        
        if best_chain is not None:
            # If we found a good solution, continue trying to improve
            pass

    if best_chain is None:
        # Fallback: simple greedy or straight line
        # Let's do a quick greedy to ensure valid output
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        rng = random.Random(999)
        for i in range(1, n):
            options = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                options.append((rng.random(), site))
            if not options:
                # Dead end, try to backtrack or just fail
                break
            _, site = max(options)
            occupied[site] = i
            chain.append(site)
        
        if len(chain) < n:
             # Failed to build full chain, use straight line as last resort
             # Note: Straight line is always valid for n>0
             best_chain = [(i, 0) for i in range(n)]
        else:
             best_chain = chain
             
    return [list(p) for p in best_chain]

def fold(sequence, dimension):
    """
    Fold a sequence into a low-energy conformation using constraint-satisfaction
    backtracking with contact map heuristics.
    """
    if len(sequence) <= 2:
        return [[0] * dimension for _ in range(len(sequence))]
    
    # Use backtracking with contact map heuristics
    try:
        # Allocate 115s to leave buffer for overhead
        result = _backtrack_fold(sequence, dimension, time_budget=115.0)
        return result
    except Exception:
        # Fallback to simple greedy if backtracking fails
        rng = random.Random(12345 + len(sequence))
        best = None
        best_energy = 0
        
        for _ in range(200):
            # Simple greedy growth with contact-aware heuristic
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
                    options.append((gain * 2.0 + 0.1 * free + rng.random(), site))
                
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

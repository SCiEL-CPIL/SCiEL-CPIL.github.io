<<<<<<< SEARCH
"""HP lattice protein folding: constraint-satisfaction backtracking with lookahead + local refinement."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp
from heapq import heappush, heappop

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _backtrack_fold(sequence, dimension, max_time=5.0):
    """
    Constraint-satisfaction backtracking with lookahead.
    Places residues one by one, using a priority queue to explore
    promising positions first. Uses time budget to limit search.
    """
    n = len(sequence)
    if n <= 2:
        return [(0, 0)] + [(i, 0) for i in range(1, n)]
    
    best_chain = None
    best_energy = 0  # We want to minimize energy (more negative is better)
    
    start_time = time.time()
    
    # Use iterative deepening with a simple backtracking approach
    # For each residue, we try placing it at neighbor positions of the previous residue
    
    def backtrack(chain, occupied, idx):
        nonlocal best_chain, best_energy
        nonlocal start_time
        
        if time.time() - start_time > max_time:
            return
        
        if idx == n:
            # Complete chain found
            e = _energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain[:]
            return
        
        # Get candidate positions (neighbors of last placed residue)
        last_pos = chain[-1]
        candidates = []
        for pos in _neighbours(last_pos):
            if pos not in occupied:
                # Calculate immediate gain (contacts formed)
                gain = 0
                if sequence[idx] == "H":
                    for q in _neighbours(pos):
                        if q in occupied:
                            j = list(occupied.keys())[list(occupied.values()).index(occupied[q])]
                            # Actually, let's store index in occupied
                            pass
                candidates.append(pos)
        
        if not candidates:
            return
        
        # Sort candidates by heuristic: prefer positions that form H-H contacts
        def score_candidate(pos):
            if sequence[idx] != "H":
                return 0
            s = 0
            for q in _neighbours(pos):
                if q in occupied:
                    j = occupied[q]
                    if j < idx - 1 and sequence[j] == "H":
                        s += 1
            return s
        
        candidates.sort(key=score_candidate, reverse=True)
        
        for pos in candidates:
            chain.append(pos)
            occupied[pos] = idx
            backtrack(chain, occupied, idx + 1)
            chain.pop()
            del occupied[pos]
            
            if time.time() - start_time > max_time:
                return
    
    # Start backtracking from origin
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    # Try multiple random orderings to escape local minima
    rng = random.Random(42)
    for attempt in range(20):
        if time.time() - start_time > max_time:
            break
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        
        # Add some randomness to the ordering
        def backtrack_random(chain, occupied, idx):
            nonlocal best_chain, best_energy
            nonlocal start_time
            
            if time.time() - start_time > max_time:
                return
            
            if idx == n:
                e = _energy(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = chain[:]
                return
            
            last_pos = chain[-1]
            candidates = []
            for pos in _neighbours(last_pos):
                if pos not in occupied:
                    candidates.append(pos)
            
            if not candidates:
                return
            
            # Randomize the order slightly
            rng.shuffle(candidates)
            
            for pos in candidates:
                chain.append(pos)
                occupied[pos] = idx
                backtrack_random(chain, occupied, idx + 1)
                chain.pop()
                del occupied[pos]
                
                if time.time() - start_time > max_time:
                    return
        
        backtrack_random(chain, occupied, 1)
    
    if best_chain is not None:
        return best_chain
    else:
        # Fallback: straight line
        return [(i, 0) for i in range(n)]
=======
"""HP lattice protein folding: contact-map-guided construction + efficient local refinement."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp
from heapq import heappush, heappop

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _compute_contact_priorities(sequence):
    """
    Compute priority scores for all potential H-H contact pairs.
    A contact between residues i and j (i < j) is feasible if |i-j| >= 3
    (since they can't be adjacent in the chain). Shorter distances are
    more likely to form contacts in compact folds.
    
    Returns a dict mapping (i, j) -> priority score.
    """
    n = len(sequence)
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    priorities = {}
    
    for a_idx, i in enumerate(h_indices):
        for b_idx in range(a_idx + 1, len(h_indices)):
            j = h_indices[b_idx]
            dist = j - i
            if dist < 3:  # Can't form a contact if too close in sequence
                continue
            # Priority: shorter distance = higher priority (more likely contact)
            # Also penalize very long distances as they're less likely
            if dist <= 10:
                priority = 100 - dist  # High priority for short-range
            elif dist <= 20:
                priority = 50 - (dist - 10)  # Medium priority
            else:
                priority = max(1, 10 - (dist - 20))  # Low priority for long-range
            
            priorities[(i, j)] = priority
    
    return priorities


def _backtrack_fold(sequence, dimension, max_time=5.0):
    """
    Contact-map-guided backtracking.
    Uses contact priorities to guide the search toward conformations
    that realize high-priority H-H contacts.
    """
    n = len(sequence)
    if n <= 2:
        return [(0, 0)] + [(i, 0) for i in range(1, n)]
    
    best_chain = None
    best_energy = 0  # We want to minimize energy (more negative is better)
    start_time = time.time()
    
    # Precompute contact priorities
    contact_priorities = _compute_contact_priorities(sequence)
    
    # For efficient contact checking, precompute for each residue which
    # other H residues it should ideally contact
    residue_contacts = {i: [] for i in range(n) if sequence[i] == 'H'}
    for (i, j), priority in contact_priorities.items():
        residue_contacts[i].append((j, priority))
        residue_contacts[j].append((i, priority))
    
    def backtrack(chain, occupied, idx):
        nonlocal best_chain, best_energy
        nonlocal start_time
        
        if time.time() - start_time > max_time:
            return
        
        if idx == n:
            # Complete chain found
            e = _energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain[:]
            return
        
        last_pos = chain[-1]
        candidates = []
        
        for pos in _neighbours(last_pos):
            if pos not in occupied:
                # Score this candidate position
                score = 0
                if sequence[idx] == 'H':
                    # Count how many high-priority contacts this position enables
                    for neighbor in _neighbours(pos):
                        if neighbor in occupied:
                            j = occupied[neighbor]
                            if j < idx - 1 and sequence[j] == 'H':
                                # Check if this is a high-priority contact
                                key = (min(idx, j), max(idx, j))
                                priority = contact_priorities.get(key, 0)
                                score += priority
                candidates.append((pos, score))
        
        if not candidates:
            return
        
        # Sort by score descending, with random tie-breaking
        rng_local = random.Random(idx * 7919 + len(chain))
        candidates.sort(key=lambda x: (-x[1], rng_local.random()))
        
        for pos, _ in candidates:
            chain.append(pos)
            occupied[pos] = idx
            backtrack(chain, occupied, idx + 1)
            chain.pop()
            del occupied[pos]
            
            if time.time() - start_time > max_time:
                return
    
    # Start backtracking from origin
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    backtrack(chain, occupied, 1)
    
    if best_chain is not None:
        return best_chain
    else:
        # Fallback: straight line
        return [(i, 0) for i in range(n)]
>>>>>>> REPLACE


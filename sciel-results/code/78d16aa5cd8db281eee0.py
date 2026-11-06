"""
Constraint-Satisfaction Backtracking Search for HP Lattice Protein Folding.
Strategy:
1. Pre-compute a contact map of high-priority H-H pairs (long-range interactions).
2. Use backtracking to place residues one by one.
3. At each step, prioritize positions that satisfy pending contact constraints.
4. Use pruning to avoid dead-ends (no free neighbors) and enforce self-avoidance.
5. Since exact backtracking can explode, we use a bounded depth/branching factor 
   with multiple random restarts and a time budget.
"""

# EVOLVE-BLOCK-START
import time
import random
import math
import sys

# Global deadline: evaluator kills after 210s wall-clock. Use a safety margin.
_DEADLINE = time.time() + 190.0  # leave 20s buffer for overhead/sandbox

def _remaining():
    return _DEADLINE - time.time()

def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

def _energy(sequence, chain):
    """Compute the HP energy: -1 for each non-consecutive H-H contact."""
    if chain is None:
        return 1
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _compute_contact_map(sequence):
    """
    Identify high-priority H-H pairs.
    We consider pairs (i, j) where i < j, sequence[i]=='H', sequence[j]=='H',
    and j - i > 1 (non-consecutive).
    We prioritize pairs with larger separation (long-range) as they are harder to satisfy
    but contribute significantly to compactness.
    Returns a list of (i, j) pairs sorted by priority (larger separation first).
    """
    n = len(sequence)
    h_indices = [i for i, res in enumerate(sequence) if res == 'H']
    pairs = []
    for idx_i, i in enumerate(h_indices):
        for j in h_indices[idx_i + 1:]:
            if j - i > 1:
                # Priority: longer range is higher priority
                pairs.append((j - i, i, j))
    # Sort by priority (descending separation)
    pairs.sort(reverse=True)
    # Return just the indices
    return [(i, j) for _, i, j in pairs]

def _backtrack_fold(sequence, dimension, rng, time_budget, max_depth=None):
    """
    Perform a backtracking search to fold the sequence.
    Uses a heuristic ordering of candidate positions based on contact satisfaction.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    # Pre-compute contact map
    contact_map = _compute_contact_map(sequence)
    # For fast lookup: for each residue i, which j's does it need to contact?
    # We'll maintain a set of "pending" contacts for the last placed residue.
    
    # Start at (0,0)
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    # We track which contacts are already satisfied
    # satisfied_contacts is a set of frozenset({i, j})
    satisfied_contacts = set()
    
    # For efficiency, we pre-compute for each residue i, the set of j's it can contact
    # contact_partners[i] = set of j such that (i,j) is in contact_map
    contact_partners = {i: set() for i in range(n)}
    for i, j in contact_map:
        contact_partners[i].add(j)
        contact_partners[j].add(i)

    def _search(depth):
        if _remaining() < 0.1:
            return None
        
        if depth == n:
            return list(chain)
        
        res = sequence[depth]
        last_pos = chain[-1]
        
        # Generate candidate positions
        candidates = []
        for pos in _neighbours(last_pos):
            if pos in occupied:
                continue
            
            # Check if this position creates any new contacts
            # A new contact is formed if pos is adjacent to an existing H residue
            # that is a partner of the current residue (or vice versa)
            # Actually, we just need to check if placing H at pos creates contacts with existing H's
            # that are part of the contact map.
            
            new_contacts = 0
            if res == 'H':
                for neighbor_pos in _neighbours(pos):
                    idx = occupied.get(neighbor_pos)
                    if idx is not None and idx < depth - 1: # idx < depth-1 because depth-1 is the previous residue
                        if sequence[idx] == 'H':
                            # Check if (idx, depth) is a high priority contact
                            # We don't strictly require it to be in the map, but it's good if it is
                            # However, for the heuristic, we prioritize satisfying the *required* map
                            if idx in contact_partners[depth]:
                                new_contacts += 1
            
            # Heuristic: Prefer positions that satisfy pending contacts
            # Also, avoid dead ends (positions with no free neighbors)
            free_neighbors = 0
            for n_pos in _neighbours(pos):
                if n_pos not in occupied:
                    free_neighbors += 1
            
            # Score:
            # 1. Strong preference for satisfying contact map constraints
            # 2. Moderate preference for avoiding dead ends
            # 3. Random noise for diversity
            score = new_contacts * 10.0 + free_neighbors * 0.5 + rng.random() * 2.0
            candidates.append((score, pos))
        
        if not candidates:
            return None
        
        # Sort candidates by score (descending)
        candidates.sort(key=lambda x: -x[0])
        
        # To avoid exponential blowup, we limit the branching factor
        # If we have too many candidates, we only try the top K
        # K depends on depth and remaining time
        max_branch = 3
        if depth < 10:
            max_branch = 4
        elif depth < 20:
            max_branch = 3
        else:
            max_branch = 2
            
        for score, pos in candidates[:max_branch]:
            # Place residue
            chain.append(pos)
            occupied[pos] = depth
            
            # Check for immediate dead end (if this is not the last residue)
            if depth < n - 1:
                has_free = False
                for n_pos in _neighbours(pos):
                    if n_pos not in occupied:
                        has_free = True
                        break
                if not has_free:
                    # Backtrack
                    chain.pop()
                    del occupied[pos]
                    continue
            
            result = _search(depth + 1)
            if result is not None:
                return result
            
            # Backtrack
            chain.pop()
            del occupied[pos]
            
            # If time is running out, stop
            if _remaining() < 0.5:
                return None

    return _search(1)

def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    Uses constraint-satisfaction backtracking with contact map heuristics.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 1 # worst possible
    
    # Allocate time for multiple restarts
    # For short sequences, we can afford more restarts
    # For long sequences, backtracking is expensive, so fewer restarts
    if n <= 25:
        num_restarts = 50
    elif n <= 40:
        num_restarts = 20
    elif n <= 60:
        num_restarts = 10
    else:
        num_restarts = 5
        
    # We will run the backtracking search multiple times with different random seeds
    # to explore different parts of the solution space.
    for restart in range(num_restarts):
        if _remaining() < 2.0:
            break
            
        # Give each restart a small time budget
        # But _backtrack_fold checks _remaining() globally, so it will stop if we run out
        # We just need to make sure we don't start a new restart if we don't have enough time
        # to complete at least a few steps
        
        chain = _backtrack_fold(sequence, dimension, rng, time_budget=_remaining() - 1.0)
        
        if chain is not None:
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
                # If we found a very good energy (close to theoretical max), we can stop early
                # Theoretical max contacts is roughly n/2 for H-rich sequences, but let's not guess
                # Just keep improving
            else:
                # If we didn't improve, we might want to stop early if time is tight
                # But for now, let's just continue to the next restart if time permits
                pass
        else:
            # Backtracking failed (likely due to dead end or time out)
            # Try a simpler greedy approach as a fallback for this restart?
            # Or just skip to next restart with a different seed
            pass

    # If backtracking failed to produce a valid chain (e.g. for very long sequences where it gets stuck),
    # we need a fallback. The previous program used greedy growth + local search.
    # Let's implement a very fast greedy fallback to ensure we always return a valid chain.
    if best_chain is None:
        # Simple greedy growth
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        for i in range(1, n):
            res = sequence[i]
            last = chain[-1]
            options = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Simple heuristic: prefer H-H contacts
                gain = 0
                if res == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain * 5.0 + free * 0.1 + rng.random()
                options.append((score, site))
            
            if not options:
                # Stuck, break
                break
                
            options.sort(key=lambda x: -x[0])
            site = options[0][1]
            occupied[site] = i
            chain.append(site)
            
        if len(chain) == n:
            best_chain = chain
            best_energy = _energy(sequence, best_chain)
        else:
            # If even greedy fails (should be rare), return a straight line
            best_chain = [(i, 0) for i in range(n)]
            best_energy = _energy(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

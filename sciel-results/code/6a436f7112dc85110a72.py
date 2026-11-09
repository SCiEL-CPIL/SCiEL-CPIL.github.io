# EVOLVE-BLOCK-START
import random
import time
import math
import sys

sys.setrecursionlimit(10000)

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _compute_energy(sequence, chain):
    """Calculate total energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _backtrack_fold(sequence, rng, time_limit, node_limit):
    """
    Heuristic Backtracking Search with Pruning.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    best_energy = float('inf')
    best_chain = None
    
    # Precompute H positions for faster counting
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    total_h = len(h_positions)
    
    # Upper bound on contacts: Each H can form at most 4 non-chain contacts.
    # But we can't have more than total_h * (total_h - 1) / 2.
    # A tighter dynamic bound: current_contacts + remaining_H * 4
    # We use a global deadline
    start_time = time.time()
    nodes_visited = 0
    max_nodes = node_limit
    
    # Initial state
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    current_contacts = 0
    
    # We will use a stack-based DFS to avoid recursion overhead and allow easy pruning
    # State: (index, chain, occupied, contacts)
    # But copying chain/occupied is expensive. We use undo/redo.
    
    # For simplicity and speed, we'll use a recursive function with global state modification.
    # This is faster in Python if we carefully manage state.
    
    def dfs(i, last_pos, current_contacts, chain, occupied):
        nonlocal best_energy, best_chain, nodes_visited, start_time
        
        # Time check
        if time.time() - start_time > time_limit:
            return
        if nodes_visited > max_nodes:
            return
            
        nodes_visited += 1
        
        if i == n:
            if current_contacts > best_energy: # Energy is negative, so -contacts. Wait.
                # Energy = -contacts. We want to minimize energy (more negative).
                # So we want to maximize contacts.
                pass # Placeholder
            
            # Correct logic:
            # Energy = -contacts.
            # We want to minimize Energy.
            # So if -current_contacts < best_energy, update.
            if -current_contacts < best_energy:
                best_energy = -current_contacts
                best_chain = list(chain)
            return
        
        # Pruning: Upper bound on additional contacts
        # Remaining H residues:
        remaining_h = sum(1 for j in range(i, n) if sequence[j] == 'H')
        # Max additional contacts: Each remaining H can form at most 4 contacts with existing Hs?
        # Actually, it can form contacts with any H in the chain.
        # A safe upper bound: current_contacts + remaining_h * 4
        # If - (current_contacts + remaining_h * 4) >= best_energy, prune
        # i.e., if current_contacts + remaining_h * 4 <= -best_energy, prune
        if current_contacts + remaining_h * 4 <= -best_energy:
            return
            
        # Generate neighbors
        last = chain[-1]
        neighbors = list(_neighbours(last))
        
        # Score neighbors: Prefer those that create contacts
        scored_neighbors = []
        for pos in neighbors:
            if pos in occupied:
                continue
            score = 0
            if sequence[i] == 'H':
                for q in _neighbours(pos):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            score += 1
            # Add small random noise to break ties and allow exploration
            score += rng.random() * 0.1
            scored_neighbors.append((score, pos))
            
        # Sort by score descending
        scored_neighbors.sort(key=lambda x: x[0], reverse=True)
        
        for score, pos in scored_neighbors:
            # Place residue
            chain.append(pos)
            occupied[pos] = i
            
            new_contacts = current_contacts
            if sequence[i] == 'H':
                for q in _neighbours(pos):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            new_contacts += 1
                            
            dfs(i + 1, pos, new_contacts, chain, occupied)
            
            # Undo
            chain.pop()
            del occupied[pos]
            
            # Time check after each step to avoid deep recursion timeout
            if time.time() - start_time > time_limit:
                return
            if nodes_visited > max_nodes:
                return
                
    # Start DFS
    dfs(1, (0, 0, 0), 0, chain, occupied)
    
    if best_chain is None:
        # Fallback: Greedy grow
        chain = [(0, 0, 0)]
        occupied = {(0, 0, 0): 0}
        for i in range(1, n):
            last = chain[-1]
            options = []
            for pos in _neighbours(last):
                if pos in occupied:
                    continue
                score = 0
                if sequence[i] == 'H':
                    for q in _neighbours(pos):
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and sequence[j] == 'H':
                                score += 1
                score += rng.random() * 0.5
                options.append((score, pos))
            if not options:
                break
            options.sort(reverse=True)
            pos = options[0][1]
            chain.append(pos)
            occupied[pos] = i
        best_chain = chain
        
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation using Heuristic Backtracking Search.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
        
    # Time budget: 26 sequences in 330s total.
    # Allocate time based on chain length.
    # Short chains: faster, allow more restarts.
    # Long chains: slower, fewer restarts.
    
    # Total time available per sequence (approx 12s)
    # We will use a dynamic allocation.
    
    if n <= 40:
        time_per_restart = 1.0
        num_restarts = 10
    elif n <= 60:
        time_per_restart = 2.0
        num_restarts = 5
    elif n <= 100:
        time_per_restart = 3.0
        num_restarts = 3
    else:
        time_per_restart = 4.0
        num_restarts = 2
        
    # Node limit per restart
    # For n=48, 10^6 nodes is fast. For n=136, 10^5 might be better.
    node_limit = 50000 if n > 80 else 200000
    
    best_chain = None
    best_energy = float('inf')
    
    # Use a deterministic seed for reproducibility if needed, but random is fine for diversity
    # We will use different seeds for each restart to explore different parts of the space
    base_seed = hash(sequence) % (2**32)
    
    for restart in range(num_restarts):
        rng = random.Random(base_seed + restart)
        chain = _backtrack_fold(sequence, rng, time_per_restart, node_limit)
        
        if chain is not None and len(chain) == n:
            e = _compute_energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain
                
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

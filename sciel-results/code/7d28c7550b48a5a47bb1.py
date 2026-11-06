"""Rewritten program using CSP-style backtracking with forward checking and branch-and-bound."""
# EVOLVE-BLOCK-START
import time


def _neighbours(point):
    """Yield all 6 neighbours in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _has_enough_space(start_pos, prev_pos, occupied, required_length, dimension):
    """
    Check if the free space connected to start_pos (excluding prev_pos and occupied)
    has at least required_length cells.
    """
    if required_length <= 0:
        return True
        
    visited = set()
    stack = [start_pos]
    visited.add(start_pos)
    
    count = 0
    while stack:
        pos = stack.pop()
        count += 1
        if count >= required_length:
            return True
            
        for nb in _neighbours(pos):
            if nb in visited:
                continue
            if nb == prev_pos:
                continue
            if nb in occupied:
                continue
            visited.add(nb)
            stack.append(nb)
            
    return False


def fold(sequence, dimension):
    """
    Find best conformation for the given HP sequence on the simple cubic lattice.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
        
    start_time = time.time()
    time_limit = 8.0 # 240s / 26 seqs ~ 9s. Leave buffer.
    
    best_energy = 0
    best_chain = None
    
    # If all P, energy is always 0. Just return a valid chain.
    if 'H' not in sequence:
        chain = []
        for i in range(n):
            chain.append([i, 0, 0])
        return chain

    # State for backtracking
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    def backtrack(k, current_energy):
        nonlocal best_energy, best_chain, time_limit, start_time
        
        # Time check
        if time.time() - start_time > time_limit:
            return False # Signal to stop
        
        # Branch and Bound
        remaining = n - k - 1
        if remaining > 0:
            # Conservative upper bound on additional energy drop
            max_possible_gain = remaining * 2
            if current_energy - max_possible_gain >= best_energy:
                return True # Pruned, continue searching other branches
        
        if k == n:
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
            return True
            
        last_pos = chain[-1]
        candidates = []
        
        for nb in _neighbours(last_pos):
            if nb in occupied:
                continue
                
            # Forward Checking: Space
            if k < n - 1:
                if not _has_enough_space(nb, last_pos, occupied, n - k - 1, dimension):
                    continue
            
            # Calculate immediate energy gain
            gain = 0
            if sequence[k] == 'H':
                for q in _neighbours(nb):
                    if q in occupied:
                        idx = occupied[q]
                        if idx < k - 1 and sequence[idx] == 'H':
                            gain += 1
            
            candidates.append((gain, nb))
        
        # Sort by gain descending to find high energy solutions early
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for gain, nb in candidates:
            # Check time again
            if time.time() - start_time > time_limit:
                return False
                
            chain.append(nb)
            occupied[nb] = k
            
            # Recurse
            result = backtrack(k + 1, current_energy - gain)
            
            if not result:
                # Timeout
                chain.pop()
                occupied.pop(nb)
                return False
            
            chain.pop()
            occupied.pop(nb)
            
        return True

    backtrack(1, 0)
    
    if best_chain is None:
        # Fallback: Straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

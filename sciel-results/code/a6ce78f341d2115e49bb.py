import random
import sys
import time

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(seq, chain):
    """Compute energy of a chain."""
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if seq[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None and abs(i - j) > 1 and seq[j] == 'H':
                if i < j:
                    contacts += 1
    return -contacts

def _dfs_fold(seq, dimension, time_budget, rng):
    """
    Depth-First Search with Lookahead and Greedy Ordering.
    """
    n = len(seq)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    start = (0, 0, 0)
    best_chain = [start]
    best_energy = 0 # Initial energy is 0 (no contacts)
    
    # Use a global flag for time limit
    deadline = time.time() + time_budget
    
    def backtrack(i, chain, occupied):
        nonlocal best_chain, best_energy
        
        if time.time() > deadline:
            return
            
        if i == n:
            e = _energy(seq, chain)
            if e < best_energy:
                best_energy = e
                best_chain = list(chain)
            return
        
        last = chain[-1]
        candidates = []
        
        # Generate valid moves
        for q in _neighbours(last):
            if q in occupied:
                continue
            
            # Lookahead: Ensure the next residue can be placed
            if i + 1 < n:
                valid_next = False
                for q2 in _neighbours(q):
                    if q2 in occupied:
                        continue
                    # Also ensure q2 is not the previous position (to avoid immediate backtrack)
                    # Actually, q2 != chain[-2] is handled by occupied? No, chain[-2] is in occupied.
                    # So if q2 is in occupied, it's skipped.
                    # But we need to ensure q2 is not the one we just came from?
                    # q is new. q2 is neighbor of q.
                    # If q2 == chain[-2], it is in occupied, so skipped.
                    # So this check is sufficient.
                    valid_next = True
                    break
                if not valid_next:
                    continue
            
            # Score: Maximize H-H contacts
            score = 0
            if seq[i] == 'H':
                for q2 in _neighbours(q):
                    if q2 in occupied:
                        idx = occupied[q2]
                        if idx < i - 1 and seq[idx] == 'H':
                            score += 1
            
            candidates.append((score, q))
            
        if not candidates:
            return
        
        # Sort by score (desc), then random
        # Add random noise to encourage exploration
        for idx, c in enumerate(candidates):
            candidates[idx] = (c[0] + rng.random() * 0.1, c[1])
            
        candidates.sort(key=lambda x: -x[0])
        
        for score, q in candidates:
            new_chain = chain + [q]
            new_occupied = occupied.copy()
            new_occupied[q] = i
            
            backtrack(i + 1, new_chain, new_occupied)
            
            if time.time() > deadline:
                return

    backtrack(0, [start], {start: 0})
    
    if len(best_chain) == n:
        return best_chain
    else:
        return None

def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice.
    """
    n = len(sequence)
    if n < 4:
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    start_time = time.time()
    # Allocate time based on sequence length
    # Shorter sequences can use more time per residue?
    # Total budget 240s for 26 sequences.
    # Let's give each sequence a fair share.
    # If we assume equal distribution: 240 / 26 ~ 9.2s.
    # But some sequences are harder.
    # Let's use a fixed budget per call, e.g., 8.0s.
    time_budget = 8.0
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    
    # Try DFS
    try:
        chain = _dfs_fold(sequence, dimension, time_budget, rng)
        if chain is not None and len(chain) == n:
            best_chain = chain
    except Exception:
        pass
    
    # Fallback if DFS fails or times out without finding a complete chain
    if best_chain is None or len(best_chain) != n:
        # Simple greedy random walk
        chain = [(0, 0, 0)]
        occupied = {chain[0]: 0}
        for i in range(1, n):
            valid = []
            for q in _neighbours(chain[-1]):
                if q not in occupied:
                    valid.append(q)
            if not valid:
                break
            q = rng.choice(valid)
            chain.append(q)
            occupied[q] = i
        
        if len(chain) == n:
            best_chain = chain
        else:
            # Last resort: linear chain
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
            
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END

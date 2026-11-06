"""
HP Lattice Protein Folding Solver
Optimized for speed and energy minimization using Simulated Annealing 
with Pivot, Crankshaft, and Reversal moves.
"""

import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 200.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _calculate_initial_energy(sequence, chain):
    """Calculate energy from scratch. O(N^2) but done once per restart."""
    n = len(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        # Check 4 neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            j = pos_to_idx.get((x + dx, y + dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _pivot_move(chain, pivot_idx, angle):
    """
    Apply a pivot move at pivot_idx.
    angle: 1 (90 deg CCW), -1 (90 deg CW), 2 (180 deg)
    Returns (new_chain, changed_indices, removed_contacts, added_contacts) 
    or None if invalid.
    We return the chain and let the caller update energy incrementally if possible,
    but for simplicity and robustness in SA, we often re-calc or do partial update.
    Here we return new_chain and the set of indices involved for potential partial update.
    Actually, to be fast, we should return the new chain and let the caller 
    handle the energy delta if we track contacts. 
    However, implementing a perfect incremental contact tracker for pivots is complex.
    Given N<=100, O(N) energy check is acceptable if moves are fast.
    Let's stick to a fast energy delta calculation by only checking neighbors of moved points.
    """
    n = len(chain)
    if pivot_idx <= 0 or pivot_idx >= n - 1:
        return None
    
    head = chain[:pivot_idx+1]
    tail = chain[pivot_idx+1:]
    px, py = chain[pivot_idx]
    
    new_tail = []
    # The points that move are indices pivot_idx+1 to n-1
    # We need to check collisions with head (0 to pivot_idx)
    # And collisions within the new_tail itself (which is a rotation, so no self-collision)
    
    occupied_head = set(head)
    
    for i, (x, y) in enumerate(tail):
        dx, dy = x - px, y - py
        if angle == 1: # CCW
            ndx, ndy = -dy, dx
        elif angle == -1: # CW
            ndx, ndy = dy, -dx
        else: # 180
            ndx, ndy = -dx, -dy
            
        nx, ny = px + ndx, py + ndy
        new_p = (nx, ny)
        
        if new_p in occupied_head:
            return None
        
        new_tail.append(new_p)
        
    return head + new_tail


def _crankshaft_move(chain, start_idx, angle):
    """
    Rotate a segment of length 2 (start_idx, start_idx+1) around start_idx-1.
    Or generally, rotate a short segment. 
    Let's implement a standard crankshaft: 
    Pick a pivot i. Rotate the sub-chain i+1, i+2 around i.
    """
    n = len(chain)
    # Segment to rotate: indices start_idx+1, start_idx+2
    # Pivot: start_idx
    if start_idx < 1 or start_idx + 2 >= n:
        return None
        
    p0 = chain[start_idx]
    p1 = chain[start_idx + 1]
    p2 = chain[start_idx + 2]
    
    # Rotate p1 and p2 around p0
    dx1, dy1 = p1[0] - p0[0], p1[1] - p0[1]
    dx2, dy2 = p2[0] - p0[0], p2[1] - p0[1]
    
    if angle == 1:
        ndx1, ndy1 = -dy1, dx1
        ndx2, ndy2 = -dy2, dx2
    elif angle == -1:
        ndx1, ndy1 = dy1, -dx1
        ndx2, ndy2 = dy2, -dx2
    else:
        return None # Crankshaft usually only 90 deg rotations for 2-mers
    
    np1 = (p0[0] + ndx1, p0[1] + ndy1)
    np2 = (p0[0] + ndx2, p0[1] + ndy2)
    
    # Check collisions with the rest of the chain (excluding the segment itself)
    # The rest of the chain is chain[:start_idx] and chain[start_idx+3:]
    # Note: np1 must not collide with chain[start_idx-1] (which is p0's neighbor in original, 
    # but p0 is fixed, so np1 just needs to be free in the global space except for the segment)
    # Actually, we just need to check if np1 or np2 are occupied by any index NOT in {start_idx+1, start_idx+2}
    
    # Build set of occupied positions excluding the moving segment
    # This is O(N). To be faster, we could maintain a global set, but for N=100 it's fine.
    occupied = set(chain)
    occupied.remove(p1)
    occupied.remove(p2)
    
    if np1 in occupied or np2 in occupied:
        return None
        
    # Also check if np1 and np2 collide with each other? 
    # Rotation preserves distance, so if they didn't collide before, they won't.
    # But wait, p1 and p2 are adjacent. Their rotated versions are also adjacent.
    # The only risk is collision with the rest.
    
    new_chain = chain[:start_idx+1] + [np1, np2] + chain[start_idx+3:]
    return new_chain


def _reversal_move(chain, start_idx, end_idx):
    """Reverse the sub-chain from start_idx to end_idx."""
    n = len(chain)
    if start_idx < 0 or end_idx >= n or start_idx >= end_idx:
        return None
        
    # The segment is chain[start_idx : end_idx+1]
    # We reverse it.
    # Check collisions.
    # The points inside the segment keep their relative distances, so no self-collision.
    # We just need to check if any point in the reversed segment collides with the rest.
    # Rest: chain[:start_idx] and chain[end_idx+1:]
    
    # Optimization: 
    # The endpoints of the segment are (start_idx, end_idx).
    # In the reversed chain, the point at start_idx becomes the old end_idx, and vice versa.
    # The points in between are reversed.
    # A reversal is valid if the new positions of the segment do not intersect the rest of the chain.
    
    # Let's construct the new chain and check.
    # To be fast, we can check if any point in the segment (excluding the endpoints which stay in the same "region" roughly? No, they swap) 
    # Actually, the set of coordinates occupied by the segment is the same! 
    # Wait, no. Reversal changes the coordinates? 
    # No, a reversal in a lattice walk usually means reflecting the coordinates? 
    # In SAW literature, "reversal" usually means reversing the sequence of steps, which is a geometric reflection?
    # No, if you reverse the order of points, the coordinates are the same set. 
    # BUT, the connectivity changes.
    # If we just reverse the list of points, the chain might self-intersect?
    # No, the set of points is identical. So no new self-intersections are created *within* the segment.
    # However, the *endpoints* of the segment are now connected to different neighbors.
    # Specifically, chain[start_idx-1] is now connected to chain[end_idx] (old).
    # And chain[end_idx+1] is now connected to chain[start_idx] (old).
    # We must ensure these new connections are valid (unit steps) and don't cause collisions.
    # Since the set of points is the same, the only collisions to check are:
    # 1. Does chain[start_idx-1] (if exists) collide with any point in the segment? 
    #    It already was adjacent to chain[start_idx]. Now it's adjacent to chain[end_idx].
    #    We must check if chain[start_idx-1] is a unit step away from chain[end_idx].
    # 2. Does chain[end_idx+1] (if exists) collide with any point in the segment?
    #    We must check if chain[end_idx+1] is a unit step away from chain[start_idx].
    
    if start_idx > 0:
        prev = chain[start_idx-1]
        new_next = chain[end_idx]
        if abs(prev[0]-new_next[0]) + abs(prev[1]-new_next[1]) != 1:
            return None
        # Also check if prev collides with any OTHER point in the segment?
        # Prev was not colliding with the segment before (except start_idx).
        # Now it is connected to end_idx.
        # Does prev touch any other point in the segment?
        # If it did, it would have been a collision before? No, prev is not part of the segment.
        # If prev touched any point in the segment, it was a collision in the original chain too (since self-avoiding).
        # So we just need to check the new adjacency.
        
    if end_idx < n - 1:
        next_ = chain[end_idx+1]
        new_prev = chain[start_idx]
        if abs(next_[0]-new_prev[0]) + abs(next_[1]-new_prev[1]) != 1:
            return None
            
    # If these checks pass, the reversal is valid because the set of points is unchanged.
    new_chain = chain[:start_idx] + chain[start_idx:end_idx+1][::-1] + chain[end_idx+1:]
    return new_chain


def _calculate_energy_delta(sequence, old_chain, new_chain, moved_indices):
    """
    Calculate energy difference efficiently.
    This is complex. For N<=100, full recalculation is ~100*4 = 400 ops.
    100,000 moves/sec * 400 ops = 40M ops/sec. Python is slow, maybe 1M moves/sec is optimistic.
    Let's try full recalculation first. If it's too slow, we optimize.
    """
    n = len(new_chain)
    pos_to_idx = {p: i for i, p in enumerate(new_chain)}
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = new_chain[i]
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            j = pos_to_idx.get((x + dx, y + dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _optimize(sequence, chain, rng, time_limit):
    """Simulated Annealing."""
    n = len(sequence)
    current_energy = _calculate_energy_delta(sequence, chain, chain, set())
    best_chain = chain[:]
    best_energy = current_energy
    
    T = 2.0
    T_min = 0.01
    cooling = 0.995
    
    start_t = time.time()
    
    while T > T_min and (time.time() - start_t) < time_limit:
        # Choose move type
        r = rng.random()
        if r < 0.6:
            # Pivot
            pivot_idx = rng.randint(1, n - 2)
            angle = rng.choice([1, -1, 2])
            new_chain = _pivot_move(chain, pivot_idx, angle)
        elif r < 0.8:
            # Crankshaft
            start_idx = rng.randint(1, n - 3)
            angle = rng.choice([1, -1])
            new_chain = _crankshaft_move(chain, start_idx, angle)
        else:
            # Reversal
            # Pick a short segment for efficiency
            length = rng.randint(2, min(5, n-2))
            start_idx = rng.randint(0, n - length - 1)
            end_idx = start_idx + length
            new_chain = _reversal_move(chain, start_idx, end_idx)
            
        if new_chain is None:
            T *= cooling
            continue
            
        new_energy = _calculate_energy_delta(sequence, chain, new_chain, set())
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                
        T *= cooling
        
    return best_chain, best_energy


def _grow(sequence, dimension, rng):
    """Randomized greedy growth."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            site = (last[0] + dx, last[1] + dy)
            if site in occupied:
                continue
            # Heuristic: prefer sites closer to other H's if current is H
            gain = 0
            if sequence[i] == 'H':
                # Check neighbors of site
                for dx2, dy2 in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nb = (site[0] + dx2, site[1] + dy2)
                    if nb in occupied:
                        idx = occupied[nb]
                        if sequence[idx] == 'H' and idx < i - 1:
                            gain += 1
            # Add some randomness
            options.append((gain + rng.random(), site))
            
        if not options:
            return None
            
        # Pick best or random from top
        rng.shuffle(options)
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
        
    return chain


def fold(sequence, dimension):
    """
    Main entry point.
    """
    rng = random.Random(42 + len(sequence))
    remaining = _time_left()
    
    if remaining <= 0:
        # Fallback: straight line (may not be self-avoiding if dimension < 2, but dim is 2)
        return [[i, 0] for i in range(len(sequence))]
        
    n = len(sequence)
    
    # Allocate time
    # We want to try multiple restarts.
    # Let's do 5 restarts.
    num_restarts = 5
    time_per_restart = remaining / num_restarts * 0.9 # 10% buffer
    
    best_chain = None
    best_energy = 0
    
    for _ in range(num_restarts):
        if _time_left() < 1.0:
            break
            
        init_chain = _grow(sequence, dimension, rng)
        if init_chain is None:
            # Try a simpler growth
            init_chain = _grow_simple(sequence, dimension, rng)
            if init_chain is None:
                continue
                
        opt_chain, e = _optimize(sequence, init_chain, rng, time_per_restart)
        
        if e < best_energy:
            best_energy = e
            best_chain = opt_chain
            
    if best_chain is None:
        # Fallback
        best_chain = _grow_simple(sequence, dimension, rng)
        if best_chain is None:
            best_chain = [[i, 0] for i in range(n)]
            
    return [list(p) for p in best_chain]


def _grow_simple(sequence, dimension, rng):
    """Fallback growth if greedy fails."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            site = (last[0] + dx, last[1] + dy)
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        occupied[site] = i
        chain.append(site)
    return chain


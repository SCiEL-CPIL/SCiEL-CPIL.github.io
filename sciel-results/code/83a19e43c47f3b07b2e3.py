import time
import random
import math

_START_TIME = time.time()
_TIME_LIMIT = 290.0

def _neighbors(p):
    x, y, z = p
    return [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1)]

def _init_chain(sequence, dimension, rng):
    n = len(sequence)
    if n == 0:
        return [], set()
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    for i in range(1, n):
        options = [n_pos for n_pos in _neighbors(chain[-1]) if n_pos not in occupied]
        if not options:
            return None, None
        next_pos = rng.choice(options)
        chain.append(next_pos)
        occupied.add(next_pos)
    return chain, occupied

def _calc_energy(sequence, chain):
    if not chain:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    energy = 0
    h_indices = [i for i in range(len(sequence)) if sequence[i] == 'H']
    for i in h_indices:
        p = chain[i]
        for n_pos in _neighbors(p):
            if n_pos in pos_to_idx:
                j = pos_to_idx[n_pos]
                if j > i + 1:
                    energy += 1
    return -energy

def _rotate_point(p, center, axis, rot_type):
    # rot_type: 0 for 90, 1 for 180, 2 for 270
    x = p[0] - center[0]
    y = p[1] - center[1]
    z = p[2] - center[2]
    
    if axis == 0: # X
        if rot_type == 0: # 90
            nx, ny, nz = x, -z, y
        elif rot_type == 1: # 180
            nx, ny, nz = x, -y, -z
        else: # 270
            nx, ny, nz = x, z, -y
    elif axis == 1: # Y
        if rot_type == 0: # 90
            nx, ny, nz = z, y, -x
        elif rot_type == 1: # 180
            nx, ny, nz = -x, y, -z
        else: # 270
            nx, ny, nz = -z, y, x
    else: # Z
        if rot_type == 0: # 90
            nx, ny, nz = -y, x, z
        elif rot_type == 1: # 180
            nx, ny, nz = -x, -y, z
        else: # 270
            nx, ny, nz = y, -x, z
            
    return (nx + center[0], ny + center[1], nz + center[2])

def _try_pivot(chain, occupied, pos_to_idx, i, rng):
    if i < 1 or i >= len(chain) - 1:
        return False, 0
    
    n = len(chain)
    dx = chain[i][0] - chain[i-1][0]
    dy = chain[i][1] - chain[i-1][1]
    dz = chain[i][2] - chain[i-1][2]
    
    if dx != 0: axis = 0
    elif dy != 0: axis = 1
    else: axis = 2
    
    # Determine rotation type
    # We pick a random rotation type (0, 1, 2)
    rot_type = rng.randint(0, 2)
    
    # Calculate new positions for the tail
    new_tail = []
    pivot = chain[i]
    for k in range(i+1, n):
        new_p = _rotate_point(chain[k], pivot, axis, rot_type)
        new_tail.append(new_p)
        
    # Check collisions with head
    head_occupied = set(chain[:i+1]) # 0..i
    for p in new_tail:
        if p in head_occupied:
            return False, 0
            
    # Check if new_tail is self-avoiding (it is, since it's a rotation of a SAW)
    # But we need to check if new_tail intersects with the OLD tail?
    # No, the old tail is removed.
    
    # Calculate Delta E
    # Old contacts involving tail (i+1..n-1) and head (0..i)
    # New contacts involving new_tail and head
    
    # We need to know which indices are H
    # Let's precompute H indices
    # This function doesn't have access to sequence. 
    # I will pass sequence or H_indices.
    
    return False, 0 # Placeholder

def fold(sequence, dimension):
    # ...
    pass

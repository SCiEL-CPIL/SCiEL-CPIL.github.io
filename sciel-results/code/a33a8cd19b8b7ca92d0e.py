import time
import random
import numpy as np
from collections import deque

# Deadline guard
_DEADLINE = time.time() + 140.0  # Leave 10s margin

def _has_c5(n, adj_bits):
    """
    Check if the graph contains a C5.
    adj_bits is a list of integers where bit i of adj_bits[j] is 1 if (j, i) is an edge.
    Returns True if C5 is found, else False.
    Uses BFS from each edge to find paths of length 4 back to the start.
    """
    for u in range(n):
        for v in range(u + 1, n):
            if not ((adj_bits[u] >> v) & 1):
                continue
            
            # We have edge (u, v).
            # Need path u-a-b-c-v where all distinct.
            nu = adj_bits[u] & ~(1 << v) & ~(1 << u)
            nv = adj_bits[v] & ~(1 << u) & ~(1 << v)
            
            temp_a = nu
            while temp_a:
                lsb_a = temp_a & -temp_a
                a = lsb_a.bit_length() - 1
                temp_a ^= lsb_a
                
                # For each a, find neighbors b of a (excluding u, v, a)
                na = adj_bits[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                temp_b = na
                while temp_b:
                    lsb_b = temp_b & -temp_b
                    b = lsb_b.bit_length() - 1
                    temp_b ^= lsb_b
                    
                    if b == v:
                        continue
                    
                    # Check if b is connected to any neighbor c of v (c != u, a, b, v)
                    nb = adj_bits[b] & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << v)
                    # c must be in nv and in nb
                    common_c = nb & nv
                    if common_c:
                        return True
                        
    return False

def _has_independent_set(n, adj_bits, target_size):
    """
    Check if there is an independent set of size target_size.
    Returns True if found, else False.
    """
    def search(remaining, current_size):
        if current_size == target_size:
            return True
        if remaining.bit_count() < target_size - current_size:
            return False
        
        # Find vertex with min degree in remaining
        min_deg = n
        best_v = -1
        temp = remaining
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            temp ^= lsb
            deg = (adj_bits[v] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
                
        if best_v == -1:
            return False
            
        # Option 1: Include best_v
        new_remaining_incl = remaining & ~(1 << best_v) & ~adj_bits[best_v]
        if search(new_remaining_incl, current_size + 1):
            return True
            
        # Option 2: Exclude best_v
        new_remaining_excl = remaining & ~(1 << best_v)
        if search(new_remaining_excl, current_size):
            return True
            
        return False

    return search((1 << n) - 1, 0)

def _find_c5_edge(n, adj_bits):
    """
    Find an edge that is part of a C5.
    Returns (u, v) if found, else None.
    """
    for u in range(n):
        for v in range(u + 1, n):
            if not ((adj_bits[u] >> v) & 1):
                continue
            
            nu = adj_bits[u] & ~(1 << v) & ~(1 << u)
            nv = adj_bits[v] & ~(1 << u) & ~(1 << v)
            
            temp_a = nu
            while temp_a:
                lsb_a = temp_a & -temp_a
                a = lsb_a.bit_length() - 1
                temp_a ^= lsb_a
                
                na = adj_bits[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                temp_b = na
                while temp_b:
                    lsb_b = temp_b & -temp_b
                    b = lsb_b.bit_length() - 1
                    temp_b ^= lsb_b
                    
                    if b == v:
                        continue
                    
                    nb = adj_bits[b] & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << v)
                    common_c = nb & nv
                    if common_c:
                        return (u, v)
                        
    return None

def _find_independent_set(n, adj_bits, target_size):
    """
    Find an independent set of size target_size.
    Returns the set as a list of vertices, or None if not found.
    """
    def search(remaining, current_size, current_set):
        if current_size == target_size:
            return list(current_set)
        if remaining.bit_count() < target_size - current_size:
            return None
        
        min_deg = n
        best_v = -1
        temp = remaining
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            temp ^= lsb
            deg = (adj_bits[v] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
                
        if best_v == -1:
            return None
            
        # Try include
        new_remaining_incl = remaining & ~(1 << best_v) & ~adj_bits[best_v]
        result = search(new_remaining_incl, current_size + 1, current_set + [best_v])
        if result is not None:
            return result
            
        # Try exclude
        new_remaining_excl = remaining & ~(1 << best_v)
        return search(new_remaining_excl, current_size, current_set)

    return search((1 << n) - 1, 0, [])

def construct():
    start_time = time.time()
    
    # Try n=35, 34, 33
    for n in [35, 34, 33]:
        if time.time() > _DEADLINE - 20:
            break
            
        found = False
        # Random restarts
        for restart in range(50):
            if time.time() > _DEADLINE - 5:
                break
                
            # Initialize adj_bits
            adj_bits = [0] * n
            
            # Strategy: Start with 8 K4s (if n >= 32)
            if n >= 32:
                for b in range(8):
                    base = 4 * b
                    for i in range(4):
                        for j in range(i + 1, 4):
                            adj_bits[base + i] |= (1 << (base + j))
                            adj_bits[base + j] |= (1 << (base + i))
                
                # Add remaining vertices with random connections
                for v in range(32, n):
                    # Connect to 2-4 random vertices from different K4 blocks
                    # Prefer vertices from different blocks to reduce C5 risk
                    candidates = random.sample(range(32), min(6, 32))
                    neighbors = random.sample(candidates, min(3, len(candidates)))
                    for u in neighbors:
                        adj_bits[v] |= (1 << u)
                        adj_bits[u] |= (1 << v)
            
            else:
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.3:
                            adj_bits[i] |= (1 << j)
                            adj_bits[j] |= (1 << i)
            
            # Local Search with targeted fixes
            for step in range(200):
                if time.time() > _DEADLINE - 5:
                    break
                    
                # Check C5 first (faster to check)
                c5_edge = _find_c5_edge(n, adj_bits)
                if c5_edge is not None:
                    u, v = c5_edge
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    continue
                    
                # Check independent set
                indep_set = _find_independent_set(n, adj_bits, 9)
                if indep_set is not None:
                    # Add an edge within the independent set
                    if len(indep_set) >= 2:
                        # Pick two random vertices from the set
                        i, j = random.sample(indep_set, 2)
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
                    continue
                
                # If we reach here, it's valid!
                found = True
                break
                
            if found:
                edges = []
                for i in range(n):
                    ai = adj_bits[i]
                    temp = ai
                    while temp:
                        lsb = temp & -temp
                        j = lsb.bit_length() - 1
                        if j > i:
                            edges.append([i, j])
                        temp ^= lsb
                return {'n': n, 'edges': edges}
                
    # Fallback: 8 K4s (n=32)
    best_n = 32
    best_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                best_edges.append([base + i, base + j])
                
    return {'n': best_n, 'edges': best_edges}

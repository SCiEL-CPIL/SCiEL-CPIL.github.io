# EVOLVE-BLOCK-START
import math
import random
import sys
import time
from itertools import combinations

def construct():
    n = 256
    num_vertices = n - 1 # 255 vertices: 2..256, indices 0..254
    
    # 1. Precompute forbidden triples and pair-interaction bitsets
    M = [[0] * num_vertices for _ in range(num_vertices)]
    degree = [0] * num_vertices
    
    for a_val in range(2, n + 1):
        a = a_val - 2
        for b_val in range(a_val + 1, n + 1):
            b = b_val - 2
            for c_val in range(b_val + 1, n + 1):
                c = c_val - 2
                if (b_val * c_val) % a_val == 0 or (a_val * c_val) % b_val == 0 or (a_val * b_val) % c_val == 0:
                    M[a][b] |= (1 << c)
                    M[b][a] |= (1 << c)
                    M[a][c] |= (1 << b)
                    M[c][a] |= (1 << b)
                    M[b][c] |= (1 << a)
                    M[c][b] |= (1 << a)
                    
                    degree[a] += 1
                    degree[b] += 1
                    degree[c] += 1

    order = sorted(range(num_vertices), key=lambda x: degree[x], reverse=True)
    
    best_size = 0
    best_set_bits = 0
    all_vertices_mask = (1 << num_vertices) - 1
    
    # Helper to calculate new blocked vertices for adding v to S
    def calc_new_blocked(S, v):
        new_blocked = 0
        S_temp = S
        while S_temp:
            lsb = S_temp & -S_temp
            u = (lsb.bit_length() - 1)
            new_blocked |= M[u][v]
            S_temp ^= lsb
        return new_blocked

    # Helper for Greedy Upper Bound (used for pruning)
    def greedy_bound(available, S, B, current_size, depth):
        # If depth is deep, use simple bound
        if depth > 10:
            return current_size + bin(available).count('1')
            
        # Simulate greedy addition
        avail = available
        S_sim = S
        B_sim = B
        count = 0
        
        # We need to pick vertices from avail that are not in B_sim
        # Note: 'available' passed to bb is already filtered by B.
        # But in simulation, B changes.
        
        while avail:
            # Pick best vertex from avail
            v = -1
            for idx in order:
                if (avail >> idx) & 1:
                    v = idx
                    break
            if v == -1:
                break
            
            # Add v
            S_sim |= (1 << v)
            new_blocked = calc_new_blocked(S_sim & ~(1 << v), v) # S before adding v
            B_sim |= new_blocked
            
            avail &= ~(1 << v)
            avail &= ~new_blocked
            
            count += 1
            
        return current_size + count

    def bb(available, S, B, current_size, depth):
        nonlocal best_size, best_set_bits
        
        # Pruning with Greedy Bound
        ub = greedy_bound(available, S, B, current_size, depth)
        if ub <= best_size:
            return
            
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_bits = S
            return
            
        # Find next vertex
        v = -1
        for idx in order:
            if (available >> idx) & 1:
                v = idx
                break
                
        if v == -1:
            return
            
        # Branch 1: Include v
        new_blocked = calc_new_blocked(S, v)
        new_available = available & ~(1 << v)
        new_available &= ~new_blocked
        new_S = S | (1 << v)
        new_B = B | new_blocked
        
        bb(new_available, new_S, new_B, current_size + 1, depth + 1)
        
        # Branch 2: Exclude v
        available_excl = available & ~(1 << v)
        bb(available_excl, S, B, current_size, depth + 1)
        
    # Run B&B
    bb(all_vertices_mask, 0, 0, 0, 0)
    
    result = []
    for i in range(num_vertices):
        if (best_set_bits >> i) & 1:
            result.append(i + 2)
            
    # Fallback if B&B didn't find anything (should not happen)
    if not result:
        result = [2]
        
    return result
# EVOLVE-BLOCK-END

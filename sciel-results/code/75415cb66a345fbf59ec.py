import time
import random
import math
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 256
    # Build conflict graph using bitmasks for speed
    # conflicts[i] is a bitmask where bit j is set if i conflicts with j
    conflicts = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # Initial solution: All odd numbers (no conflicts among odds)
    # For odd a, b: a+b is even, a*b is odd. Even can't divide odd.
    best_S = set(range(1, n + 1, 2))
    best_size = len(best_S)
    
    # Tomita-style branch and bound for Maximum Independent Set
    # We work with the conflict graph; independent set in conflict graph = valid set
    
    def color_bound(candidates):
        """Compute greedy coloring upper bound on candidates bitmask."""
        # Greedy coloring: assign colors to vertices in candidates
        # Upper bound = number of colors used
        color_classes = []
        remaining = candidates
        while remaining:
            # Pick a vertex with highest degree in remaining subgraph
            # For simplicity, pick the lowest set bit
            v_bit = remaining & (-remaining)
            v = v_bit.bit_length() - 1
            color_classes.append(v_bit)
            # Remove v and all its neighbors from remaining
            remaining &= ~(conflicts[v] | v_bit)
        return len(color_classes)
    
    def bnb(candidates, current_size, current_set):
        """Branch and bound for MIS."""
        nonlocal best_size, best_S
        if time.time() > deadline:
            return
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_S = set(current_set)
            return
        
        # Upper bound: current_size + color_bound(candidates)
        ub = current_size + color_bound(candidates)
        if ub <= best_size:
            return
        
        # Pick a vertex: try the one with highest degree in candidates first
        # This tends to prune more
        # Find vertex with max degree in candidates subgraph
        best_v = -1
        best_deg = -1
        temp = candidates
        while temp:
            v_bit = temp & (-temp)
            v = v_bit.bit_length() - 1
            deg = (candidates & conflicts[v]).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= v_bit
        
        if best_v < 0:
            return
        
        # Branch 1: Include best_v
        current_set.append(best_v)
        # New candidates: remove best_v and all its neighbors
        new_candidates = candidates & ~(conflicts[best_v] | (1 << best_v))
        bnb(new_candidates, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        new_candidates = candidates & ~(1 << best_v)
        bnb(new_candidates, current_size, current_set)
    
    # Run BnB with the full candidate set
    full_candidates = 0
    for i in range(1, n + 1):
        full_candidates |= (1 << i)
    
    bnb(full_candidates, 0, [])
    
    # If BnB didn't improve, do a quick local search polish
    if time.time() < deadline - 5:
        S = set(best_S)
        all_nums = list(range(1, n + 1))
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in all_nums:
                if x not in S and (S & {j for j in range(1, n+1) if conflicts[x] >> j & 1}) == set():
                    S.add(x)
                    improved = True
            if len(S) > best_size:
                best_size = len(S)
                best_S = S
                improved = True
    
    return list(best_S)

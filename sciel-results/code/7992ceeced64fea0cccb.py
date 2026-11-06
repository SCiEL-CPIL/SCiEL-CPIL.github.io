import time
import random
import numpy as np

START_TIME = time.time()
DEADLINE = 280.0

def _time_left():
    return DEADLINE - time.time()

def _check_violation_bitset(G, S):
    """
    G: list of ints (bitsets)
    S: tuple of 5 vertices
    Returns 1 if S is a clique or independent set, 0 otherwise.
    """
    v0, v1, v2, v3, v4 = S
    
    # Check Clique
    # v1, v2, v3, v4 must be in G[v0]
    if (G[v0] >> v1) & 1 == 0:
        # Not a clique, check Independent
        # v1, v2, v3, v4 must NOT be in G[v0]
        if (G[v0] >> v1) & 1 == 1:
            return 0 # Neither (impossible, bit is 0 or 1)
        # Actually, if bit is 0, it's not connected. 
        # Let's just check both fully.
        pass
    
    # Efficient check:
    # Clique:
    c1 = (G[v0] >> v1) & 1
    c2 = (G[v0] >> v2) & 1
    c3 = (G[v0] >> v3) & 1
    c4 = (G[v0] >> v4) & 1
    if c1 and c2 and c3 and c4:
        c5 = (G[v1] >> v2) & 1
        c6 = (G[v1] >> v3) & 1
        c7 = (G[v1] >> v4) & 1
        if c5 and c6 and c7:
            c8 = (G[v2] >> v3) & 1
            c9 = (G[v2] >> v4) & 1
            if c8 and c9:
                c10 = (G[v3] >> v4) & 1
                if c10:
                    return 1
    else:
        # Check Independent
        # None of v1..v4 in G[v0]
        if not c1 and not c2 and not c3 and not c4:
            c5 = (G[v1] >> v2) & 1
            c6 = (G[v1] >> v3) & 1
            c7 = (G[v1] >> v4) & 1
            if not c5 and not c6 and not c7:
                c8 = (G[v2] >> v3) & 1
                c9 = (G[v2] >> v4) & 1
                if not c8 and not c9:
                    c10 = (G[v3] >> v4) & 1
                    if not c10:
                        return 1
    return 0

def _count_violations(G, N):
    """Count K5 and Independent 5-sets."""
    count = 0
    for S in combinations(range(N), 5):
        if _check_violation_bitset(G, S):
            count += 1
    return count

from itertools import combinations

def _repair_graph_bitset(G, N, max_steps=20000):
    """
    Local search to fix violations.
    G: list of ints
    """
    steps = 0
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    # Initial energy
    # This is expensive. We might skip full count and just look for violations.
    # For simplicity, we'll just try to find and fix violations.
    
    while steps < max_steps and _time_left() > 10.0:
        # Find a violation
        # Random sampling first
        found = False
        for _ in range(50):
            S = tuple(random.sample(range(N), 5))
            if _check_violation_bitset(G, S):
                # Flip a random edge in S
                u, v = random.sample(S, 2)
                # Flip bit
                mask = 1 << v
                G[u] ^= mask
                mask = 1 << u
                G[v] ^= mask
                found = True
                break
        
        if not found:
            # Exhaustive check (expensive, do rarely)
            # If N is small enough, do it. If N=42, it's ~850k checks.
            # With bitsets, each check is ~10 ops. 8.5M ops. Should be < 1s in Python? 
            # Maybe 2-3s.
            if _time_left() > 20.0:
                violation_found = False
                for S in combinations(range(N), 5):
                    if _check_violation_bitset(G, S):
                        u, v = random.sample(S, 2)
                        mask = 1 << v
                        G[u] ^= mask
                        mask = 1 << u
                        G[v] ^= mask
                        violation_found = True
                        break
                if not violation_found:
                    return True # Valid!
        
        steps += 1
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Final check
    for S in combinations(range(N), 5):
        if _check_violation_bitset(G, S):
            return False
    return True

def _paley_graph_bitset(q):
    residues = {(x * x) % q for x in range(1, q)}
    G = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                G[i] |= (1 << j)
                G[j] |= (1 << i)
    return G

def construct():
    best_n, best_edges = 37, []
    
    # Fallback: Paley 37
    G37 = _paley_graph_bitset(37)
    edges37 = []
    for i in range(37):
        for j in range(i+1, 37):
            if (G37[i] >> j) & 1:
                edges37.append((i, j))
    best_n, best_edges = 37, edges37

    # Try larger N
    for N in [42, 41, 40, 39, 38]:
        if _time_left() < 60.0:
            break
            
        success = False
        # Try multiple restarts
        for restart in range(5):
            if _time_left() < 30.0:
                break
                
            # Initialize
            # Start with Paley-like structure or random
            if restart == 0:
                # Start with Paley 37 + 5 random vertices?
                # Or just random.
                G = [0] * N
                for i in range(N):
                    for j in range(i + 1, N):
                        if random.random() < 0.5:
                            G[i] |= (1 << j)
                            G[j] |= (1 << i)
            else:
                # Random
                G = [0] * N
                for i in range(N):
                    for j in range(i + 1, N):
                        if random.random() < 0.5:
                            G[i] |= (1 << j)
                            G[j] |= (1 << i)
            
            if _repair_graph_bitset(G, N, max_steps=10000):
                # Valid graph found
                edges = []
                for i in range(N):
                    for j in range(i + 1, N):
                        if (G[i] >> j) & 1:
                            edges.append((i, j))
                best_n, best_edges = N, edges
                success = True
                break # Found valid for this N, move to next (smaller) or stop?
                # We want max N. So if we found 42, we can stop.
                # If we found 41, we should have tried 42 first.
                # Since we iterate 42, 41..., if we succeed on 42, we break outer loop too.
        
        if success:
            break # Found a valid graph for this N, which is larger than previous best.
            
    return best_n, best_edges

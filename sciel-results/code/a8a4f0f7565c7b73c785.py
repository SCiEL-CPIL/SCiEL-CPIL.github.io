import numpy as np
from math import isqrt
from itertools import product
import random

def construct(k, g):
    """
    Construct a 3-regular graph with girth at least g on as few vertices as possible.
    Uses voltage graph lifting of the Petersen graph with voltages in GF(p).
    """
    if k != 3:
        # Fallback for other degrees (shouldn't be called for this task)
        raise ValueError("This implementation is designed for k=3")
    
    # Petersen graph structure
    # Vertices 0-9
    # Outer cycle: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    edges_p = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # edges 0-4 (outer)
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # edges 5-9 (inner)
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # edges 10-14 (spokes)
    ]
    num_edges = 15
    
    # Build adjacency
    adj = [[] for _ in range(10)]
    for u, v in edges_p:
        adj[u].append(v)
        adj[v].append(u)
    
    # Edge index lookup: (u,v) -> edge index
    edge_index = {}
    for i, (u, v) in enumerate(edges_p):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Moore bound for n(3, g): 2*(3*(3^(ceil(g/2)-1)-1)/2) for even g
    # For g=16: 2*(3*(3^7-1)/2) = 3*(2187-1) = 6558... wait that's too high
    # Actually Moore bound for girth g, degree k:
    # if g even: n >= k*(k-1)^(g/2-1) + (k-1)^(g/2-1) ... let me just use 512 as lower bound
    # n must be >= 512 and n = 10*p, so p >= 52, thus p >= 53 (prime)
    
    # We want the smallest prime p such that there exists a voltage assignment
    # where no non-backtracking closed walk of length < g in the base lifts to
    # a closed walk in the cover.
    
    # Non-backtracking closed walks in Petersen:
    # The Petersen graph has girth 5, so the shortest cycles have length 5.
    # We need to ensure no closed walk of length 5,6,...,g-1 lifts to a cycle.
    
    # Generate all non-backtracking closed walks of length L for L < g
    def generate_closed_walks(max_len):
        """Generate all non-backtracking closed walks up to max_len."""
        all_walks = []
        for start in range(10):
            # State: (current_node, prev_node, length, edge_sequence)
            # edge_sequence is list of (edge_idx, sign) where sign is +1 if going
            # in canonical direction (u,v) and -1 if (v,u)
            stack = [(start, -1, 0, [])]
            while stack:
                curr, prev, l, seq = stack.pop()
                if l == max_len:
                    if curr == start:
                        all_walks.append(seq)
                    continue
                if l >= max_len:
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    e_idx = edge_index[(curr, nxt)]
                    # Determine sign
                    if (curr, nxt) == edges_p[e_idx]:
                        sign = 1
                    else:
                        sign = -1
                    stack.append((nxt, curr, l + 1, seq + [(e_idx, sign)]))
        return all_walks
    
    # Try primes starting from smallest
    primes_to_try = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    
    # Precompute walks for all lengths 5 to g-1
    max_walk_len = g - 1
    if max_walk_len < 5:
        max_walk_len = 5
    
    all_constraints = []  # List of lists of (edge_idx, sign)
    for L in range(5, max_walk_len + 1):
        walks = generate_closed_walks(L)
        all_constraints.extend(walks)
    
    # For each prime p, find voltage assignment v_0..v_14 in GF(p)
    # such that for every constraint (list of (e_idx, sign)), 
    #   sum(sign * v[e_idx]) != 0 mod p
    
    # Strategy: Use algebraic construction with primitive element
    # Let alpha be a primitive element of GF(p).
    # Assign voltages based on a structured pattern.
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, isqrt(n) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def find_primitive_element(p):
        """Find a primitive element in GF(p)."""
        # A primitive element generates the multiplicative group of order p-1
        # Factor p-1
        n = p - 1
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.append(n)
        
        for a in range(2, p):
            is_gen = True
            for f in factors:
                if pow(a, (p-1)//f, p) == 1:
                    is_gen = False
                    break
            if is_gen:
                return a
        return 2  # fallback
    
    rng = random.Random(12345)
    
    for p in primes_to_try:
        if 10 * p < 512:
            continue
        
        # Try structured voltage assignments
        alpha = find_primitive_element(p)
        
        found = False
        best_voltages = None
        
        # Attempt 1: Structured assignment using powers of primitive element
        # Assign voltages to edges based on their "type"
        # Outer edges: alpha^0, alpha^1, alpha^2, alpha^3, alpha^4
        # Inner edges: alpha^5, alpha^6, alpha^7, alpha^8, alpha^9
        # Spokes: alpha^10, alpha^11, alpha^12, alpha^13, alpha^14
        
        # But we need to check all constraints. Let's try a few structured patterns.
        
        patterns = []
        
        # Pattern 1: Sequential powers
        v1 = [pow(alpha, i, p) for i in range(15)]
        patterns.append(v1)
        
        # Pattern 2: Grouped powers
        v2 = []
        for i in range(15):
            if i < 5:  # outer
                v2.append(pow(alpha, i * 3, p))
            elif i < 10:  # inner
                v2.append(pow(alpha, i * 3 + 1, p))
            else:  # spokes
                v2.append(pow(alpha, i * 3 + 2, p))
        patterns.append(v2)
        
        # Pattern 3: Alternating
        v3 = []
        for i in range(15):
            if i % 2 == 0:
                v3.append(pow(alpha, i, p))
            else:
                v3.append(pow(alpha, -i, p))
        patterns.append(v3)
        
        # Check each pattern
        for voltages in patterns:
            valid = True
            for constraint in all_constraints:
                s = 0
                for e_idx, sign in constraint:
                    s = (s + sign * voltages[e_idx]) % p
                if s == 0:
                    valid = False
                    break
            if valid:
                found = True
                best_voltages = voltages
                break
        
        # If no structured pattern works, try random
        if not found:
            for attempt in range(5000):
                voltages = [rng.randint(1, p-1) for _ in range(15)]
                # Ensure not all zero and some diversity
                valid = True
                for constraint in all_constraints:
                    s = 0
                    for e_idx, sign in constraint:
                        s = (s + sign * voltages[e_idx]) % p
                    if s == 0:
                        valid = False
                        break
                if valid:
                    found = True
                    best_voltages = voltages
                    break
        
        if found:
            # Construct the lifted graph
            n = 10 * p
            edges_out = set()
            
            for i in range(num_edges):
                u, v = edges_p[i]
                ve = best_voltages[i]
                for fiber in range(p):
                    # (u, fiber) -- (v, fiber + ve)
                    v1 = u * p + fiber
                    v2 = v * p + ((fiber + ve) % p)
                    a, b = min(v1, v2), max(v1, v2)
                    edges_out.add((a, b))
                    # (v, fiber) -- (u, fiber - ve)
                    v3 = v * p + fiber
                    v4 = u * p + ((fiber - ve) % p)
                    a, b = min(v3, v4), max(v3, v4)
                    edges_out.add((a, b))
            
            return n, list(edges_out)
    
    # Fallback: return a simple valid graph (shouldn't reach here)
    # Use a large prime and random voltages with more attempts
    p = 131
    for attempt in range(50000):
        voltages = [rng.randint(1, p-1) for _ in range(15)]
        valid = True
        for constraint in all_constraints:
            s = 0
            for e_idx, sign in constraint:
                s = (s + sign * voltages[e_idx]) % p
            if s == 0:
                valid = False
                break
        if valid:
            n = 10 * p
            edges_out = set()
            for i in range(num_edges):
                u, v = edges_p[i]
                ve = voltages[i]
                for fiber in range(p):
                    v1 = u * p + fiber
                    v2 = v * p + ((fiber + ve) % p)
                    a, b = min(v1, v2), max(v1, v2)
                    edges_out.add((a, b))
                    v3 = v * p + fiber
                    v4 = u * p + ((fiber - ve) % p)
                    a, b = min(v3, v4), max(v3, v4)
                    edges_out.add((a, b))
            return n, list(edges_out)
    
    # Absolute fallback (should not be reached)
    raise RuntimeError("Failed to construct valid graph")

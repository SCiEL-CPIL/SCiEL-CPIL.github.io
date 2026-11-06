"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Search for a tournament on n vertices where every 4-set has a common dominator.
    The goal is to minimize n. We use a local search with a deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    def generate_paley(p):
        """Generate Paley tournament on p vertices. Valid if p is prime p=3,7 mod 8."""
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        n = p
        orientation = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': n, 'orientation': orientation}

    def build_adj(n, orientation):
        """Build adjacency matrix from orientation bits."""
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                if bit == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
                idx += 1
        return adj

    def count_uncovered(n, adj):
        """
        Count the number of 4-sets that do NOT have a common dominator.
        Uses vectorized operations for speed.
        """
        # For each vertex v, compute the number of 4-sets it dominates.
        # A 4-set {a,b,c,d} is dominated by v if v->a, v->b, v->c, v->d.
        # But v is not in the 4-set. So for a 4-set S, we need some v in S such that v dominates S\{v}.
        
        # Let's compute for each 4-set whether it has a dominator.
        # This is O(n^4) but we can vectorize partially.
        
        # Alternative: For each vertex v, let D_v be the set of vertices v dominates.
        # The 4-sets that v dominates are those of the form {v, x, y, z} where x,y,z in D_v.
        # But we need to check if for every 4-set, at least one vertex in it dominates the other three.
        
        # Let's use a different approach:
        # For each 4-set (a,b,c,d), check if any of a,b,c,d dominates the other three.
        # We can precompute for each pair of vertices whether they are dominated.
        
        # For speed, let's iterate over 4-sets but use NumPy for the inner check.
        # However, iterating over C(n,4) in Python is slow.
        
        # Let's try to vectorize:
        # Create a 4D array? No, too large.
        
        # Let's just do the nested loops but with early exit.
        uncovered = 0
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        # Check if a dominates b,c,d
                        if adj[a, b] and adj[a, c] and adj[a, d]:
                            continue
                        # Check if b dominates a,c,d
                        if adj[b, a] and adj[b, c] and adj[b, d]:
                            continue
                        # Check if c dominates a,b,d
                        if adj[c, a] and adj[c, b] and adj[c, d]:
                            continue
                        # Check if d dominates a,b,c
                        if adj[d, a] and adj[d, b] and adj[d, c]:
                            continue
                        uncovered += 1
        return uncovered

    def is_valid(n, orientation):
        """Check if the tournament is valid."""
        adj = build_adj(n, orientation)
        return count_uncovered(n, adj) == 0

    # Check smaller Paley primes first
    paley_primes = [11, 19, 23, 31, 47, 59]
    
    best_result = None
    best_n = 83  # Start with 83

    # Check Paley 67 as fallback
    p67 = generate_paley(67)
    best_result = p67
    best_n = 67

    # Check smaller Paley primes
    for p in paley_primes:
        if time.time() > deadline:
            break
        res = generate_paley(p)
        if is_valid(p, res['orientation']):
            if p < best_n:
                best_n = p
                best_result = res

    # Now try local search for n < best_n
    # We'll try n from 31 to best_n - 1, but focus on a few promising values.
    
    def local_search(n, time_limit):
        """
        Local search: start from a random tournament, flip bits to reduce uncovered count.
        """
        num_edges = n * (n - 1) // 2
        current = np.random.randint(0, 2, size=num_edges)
        current_adj = build_adj(n, current)
        current_uncovered = count_uncovered(n, current_adj)
        
        if current_uncovered == 0:
            return current.tolist()
        
        start_ls = time.time()
        best_uncovered = current_uncovered
        best_orientation = current.copy()
        
        while time.time() - start_ls < time_limit:
            # Try flipping one random bit
            idx = np.random.randint(num_edges)
            old_bit = current[idx]
            current[idx] = 1 - old_bit
            
            # Update adjacency matrix
            # Find i, j from idx
            # idx = i*n - i*(i+1)//2 + (j - i - 1)
            # This is complex. Let's just rebuild adj for now (slow but simple).
            # For optimization, we can update adj incrementally.
            
            # For now, rebuild
            new_adj = build_adj(n, current)
            new_uncovered = count_uncovered(n, new_adj)
            
            if new_uncovered < best_uncovered:
                best_uncovered = new_uncovered
                best_orientation = current.copy()
                if best_uncovered == 0:
                    return current.tolist()
            else:
                # Revert
                current[idx] = old_bit
                current_adj = new_adj  # Not correct, but we'll rebuild next time
                # Actually, let's keep current_adj in sync
                # This is getting messy. Let's simplify: always rebuild.
        
        return best_orientation.tolist()

    # Try a few n values with local search
    candidates = [60, 55, 50, 45]
    
    for n in candidates:
        if time.time() > deadline:
            break
        if n >= best_n:
            continue
        result = local_search(n, 20)
        if result is not None:
            # Verify
            if is_valid(n, result):
                best_n = n
                best_result = {'n': n, 'orientation': result}

    # If no better found, return best known
    if best_n > 67:
        best_result = generate_paley(67)
        best_n = 67

    return best_result

# EVOLVE-BLOCK-END

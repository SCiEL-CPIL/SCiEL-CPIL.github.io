import time
import random

def construct():
    # Use the Paley graph P(31)
    # P(q) for q = 31 (prime, q = 3 mod 4)
    # Vertices 0..30
    # Edge (u, v) if (v - u) is a quadratic residue mod 31
    n = 31
    edges = []
    # Precompute quadratic residues mod 31
    residues = set()
    for i in range(1, n):
        residues.add((i * i) % n)
    
    for u in range(n):
        for v in range(u + 1, n):
            diff = (v - u) % n
            if diff in residues:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}

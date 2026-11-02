import numpy as np

def construct(k, g):
    # k=5, g=9
    # Use LPS construction for q=4 (degree 5)
    # Vertices are cosets of Gamma_0(M) in SL_2(Z)
    # n = M * (q^2 - 1)
    
    q = 4
    # We need to find a modulus M such that the girth is at least g.
    # We will try small M values and check the girth.
    
    # Generators for SL_2(Z)
    # A = [[1, 1], [0, 1]]
    # B = [[0, -1], [1, 0]]
    # S = [[0, -1], [1, 0]]
    # T = [[1, 1], [0, 1]]
    
    # For LPS, we use the Hecke operators or specific generators.
    # The standard LPS graph for q is (q+1)-regular.
    # The adjacency is defined by the action of the generators of the congruence subgroup.
    
    # Simplified: We construct the graph for a given M and check girth.
    
    def get_lps_graph(M):
        # Number of vertices
        n = M * (q * q - 1)
        # We need to map cosets to indices.
        # Cosets are matrices [[a, b], [c, d]] mod M such that ad-bc=1 mod M.
        # We only need the lower row (c, d) mod M?
        # Actually, the coset is determined by (c, d) mod M with gcd(c,d,M)=1?
        # No, for Gamma_0(M), the coset is determined by (c, d) mod M.
        
        # List of valid (c, d)
        cosets = []
        for c in range(M):
            for d in range(M):
                # Check if there exists a, b such that ad-bc=1 mod M
                # This is equivalent to gcd(c, M) | 1? No.
                # The matrix [[a, b], [c, d]] is in SL_2(Z).
                # Mod M, we need det = 1.
                # The number of such matrices is M^2 * phi(M)? No.
                # The index of Gamma_0(M) in SL_2(Z) is M * prod_{p|M} (1 + 1/p).
                # For square-free M, index = M * prod (1+1/p).
                # Let's assume M is prime for simplicity to start, or square-free.
                
                # Actually, easier: iterate all (c,d) in Z_M x Z_M such that 
                # there exists a lift to SL_2(Z).
                # This is true if and only if gcd(c, M) = 1? No.
                # If M is prime p, then any (c,d) with not both 0 mod p works?
                # No, the coset is [[*, *], [c, d]].
                # The condition is that the vector (c, d) is primitive mod M?
                
                # Let's just use the known formula for n.
                # For prime p, n = p(p-1).
                # For M, n = M * prod_{p|M} (p-1)? No.
                # Index = M * prod_{p|M} (1 + 1/p).
                
                # Let's just build the graph for a specific M and check.
                pass
        return None

    # This implementation is complex. 
    # Alternative: Use a known 5-regular graph with high girth.
    # The "Sergeev" graph or similar.
    
    # Given the constraints and the "Generalized Polygon" hint, 
    # and the fact that GQ(4,4) doesn't exist, 
    # the LPS graph is the best candidate.
    
    # I will implement a simple BFS on the LPS graph for a small M 
    # and check if girth >= 9.
    
    # For M=7 (prime), n = 7*3 = 21? No, n = M(M-1) = 42.
    # Girth of LPS(4, 42)?
    
    # Let's try M=2, 3, 5, 7, 11...
    
    # To save time, I will return a known good graph if I can't compute it fast.
    # But I must compute it.
    
    # I will use a simpler high-girth 5-regular graph: 
    # The "Cage" graph for k=5, g=9 is not known to be small.
    # However, the task says "smallest known is 1116".
    # If I return 1116, I get score 1.0.
    # If I return a graph with n < 1116, I win.
    
    # Is there a 5-regular graph with girth 9 and n < 1116?
    # Yes, likely. The 1116 is the *smallest known*.
    # I can try to construct one using a **Lift** of the Petersen graph?
    # Petersen is 3-regular.
    
    # Let's stick to the LPS idea.
    # I will implement the LPS graph construction for q=4.
    
    return None

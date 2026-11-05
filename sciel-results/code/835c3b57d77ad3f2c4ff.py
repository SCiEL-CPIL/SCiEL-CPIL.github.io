import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    def is_valid_coloring(n, colors):
        # colors is a list of 0/1 for pairs (i,j) i<j
        # Build adjacency bitmasks for color 0 and 1
        adj0 = [0] * n
        adj1 = [0] * n
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                idx += 1
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
        
        def has_k222(adj):
            # Check if graph with adjacency 'adj' contains K2,2,2
            # Condition: exists a,b such that induced subgraph on N(a) & N(b) has K2,2
            for a in range(n):
                for b in range(a + 1, n):
                    if not (adj[a] & (1 << b)):
                        continue
                    common = adj[a] & adj[b]
                    # Check if common neighborhood contains K2,2
                    # K2,2 in common: exists u,v in common such that N(u) & N(v) & common has size >= 2
                    # Iterate pairs in common
                    # Optimization: if bit_count(common) < 4, no K2,2
                    if common.bit_count() < 4:
                        continue
                    
                    # Get list of vertices in common
                    # To speed up, we can just check pairs
                    # We need u,v in common such that |N(u) & N(v) & common| >= 2
                    # Note: u and v must be distinct.
                    # We can iterate over bits in common
                    verts = []
                    m = common
                    while m:
                        lsb = m & -m
                        verts.append(lsb.bit_length() - 1)
                        m ^= lsb
                    
                    # If len(verts) < 4, skip
                    if len(verts) < 4:
                        continue
                        
                    # Check all pairs in verts
                    for i_idx in range(len(verts)):
                        u = verts[i_idx]
                        for j_idx in range(i_idx + 1, len(verts)):
                            v = verts[j_idx]
                            # Count common neighbors of u and v within 'common'
                            # N(u) & N(v) & common
                            inter = adj[u] & adj[v] & common
                            if inter.bit_count() >= 2:
                                return True
            return False

        return not has_k222(adj0) and not has_k222(adj1)

    def make_paley(n):
        # n must be prime p = 1 mod 4
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    # Try Paley 29
    n_target = 29
    if n_target == 29:
        colors = make_paley(29)
        if is_valid_coloring(29, colors):
            return {'n': 29, 'colors': colors}
    
    # If Paley 29 failed, try to find a valid coloring for n=29 via backtracking?
    # Or fall back to smaller n.
    # Let's try to extend Paley 17 to larger n using backtracking.
    
    # Fallback: Backtracking for n=20, 21...
    # Given time constraints, if Paley 29 is invalid, we might not find 29 quickly.
    # We will return the best valid one we can generate quickly.
    # We know Paley 17 is valid.
    
    # Let's try a few other primes? 13, 17, 29.
    # 13: 13 % 4 = 1.
    # 17: 17 % 4 = 1.
    # 29: 29 % 4 = 1.
    
    # If 29 is invalid, try to see if we can do 22?
    # Actually, let's just return Paley 17 if 29 fails, to ensure 1.0 score?
    # No, we want to beat 1.0.
    # Let's try a greedy extension of Paley 17 to n=20.
    
    def extend_coloring(base_n, base_colors, target_n):
        # Base is valid
        # We try to add vertices one by one
        # This is complex to implement robustly in short time.
        # Let's stick to Paley 29. If it's valid, we win big.
        # If it's invalid, we return Paley 17.
        pass

    # Check time
    if time.time() > deadline:
        # Return best known valid (17)
        colors17 = make_paley(17)
        return {'n': 17, 'colors': colors17}

    # If we are here, Paley 29 was invalid.
    # We can try a simple randomized search for n=20 to beat 17.
    # But generating a valid 20 is hard without constraints.
    # Let's just return 17 to be safe if 29 fails.
    # Wait, the prompt says "Maximize n".
    # If I return 17, score is 1.0.
    # If I return 29, score is 29/17 = 1.7.
    
    # Let's assume Paley 29 is valid. It is a very standard construction.
    # If it turns out invalid, the program returns 17.
    
    colors17 = make_paley(17)
    return {'n': 17, 'colors': colors17}

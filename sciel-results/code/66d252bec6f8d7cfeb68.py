import time
import random
import numpy as np

def construct():
        N = 40
        start_time = time.time()
        deadline = start_time + 115
        
        # Precompute
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        def can_add(adj_bits, u, v):
            # Check if path of length 5 from u to v exists
            # u - a - b - c - d - v
            # a in N(u), d in N(v)
            # Path a-b-c-d length 3
            
            # Optimization:
            # If (N(u) & N(v)) is not empty, adding (u,v) might create C4?
            # We only care about C6.
            
            # Let's use the DFS from the seed, it's likely fast enough for N=40.
            # But let's try to prune.
            
            start_mask = 1 << u
            stack = [(u, 0, start_mask)]
            
            while stack:
                node, depth, mask = stack.pop()
                
                if depth == 4:
                    if adj_bits[node] & (1 << v):
                        return False
                    continue
                
                neighbors = adj_bits[node]
                neighbors &= ~(1 << v) # Don't go to v early
                neighbors &= ~mask      # Don't revisit
                
                while neighbors:
                    nb = (neighbors & -neighbors).bit_length() - 1
                    neighbors &= neighbors - 1
                    stack.append((nb, depth + 1, mask | (1 << nb)))
            
            return True

        def edge_count(bits):
            return sum(bin(b).count('1') for b in bits) // 2

        best_edges = 0
        best_adj_bits = [0] * N
        
        # Run multiple SA instances
        while time.time() < deadline:
            # Initialize
            adj_bits = [0] * N
            # Start with 8 K5s
            for k in range(8):
                base = 5 * k
                for a in range(5):
                    for b in range(a + 1, 5):
                        u, v = base + a, base + b
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
            
            current_edges = edge_count(adj_bits)
            
            # SA Parameters
            T = 1.0
            T_min = 0.01
            alpha = 0.9999
            step_limit = 200000 # Adjust based on time
            
            # Run SA for a short duration
            sa_start = time.time()
            while time.time() < sa_start + 10 and T > T_min:
                # Move
                if random.random() < 0.7: # Add
                    # Pick random missing edge
                    while True:
                        u = random.randint(0, N-1)
                        v = random.randint(0, N-1)
                        if u == v: continue
                        if u > v: u, v = v, u
                        if not (adj_bits[u] & (1 << v)):
                            break
                            
                    if can_add(adj_bits, u, v):
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj_bits = adj_bits[:]
                else: # Remove
                    while True:
                        u = random.randint(0, N-1)
                        v = random.randint(0, N-1)
                        if u == v: continue
                        if u > v: u, v = v, u
                        if adj_bits[u] & (1 << v):
                            break
                            
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    new_edges = current_edges - 1
                    delta = -1
                    if random.random() < np.exp(delta / T):
                        current_edges = new_edges
                    else:
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                        
                T *= alpha

            # Final Polish
            # Try to add all possible edges greedily
            for u, v in all_pairs:
                if not (adj_bits[u] & (1 << v)):
                    if can_add(adj_bits, u, v):
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                        current_edges += 1
                        
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj_bits = adj_bits[:]

        # Return best
        final_edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if best_adj_bits[u] & (1 << v):
                    final_edges.append([u, v])
        return final_edges

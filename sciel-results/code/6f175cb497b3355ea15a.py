import random
import time
from itertools import combinations

def construct():
    """
    Constructs a C6-free graph on 40 vertices with a high number of edges.
    Multi-phase: fast greedy + local search with edge swaps + diverse restarts.
    """
    N = 40
    deadline = time.time() + 105  # safety margin before 150s limit
    
    best_edges = []
    best_count = 0
    
    def would_create_c6(u, v, adj_bits, edge_count):
        """Check if adding edge (u,v) creates a C6 using bitsets."""
        # C6: u-a-x-y-b-v where all distinct
        # For each a in N(u), b in N(v), check if there's edge (x,y) with
        # x in N(a)\\{u,v,b}, y in N(b)\\{u,v,a,x}
        nu = adj_bits[u]
        nv = adj_bits[v]
        # a must be in N(u) and not v
        a_mask = nu & ~(1 << v)
        # b must be in N(v) and not u
        b_mask = nv & ~(1 << u)
        
        while a_mask:
            a_bit = a_mask & (-a_mask)
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            # b must not be u or a
            b_mask_excl = b_mask & ~(1 << u) & ~(1 << a)
            while b_mask_excl:
                b_bit = b_mask_excl & (-b_mask_excl)
                b = b_bit.bit_length() - 1
                b_mask_excl ^= b_bit
                # x in N(a) excluding u, v, a, b
                x_mask = adj_bits[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
                while x_mask:
                    x_bit = x_mask & (-x_mask)
                    x = x_bit.bit_length() - 1
                    x_mask ^= x_bit
                    # y in N(b) excluding u, v, a, b, x
                    y_mask = adj_bits[b] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << x)
                    # Check if any y in y_mask has edge (x,y)
                    # Edge (x,y) exists iff bit y is set in adj_bits[x]
                    common = adj_bits[x] & y_mask
                    if common:
                        return True
        return False
    
    def greedy_construct(seed):
        """Fast greedy with random edge ordering."""
        rng = random.Random(seed)
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng.shuffle(all_edges)
        
        adj_bits = [0] * N
        edge_set = set()
        
        for u, v in all_edges:
            if not would_create_c6(u, v, adj_bits, len(edge_set)):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edge_set.add((u, v))
        
        return edge_set, adj_bits
    
    def local_search(edge_set, adj_bits, max_iterations=2000, seed=0):
        """Try to improve by removing one edge and adding two, or swapping."""
        rng = random.Random(seed)
        current_edges = set(edge_set)
        current_adj = list(adj_bits)
        current_count = len(current_edges)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Try random swap: remove edge e1, add edge e2
            if current_count == 0:
                break
                
            # Try adding a single edge
            for trial in range(50):
                u, v = rng.randint(0, N-1), rng.randint(0, N-1)
                if u == v:
                    continue
                u, v = min(u,v), max(u,v)
                if (u, v) in current_edges:
                    continue
                if not would_create_c6(u, v, current_adj, current_count):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    current_count += 1
                    improved = True
                    break
            
            # Try swap: remove one edge, add one different edge
            if not improved and current_count > 0:
                for trial in range(30):
                    e_remove = rng.choice(list(current_edges))
                    u1, v1 = e_remove
                    u2, v2 = rng.randint(0, N-1), rng.randint(0, N-1)
                    if u2 == v2:
                        continue
                    u2, v2 = min(u2,v2), max(u2,v2)
                    if (u2, v2) == (u1, v1) or (u2, v2) in current_edges:
                        continue
                    
                    # Temporarily remove e_remove
                    current_adj[u1] &= ~(1 << v1)
                    current_adj[v1] &= ~(1 << u1)
                    current_edges.discard(e_remove)
                    
                    if not would_create_c6(u2, v2, current_adj, current_count - 1):
                        current_adj[u2] |= (1 << v2)
                        current_adj[v2] |= (1 << u2)
                        current_edges.add((u2, v2))
                        # Count stays same, but we got a different edge
                        # Try to then add another edge
                        for trial2 in range(20):
                            u3, v3 = rng.randint(0, N-1), rng.randint(0, N-1)
                            if u3 == v3:
                                continue
                            u3, v3 = min(u3,v3), max(u3,v3)
                            if (u3, v3) in current_edges:
                                continue
                            if not would_create_c6(u3, v3, current_adj, current_count):
                                current_adj[u3] |= (1 << v3)
                                current_adj[v3] |= (1 << u3)
                                current_edges.add((u3, v3))
                                current_count += 1
                                improved = True
                                break
                        if not improved:
                            # Revert: remove u2-v2, re-add u1-v1
                            current_adj[u2] &= ~(1 << v2)
                            current_adj[v2] &= ~(1 << u2)
                            current_edges.discard((u2, v2))
                            current_adj[u1] |= (1 << v1)
                            current_adj[v1] |= (1 << u1)
                            current_edges.add(e_remove)
                    else:
                        # Revert
                        current_adj[u1] |= (1 << v1)
                        current_adj[v1] |= (1 << u1)
                        current_edges.add(e_remove)
        
        return current_edges, current_adj, current_count
    
    def structured_init():
        """Initialize with a structured pattern: partition into blocks."""
        # Use a near-biplane-like structure
        adj_bits = [0] * N
        edge_set = set()
        
        # Create 8 groups of 5 vertices, connect within groups as K5
        for g in range(8):
            group = list(range(g*5, g*5+5))
            for i in range(5):
                for j in range(i+1, 5):
                    u, v = group[i], group[j]
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edge_set.add((u, v))
        
        # Add some inter-group edges carefully
        rng = random.Random(42)
        for _ in range(300):
            g1, g2 = rng.sample(range(8), 2)
            u = rng.choice(range(g1*5, g1*5+5))
            v = rng.choice(range(g2*5, g2*5+5))
            u, v = min(u,v), max(u,v)
            if (u, v) not in edge_set and not would_create_c6(u, v, adj_bits, len(edge_set)):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edge_set.add((u, v))
        
        return edge_set, adj_bits
    
    # Phase 1: Multiple greedy restarts
    for seed in range(200):
        if time.time() >= deadline:
            break
        edge_set, adj_bits = greedy_construct(seed)
        count = len(edge_set)
        if count > best_count:
            best_count = count
            best_edges = list(edge_set)
            best_adj = list(adj_bits)
    
    # Phase 2: Structured initialization + local search
    if time.time() < deadline:
        edge_set, adj_bits = structured_init()
        count = len(edge_set)
        if count > best_count:
            best_count = count
            best_edges = list(edge_set)
            best_adj = list(adj_bits)
        
        # Local search on best
        if time.time() < deadline:
            new_edges, new_adj, new_count = local_search(edge_set, adj_bits, max_iterations=3000, seed=12345)
            if new_count > best_count:
                best_count = new_count
                best_edges = list(new_edges)
                best_adj = list(new_adj)
    
    # Phase 3: Local search on the best greedy result
    if best_edges and time.time() < deadline:
        # Reconstruct adj from best_edges
        adj_bits = [0] * N
        for u, v in best_edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        new_edges, new_adj, new_count = local_search(best_edges, adj_bits, max_iterations=5000, seed=99999)
        if new_count > best_count:
            best_count = new_count
            best_edges = list(new_edges)
    
    # Phase 4: More local search with different seeds
    if best_edges and time.time() < deadline:
        adj_bits = [0] * N
        for u, v in best_edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        for ls_seed in range(5):
            if time.time() >= deadline:
                break
            new_edges, new_adj, new_count = local_search(best_edges, adj_bits, max_iterations=3000, seed=1000+ls_seed)
            if new_count > best_count:
                best_count = new_count
                best_edges = list(new_edges)
                adj_bits = list(new_adj)
    
    return [[u, v] for u, v in best_edges]

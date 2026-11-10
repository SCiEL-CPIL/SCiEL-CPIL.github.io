import time
import random
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices maximizing edges.
    Uses greedy insertion with kicks and optimized C6 detection.
    """
    N = 40
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def can_add(adj_bits, u, v):
        """Check if adding edge (u,v) creates a C6.
        A C6 forms if there's a path u-a-b-c-d-v of length 5 (4 intermediate nodes).
        """
        neighbors_u = adj_bits[u] & ~(1 << v)
        neighbors_v = adj_bits[v] & ~(1 << u)
        
        if neighbors_u == 0 or neighbors_v == 0:
            return True
        
        # For each a in N(u)\{v}, for each d in N(v)\{u},
        # check if there exists b, c with a-b-c-d, all distinct from u,v,a,d
        # and b != d, c != a, etc.
        
        # Precompute: for each node x, the set of neighbors as a bitmask
        # For efficiency, iterate over bits
        
        nu = neighbors_u
        while nu:
            lsb_u = nu & (-nu)
            a = lsb_u.bit_length() - 1
            nu ^= lsb_u
            
            # For this a, find all b in N(a) \ {u, v, a}
            ba_mask = adj_bits[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
            
            if ba_mask == 0:
                continue
            
            # For each d in N(v)\{u}, check if there's a valid b-c-d path
            nv = neighbors_v & ~(1 << a)  # d != a
            while nv:
                lsb_v = nv & (-nv)
                d = lsb_v.bit_length() - 1
                nv ^= lsb_v
                
                # b must be in N(a) \ {u,v,a,d}
                ba = ba_mask & ~(1 << d)
                if ba == 0:
                    continue
                
                # For each b, check if N(b) \ {u,v,a,b,d} intersects N(d)
                bb = ba
                while bb:
                    lsb_b = bb & (-bb)
                    b = lsb_b.bit_length() - 1
                    bb ^= lsb_b
                    
                    # c in N(b) \ {u,v,a,b,d}
                    cb = adj_bits[b] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << d)
                    if cb & adj_bits[d]:
                        return False
        
        return True
    
    # Initialize with a denser starting structure
    # Start with 8 K5s (80 edges) and try to add inter-clique edges
    adj_bits = [0] * N
    
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
    
    def edge_count(bits):
        return sum(bin(b).count('1') for b in bits) // 2
    
    current_edges = edge_count(adj_bits)
    best_edges = current_edges
    best_adj_bits = adj_bits[:]
    
    # Build missing edges list
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    missing_edges = [p for p in all_pairs if not (adj_bits[p[0]] & (1 << p[1]))]
    
    # Sort missing edges by a heuristic: prefer edges between nodes in different cliques
    # and with lower degree (more likely to be addable)
    def edge_priority(u, v):
        clique_u = u // 5
        clique_v = v // 5
        inter_clique = 0 if clique_u != clique_v else 1
        deg_u = bin(adj_bits[u]).count('1')
        deg_v = bin(adj_bits[v]).count('1')
        return (inter_clique, deg_u + deg_v)
    
    # Track pass count for adaptive kick strategy
    pass_count = 0
    consecutive_no_add = 0
    
    while time.time() < deadline:
        # Sort missing edges by priority
        missing_edges.sort(key=lambda e: edge_priority(e[0], e[1]))
        
        added_in_pass = 0
        new_missing = []
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
                
            if can_add(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                added_in_pass += 1
                current_edges += 1
            else:
                new_missing.append((u, v))
                
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj_bits = adj_bits[:]
        
        missing_edges = new_missing
        pass_count += 1
        
        if added_in_pass == 0:
            consecutive_no_add += 1
        else:
            consecutive_no_add = 0
        
        # Kick strategy: increase kick size as we get more stuck
        if added_in_pass == 0 and current_edges > 10:
            # Adaptive kick size: larger kicks when stuck longer
            base_kick = 3 + min(consecutive_no_add, 5)
            kick_size = random.randint(base_kick, base_kick + 5)
            
            existing_edges = []
            for u in range(N):
                for v in range(u + 1, N):
                    if adj_bits[u] & (1 << v):
                        existing_edges.append((u, v))
            
            if not existing_edges:
                break
            
            # Prefer removing edges between high-degree nodes
            existing_edges.sort(key=lambda e: -(bin(adj_bits[e[0]]).count('1') + bin(adj_bits[e[1]]).count('1')))
            
            # Take a mix: some from top, some random
            kick_set = set()
            for i in range(min(kick_size, len(existing_edges))):
                if i < kick_size // 2 and i < len(existing_edges):
                    u, v = existing_edges[i]
                else:
                    idx = random.randint(0, len(existing_edges) - 1)
                    u, v = existing_edges[idx]
                if (u, v) not in kick_set:
                    kick_set.add((u, v))
            
            for (u, v) in kick_set:
                if adj_bits[u] & (1 << v):  # Still present
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    current_edges -= 1
                    missing_edges.append((u, v))
            
            # After kick, try to immediately re-add some edges
            for (u, v) in list(missing_edges):
                if time.time() > deadline:
                    break
                if can_add(adj_bits, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    current_edges += 1
                    missing_edges.remove((u, v))
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj_bits = adj_bits[:]
    
    # Final polish: try one more pass
    if time.time() < deadline - 5:
        missing_edges.sort(key=lambda e: edge_priority(e[0], e[1]))
        for u, v in missing_edges:
            if time.time() > deadline - 2:
                break
            if can_add(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                current_edges += 1
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj_bits = adj_bits[:]
    
    # Return best graph
    final_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj_bits[u] & (1 << v):
                final_edges.append([u, v])
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")

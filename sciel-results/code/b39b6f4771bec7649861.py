"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 15
    deadline = time.time() + 75  # safety margin before 90s kill
    
    # Use bitmask representation for fast operations
    # adj_masks[i] is a bitmask of neighbors of vertex i
    
    # Start with K_{7,8}
    adj_masks = [0] * n
    for i in range(7):
        for j in range(7, 15):
            adj_masks[i] |= (1 << j)
            adj_masks[j] |= (1 << i)
    
    def get_degrees():
        return [bin(m).count('1') for m in adj_masks]
    
    def check_valid():
        """Check if no two vertices of equal degree are connected by a 3-edge path."""
        degs = get_degrees()
        # For each pair (s,t) with same degree, check if 3-path exists
        for s in range(n):
            for t in range(s + 1, n):
                if degs[s] == degs[t]:
                    # 3-path s-u-v-t: u in N(s), v in N(t), (u,v) edge, u!=t, v!=s
                    # u must be neighbor of s (u != s always since no loops)
                    # v must be neighbor of t
                    # (u,v) must be an edge
                    # u != t, v != s
                    ns = adj_masks[s]
                    nt = adj_masks[t]
                    # u in N(s), u != t
                    # v in N(t), v != s
                    # adj[u] & (1 << v) != 0
                    u_mask = ns & ~(1 << t)  # neighbors of s, excluding t
                    if u_mask == 0:
                        continue
                    # For each u in u_mask, check if any v in N(t)\{s} is adjacent to u
                    # Equivalently: exists u in N(s)\{t} such that adj[u] & (nt & ~(1<<s)) != 0
                    v_mask = nt & ~(1 << s)  # neighbors of t, excluding s
                    if v_mask == 0:
                        continue
                    # Check if any u in u_mask has an edge to any v in v_mask
                    # This is: OR over u in u_mask of (adj[u] & v_mask) != 0
                    # We can iterate over bits in u_mask
                    um = u_mask
                    while um:
                        lb = um & (-um)
                        u = lb.bit_length() - 1
                        if adj_masks[u] & v_mask:
                            return False
                        um ^= lb
        return True
    
    def count_edges():
        return sum(bin(m).count('1') for m in adj_masks) // 2
    
    best_edges = count_edges()
    
    # Greedy: try adding intra-partition edges
    candidate_edges = []
    for i in range(7):
        for j in range(i + 1, 7):
            candidate_edges.append((i, j))
    for i in range(7, 15):
        for j in range(i + 1, 15):
            candidate_edges.append((i, j))
    
    random.seed(42)
    random.shuffle(candidate_edges)
    
    for (i, j) in candidate_edges:
        if time.time() > deadline:
            break
        if not (adj_masks[i] & (1 << j)):
            adj_masks[i] |= (1 << j)
            adj_masks[j] |= (1 << i)
            if check_valid():
                best_edges += 1
            else:
                adj_masks[i] &= ~(1 << j)
                adj_masks[j] &= ~(1 << i)
    
    # Local search: try removing one edge and adding another
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
    
    # Multiple passes of local search
    for _ in range(20):
        if time.time() > deadline:
            break
        improved = False
        for (i, j) in all_edges:
            if time.time() > deadline:
                break
            if adj_masks[i] & (1 << j):
                # Try removing (i,j)
                adj_masks[i] &= ~(1 << j)
                adj_masks[j] &= ~(1 << i)
                for (k, l) in all_edges:
                    if time.time() > deadline:
                        break
                    if not (adj_masks[k] & (1 << l)):
                        adj_masks[k] |= (1 << l)
                        adj_masks[l] |= (1 << k)
                        if check_valid():
                            best_edges += 1
                            improved = True
                            break
                        else:
                            adj_masks[k] &= ~(1 << l)
                            adj_masks[l] &= ~(1 << k)
                if not improved:
                    # Restore (i,j)
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
                else:
                    # Keep the swap, break out to restart
                    break
        if not improved:
            break
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_masks[i] & (1 << j):
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END

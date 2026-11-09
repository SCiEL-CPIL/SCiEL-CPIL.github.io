import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n_base = 29
    exponent = (n_base - 1) // 2
    
    # Precompute Paley graph adjacency for 29 vertices
    # adj[i][j] = 1 if (j-i) is QR mod 29, else 0
    # Note: (i,j) is symmetric.
    adj = [[0] * n_base for _ in range(n_base)]
    for i in range(n_base):
        for j in range(i + 1, n_base):
            diff = (j - i) % n_base
            res = pow(diff, exponent, n_base)
            if res == 1:
                adj[i][j] = 1
                adj[j][i] = 1
            else:
                adj[i][j] = 0
                adj[j][i] = 0

    # Function to count K2,2 in a graph induced by subset S
    # S is a bitmask of 29 bits
    def count_k22(mask, adj_mat):
        # Get list of vertices in mask
        verts = [i for i in range(n_base) if (mask >> i) & 1]
        if len(verts) < 4:
            return 0
        
        count = 0
        # For each pair in verts, count common neighbors in verts
        # common = { w in verts | adj[u][w] and adj[v][w] }
        # Add C(common, 2)
        for i_idx in range(len(verts)):
            u = verts[i_idx]
            for j_idx in range(i_idx + 1, len(verts)):
                v = verts[j_idx]
                common_count = 0
                for k_idx in range(len(verts)):
                    w = verts[k_idx]
                    if w != u and w != v:
                        if adj_mat[u][w] and adj_mat[v][w]:
                            common_count += 1
                count += common_count * (common_count - 1) // 2
        return count

    # We want to find S such that count_k22(S, adj) == 0 and count_k22(~S, 1-adj) == 0
    # Note: adj is 0/1. Complement adj_comp has 1 where adj has 0 (and i!=j).
    adj_comp = [[0] * n_base for _ in range(n_base)]
    for i in range(n_base):
        for j in range(n_base):
            if i != j:
                adj_comp[i][j] = 1 - adj[i][j]

    def get_cost(mask):
        # mask is int
        cost1 = count_k22(mask, adj)
        # ~mask needs to be masked to 29 bits
        mask_inv = ((1 << n_base) - 1) ^ mask
        cost2 = count_k22(mask_inv, adj_comp)
        return cost1 + cost2

    # Local Search
    # Start with random or specific
    # Try Q U {0}
    # Q = {1, 2, 4, 8, 16, 7, 14, 28, 27, 25, 21, 13, 9, 18}
    q_set = {1, 2, 4, 8, 16, 7, 14, 28, 27, 25, 21, 13, 9, 18}
    init_mask = 1 # vertex 0
    for v in q_set:
        init_mask |= (1 << v)
        
    best_mask = init_mask
    best_cost = get_cost(best_mask)
    
    if best_cost > 0:
        # Hill Climbing
        current_mask = init_mask
        current_cost = best_cost
        max_iter = 5000
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try flipping each bit
            # Shuffle order to break symmetry?
            bits = list(range(n_base))
            random.shuffle(bits)
            for b in bits:
                new_mask = current_mask ^ (1 << b)
                new_cost = get_cost(new_mask)
                if new_cost < current_cost:
                    current_mask = new_mask
                    current_cost = new_cost
                    improved = True
                    if current_cost == 0:
                        break
            if current_cost == 0:
                break
            if not improved:
                # Random restart or kick
                # Just pick a random mask
                current_mask = random.getrandbits(n_base)
                current_cost = get_cost(current_mask)
                
        if current_cost < best_cost:
            best_mask = current_mask
            best_cost = current_cost

    if best_cost == 0:
        n = 30
        colors = []
        # Edges 0..28
        for i in range(n_base):
            for j in range(i + 1, n_base):
                colors.append(adj[i][j])
        # Edges 29..i
        for i in range(n_base):
            if (best_mask >> i) & 1:
                colors.append(0)
            else:
                colors.append(1)
        return {'n': n, 'colors': colors}
    else:
        # Fallback to n=29
        n = 29
        colors = []
        for i in range(n_base):
            for j in range(i + 1, n_base):
                colors.append(adj[i][j])
        return {'n': n, 'colors': colors}

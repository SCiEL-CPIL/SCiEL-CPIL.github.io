"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n = 20
    
    # Precompute popcount for 20-bit masks using a lookup table
    # 2^20 = 1048576 entries, each a small int - fits in memory
    POPCOUNT = [0] * (1 << n)
    for i in range(1, 1 << n):
        POPCOUNT[i] = POPCOUNT[i >> 1] + (i & 1)
    
    def popcount(x):
        return POPCOUNT[x]
    
    def can_add(adj, u, v, common):
        """Check if adding edge (u,v) keeps graph K2,3-free.
        common[i][j] = popcount(adj[i] & adj[j]) for i != j
        """
        if (adj[u] >> v) & 1:
            return False
        
        # 1. Pair (u,v): common neighbors must be <= 2
        if common[u][v] > 2:
            return False
        
        # 2. For each w in N(v), w != u: pair (u,w) gains v as common neighbor
        #    Need common[u][w] <= 1 (since v will be added to N(u) ∩ N(w))
        nv_mask = adj[v]
        # Iterate through neighbors of v
        mask = nv_mask
        while mask:
            lsb = mask & (-mask)
            w = lsb.bit_length() - 1
            mask ^= lsb
            if w != u:
                if common[u][w] > 1:
                    return False
        
        # 3. For each w in N(u), w != v: pair (v,w) gains u as common neighbor
        #    Need common[v][w] <= 1
        nu_mask = adj[u]
        mask = nu_mask
        while mask:
            lsb = mask & (-mask)
            w = lsb.bit_length() - 1
            mask ^= lsb
            if w != v:
                if common[v][w] > 1:
                    return False
        
        return True
    
    def add_edge(adj, u, v, common):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        # Update common neighbor counts
        # For each w in N(u) \ {v}: common[v][w] += 1 (v and w now share u)
        # For each w in N(v) \ {u}: common[u][w] += 1 (u and w now share v)
        # common[u][v] unchanged (was popcount(N(u)&N(v)), still same after adding)
        # Actually common[u][v] = popcount(N(u) & N(v)) which doesn't change when we add (u,v)
        # because u is not in N(u) and v is not in N(v)
        
        nu_mask = adj[u] & ~(1 << v)  # N(u) without v
        mask = nu_mask
        while mask:
            lsb = mask & (-mask)
            w = lsb.bit_length() - 1
            mask ^= lsb
            common[v][w] += 1
            common[w][v] += 1
        
        nv_mask = adj[v] & ~(1 << u)  # N(v) without u
        mask = nv_mask
        while mask:
            lsb = mask & (-mask)
            w = lsb.bit_length() - 1
            mask ^= lsb
            common[u][w] += 1
            common[w][u] += 1
    
    def remove_edge(adj, u, v, common):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        # Reverse the updates
        nu_mask = adj[u]  # N(u) after removal (doesn't include v)
        mask = nu_mask
        while mask:
            lsb = mask & (-mask)
            w = lsb.bit_length() - 1
            mask ^= lsb
            common[v][w] -= 1
            common[w][v] -= 1
        
        nv_mask = adj[v]  # N(v) after removal (doesn't include u)
        mask = nv_mask
        while mask:
            lsb = mask & (-mask)
            w = lsb.bit_length() - 1
            mask ^= lsb
            common[u][w] -= 1
            common[w][u] -= 1
    
    def build_common(adj):
        """Build the common neighbor count matrix."""
        common = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = POPCOUNT[adj[i] & adj[j]]
                common[i][j] = c
                common[j][i] = c
        return common
    
    def get_edge_count(adj):
        total = 0
        for i in range(n):
            total += POPCOUNT[adj[i]]
        return total // 2
    
    def is_valid(common):
        for i in range(n):
            for j in range(i + 1, n):
                if common[i][j] > 2:
                    return False
        return True
    
    def make_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    # --- Structured initial construction using quadratic residues mod 19 ---
    # Paley graph on 19 vertices: edge (i,j) iff (i-j) mod 19 is a quadratic residue
    # Then add vertex 20 connected strategically
    def paley_initial():
        """Create a Paley-type graph on 19 vertices + 1 extra vertex."""
        adj = [0] * n
        # Quadratic residues mod 19
        residues = set()
        for x in range(1, 19):
            residues.add((x * x) % 19)
        # Build Paley graph on vertices 0..18
        for i in range(19):
            for j in range(i + 1, 19):
                diff = (i - j) % 19
                if diff in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Vertex 19: connect to some vertices
        # Connect vertex 19 to vertices 0,1,2,3 (4 edges) - will be optimized
        for j in range(4):
            adj[19] |= (1 << j)
            adj[j] |= (1 << 19)
        return adj
    
    # --- Greedy edge addition from sparse start ---
    def greedy_initial():
        """Start empty and greedily add edges."""
        adj = [0] * n
        common = build_common(adj)
        # Try all pairs in order, add if feasible
        for i in range(n):
            for j in range(i + 1, n):
                if can_add(adj, i, j, common):
                    add_edge(adj, i, j, common)
        return adj
    
    # --- Random initial ---
    def random_initial(density=0.3):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Fix violations
        common = build_common(adj)
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if common[i][j] > 2 and ((adj[i] >> j) & 1):
                        remove_edge(adj, i, j, common)
                        changed = True
        return adj
    
    # Try multiple initializations and keep the best
    best_adj = None
    best_edges = -1
    best_common = None
    
    initializations = [
        paley_initial(),
        greedy_initial(),
        random_initial(0.3),
        random_initial(0.4),
    ]
    
    for adj in initializations:
        common = build_common(adj)
        if is_valid(common):
            ec = get_edge_count(adj)
            if ec > best_edges:
                best_edges = ec
                best_adj = adj[:]
                best_common = common
    
    if best_adj is None:
        # Fallback: empty graph
        best_adj = [0] * n
        best_common = build_common(best_adj)
        best_edges = 0
    
    current_adj = best_adj[:]
    current_common = best_common
    current_edges = best_edges
    
    # --- Local search: hill climbing with random restarts ---
    def local_search(adj, common, edges, deadline):
        """Hill climbing: try to add edges, then do swaps."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each possible edge
            for i in range(n):
                for j in range(i + 1, n):
                    if not ((adj[i] >> j) & 1):
                        if can_add(adj, i, j, common):
                            add_edge(adj, i, j, common)
                            edges += 1
                            improved = True
            # If no edges can be added, try swaps
            if not improved:
                # Try removing one edge and adding another
                existing = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            existing.append((i, j))
                
                for rem_i, rem_j in existing:
                    if time.time() >= deadline:
                        break
                    # Temporarily remove
                    remove_edge(adj, rem_i, rem_j, common)
                    added_any = False
                    for add_i in range(n):
                        for add_j in range(add_i + 1, n):
                            if (add_i, add_j) == (rem_i, rem_j) or (add_i, add_j) == (rem_j, rem_i):
                                continue
                            if not ((adj[add_i] >> add_j) & 1):
                                if can_add(adj, add_i, add_j, common):
                                    add_edge(adj, add_i, add_j, common)
                                    added_any = True
                                    break
                        if added_any:
                            break
                    if not added_any:
                        # Revert
                        add_edge(adj, rem_i, rem_j, common)
                    else:
                        # Keep the swap (net zero edge count change)
                        # But this might enable future additions
                        # Re-run the add phase
                        improved = True
                        break
        
        return adj, common, edges
    
    # Phase 1: Hill climbing
    current_adj, current_common, current_edges = local_search(
        current_adj, current_common, current_edges, deadline - 30
    )
    
    if current_edges > best_edges:
        best_edges = current_edges
        best_adj = current_adj[:]
        best_common = current_common
    
    # Phase 2: Simulated annealing / random restarts for more exploration
    T = 5.0
    cooling = 0.9995
    min_T = 0.05
    
    while time.time() < deadline:
        # Random perturbation: remove 1-3 random edges
        perturb_count = random.randint(1, 3)
        adj_pert = current_adj[:]
        common_pert = current_common
        edges_pert = current_edges
        
        # Collect existing edges
        existing = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_pert[i] >> j) & 1:
                    existing.append((i, j))
        
        for _ in range(perturb_count):
            if not existing:
                break
            idx = random.randint(0, len(existing) - 1)
            u, v = existing[idx]
            remove_edge(adj_pert, u, v, common_pert)
            existing.pop(idx)
            edges_pert -= 1
        
        # Try to recover by adding edges greedily
        for i in range(n):
            for j in range(i + 1, n):
                if not ((adj_pert[i] >> j) & 1):
                    if can_add(adj_pert, i, j, common_pert):
                        add_edge(adj_pert, i, j, common_pert)
                        edges_pert += 1
        
        # Accept if better or with probability exp(-delta/T)
        delta = current_edges - edges_pert
        if edges_pert >= current_edges or random.random() < min(1.0, 2.718281828 ** (-max(0, delta) / T)):
            current_adj = adj_pert
            current_common = common_pert
            current_edges = edges_pert
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = current_adj[:]
            best_common = current_common
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Final polish: exhaustive edge addition
    final_adj = best_adj[:]
    final_common = best_common
    final_edges = best_edges
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if not ((final_adj[i] >> j) & 1):
                    if can_add(final_adj, i, j, final_common):
                        add_edge(final_adj, i, j, final_common)
                        final_edges += 1
                        improved = True
    
    if final_edges > best_edges:
        best_edges = final_edges
        best_adj = final_adj[:]
        best_common = final_common
    
    return make_edges(best_adj)
# EVOLVE-BLOCK-END

import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit
    
    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj_masks = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
    
    def has_k4_in_subset(adj, subset_mask):
        """Check if induced subgraph on subset_mask contains a K4."""
        s_list = []
        temp = subset_mask
        while temp:
            lsb = temp & -temp
            idx = (lsb.bit_length() - 1)
            s_list.append(idx)
            temp ^= lsb
        m = len(s_list)
        if m < 4:
            return False
        for i in range(m):
            u = s_list[i]
            for j in range(i+1, m):
                v = s_list[j]
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v] & subset_mask & ~(1 << u) & ~(1 << v)
                    if common == 0:
                        continue
                    c_list = []
                    temp_c = common
                    while temp_c:
                        lsb = temp_c & -temp_c
                        idx = (lsb.bit_length() - 1)
                        c_list.append(idx)
                        temp_c ^= lsb
                    for a in c_list:
                        if adj[a] & common & ~(1 << a):
                            return True
        return False

    def has_i4_in_subset(adj, subset_mask):
        """Check if induced subgraph on subset_mask contains an independent set of size 4."""
        s_list = []
        temp = subset_mask
        while temp:
            lsb = temp & -temp
            idx = (lsb.bit_length() - 1)
            s_list.append(idx)
            temp ^= lsb
        m = len(s_list)
        if m < 4:
            return False
        for i in range(m):
            u = s_list[i]
            for j in range(i+1, m):
                v = s_list[j]
                if not (adj[u] & (1 << v)):
                    cand_mask = subset_mask & ~(adj[u] | adj[v] | (1 << u) | (1 << v))
                    if cand_mask == 0:
                        continue
                    c_list = []
                    temp_c = cand_mask
                    while temp_c:
                        lsb = temp_c & -temp_c
                        idx = (lsb.bit_length() - 1)
                        c_list.append(idx)
                        temp_c ^= lsb
                    for a in c_list:
                        for b in c_list:
                            if a != b and not (adj[a] & (1 << b)):
                                return True
        return False

    def has_k5(adj, n):
        """Check if graph has a K5."""
        for u in range(n):
            neighbors_mask = adj[u]
            if has_k4_in_subset(adj, neighbors_mask):
                return True
        return False

    def has_i5(adj, n):
        """Check if graph has an independent set of size 5."""
        all_bits = (1 << n) - 1
        for u in range(n):
            non_neighbors = all_bits & ~(adj[u] | (1 << u))
            if has_i4_in_subset(adj, non_neighbors):
                return True
        return False

    def is_valid(adj, n):
        if has_k5(adj, n):
            return False
        if has_i5(adj, n):
            return False
        return True

    # Verify seed
    if not is_valid(adj_masks, n):
        return n, []

    best_n = n
    best_adj = adj_masks[:]

    # 2. Try to extend using multiple strategies
    current_n = n
    current_adj = adj_masks[:]
    
    # Strategy: Try adding vertices with various neighbor set constructions
    # Key insight: For a new vertex v, N(v) must be K4-free and co-N(v) must be I4-free
    # We can try to find such sets by starting from existing vertex neighborhoods
    
    attempts_per_target = 200
    
    for target_n in range(current_n + 1, 43):
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(attempts_per_target):
            if time.time() > deadline:
                break
                
            # Strategy: Pick a random existing vertex and use its neighborhood as base
            # Then modify it slightly
            base_vertex = random.randint(0, current_n - 1)
            base_neighbors_mask = current_adj[base_vertex]
            
            # Convert to list
            base_neighbors = []
            temp = base_neighbors_mask
            while temp:
                lsb = temp & -temp
                idx = (lsb.bit_length() - 1)
                base_neighbors.append(idx)
                temp ^= lsb
            
            # Modify the neighbor set: add/remove some vertices
            # Try different modification strategies
            strategy = random.choice(['add', 'remove', 'swap', 'random'])
            
            if strategy == 'add':
                # Add 1-3 random non-neighbors
                non_neighbors = [v for v in range(current_n) if not (base_neighbors_mask & (1 << v))]
                num_add = random.randint(0, min(3, len(non_neighbors)))
                to_add = random.sample(non_neighbors, num_add) if non_neighbors else []
                neighbors_set = set(base_neighbors) | set(to_add)
            elif strategy == 'remove':
                # Remove 1-3 random neighbors
                num_remove = random.randint(0, min(3, len(base_neighbors)))
                to_remove = set(random.sample(base_neighbors, num_remove)) if base_neighbors else set()
                neighbors_set = set(base_neighbors) - to_remove
            elif strategy == 'swap':
                # Swap some neighbors with non-neighbors
                non_neighbors = [v for v in range(current_n) if not (base_neighbors_mask & (1 << v))]
                num_swap = random.randint(0, min(2, len(base_neighbors), len(non_neighbors)))
                to_remove = random.sample(base_neighbors, num_swap) if base_neighbors else []
                to_add = random.sample(non_neighbors, num_swap) if non_neighbors else []
                neighbors_set = set(base_neighbors) - set(to_remove) | set(to_add)
            else:
                # Pure random with bias toward balanced degrees
                size = random.randint(current_n // 3, (2 * current_n) // 3)
                neighbors_set = set(random.sample(range(current_n), size))
            
            # Check validity
            neighbors_mask = 0
            for v in neighbors_set:
                neighbors_mask |= (1 << v)
            
            # Condition 1: No K4 in neighbors
            if has_k4_in_subset(current_adj, neighbors_mask):
                continue
            
            # Condition 2: No I4 in non-neighbors
            all_mask = (1 << current_n) - 1
            non_neighbors_mask = all_mask & ~neighbors_mask
            if has_i4_in_subset(current_adj, non_neighbors_mask):
                continue
            
            # Valid! Add the vertex
            found = True
            new_n = current_n + 1
            new_adj = current_adj + [0]
            for v in neighbors_set:
                new_adj[v] |= (1 << new_n)
                new_adj[new_n] |= (1 << v)
            
            current_n = new_n
            current_adj = new_adj
            best_n = current_n
            best_adj = current_adj[:]
            break

        if not found:
            break

    # Reconstruct edges from best_adj
    final_edges = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            if best_adj[i] & (1 << j):
                final_edges.append((i, j))
                
    return best_n, final_edges
# EVOLVE-BLOCK-END

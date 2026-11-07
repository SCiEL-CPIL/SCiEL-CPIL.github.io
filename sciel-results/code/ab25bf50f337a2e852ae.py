"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    values = list(range(2, N + 1))
    n_nodes = len(values)
    
    # 1. Build Adjacency Masks
    # adj[i] is a bitmask of nodes conflicting with node i
    adj = [0] * n_nodes
    for i in range(n_nodes):
        v = values[i]
        mask = 0
        # Optimization: only check j > i and set bits for both
        for j in range(i + 1, n_nodes):
            w = values[j]
            if math.gcd(v, w) > 1:
                mask |= (1 << j)
                adj[j] |= (1 << i)
        adj[i] |= mask
        
    # 2. Identify and Remove Isolated Nodes
    # A node is isolated if adj[i] == 0
    base_sum = 1 # Always include 1
    active_mask = (1 << n_nodes) - 1
    isolated_mask = 0
    for i in range(n_nodes):
        if adj[i] == 0:
            base_sum += values[i]
            isolated_mask |= (1 << i)
    
    active_mask &= ~isolated_mask
    
    # 3. Remove Dominated Nodes
    # If N(u) is a superset of N(v) (in active nodes) and w(u) <= w(v), remove u.
    # Note: N(u) should be considered within the active set.
    # Let's iterate and mark nodes to remove.
    
    # To make this efficient, we can sort by weight and check.
    # But N=255 is small.
    
    removed_mask = 0
    # We need to be careful: if u dominates v, and v dominates w, etc.
    # Simple pass:
    # For each pair (u, v), if u is dominated by v, mark u.
    
    # Optimization: Only check if w(u) < w(v) or (w(u)==w(v) and some tie break)
    # Actually, if w(u) == w(v) and N(u) == N(v), one can be removed.
    
    # Let's just do a simple check.
    # To avoid O(N^2) with heavy bit operations, we can group by weight?
    # N is small enough for O(N^2) bit checks.
    
    dominated = [False] * n_nodes
    
    # Sort indices by weight descending to find dominators easily?
    # If we process in descending weight, when we see u, all v with w(v) >= w(u) have been seen.
    # If any such v has N(v) subset of N(u), then u is dominated.
    
    indices_by_weight = sorted(range(n_nodes), key=lambda i: values[i], reverse=True)
    
    # We only care about active nodes
    active_indices = [i for i in range(n_nodes) if (active_mask >> i) & 1]
    
    # Re-sort active indices
    active_indices.sort(key=lambda i: values[i], reverse=True)
    
    for i_idx, u in enumerate(active_indices):
        if dominated[u]:
            continue
            
        # Check against all v processed so far (which have weight >= w(u))
        # v is in active_indices[0 : i_idx]
        for j_idx in range(i_idx):
            v = active_indices[j_idx]
            if dominated[v]:
                continue
            
            # Check if N(u) is a superset of N(v)
            # N(u) in active set: adj[u] & active_mask
            # N(v) in active set: adj[v] & active_mask
            
            # If (adj[u] & active_mask) | (1<<u) ? No, just neighbors.
            # Condition: (adj[v] & active_mask) & ~(adj[u] & active_mask) == 0
            # i.e., N(v) \subseteq N(u)
            
            # Also, we must ensure u and v are not neighbors?
            # If they are neighbors, we can't pick both.
            # The domination logic holds for independent sets:
            # If we pick v, we block N(v). If we pick u, we block N(u).
            # If N(v) \subseteq N(u), then picking v leaves a superset of available nodes compared to picking u.
            # And w(v) >= w(u). So v is strictly better.
            
            # One caveat: if u and v are connected, does it matter?
            # If we pick v, u is blocked (since u in N(v)? No, u is not in N(v) unless they are neighbors).
            # If they are neighbors, u \in N(v) and v \in N(u).
            # Then N(v) includes u. N(u) includes v.
            # If N(v) \subseteq N(u), then u \in N(u) (false) or u \in N(u) via other path?
            # Bitsets don't include self.
            # If u and v are neighbors:
            # N(v) contains u.
            # N(u) contains v.
            # For N(v) \subseteq N(u), we need u \in N(u), which is false.
            # So neighbors cannot dominate each other in this strict sense unless we adjust definitions.
            # Actually, if u and v are neighbors, we can't pick both.
            # The comparison is: "Is it ever better to pick u than v?"
            # If we pick v, we gain w(v) and lose N(v).
            # If we pick u, we gain w(u) and lose N(u).
            # If N(v) \setminus \{u\} \subseteq N(u) \setminus \{v\} and w(v) >= w(u), then v is better.
            
            # Let's use the safer condition:
            # (adj[v] & active_mask) & ~(adj[u] | (1<<u)) == 0 ?
            # This checks if all neighbors of v (except u) are neighbors of u.
            
            neighbors_v = adj[v] & active_mask
            neighbors_u = adj[u] & active_mask
            
            # If v is a neighbor of u, u is in neighbors_v. We exclude u from the check.
            # If u is a neighbor of v, v is in neighbors_u.
            
            # Check: Are all neighbors of v (that are not u) also neighbors of u?
            # mask_to_check = neighbors_v & ~(1 << u)
            # if mask_to_check & ~neighbors_u == 0:
            
            mask_to_check = neighbors_v & ~(1 << u)
            if mask_to_check & ~neighbors_u == 0:
                # v dominates u
                dominated[u] = True
                break
                
    # Apply removal
    for i in range(n_nodes):
        if dominated[i]:
            active_mask &= ~(1 << i)
            
    # 4. Branch and Bound on Active Nodes
    # Re-map active nodes to a smaller bitset?
    # Or just use the original bitset with active_mask constraint.
    
    # Get list of active node indices
    active_node_indices = [i for i in range(n_nodes) if (active_mask >> i) & 1]
    
    # Precompute sums for bounds?
    # We will use a simple recursive B&B.
    
    best_sum = 0
    
    # To speed up, order nodes by weight descending in the recursion?
    # We can just pick the max weight available node.
    
    def bnb(avail_mask, current_sum):
        nonlocal best_sum
        
        if time.time() > deadline:
            return
            
        if avail_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
            return
            
        # Bound
        # Calculate sum of values in avail_mask
        # Optimization: if we precomputed prefix sums? No, mask is arbitrary.
        # Just loop.
        temp = avail_mask
        s = 0
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            s += values[idx]
            temp ^= lsb
            
        if current_sum + s <= best_sum:
            return
            
        # Pick node with max value
        max_val = -1
        max_idx = -1
        temp = avail_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            if values[idx] > max_val:
                max_val = values[idx]
                max_idx = idx
            temp ^= lsb
            
        if max_idx == -1:
            return
            
        # Branch Include
        # New avail: remove max_idx and its neighbors
        new_mask_inc = avail_mask & ~(1 << max_idx) & ~adj[max_idx]
        bnb(new_mask_inc, current_sum + values[max_idx])
        
        # Branch Exclude
        new_mask_exc = avail_mask & ~(1 << max_idx)
        bnb(new_mask_exc, current_sum)

    bnb(active_mask, 0)
    
    return base_sum + best_sum

# EVOLVE-BLOCK-END

import time
from math import gcd
from fractions import Fraction

def construct():
    """
    Find a set of distinct pairwise coprime integers in 1..511 that maximizes
    the sum of 1/(512 - a) using a graph-based branch-and-bound algorithm
    with dynamic node selection, Moon-Moser clique cover bound, and
    structural pruning.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    N = 512
    
    # Get all primes < 512
    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute prime factor masks for all candidates 1..511
    def get_factor_mask(n):
        mask = 0
        temp = n
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << i)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
        return mask
    
    factor_masks = [0] * N
    for a in range(1, N):
        factor_masks[a] = get_factor_mask(a)
    
    # Precompute scores
    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)
    
    # Sort nodes by score descending for better pruning
    nodes = list(range(1, N))
    nodes.sort(key=lambda x: -scores[x])
    n = len(nodes)
    
    # Map original value to sorted index
    val_to_idx = {nodes[i]: i for i in range(n)}
    
    # Build adjacency: adj[i] is a bitmask of nodes that conflict with node i
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (factor_masks[nodes[i]] & factor_masks[nodes[j]]) != 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Build SPF-based groups for Moon-Moser clique cover bound
    spf_groups = {}  # spf -> list of indices
    node_spf = [0] * n
    for i in range(n):
        val = nodes[i]
        if val == 1:
            spf = -1
        else:
            temp = val
            spf = temp
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    spf = p
                    break
            if spf == temp and temp > 1:
                spf = temp
        node_spf[i] = spf
        if spf not in spf_groups:
            spf_groups[spf] = []
        spf_groups[spf].append(i)
    
    # For each SPF group, precompute sorted list by weight descending
    # and map from index to position in sorted order
    spf_group_sorted = {}  # spf -> list of indices sorted by weight desc
    spf_group_pos = {}  # (spf, idx) -> position in sorted list
    spf_group_count = {}  # spf -> number of nodes in group
    for spf, group in spf_groups.items():
        sorted_group = sorted(group, key=lambda i: -scores[nodes[i]])
        spf_group_sorted[spf] = sorted_group
        spf_group_count[spf] = len(sorted_group)
        for pos, idx in enumerate(sorted_group):
            spf_group_pos[(spf, idx)] = pos
    
    # Precompute the list of SPF values for fast iteration
    spf_list = list(spf_groups.keys())
    num_spf_groups = len(spf_list)
    
    # Greedy initialization
    def greedy_init():
        chosen = []
        chosen_mask = 0
        for i in range(n):
            if (adj[i] & chosen_mask) == 0:
                chosen.append(nodes[i])
                # Add conflicts of i to chosen_mask
                chosen_mask |= adj[i]
        return chosen, sum(scores[v] for v in chosen)
    
    greedy_sol, greedy_weight = greedy_init()
    best_weight_val = greedy_weight
    best_solution_list = greedy_sol[:]
    
    # Incremental bound tracking
    # For each SPF group, track the current max weight among available nodes
    # and the count of available nodes
    # We'll maintain these incrementally as we recurse
    
    # Initial state: all nodes available
    group_max_w = [0.0] * num_spf_groups
    group_max_idx = [-1] * num_spf_groups
    group_avail_count = [0] * num_spf_groups
    current_bound = 0.0
    
    for si, spf in enumerate(spf_list):
        sorted_grp = spf_group_sorted[spf]
        if sorted_grp:
            top_idx = sorted_grp[0]
            group_max_w[si] = scores[nodes[top_idx]]
            group_max_idx[si] = top_idx
            group_avail_count[si] = len(sorted_grp)
            current_bound += scores[nodes[top_idx]]
    
    # Map spf value to its index in spf_list
    spf_to_si = {spf: i for i, spf in enumerate(spf_list)}
    
    import sys
    sys.setrecursionlimit(100000)
    
    # Precompute: for each node, which SPF group it belongs to (as si index)
    node_spf_si = [0] * n
    for i in range(n):
        node_spf_si[i] = spf_to_si[node_spf[i]]
    
    # Precompute: for each node, the set of nodes it conflicts with (as a list for fast iteration)
    # Actually, we use adj bitmask. Let's precompute neighbor lists for the "include" branch
    # neighbors[i] = list of indices that conflict with i
    neighbors = []
    for i in range(n):
        nbrs = []
        m = adj[i]
        j = 0
        while m:
            if m & 1:
                nbrs.append(j)
            m >>= 1
            j += 1
        neighbors.append(nbrs)
    
    # For the "exclude" branch, we just remove one node.
    # For the "include" branch, we remove the node and all its neighbors.
    # We need to update group_max_w incrementally.
    
    # To efficiently find the new max in a group after removing nodes,
    # we precompute for each group a "next candidate" pointer.
    # When the current max is removed, we advance to the next available.
    
    # For each group, maintain a pointer to the current max position in sorted order
    group_ptr = [0] * num_spf_groups  # position in sorted_group_sorted[spf] of current max
    
    # We also need to track which nodes are available per group.
    # Since we're doing DFS, we need to be able to undo removals.
    # Use a stack of changes.
    
    # Alternative: pass the available mask and recompute bound only when needed,
    # but do it efficiently by scanning only affected groups.
    
    # Let's use a different approach: maintain available_mask and compute bound
    # by only looking at groups that might have changed.
    
    # Actually, the simplest efficient approach: 
    # For each group, precompute the sorted list. To find max available in a group,
    # scan from the front until we find an available node. This is O(group_size) worst case
    # but typically fast since high-weight nodes are near the front.
    
    # But we can do better: maintain for each group a pointer that only moves forward
    # (since we only remove nodes, never add them back in a single branch).
    # In DFS, we need to backtrack, so we save/restore pointers.
    
    # Let's implement with explicit save/restore of group state.
    
    def branch_and_bound(available_mask, current_weight, chosen_indices):
        nonlocal best_weight_val, best_solution_list
        nonlocal group_max_w, group_max_idx, group_avail_count, current_bound, group_ptr
        
        if time.time() > deadline:
            return
        
        # Update best if current solution is better
        if current_weight > best_weight_val + 1e-12:
            best_weight_val = current_weight
            best_solution_list = [nodes[i] for i in chosen_indices]
        
        if available_mask == 0:
            return
        
        # Compute upper bound using current group state
        ub = current_bound
        if current_weight + ub <= best_weight_val + 1e-12:
            return
        
        # Find the highest-weight available node
        # Scan through nodes in sorted order (by weight desc) to find first available
        # Since nodes are sorted by weight desc, the first available is the best
        v = -1
        for i in range(n):
            if (available_mask >> i) & 1:
                v = i
                break
        
        if v == -1:
            return
        
        # Save state for backtracking
        saved_group_max_w = group_max_w[:]
        saved_group_max_idx = group_max_idx[:]
        saved_group_avail_count = group_avail_count[:]
        saved_current_bound = current_bound
        saved_group_ptr = group_ptr[:]
        
        # Branch 1: Include v
        # Remove v and all its neighbors from available
        removed_nodes = [v] + neighbors[v]
        new_available = available_mask
        for r in removed_nodes:
            new_available &= ~(1 << r)
        
        # Update group state for removed nodes
        for r in removed_nodes:
            si = node_spf_si[r]
            spf = spf_list[si]
            sorted_grp = spf_group_sorted[spf]
            
            # Check if r was the current max
            if r == group_max_idx[si]:
                # Advance pointer to find next available in this group
                ptr = group_ptr[si] + 1
                found = False
                while ptr < len(sorted_grp):
                    cand = sorted_grp[ptr]
                    if (new_available >> cand) & 1:
                        group_max_w[si] = scores[nodes[cand]]
                        group_max_idx[si] = cand
                        group_ptr[si] = ptr
                        found = True
                        break
                    ptr += 1
                if not found:
                    group_max_w[si] = 0.0
                    group_max_idx[si] = -1
                    group_ptr[si] = len(sorted_grp)
                group_avail_count[si] -= 1
            else:
                # r is not the max, just decrement count
                group_avail_count[si] -= 1
            
            # Update current_bound
            # We need to recompute the contribution of group si
            # Old contribution was group_max_w[si] before update
            # Actually, we already updated group_max_w[si], so:
            # The change in bound = new_max - old_max
            # But we already modified group_max_w[si] above.
            # Let's track old max before modification.
        
        # The above approach has a bug: we modify group_max_w before computing the delta.
        # Let me redo this more carefully.
        
        # Restore state and redo
        group_max_w = saved_group_max_w[:]
        group_max_idx = saved_group_max_idx[:]
        group_avail_count = saved_group_avail_count[:]
        current_bound = saved_current_bound
        group_ptr = saved_group_ptr[:]
        
        # Process removals carefully
        old_bound = current_bound
        for r in removed_nodes:
            si = node_spf_si[r]
            spf = spf_list[si]
            sorted_grp = spf_group_sorted[spf]
            old_max = group_max_w[si]
            
            if r == group_max_idx[si]:
                # Advance to next available
                ptr = group_ptr[si] + 1
                found = False
                while ptr < len(sorted_grp):
                    cand = sorted_grp[ptr]
                    if (new_available >> cand) & 1:
                        group_max_w[si] = scores[nodes[cand]]
                        group_max_idx[si] = cand
                        group_ptr[si] = ptr
                        found = True
                        break
                    ptr += 1
                if not found:
                    group_max_w[si] = 0.0
                    group_max_idx[si] = -1
                    group_ptr[si] = len(sorted_grp)
                group_avail_count[si] -= 1
            else:
                group_avail_count[si] -= 1
            
            # Update current_bound: subtract old max, add new max
            current_bound -= old_max
            current_bound += group_max_w[si]
        
        branch_and_bound(new_available, current_weight + scores[nodes[v]], chosen_indices + [v])
        
        # Restore state for branch 2
        group_max_w = saved_group_max_w[:]
        group_max_idx = saved_group_max_idx[:]
        group_avail_count = saved_group_avail_count[:]
        current_bound = saved_current_bound
        group_ptr = saved_group_ptr[:]
        
        # Branch 2: Exclude v
        # Just remove v from available
        new_available = available_mask & ~(1 << v)
        
        # Update group state for v
        si = node_spf_si[v]
        spf = spf_list[si]
        sorted_grp = spf_group_sorted[spf]
        old_max = group_max_w[si]
        
        if v == group_max_idx[si]:
            ptr = group_ptr[si] + 1
            found = False
            while ptr < len(sorted_grp):
                cand = sorted_grp[ptr]
                if (new_available >> cand) & 1:
                    group_max_w[si] = scores[nodes[cand]]
                    group_max_idx[si] = cand
                    group_ptr[si] = ptr
                    found = True
                    break
                ptr += 1
            if not found:
                group_max_w[si] = 0.0
                group_max_idx[si] = -1
                group_ptr[si] = len(sorted_grp)
            group_avail_count[si] -= 1
        else:
            group_avail_count[si] -= 1
        
        current_bound -= old_max
        current_bound += group_max_w[si]
        
        branch_and_bound(new_available, current_weight, chosen_indices)
    
    # Initial available mask: all nodes
    initial_mask = (1 << n) - 1
    branch_and_bound(initial_mask, 0.0, [])
    
    # Final local search to polish
    if time.time() < deadline - 5:
        chosen_set = set(best_solution_list)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for v in list(best_solution_list):
                # Try removing v and adding a better node
                new_mask = 0
                for u in best_solution_list:
                    if u != v:
                        new_mask |= factor_masks[u]
                
                best_b = -1
                best_gain = 0.0
                for b in range(1, N):
                    if b in chosen_set:
                        continue
                    if (factor_masks[b] & new_mask) == 0:
                        gain = scores[b] - scores[v]
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b
                
                if best_b != -1 and best_gain > 1e-12:
                    best_solution_list.remove(v)
                    best_solution_list.append(best_b)
                    chosen_set.discard(v)
                    chosen_set.add(best_b)
                    improved = True
        
        final_weight = sum(scores[v] for v in best_solution_list)
        if final_weight > best_weight_val:
            best_weight_val = final_weight
    
    return best_solution_list

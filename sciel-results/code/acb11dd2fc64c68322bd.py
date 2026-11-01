import time
import random
import itertools

def construct(n):
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    if n <= 1:
        return "1"
    
    # Generate all permutations
    symbols = list(range(1, n + 1))
    perms = list(itertools.permutations(symbols))
    perm_set = set(perms)
    # Map permutation to index for quick lookup?
    # We need to work with cycles.
    
    # Build the Permutation Graph cycles
    # Node: tuple of length n
    # Edge: u -> v if u[1:] == v[:-1]
    # Actually, v is determined by u[1:] and the last element.
    # v = u[1:] + (x,) where x is the missing element.
    
    visited = set()
    cycles = []
    
    for p in perms:
        if p in visited:
            continue
        cycle = []
        curr = p
        while curr not in visited:
            visited.add(curr)
            cycle.append(curr)
            # Find next node
            # next node must start with curr[1:]
            # The last element is the one not in curr[1:]
            # curr is a permutation, so missing element is unique
            missing = [x for x in symbols if x not in set(curr[1:])]
            # Actually, set(curr[1:]) has n-1 elements.
            # The missing one is unique.
            # But wait, curr[1:] is a sequence.
            # v = curr[1:] + (missing,)
            # Is this the ONLY next node?
            # Yes, because v[:-1] must equal curr[1:].
            # So v is uniquely determined.
            v = curr[1:] + tuple(missing)
            curr = v
        cycles.append(cycle)
        
    # We have n-1! cycles?
    # For n=3, 2 cycles.
    # For n=7, 720 cycles.
    
    num_cycles = len(cycles)
    
    # Precompute jump costs between cycles
    # C[i][j] = min cost to jump from any node in cycles[i] to any node in cycles[j]
    # Cost = n - overlap(u, v)
    
    # Optimization: Overlap calculation
    def get_overlap(u, v):
        # Max k such that u[-k:] == v[:k]
        # Max possible overlap is n-1
        max_k = n - 1
        for k in range(max_k, -1, -1):
            if u[-k:] == v[:k]:
                return k
        return 0

    # Precompute min jump cost matrix
    # This might be slow if done naively for 720x720x49
    # 25 million iterations is okay.
    
    jump_costs = [[0] * num_cycles for _ in range(num_cycles)]
    
    for i in range(num_cycles):
        if time.time() > deadline:
            break
        for j in range(num_cycles):
            if i == j:
                jump_costs[i][j] = 0 # Should not happen in TSP path, but just in case
                continue
            min_cost = n # Max cost is n (no overlap)
            for u in cycles[i]:
                for v in cycles[j]:
                    ov = get_overlap(u, v)
                    cost = n - ov
                    if cost < min_cost:
                        min_cost = cost
            jump_costs[i][j] = min_cost

    # TSP Heuristic: Nearest Neighbor + 2-Opt
    # We need a path that visits all cycles.
    # Since it's a cycle of cycles, we can treat it as a TSP cycle.
    
    # Init: Random start
    order = [random.randint(0, num_cycles - 1)]
    unvisited = set(range(num_cycles))
    unvisited.remove(order[0])
    
    while unvisited:
        last = order[-1]
        # Find nearest neighbor
        best_next = -1
        best_cost = float('inf')
        for j in unvisited:
            cost = jump_costs[last][j]
            if cost < best_cost:
                best_cost = cost
                best_next = j
        order.append(best_next)
        unvisited.remove(best_next)
        
    # 2-Opt Improvement
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(1, num_cycles - 1):
            for j in range(i + 1, num_cycles):
                # Check if reversing order[i:j+1] improves
                # Old cost: cost(order[i-1], order[i]) + cost(order[j], order[j+1])
                # New cost: cost(order[i-1], order[j]) + cost(order[i], order[j+1])
                # Note: It's a cycle, so j+1 might be 0?
                # Our order is a path 0..N-1.
                # We treat it as a cycle for 2-opt?
                # Yes, the jumps are cyclic.
                
                c1 = jump_costs[order[i-1]][order[i]]
                c2 = jump_costs[order[j]][order[(j+1) % num_cycles]]
                
                c1_new = jump_costs[order[i-1]][order[j]]
                c2_new = jump_costs[order[i]][order[(j+1) % num_cycles]]
                
                if c1_new + c2_new < c1 + c2:
                    order[i:j+1] = reversed(order[i:j+1])
                    improved = True
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break

    # Construct the word
    # Start with the first node of the first cycle
    # We need to pick the specific nodes to enter/exit to achieve the min cost.
    # This requires a second pass to reconstruct the actual path.
    
    # For each cycle in the order, we need to find the best entry and exit nodes.
    # Entry node $v_{in}$ in $C_i$ and exit node $u_{out}$ in $C_i$.
    # The cost from $C_{i-1}$ to $C_i$ depends on $u_{out, i-1}$ and $v_{in, i}$.
    # The internal path in $C_i$ goes from $v_{in}$ to $u_{out}$ visiting all nodes.
    # Since $C_i$ is a cycle, the path from $v_{in}$ to $u_{out}$ is unique (length $n-1$ steps).
    # So we just need to choose $v_{in}$ and $u_{out}$ to minimize the sum of jump costs.
    
    # This is a DP problem on the cycles?
    # Or we can just pick the pair $(v_{in}, u_{out})$ that minimizes the sum of incoming and outgoing jump costs?
    # Jump cost is not additive in that simple way because it depends on the specific nodes.
    
    # Let's just pick the nodes that give the min cost for the specific transition.
    # For transition $C_{i-1} \to C_i$, we used $min(u \in C_{i-1}, v \in C_i) cost(u,v)$.
    # But we must be consistent: the exit node of $C_{i-1}$ must be the same for the transition to $C_i$ and from $C_{i-2}$.
    
    # This is a constraint satisfaction.
    # However, since the gap is small, we can just use the nodes that achieved the min cost for the *incoming* jump,
    # and assume the internal path leads to the exit node that is best for the *outgoing* jump.
    # This might not be optimal, but it's a good heuristic.
    
    # Better: For each cycle $C_i$, and for each possible entry node $v$, calculate the exit node $u$ (determined by the cycle structure).
    # Then check the cost from previous exit to $v$ and from $u$ to next entry.
    
    # Let's simplify:
    # We will iterate through the cycles.
    # For each cycle, we try all $n$ possible entry points.
    # For each entry point, the exit point is fixed (traverse the cycle).
    # We choose the entry point that minimizes the total cost (prev_jump + next_jump).
    
    # This is a local choice, might not be globally optimal, but should be close.
    
    word = []
    # Start with first cycle
    # Pick best entry for first cycle (min cost from "start" to entry? No, we start with the node itself)
    # Actually, the first cycle is just added. Cost is $n$.
    # Let's pick the entry node that minimizes the cost to the next cycle.
    
    prev_exit_node = None
    for idx, c_idx in enumerate(order):
        cycle = cycles[c_idx]
        # Determine best entry node in this cycle
        best_entry_idx = 0
        best_total_cost = float('inf')
        
        for k, entry_node in enumerate(cycle):
            # Exit node is the node after traversing the cycle from entry_node
            # The cycle is a list. Entry is at k.
            # We traverse n-1 steps to visit all nodes.
            # Exit is at (k + n - 1) % n
            exit_idx = (k + n - 1) % n
            exit_node = cycle[exit_idx]
            
            # Cost to enter this cycle
            if idx == 0:
                cost_in = 0 # We start here
            else:
                # Cost from prev_exit_node to entry_node
                ov = get_overlap(prev_exit_node, entry_node)
                cost_in = n - ov
            
            # Cost to exit this cycle (to next)
            if idx == num_cycles - 1:
                cost_out = 0 # End
            else:
                next_c_idx = order[(idx + 1) % num_cycles]
                # We don't know the next entry node yet.
                # This local optimization is flawed.
                # Let's just pick the entry that minimizes cost_in.
                # Or average?
                pass
        
        # Fallback: Just pick the first node, or the one that minimizes cost_in
        # Let's pick the entry node that minimizes cost_in.
        best_entry_idx = 0
        min_cost_in = float('inf')
        for k, entry_node in enumerate(cycle):
            if idx == 0:
                c = 0
            else:
                ov = get_overlap(prev_exit_node, entry_node)
                c = n - ov
            if c < min_cost_in:
                min_cost_in = c
                best_entry_idx = k
                
        entry_node = cycle[best_entry_idx]
        exit_idx = (best_entry_idx + n - 1) % n
        exit_node = cycle[exit_idx]
        
        # Append to word
        if idx == 0:
            word.extend(entry_node)
        else:
            # Append the non-overlapping part of entry_node
            ov = get_overlap(prev_exit_node, entry_node)
            word.extend(entry_node[ov:])
            
        # Traverse the rest of the cycle
        # We are at entry_node.
        # Next nodes are cycle[best_entry_idx + 1], ..., cycle[exit_idx]
        for step in range(1, n):
            curr_idx = (best_entry_idx + step) % n
            curr_node = cycle[curr_idx]
            # Overlap with previous node is n-1
            # So we append the last character
            word.append(curr_node[-1])
            
        prev_exit_node = exit_node
        
    return "".join(map(str, word))

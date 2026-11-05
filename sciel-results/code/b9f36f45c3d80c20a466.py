import itertools
import sys

def construct(n):
    if n <= 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Group permutations into cycles based on left-shift
    # shift(P) = P[1:] + P[0]
    visited = [False] * len(perms)
    cycles = []
    
    for i in range(len(perms)):
        if not visited[i]:
            cycle = []
            curr = perms[i]
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                # Next in cycle is the left shift
                curr = curr[1:] + curr[0]
            cycles.append(cycle)
            
    # There are n! / n cycles
    num_cycles = len(cycles)
    
    # Precompute suffixes of length n-1 and prefixes of length n-1 for each cycle
    # To find min jump cost between cycles
    # We assume max overlap n-1 (cost 1) is impossible between distinct cycles.
    # We check overlap n-2 (cost 2), n-3 (cost 3), etc.
    
    # For efficiency, we'll check overlap n-2 (length n-2) first.
    # If found, cost is 2.
    # If not, cost is higher.
    
    # Store sets of suffixes and prefixes for each cycle
    cycle_suffixes = []
    cycle_prefixes = []
    for c in cycles:
        suffs = set()
        prefs = set()
        for p in c:
            suffs.add(p[1:]) # Suffix of length n-1
            prefs.add(p[:-1]) # Prefix of length n-1
        cycle_suffixes.append(suffs)
        cycle_prefixes.append(prefs)
        
    # Calculate min jump cost between every pair of cycles
    # cost[i][j] = min cost to go from cycle i to cycle j
    # We want to minimize total path length.
    # Inside a cycle, moving from start to end costs (len(cycle)-1) * 1 = n-1?
    # No, inside a cycle of length n, to visit all n nodes, we take n-1 steps of cost 1.
    # So the internal cost is fixed at n-1 for each cycle visited.
    # The variable cost is the jump between cycles.
    
    # We need to find a permutation of the cycles $C_{\pi_1}, \dots, C_{\pi_k}$
    # such that sum of jump costs is minimized.
    
    # Build adjacency matrix of min jump costs
    # If overlap n-2 exists, cost is 2.
    # If not, we might need to check lower overlaps.
    # For n=7, n-2=5.
    
    min_jump_cost = [[0] * num_cycles for _ in range(num_cycles)]
    
    for i in range(num_cycles):
        for j in range(num_cycles):
            if i == j:
                min_jump_cost[i][j] = 0
                continue
            # Check for overlap of length n-2
            # Intersection of suffixes of i and prefixes of j
            # Note: Suffix of P in C_i is P[1:]. Prefix of Q in C_j is Q[:-1].
            # If P[1:] == Q[:-1], then overlap is n-1?
            # Wait.
            # Overlap k means P[-k:] == Q[:k].
            # If k=n-1, P[1:] == Q[:n-1].
            # This implies Q is the shift of P.
            # If P and Q are in different cycles, this is impossible.
            # So max overlap between distinct cycles is n-2.
            
            # Check if any suffix of C_i matches any prefix of C_j with length n-2
            # Suffix of length n-2 is P[2:]
            # Prefix of length n-2 is Q[:n-2]
            
            # We stored suffixes of length n-1.
            # Let's re-calculate for length n-2 to be precise or use the stored ones.
            # If P[1:] == Q[:n-1] is impossible, then P[2:] == Q[:n-2] is the next best.
            
            # Let's just check intersection of sets of length n-2.
            # To save memory/time, we can compute on the fly or precompute.
            pass
            
    # Recompute with length n-2
    cycle_suffixes_n2 = []
    cycle_prefixes_n2 = []
    for c in cycles:
        suffs = set()
        prefs = set()
        for p in c:
            suffs.add(p[2:])
            prefs.add(p[:n-2])
        cycle_suffixes_n2.append(suffs)
        cycle_prefixes_n2.append(prefs)
        
    for i in range(num_cycles):
        for j in range(num_cycles):
            if i == j:
                min_jump_cost[i][j] = 0
                continue
            # Check overlap n-2
            if cycle_suffixes_n2[i] & cycle_prefixes_n2[j]:
                min_jump_cost[i][j] = 2 # Cost = n - (n-2) = 2
            else:
                # Check overlap n-3
                # This is expensive. Assume cost 3 for now if n-2 fails?
                # Or compute.
                # For n=7, n-3=4.
                # Let's assume most are 2.
                min_jump_cost[i][j] = 3 # Fallback, likely rare
                
    # Solve TSP on cycles using Nearest Neighbor
    visited_cycles = [False] * num_cycles
    current = 0
    visited_cycles[0] = True
    path = [0]
    
    for _ in range(num_cycles - 1):
        next_cycle = -1
        min_cost = float('inf')
        for j in range(num_cycles):
            if not visited_cycles[j]:
                if min_jump_cost[current][j] < min_cost:
                    min_cost = min_jump_cost[current][j]
                    next_cycle = j
        if next_cycle == -1:
            break
        visited_cycles[next_cycle] = True
        path.append(next_cycle)
        current = next_cycle
        
    # Construct the string
    # For each cycle in path, we traverse the entire cycle.
    # We need to choose the start and end of the traversal to match the jump.
    # This requires knowing WHICH permutations in the cycle are used for the jump.
    # The simple TSP above only gave the order of cycles.
    # To construct the string, we need to:
    # 1. For the first cycle, pick a start permutation.
    # 2. Traverse the cycle.
    # 3. At the end of the cycle, we are at some permutation P_end.
    # 4. We need to jump to the next cycle C_next.
    # 5. We need to find P_end in C_curr and P_start in C_next such that overlap is max.
    # 6. Then traverse C_next starting from P_start.
    
    # This is more complex. The Nearest Neighbor on cycles doesn't fix the specific nodes.
    # However, if we assume we can always find a cost-2 jump, we can just pick any valid pair.
    
    # Let's refine the construction:
    # 1. Order the cycles: c_0, c_1, ..., c_{k-1}
    # 2. For c_0, pick an arbitrary start p_start.
    # 3. Traverse c_0 completely. The end permutation is p_end.
    #    (Note: In a cycle, if you start at p, you visit all nodes and end at the predecessor of p).
    # 4. Find the best jump from p_end to any q in c_1.
    #    We want max overlap(p_end, q).
    # 5. Start c_1 at q. Traverse c_1 completely.
    # 6. Repeat.
    
    # This greedy approach on the specific nodes is better.
    
    # Let's implement the node-level greedy:
    # 1. Pick a random start permutation.
    # 2. Mark it visited.
    # 3. Current string = start.
    # 4. While unvisited perms remain:
    #    a. Find the unvisited permutation Q that maximizes overlap with the current end P.
    #    b. Append Q to the string (with overlap).
    #    c. Mark Q visited.
    #    d. Set P = Q.
    
    # This is O(N^2) where N=5040. 25M operations.
    # In Python, this might be too slow (120s limit).
    # 25M * (string ops) is likely > 120s.
    
    # Optimization:
    # Use the cycle structure.
    # We know we can stay in a cycle with cost 1.
    # So we should stay in the current cycle as long as possible?
    # No, we must visit ALL perms.
    # The cycle structure means we are forced to jump every n steps.
    
    # Let's stick to the Cycle TSP + Node Selection.
    # 1. Order cycles using the min_jump_cost matrix.
    # 2. For the sequence of cycles, determine the specific entry/exit nodes.
    #    For cycle i, we enter at $E_i$ and exit at $X_i$.
    #    The internal path is the segment of the cycle from $E_i$ to $X_i$ visiting all nodes?
    #    No, a cycle is a loop. To visit all nodes, we must traverse the whole loop.
    #    So we enter at $E_i$, go around, and exit at the node BEFORE $E_i$ (call it $X_i$).
    #    So $X_i$ is fixed by $E_i$.
    #    We want to maximize overlap($X_i, E_{i+1}$).
    
    # Algorithm:
    # 1. Get cycle order $C_0, \dots, C_{K-1}$.
    # 2. For $C_0$, pick arbitrary $E_0$. $X_0$ is determined.
    # 3. For $k=1$ to $K-1$:
    #    Find $E_k$ in $C_k$ that maximizes overlap($X_{k-1}, E_k$).
    #    $X_k$ is determined by $E_k$.
    # 4. Construct string.
    
    # This is efficient!
    
    # Step 1: Cycle Order (Nearest Neighbor on cycles)
    # We already have min_jump_cost.
    
    # Step 2: Node Selection
    # Precompute for each cycle a list of its permutations.
    # And a mapping from permutation to its index in the cycle.
    
    cycle_perms = []
    for c in cycles:
        cycle_perms.append(c)
        
    # For fast lookup of next/prev in cycle
    # cycle_next[i][j] = next perm in cycle i after j
    
    # Let's just use the list.
    
    # Construct String
    word = []
    
    # Start with first cycle
    curr_cycle_idx = path[0]
    curr_cycle = cycle_perms[curr_cycle_idx]
    # Pick arbitrary start
    start_perm = curr_cycle[0]
    word.extend(start_perm)
    
    # Traverse the rest of the cycle
    # We are at start_perm. We need to visit all others.
    # The cycle is a loop. We can just append the rest of the list.
    # But we need to exit at the predecessor of start_perm to jump?
    # No, we can choose ANY start_perm.
    # The exit node is the one before start_perm in the cycle order.
    # So if we start at index 0, we end at index len-1.
    # The permutation at index len-1 is the one that shifts to index 0.
    
    # Let's just append the whole cycle starting from start_perm.
    # The sequence is start_perm, next, next, ..., last.
    # The last perm is the one that has overlap 6 with start_perm.
    # So the end of the string is `last`.
    
    for p in curr_cycle[1:]:
        # Append p with max overlap
        # Since it's the next in the cycle, overlap is n-1.
        # So we just append the last symbol of p.
        word.append(p[-1])
        
    # Now we are at the end of cycle 0.
    # Current end permutation is curr_cycle[-1].
    
    for k in range(1, len(path)):
        next_cycle_idx = path[k]
        next_cycle = cycle_perms[next_cycle_idx]
        
        # Find best entry E_k in next_cycle that maximizes overlap with current end
        curr_end = curr_cycle[-1]
        
        best_entry = None
        best_overlap = -1
        
        for q in next_cycle:
            # Calculate overlap
            # Max possible n-1? No, different cycles.
            # Check n-2, n-3...
            # For speed, check n-2 first.
            if curr_end[1:] == q[:-1]: # Overlap n-1? Impossible.
                overlap = n - 1
            elif curr_end[2:] == q[:n-2]: # Overlap n-2
                overlap = n - 2
            else:
                # Find actual overlap
                overlap = 0
                for l in range(n-3, 0, -1):
                    if curr_end[n-l:] == q[:l]:
                        overlap = l
                        break
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_entry = q
                
        # Append best_entry
        # We need to append it to the word.
        # Overlap is best_overlap.
        # So we append best_entry[best_overlap:]
        word.extend(best_entry[best_overlap:])
        
        # Now traverse the rest of next_cycle
        # We started at best_entry.
        # We need to visit all nodes in next_cycle.
        # Find index of best_entry in next_cycle
        try:
            start_idx = next_cycle.index(best_entry)
        except ValueError:
            start_idx = 0
            
        # Append the rest of the cycle
        for idx in range(1, len(next_cycle)):
            p = next_cycle[(start_idx + idx) % len(next_cycle)]
            # Next in cycle, overlap n-1
            word.append(p[-1])
            
        # Update curr_cycle for next iteration
        curr_cycle = next_cycle
        # The end of this cycle is the node before best_entry
        # Which is next_cycle[(start_idx - 1) % len(next_cycle)]
        # But we just appended the whole cycle, so the last appended was the one before best_entry.
        # So curr_end is effectively next_cycle[(start_idx - 1) % len(next_cycle)]
        
    return "".join(map(str, word))

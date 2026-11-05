# EVOLVE-BLOCK-START
import sys
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Build the graph:
    # Nodes: (n-1)-tuples of distinct symbols from {1..n}
    # Edge from u to v where v = u[1:] + (missing_symbol,)
    # Each edge corresponds to an n-permutation: u + (missing_symbol,)
    
    all_syms = set(range(1, n+1))
    
    def get_missing(node):
        node_set = set(node)
        for s in all_syms:
            if s not in node_set:
                return s
        return None
    
    def get_next(node):
        s = get_missing(node)
        return node[1:] + (s,)
    
    def node_to_perm(node):
        s = get_missing(node)
        return node + (s,)
    
    # Find all cycles
    all_nodes = set(permutations(range(1, n+1), n-1))
    visited = set()
    cycles = []
    
    for node in all_nodes:
        if node not in visited:
            cycle = []
            cur = node
            while cur not in visited:
                visited.add(cur)
                cycle.append(cur)
                cur = get_next(cur)
            cycles.append(cycle)
    
    # For each cycle, compute the list of n-permutations
    cycle_perms = []
    for cycle in cycles:
        perms = [node_to_perm(node) for node in cycle]
        cycle_perms.append(perms)
        
    # Store cycle data: (perms, length)
    m = len(cycles)
    
    if m == 1:
        # Single cycle, just output it
        word = []
        perms = cycle_perms[0]
        for perm in perms:
            if not word:
                word.extend(perm)
            else:
                # Overlap with previous
                max_ovl = 0
                for k in range(1, n):
                    if tuple(word[-k:]) == perm[:k]:
                        max_ovl = k
                word.extend(perm[max_ovl:])
        return "".join(map(str, word))
        
    # Precompute max overlaps between any edge in cycle i and any edge in cycle j
    # overlap_info[i][j] = (max_ovl, idx_end_in_i, idx_start_in_j)
    overlap_info = [[None]*m for _ in range(m)]
    
    for i in range(m):
        perms_i = cycle_perms[i]
        len_i = len(perms_i)
        for j in range(m):
            if i == j:
                continue
            perms_j = cycle_perms[j]
            len_j = len(perms_j)
            
            best_ovl = 0
            best_idx_i = 0
            best_idx_j = 0
            
            for idx_i, p_i in enumerate(perms_i):
                # Optimization: if max possible overlap (n-1) is reached, break?
                # But we need indices.
                for idx_j, p_j in enumerate(perms_j):
                    # Calculate overlap between p_i (suffix) and p_j (prefix)
                    # p_i is tuple of length n, p_j is tuple of length n
                    # Overlap k: p_i[n-k:] == p_j[:k]
                    max_k = 0
                    # Max possible overlap is n-1
                    for k in range(1, n):
                        if p_i[n-k:] == p_j[:k]:
                            max_k = k
                    if max_k > best_ovl:
                        best_ovl = max_k
                        best_idx_i = idx_i
                        best_idx_j = idx_j
                    elif max_k == best_ovl and max_k > 0:
                        # Tie-breaking: doesn't matter much, keep first
                        pass
            
            overlap_info[i][j] = (best_ovl, best_idx_i, best_idx_j)
            
    # Greedy ordering
    used = [False] * m
    order = []
    idx_in_cycle = [0] * m # Start index for each cycle in the final word
    
    # Start with the cycle that has the best average outgoing overlap
    best_start = 0
    best_score = -1
    for i in range(m):
        score = sum(overlap_info[i][j][0] for j in range(m) if j != i)
        if score > best_score:
            best_score = score
            best_start = i
            
    order.append(best_start)
    used[best_start] = True
    current = best_start
    
    for _ in range(m - 1):
        best_next = -1
        best_ovl = -1
        for j in range(m):
            if not used[j]:
                ovl = overlap_info[current][j][0]
                if ovl > best_ovl:
                    best_ovl = ovl
                    best_next = j
        if best_next == -1:
            for j in range(m):
                if not used[j]:
                    best_next = j
                    break
        order.append(best_next)
        used[best_next] = True
        current = best_next
        
    # Local Search Improvement (if number of cycles is small)
    if m < 30:
        # Calculate current total overlap
        def get_total_overlap(ord):
            total = 0
            for k in range(len(ord) - 1):
                total += overlap_info[ord[k]][ord[k+1]][0]
            return total
            
        current_score = get_total_overlap(order)
        improved = True
        max_iters = 100
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for k in range(len(order) - 1):
                # Try swapping order[k] and order[k+1]
                new_order = order[:]
                new_order[k], new_order[k+1] = new_order[k+1], new_order[k]
                new_score = get_total_overlap(new_order)
                if new_score > current_score:
                    order = new_order
                    current_score = new_score
                    improved = True
                    
    # Build the word
    word = []
    
    # For the first cycle, we need to determine the start index.
    # The "start" index is the one that overlaps with the *previous* cycle.
    # For the first cycle, there is no previous cycle, so we can pick any start?
    # Actually, the start of the first cycle determines the "end" of the first cycle.
    # The end of the first cycle must overlap with the start of the second cycle.
    # So for cycle 0 in the order, the "end" index is determined by overlap_info[order[0]][order[1]].
    # The "start" index of cycle 0 is free?
    # No, the cycle is circular. If we fix the end, the start is fixed (it's the next edge).
    # Wait.
    # If we break the cycle at edge $u_{start}$, the sequence is $u_{start}, ..., u_{end}$.
    # $u_{end}$ is the predecessor of $u_{start}$.
    # So if we know $u_{end}$ (from overlap with next cycle), we know $u_{start}$ (successor of $u_{end}$).
    
    # For the last cycle, the "start" index is determined by overlap with previous cycle.
    # The "end" index is free?
    # No, the word just ends there.
    
    # So:
    # For cycle at position $k$ in order:
    # If $k > 0$: Start index is determined by overlap_info[order[k-1]][order[k]] (idx_start_in_j).
    #             End index is determined by overlap_info[order[k]][order[k+1]] (idx_end_in_i) if $k < m-1$.
    #             If $k == m-1$: End index is free. We can pick any? 
    #             Actually, the end of the last cycle doesn't overlap with anything.
    #             So we can pick the end index to be anything?
    #             But the start is fixed. The end is the predecessor of the start.
    #             So the whole cycle sequence is fixed by the start index.
    
    # Let's re-evaluate.
    # We have a cycle of nodes $0, 1, ..., L-1$.
    # If we start at $s$, the sequence is $s, s+1, ..., s+L-1$ (mod L).
    # The last element is $s+L-1 \equiv s-1$.
    # So the "end" edge is $s-1$.
    # The "start" edge is $s$.
    
    # Constraints:
    # For $k=1$ to $m-1$:
    # Start of cycle $k$ ($s_k$) must be $idx\_start\_in\_j$ from $overlap\_info[order[k-1]][order[k]]$.
    # End of cycle $k-1$ ($e_{k-1}$) must be $idx\_end\_in\_i$ from $overlap\_info[order[k-1]][order[k]]$.
    # Note: $e_{k-1}$ must be consistent with $s_{k-1}$.
    # $e_{k-1} = s_{k-1} - 1 \pmod L$.
    # So $s_{k-1} = e_{k-1} + 1 \pmod L$.
    # This means the start of cycle $k-1$ is determined by the end of cycle $k-1$.
    # And the end of cycle $k-1$ is determined by the overlap with cycle $k$.
    # So $s_{k-1}$ is determined by the overlap with cycle $k$.
    
    # For the first cycle ($k=0$):
    # Its end $e_0$ is determined by overlap with cycle 1.
    # So its start $s_0 = e_0 + 1$.
    # So the start of the first cycle is also determined!
    
    # For the last cycle ($k=m-1$):
    # Its start $s_{m-1}$ is determined by overlap with cycle $m-2$.
    # Its end $e_{m-1}$ is free?
    # No, $e_{m-1} = s_{m-1} - 1$.
    # So the last cycle is also fully determined.
    
    # So, the entire sequence of edges for every cycle is fixed by the pairwise overlaps.
    
    # Let's verify consistency.
    # Overlap(i, j) gives $e_i, s_j$.
    # Consistency check: $s_i = e_i + 1$?
    # $e_i$ comes from Overlap(i, i+1).
    # $s_i$ comes from Overlap(i-1, i).
    # These are independent choices in the matrix.
    # Is it possible that $s_i \neq e_i + 1$?
    # Yes! The max overlap for (i, j) might use a different edge in i than the max overlap for (i-1, i) implies.
    # Example:
    # Cycle i has edges 0, 1, 2.
    # Overlap(i-1, i) says: Start i at 1. So End i must be 0.
    # Overlap(i, i+1) says: End i is 2. So Start i must be 0.
    # Contradiction: Start i cannot be both 1 and 0.
    
    # So we cannot independently maximize overlaps for each pair.
    # We must choose a start index $s_k$ for each cycle $k$.
    # Then $e_k = s_k - 1$.
    # The overlap with next cycle is $overlap(Perm(e_k), Perm(s_{k+1}))$.
    # We want to maximize $\sum overlap(Perm(s_k-1), Perm(s_{k+1}))$.
    
    # This is a different problem.
    # We can use the greedy ordering of cycles, but then optimize the start indices.
    # Or, simpler:
    # Just pick the start index for each cycle that maximizes the overlap with the *previous* cycle's end.
    # But the previous cycle's end depends on *its* start.
    
    # Simplified Approach for Implementation:
    # 1. Order the cycles greedily based on max possible overlaps (as before).
    # 2. Iterate through the ordered cycles.
    # 3. For the first cycle, pick the start index that maximizes overlap with the second cycle's *potential* start?
    #    Actually, let's just pick the start index for cycle 0 that maximizes overlap with cycle 1's *best* start?
    #    This is getting complex.
    
    # Alternative Simple Improvement:
    # Instead of complex consistency checks, just use the greedy order.
    # For each cycle in the order, pick the start index that maximizes overlap with the *previous* cycle's *actual* end.
    # For the first cycle, pick the start index that maximizes overlap with the *next* cycle's *actual* start?
    # No, let's just pick the start of the first cycle to be 0 (arbitrary).
    # Then the end of cycle 0 is fixed.
    # Then pick the start of cycle 1 to maximize overlap with end of cycle 0.
    # Then the end of cycle 1 is fixed.
    # Then pick the start of cycle 2 to maximize overlap with end of cycle 1.
    # And so on.
    # This is a sequential greedy assignment of start indices.
    # It might not be globally optimal, but it's consistent and simple.
    
    # Let's implement this sequential assignment.
    
    # Re-calculate overlaps for specific pairs of edges?
    # We already have overlap_info[i][j] which is max over ALL edges.
    # But now we need overlap between SPECIFIC edges.
    # So we need a function `get_overlap(p1, p2)`.
    
    def get_overlap(p1, p2):
        max_k = 0
        for k in range(1, n):
            if p1[n-k:] == p2[:k]:
                max_k = k
        return max_k

    # Sequential Construction
    # Order is fixed.
    # Start indices:
    start_indices = [0] * m
    
    # For the first cycle, we can choose the start to maximize overlap with the second cycle.
    # But the second cycle's start is not known yet.
    # Let's just start the first cycle at index 0.
    # Or, better: Start the first cycle at the index that maximizes overlap with the *next* cycle's *best* start?
    # Let's just use 0 for the first one to keep it simple, or pick the one that gives best overlap with the next cycle's *max overlap start*?
    # Actually, let's just pick the start of cycle 0 to be the one that maximizes overlap with cycle 1's *corresponding* end?
    # No, let's just iterate.
    
    # Step 1: Determine start for cycle 0.
    # Try all starts for cycle 0, calculate resulting end, then calculate overlap with cycle 1's *best* start?
    # Too slow.
    # Just pick start[0] = 0.
    
    # Step 2: For k = 1 to m-1:
    # Pick start[k] to maximize overlap( end[k-1], start[k] ).
    # end[k-1] is start[k-1] - 1 (mod len).
    
    # This is O(m * L * n). L is cycle length. m is num cycles.
    # This is fast.
    
    # Let's do this.
    
    # Note: The "Order" of cycles is still determined by the greedy max-overlap matrix.
    # This ensures we connect "compatible" cycles.
    
    # Implementation of Sequential Start Assignment:
    
    # We need the perms for each cycle.
    # cycle_perms[i] is the list of perms in cycle order starting from node 0.
    
    # start_indices[0] = 0
    start_indices[0] = 0
    
    for k in range(1, m):
        prev_cycle = order[k-1]
        curr_cycle = order[k]
        
        prev_start = start_indices[prev_cycle]
        prev_len = len(cycle_perms[prev_cycle])
        prev_end_idx = (prev_start - 1) % prev_len
        prev_end_perm = cycle_perms[prev_cycle][prev_end_idx]
        
        curr_perms = cycle_perms[curr_cycle]
        curr_len = len(curr_perms)
        
        best_start = 0
        best_ovl = -1
        
        for s in range(curr_len):
            curr_start_perm = curr_perms[s]
            ovl = get_overlap(prev_end_perm, curr_start_perm)
            if ovl > best_ovl:
                best_ovl = ovl
                best_start = s
                
        start_indices[curr_cycle] = best_start
        
    # Build Word
    word = []
    
    for k in range(m):
        ci = order[k]
        perms = cycle_perms[ci]
        start_idx = start_indices[ci]
        L = len(perms)
        
        # Sequence: start_idx, start_idx+1, ..., start_idx+L-1
        for offset in range(L):
            idx = (start_idx + offset) % L
            perm = perms[idx]
            
            if not word:
                word.extend(perm)
            else:
                max_ovl = 0
                # Overlap with previous perm
                # We can just check against the last perm added
                # But we know the exact overlap from the start index selection?
                # No, the overlap might be different if we didn't pick the max?
                # We picked the max, so we can just use that.
                # But to be safe, recalculate.
                for k_ovl in range(1, n):
                    if tuple(word[-k_ovl:]) == perm[:k_ovl]:
                        max_ovl = k_ovl
                word.extend(perm[max_ovl:])
                
    return "".join(map(str, word))

# EVOLVE-BLOCK-END

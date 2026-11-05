import time
import itertools
import sys

# EVOLVE-BLOCK-START
def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    if n <= 1:
        return "1"
    
    # For small n, standard is optimal
    if n <= 4:
        return _standard_superpermutation(n)
    
    # For n=5, 6, standard is optimal
    if n <= 6:
        return _standard_superpermutation(n)
    
    # For n=7, try to improve using cycle stitching
    # 1. Identify cycles in the graph of (n-1)-perms
    # Nodes: permutations of 1..n-1
    # Edge from u to v if v is u shifted left by 1 and appended with the missing symbol
    
    symbols = list(range(1, n))
    perms = list(itertools.permutations(symbols))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Build successor map
    # For a perm p of 1..n-1, the missing symbol is m
    # Next node is p[1:] + (m,)
    successors = {}
    for p in perms:
        s = set(p)
        missing = [x for x in range(1, n) if x not in s][0]
        next_p = p[1:] + (missing,)
        successors[p] = next_p
        
    # Find cycles
    visited = set()
    cycles = []
    for p in perms:
        if p not in visited:
            cycle = []
            curr = p
            while curr not in visited:
                visited.add(curr)
                cycle.append(curr)
                curr = successors[curr]
            if cycle:
                cycles.append(cycle)
    
    # We have a list of cycles. Each cycle is a list of nodes (perms of n-1).
    # Traversing a cycle of length L uses L edges (perms of n) and produces a word of length L + (n-1) ?
    # Actually, a cycle of k nodes corresponds to k edges.
    # The word segment for a cycle is the concatenation of the edges.
    # Edge for node u is u + (missing_symbol(u),)
    
    # We need to stitch these cycles together.
    # We can break a cycle at any point.
    # When stitching Cycle A to Cycle B, we end A and start B.
    # The cost of stitching is the length of the "bridge" word.
    # If we end A with node u_last, the last edge is E_last.
    # If we start B with node v_first, the first edge is E_first.
    # The bridge is the sequence of symbols connecting E_last to E_first.
    # However, we are allowed to use ANY symbols in the bridge, not just permutations.
    # To minimize length, we want the maximum overlap between the suffix of the A-segment and prefix of the B-segment.
    
    # This is a complex optimization.
    # Given time constraints, we will use a simple Greedy Nearest Neighbor on the cycles.
    # Distance between two cycles is the minimum bridge length required to connect any end of A to any start of B.
    
    # Precompute the "word" representation of each cycle?
    # No, just the nodes.
    
    # To keep it fast, we will limit the search.
    # If n=7, there are ~720 nodes.
    # Number of cycles is likely small (e.g., 1-10 cycles).
    
    # Let's assume the number of cycles is small enough for O(C^2) or O(C^3) search.
    
    if time.time() > deadline:
        return _standard_superpermutation(n)
        
    # Calculate bridge cost between two nodes u (end of A) and v (start of B)
    # u is a perm of n-1. v is a perm of n-1.
    # We need a word W such that:
    # 1. W contains the edge from u (which is u + missing(u)) as a suffix? No.
    #    The segment for A ends with the edge E_u = u + (m_u,).
    #    So the word ends with E_u.
    # 2. W starts with the edge E_v = v + (m_v,)?
    #    The segment for B starts with E_v.
    #    So the word starts with E_v.
    # 3. We want to merge End(A) and Start(B).
    #    We append Start(B) to End(A) with maximum overlap.
    #    Overlap between string X (suffix of A) and string Y (prefix of B).
    
    def overlap_len(s1, s2):
        # Max k such that s1[-k:] == s2[:k]
        max_k = min(len(s1), len(s2))
        for k in range(max_k, 0, -1):
            if s1[-k:] == s2[:k]:
                return k
        return 0

    # Convert nodes to strings for overlap calculation
    # Node u -> Edge string E_u = str(u) + str(missing)
    # We need the full edge string.
    
    edge_strs = {}
    for p in perms:
        s = set(p)
        missing = [x for x in range(1, n) if x not in s][0]
        edge_strs[p] = "".join(map(str, p)) + str(missing)
        
    # For each cycle, we can choose any start node.
    # We will represent a cycle by its list of nodes.
    # Cost to go from end of Cycle A (node u) to start of Cycle B (node v) is:
    # len(edge_strs[v]) - overlap_len(edge_strs[u], edge_strs[v])
    # Note: We are appending the edge of v.
    
    # Greedy Stitching:
    # 1. Start with the largest cycle (or any).
    # 2. Repeatedly pick the next cycle that minimizes the bridge cost from the current end.
    # 3. Within the chosen cycle, pick the start node that minimizes the bridge cost.
    # 4. Update the current end to the end node of the chosen cycle (which is the node before the start node in the cycle order? No, we traverse the whole cycle, so end is the node before start).
    
    # Actually, if we traverse the whole cycle starting at v_start, we end at v_prev (the node before v_start in the cycle).
    
    # Let's implement this greedy.
    
    # If time is running out, return standard.
    if time.time() > deadline - 10:
        return _standard_superpermutation(n)
        
    # Sort cycles by length descending (larger first is often better to start)
    cycles.sort(key=lambda c: len(c), reverse=True)
    
    # State: current_end_node, list of visited_cycles, current_word
    # To avoid building the full word (expensive), we just track the end node and the accumulated length.
    # But we need to verify validity at the end.
    
    # Heuristic:
    # Pick a starting cycle.
    # For each remaining cycle, calculate min bridge cost from current_end to any node in that cycle.
    # Pick the cycle with min cost.
    # Pick the specific start node in that cycle that gave min cost.
    # Traverse that cycle, update current_end to the node before the start node.
    
    # Try multiple starting cycles (e.g., top 5 largest) to find a better global solution.
    
    best_len = float('inf')
    best_word = None
    
    num_starts = min(5, len(cycles))
    
    for start_c_idx in range(num_starts):
        current_end = cycles[start_c_idx][0] # Start at first node of this cycle? 
        # Actually, we can choose the start node of the first cycle too.
        # For simplicity, fix start node of first cycle to its first element for now, or try all?
        # Trying all start nodes for the first cycle is O(L).
        
        visited_cycles = set([start_c_idx])
        curr_end = cycles[start_c_idx][0]
        # We need to know the "previous" node in the cycle to determine the end after traversal.
        # If we start at index i in cycle C, we traverse C[i], C[i+1], ..., C[i-1].
        # The end node is C[i-1].
        
        # Let's just pick the start node that minimizes the "cost to enter" (0 for first) 
        # and then greedily pick next.
        
        # To save time, we just pick the first node of the first cycle.
        # And for subsequent cycles, pick the best entry.
        
        word_parts = []
        # Add the first cycle
        c = cycles[start_c_idx]
        # We need to decide where to start in c.
        # Let's start at c[0].
        start_node = c[0]
        # The word for this cycle is the concatenation of edges from start_node to its predecessor.
        # Edge of u is E_u.
        # Sequence: E_start, E_next(start), ..., E_prev(start).
        
        # Build string for this cycle
        s = ""
        curr = start_node
        for _ in range(len(c)):
            s += edge_strs[curr]
            curr = successors[curr]
            if time.time() > deadline:
                return _standard_superpermutation(n)
        
        word_parts.append(s)
        curr_end = successors[start_node] # Wait, if we traverse len(c) steps, we end at the node BEFORE start_node.
        # Let's trace:
        # Nodes: n0, n1, n2. Successors: n0->n1, n1->n2, n2->n0.
        # Start n0. Edges: E_n0, E_n1, E_n2.
        # Word: E_n0 E_n1 E_n2 (overlapped).
        # The last edge is E_n2. The node associated with the end of E_n2 is n2.
        # So curr_end should be n2 (which is predecessor of n0).
        curr_end = successors[successors[start_node]] if len(c) > 2 else successors[start_node]
        # Actually, just take the last node in the traversal.
        # Traversal: start_node, succ(start), ..., succ^(L-1)(start).
        # The last node visited is succ^(L-1)(start).
        # This is the predecessor of start_node in the cycle.
        
        # Calculate curr_end properly
        curr = start_node
        for _ in range(len(c) - 1):
            curr = successors[curr]
        curr_end = curr
        
        # Now greedily add other cycles
        remaining = [i for i in range(len(cycles)) if i != start_c_idx]
        
        while remaining:
            best_next_idx = -1
            best_start_node = None
            min_cost = float('inf')
            
            end_str = edge_strs[curr_end]
            
            for idx in remaining:
                c_next = cycles[idx]
                for node in c_next:
                    # Cost to bridge from curr_end to node
                    # We append E_node to the current word.
                    # Overlap between end_str (suffix of current word) and E_node (prefix of new segment)
                    ov = overlap_len(end_str, edge_strs[node])
                    cost = len(edge_strs[node]) - ov
                    if cost < min_cost:
                        min_cost = cost
                        best_next_idx = idx
                        best_start_node = node
            
            if best_next_idx == -1:
                break
                
            # Add this cycle
            c_next = cycles[best_next_idx]
            start_node = best_start_node
            
            # Find position of start_node in c_next to determine traversal order
            # We traverse from start_node to its predecessor.
            s = ""
            curr = start_node
            for _ in range(len(c_next)):
                s += edge_strs[curr]
                curr = successors[curr]
                if time.time() > deadline:
                    return _standard_superpermutation(n)
            
            word_parts.append(s)
            
            # Update curr_end
            curr = start_node
            for _ in range(len(c_next) - 1):
                curr = successors[curr]
            curr_end = curr
            
            remaining.remove(best_next_idx)
            
        # Construct full word
        # Overlap adjacent parts
        full_word = word_parts[0]
        for part in word_parts[1:]:
            ov = overlap_len(full_word, part)
            full_word += part[ov:]
            
        if len(full_word) < best_len:
            best_len = len(full_word)
            best_word = full_word
            
        if time.time() > deadline - 5:
            break
            
    if best_word and len(best_word) < len(_standard_superpermutation(n)):
        # Verify validity (optional but recommended)
        if _is_valid_superpermutation(best_word, n):
            return best_word
            
    return _standard_superpermutation(n)

def _standard_superpermutation(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _is_valid_superpermutation(word, n):
    if len(word) < n:
        return False
    perms = set()
    for i in range(len(word) - n + 1):
        w = word[i:i+n]
        if len(set(w)) == n:
            perms.add(w)
    return len(perms) == len(list(itertools.permutations(range(1, n+1))))
# EVOLVE-BLOCK-END

import random
import time
import sys
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses cyclic (Z_n-regular) construction with non-backtracking girth check.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def check_girth_nb(adj, n, g):
        """
        Check girth using non-backtracking walks from every vertex.
        Returns True if no cycle shorter than g exists.
        """
        # For each starting vertex and each outgoing edge, do non-backtracking BFS
        # to depth (g-1)//2. If we return to the start via a different path,
        # we've found a short cycle.
        
        max_depth = (g - 1) // 2
        
        for start in range(n):
            for first_neighbor in adj[start]:
                # Non-backtracking walk: start -> first_neighbor -> ...
                # State: (current_node, prev_node, depth)
                # We want to detect if we can reach 'start' from a non-backtracking
                # walk of length L where L < g and the last step isn't backtracking
                # to the first step (which would be a valid cycle of length L+1... 
                # actually let me think about this more carefully).
                
                # A non-backtracking walk of length L from start back to start
                # where the walk doesn't immediately reverse any step.
                # If L < g, we have a cycle of length L (or a subcycle).
                
                # Actually, simpler: do BFS in the directed line graph (non-backtracking).
                # Nodes in line graph: directed edges (u,v) where u!=v and (u,v) is an edge.
                # Transition: (u,v) -> (v,w) where w != u and (v,w) is an edge.
                
                # For each directed edge (start, first_neighbor), check if we can
                # reach any directed edge (x, start) in fewer than g-1 steps.
                # If so, there's a non-backtracking walk of length < g from start
                # back to start, implying a cycle of length < g.
                
                # More precisely: a non-backtracking walk of length L from (start, fn)
                # to (x, start) means we have a walk start -> fn -> ... -> x -> start
                # of length L+1. If L+1 < g, we have a cycle of length <= L+1 < g.
                
                # Let's just do BFS from each directed edge (start, fn) and check
                # if we can reach any edge ending at 'start' within g-2 steps.
                
                # visited: set of directed edges (u, v)
                visited = set()
                queue = deque()
                queue.append((first_neighbor, start, 0))  # (current, prev, depth)
                visited.add((first_neighbor, start))
                
                found_short_cycle = False
                while queue:
                    curr, prev, depth = queue.popleft()
                    if depth >= g - 1:
                        continue
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue  # non-backtracking
                        # Check if this completes a short cycle
                        if nxt == start and depth + 1 < g:
                            # We have a non-backtracking walk of length depth+2
                            # from start back to start. Cycle length = depth+2.
                            # But wait, we need to make sure this isn't just
                            # the trivial 2-step walk start->fn->start if fn=start's neighbor
                            # Actually depth+2 is the cycle length. If depth+2 < g, short cycle.
                            # But we also need to ensure it's not just going back and forth.
                            # Since we're non-backtracking, this should be fine.
                            # However, the walk start->fn->...->curr->start has length depth+2.
                            # If depth+2 < g, we have a cycle of length <= depth+2 < g.
                            found_short_cycle = True
                            break
                        if (nxt, curr) not in visited:
                            visited.add((nxt, curr))
                            queue.append((nxt, curr, depth + 1))
                    if found_short_cycle:
                        break
                if found_short_cycle:
                    return False
        
        return True

    def make_cyclic_graph(n, offsets):
        """
        Create a cyclic Z_n-regular graph with given offsets.
        offsets should have k elements (k/2 pairs if we use symmetric, but here
        we just use k offsets and add edges i -> (i+offset)%n).
        For k=4, we use 4 offsets. To ensure simplicity, offsets should be
        distinct modulo n and no offset should be 0 or n/2 (to avoid loops/multiedges
        when n is even).
        """
        edges = []
        edge_set = set()
        for i in range(n):
            for off in offsets:
                j = (i + off) % n
                if i == j:
                    return None  # loop
                key = (min(i, j), max(i, j))
                if key in edge_set:
                    return None  # multiedge
                edge_set.add(key)
                edges.append((i, j))
        
        # Check degree
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if any(d != k for d in degree):
            return None
        
        return edges

    def check_girth_from_edges(edges, n, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return check_girth_nb(adj, n, g)

    # Try cyclic constructions
    # For k=4, we need 4 offsets. The graph is Z_n-regular.
    # We search over offset sets.
    
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for (4,10): n >= 2*(4-1)*(3^3 + 3^2 + 3 + 1) = 6*91 = 546? 
    # Wait, Moore bound for regular graphs: n >= 2*(k-1)*sum_{i=0}^{(g-2)/2} (k-1)^i for even g.
    # For k=4, g=10: n >= 2*3*(1 + 3 + 9 + 27) = 6*40 = 240.
    # Actually the formula is: n >= 2(k-1) * ((k-1)^{(g/2)} - 1) / (k-2) for even g?
    # Let me use the standard: for girth g and degree k, 
    # n >= 2(k-1) * sum_{i=0}^{floor((g-2)/2)} (k-1)^i
    # For g=10: floor((10-2)/2) = 4, so sum_{i=0}^{4} 3^i = 1+3+9+27+81 = 121
    # n >= 2*3*121 = 726? That seems high.
    
    # Actually the Moore bound for cages: for even girth g=2r, 
    # n >= 2(k-1) * ((k-1)^r - 1)/(k-2)
    # For k=4, g=10, r=5: n >= 2*3*(3^5 - 1)/2 = 3*(243-1) = 3*242 = 726.
    # Hmm, but the problem says lower bound is 243. Let me re-check.
    
    # The Moore bound: for a k-regular graph of girth g,
    # if g is even, g=2r: n >= 2(k-1) * sum_{i=0}^{r-1} (k-1)^i
    # = 2(k-1) * ((k-1)^r - 1)/(k-2)
    # For k=4, r=5: 2*3*(3^5-1)/2 = 3*242 = 726.
    
    # But the problem statement says "best lower bound is 243". Maybe they use a different bound.
    # Regardless, we'll search from a reasonable lower bound.
    
    # Let's try n from 243 upward, but cyclic graphs have constraints.
    # For a Z_n-regular graph with k=4, we need 4 offsets a1,a2,a3,a4 such that
    # all 4n directed edges (i, i+ai) are distinct and no loops.
    # This requires ai != 0 mod n and ai != aj mod n for i != j, and also
    # ai != -aj mod n for i != j (to avoid multiedges from different offsets).
    # Actually, edge (i, i+a) and edge (j, j+b) are the same undirected edge if
    # i = j+b and i+a = j, i.e., b = a and i = j, or i = j+a and i+a = j+b...
    # More carefully: (i, i+a) = (j, j+b) as undirected edge iff
    # {i, i+a} = {j, j+b}. This happens if i=j and a=b, or i=j+b and i+a=j.
    # The second case: i = j+b, i+a = j => j+b+a = j => a+b = 0 mod n.
    # So we need a_i + a_j != 0 mod n for all i != j.
    
    # Also a_i != 0 mod n (no loops).
    
    # For k=4, we need 4 offsets a1,a2,a3,a4 with:
    # - a_i != 0 mod n
    # - a_i != a_j mod n for i != j  
    # - a_i + a_j != 0 mod n for i != j
    
    # Let's try small n values and search for valid offset sets.
    
    # Start from a reasonable lower bound. The problem says 243 is the best lower bound.
    # Let's try n from 243 to some upper limit.
    
    # For efficiency, we'll first filter offset sets that satisfy the basic
    # constraints, then check girth.
    
    # To make this tractable, we'll try a limited set of n values and
    # sample offset sets.
    
    # Actually, for cyclic graphs, the girth can be computed more efficiently.
    # A cycle in a Z_n-regular graph corresponds to a sequence of offsets
    # a_{i1}, a_{i2}, ..., a_{il} such that their sum is 0 mod n, and no
    # partial sum is 0 mod n (for a simple cycle), and the non-backtracking
    # condition means a_{ij} != -a_{i(j-1)}.
    
    # So girth >= g iff there is no sequence of < g offsets (with the
    # non-backtracking condition) that sums to 0 mod n.
    
    # This is equivalent to: the shortest non-backtracking walk in the
    # offset graph that sums to 0 mod n has length >= g.
    
    # We can check this by BFS in the state space (sum_so_far, last_offset)
    # where sum_so_far is in Z_n and last_offset is one of the offsets.
    
    def check_girth_cyclic(n, offsets):
        """
        Check if the cyclic graph with given offsets has girth >= g.
        Uses BFS on (current_sum, last_offset_index) states.
        """
        if len(offsets) != k:
            return False
        
        # Basic validity checks
        if any(o % n == 0 for o in offsets):
            return False
        for i in range(k):
            for j in range(i+1, k):
                if offsets[i] % n == offsets[j] % n:
                    return False
                if (offsets[i] + offsets[j]) % n == 0:
                    return False
        
        # BFS: state is (sum, last_offset_idx)
        # Start: for each offset, state (offset, idx) at depth 1
        # Transition: from (s, i), for each j != i such that offsets[j] != -offsets[i] mod n:
        #   new state ((s + offsets[j]) % n, j) at depth+1
        # We want to know if any state (0, j) is reachable at depth < g.
        # But (0, j) means sum is 0 and last offset is j. This corresponds to
        # a non-backtracking walk of length depth that sums to 0.
        
        # Actually, we need to be careful. The walk is:
        # offset_1, offset_2, ..., offset_L
        # where offset_{i+1} != -offset_i (non-backtracking)
        # and sum = 0 mod n.
        # The cycle length is L.
        
        # BFS from each starting offset:
        # State: (sum, last_idx)
        # Initial states: (offsets[i], i) for each i, depth 1
        # If we reach (0, j) at depth L, we have a cycle of length L.
        # But we need L >= g for valid girth.
        
        # visited: set of (sum, last_idx) to avoid revisiting
        # We do BFS from all starting states simultaneously.
        
        visited = set()
        queue = deque()
        
        for i in range(k):
            s = offsets[i] % n
            state = (s, i)
            visited.add(state)
            queue.append((s, i, 1))
        
        while queue:
            s, last_idx, depth = queue.popleft()
            if depth >= g:
                continue
            for j in range(k):
                if j == last_idx:
                    continue
                # Non-backtracking: offsets[j] != -offsets[last_idx] mod n
                if (offsets[j] + offsets[last_idx]) % n == 0:
                    continue
                new_s = (s + offsets[j]) % n
                new_state = (new_s, j)
                if new_state in visited:
                    continue
                # If new_s == 0, we have a cycle of length depth+1
                if new_s == 0:
                    if depth + 1 < g:
                        return False
                    # If depth+1 >= g, it's fine, but we don't need to explore further
                    # from this state since any extension would be longer.
                    # Actually, we should still mark it visited to avoid reprocessing.
                    visited.add(new_state)
                    continue
                visited.add(new_state)
                queue.append((new_s, j, depth + 1))
        
        return True

    # Search over n values
    # Start from the theoretical lower bound. For (4,10), let's start from 243.
    # But cyclic graphs may not exist for all n. Let's try a range.
    
    # To keep runtime manageable, we'll try n values and for each, search
    # over offset sets. For k=4, we need 4 offsets from 1..n-1.
    # The search space is large, so we'll use a heuristic: try offsets
    # that are "spread out" and use a limited number of random trials.
    
    # First, let's try some known good constructions or small n values.
    # For k=4, g=10, the smallest known cyclic graph might be large.
    # Let's try n from 243 to 500, with a limited number of offset sets per n.
    
    max_n_to_try = 600
    min_n = 243
    
    # For each n, we'll try a few random offset sets and some structured ones.
    
    for n in range(min_n, min(max_n_to_try, 500) + 1):
        if time.time() > deadline - 20:
            break
        
        # Try some structured offset sets first
        # For even n, avoid offsets that are n/2 (since a + a = 0 if a = n/2)
        # Actually, we already check a_i + a_j != 0, so if a_i = n/2, then
        # a_i + a_i = 0, but that's the same offset used twice, which isn't
        # allowed since offsets must be distinct. So it's fine.
        
        # Try offsets based on a primitive root or just random
        found = False
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Generate random offsets
            offsets = []
            used = set()
            valid = True
            for _ in range(k):
                o = random.randint(1, n - 1)
                if o in used:
                    valid = False
                    break
                # Check a_i + a_j != 0 for existing offsets
                for existing in offsets:
                    if (o + existing) % n == 0:
                        valid = False
                        break
                if not valid:
                    break
                offsets.append(o)
                used.add(o)
            
            if not valid or len(offsets) != k:
                continue
            
            if check_girth_cyclic(n, offsets):
                # Found a valid cyclic graph!
                edges = make_cyclic_graph(n, offsets)
                if edges is not None:
                    # Double-check with full girth verification
                    if check_girth_from_edges(edges, n, g):
                        if n < best_n:
                            best_n = n
                            best_edges = edges
                        found = True
                        break
        
        if found:
            # Found a valid n, but let's see if we can find a smaller one
            # by continuing to search (but we're going in increasing n, so
            # the first one we find is the smallest for this search order)
            # Actually, we're iterating n from min_n upward, so the first
            # valid n we find is the smallest. We can break.
            # But wait, we might have skipped some n values or not found
            # a valid offset set for smaller n. Let's continue searching
            # a bit more to ensure we don't miss a smaller valid n.
            # Actually, since we're going in order, the first valid n is
            # the smallest we'll find. But we might not have found a valid
            # offset set for a smaller n due to limited attempts.
            # Let's continue for a few more n values to be safe.
            continue
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback: LW Graph
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

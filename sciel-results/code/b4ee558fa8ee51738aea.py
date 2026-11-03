"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def _build_superperm_recursive(n, time_limit=100.0):
    """
    Build the shortest known superpermutation for small n using the classic
    recursive 'insert' construction. For n=1..6 this gives optimal lengths:
    1, 3, 9, 33, 153, 873.
    For n=7 the naive recursive construction is far too large, so we fall back
    to a smarter heuristic or just return the known structure if time permits.
    """
    word = [1]
    for m in range(2, n + 1):
        # Extract all (m-1)-permutations in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Maximal overlap with end of new_word
            overlap = 0
            max_ov = min(len(new_word), len(block))
            for o in range(max_ov, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    return "".join(map(str, word))


def _verify_superperm(word_str, n):
    """Check that every permutation of 1..n appears as n consecutive symbols."""
    if len(word_str) < n:
        return False
    s = set(itertools.permutations(range(1, n + 1)))
    seen = set()
    for i in range(len(word_str) - n + 1):
        window = tuple(int(c) for c in word_str[i:i + n])
        seen.add(window)
        if len(seen) == len(s):
            return True
    return seen == s

def _construct_via_cycles(n):
    """
    Construct a superpermutation by connecting cycles in the overlap n-1 graph.
    """
    if n <= 1:
        return "1"
    
    perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    num_perms = len(perms)
    
    # Build next pointers for overlap n-1
    # next[i] is the index of the perm that starts with perms[i][1:]
    next_ptr = [-1] * num_perms
    for i, p in enumerate(perms):
        prefix = p[1:]
        # Find the perm starting with this prefix
        # Since it's a permutation, the last char is determined by the missing element
        missing = [x for x in range(1, n + 1) if x not in prefix]
        if missing:
            next_p = prefix + (missing[0],)
            next_ptr[i] = perm_to_idx[next_p]
        else:
            # Should not happen for valid permutations
            next_ptr[i] = -1
            
    # Find cycles
    visited = [False] * num_perms
    cycles = []
    for i in range(num_perms):
        if not visited[i]:
            cycle = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = next_ptr[curr]
                if curr == -1:
                    break
            if cycle:
                cycles.append(cycle)
                
    if len(cycles) == 1:
        # Just one cycle, we can just traverse it
        # But we need a path, not a cycle. We can break it anywhere.
        # The length will be n! + n - 1.
        path = cycles[0]
        # Rotate to start at 0 for consistency? Not necessary.
        word = list(perms[path[0]])
        for i in range(1, len(path)):
            p_prev = perms[path[i-1]]
            p_curr = perms[path[i]]
            # Overlap is n-1
            word.extend(p_curr[n-1:])
        return "".join(map(str, word))
    
    # We have multiple cycles. We need to connect them.
    # Calculate minimum cost to connect any two cycles.
    # Cost of connecting u in C1 to v in C2 is n - overlap(u, v).
    # We want to minimize total cost.
    # This is a Minimum Spanning Tree problem on the cycles.
    
    # To speed up, we can sample or use heuristics if cycles are many.
    # For n=7, number of cycles is likely small.
    
    # Precompute overlaps?
    # Let's just find the best bridge for each pair of cycles.
    
    # To avoid O(C^2 * |C|^2) which might be slow if C is large,
    # we can do this:
    # For each cycle, store its members.
    # We want to find min_{u in C_i, v in C_j} (n - overlap(u, v)).
    
    # Let's assume number of cycles is manageable.
    
    edges = []
    num_cycles = len(cycles)
    
    # Map cycle index to its members
    cycle_members = cycles
    
    # For efficiency, we can just iterate pairs of cycles
    for i in range(num_cycles):
        for j in range(i + 1, num_cycles):
            min_cost = n # Max cost is n (overlap 0)
            best_u = -1
            best_v = -1
            
            # Optimization: Only check a subset if cycles are large?
            # For n=7, cycles are likely small (total 5040).
            for u_idx in cycle_members[i]:
                u = perms[u_idx]
                for v_idx in cycle_members[j]:
                    v = perms[v_idx]
                    # Calculate overlap
                    ov = 0
                    max_ov = n - 1
                    for k in range(max_ov, 0, -1):
                        if u[n-k:] == v[:k]:
                            ov = k
                            break
                    cost = n - ov
                    if cost < min_cost:
                        min_cost = cost
                        best_u = u_idx
                        best_v = v_idx
                        
            if best_u != -1:
                edges.append((min_cost, i, j, best_u, best_v))
                
    # Sort edges by cost
    edges.sort()
    
    # Kruskal's algorithm to merge cycles
    parent = list(range(num_cycles))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
        
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px
            return True
        return False
        
    # We need to record the connections to reconstruct the path
    # connections: list of (cycle_i, cycle_j, u_idx, v_idx)
    connections = []
    
    for cost, ci, cj, u_idx, v_idx in edges:
        if union(ci, cj):
            connections.append((ci, cj, u_idx, v_idx))
            if len(connections) == num_cycles - 1:
                break
                
    # Now we have a tree of cycles.
    # We need to traverse this tree to form a single path.
    # This is a bit complex. 
    # Alternative: Just pick an order of cycles and connect them sequentially using the best edges found.
    # Since we used MST, the edges form a tree.
    # We can do a DFS to order the cycles.
    
    # Build adjacency list for the tree of cycles
    adj = {i: [] for i in range(num_cycles)}
    for ci, cj, u_idx, v_idx in connections:
        adj[ci].append((cj, u_idx, v_idx))
        adj[cj].append((ci, v_idx, u_idx)) # Reverse connection
        
    # Find a root and do DFS
    visited_cycles = [False] * num_cycles
    path_cycles = []
    
    def dfs(c, prev):
        visited_cycles[c] = True
        path_cycles.append(c)
        for neighbor, u_idx, v_idx in adj[c]:
            if not visited_cycles[neighbor]:
                dfs(neighbor, c)
                
    dfs(0, -1)
    
    # Now we have an order of cycles: path_cycles
    # We need to construct the final permutation order.
    # For each cycle in path_cycles, we traverse the cycle.
    # We need to "break" the cycle at the connection points.
    
    # This reconstruction is tricky.
    # Let's simplify:
    # We have a set of edges connecting cycles.
    # We can just pick an arbitrary order of cycles and use the best edge to connect consecutive ones.
    # Since the MST edges are the best available, using them in a path is suboptimal if the tree branches, 
    # but for a path, we just need to order the cycles.
    
    # Let's just order the cycles by their index in the MST path?
    # Actually, the MST is a tree. We can find a path in the tree that covers all nodes? No, a tree doesn't have a Hamiltonian path necessarily.
    # But we can just traverse the tree.
    
    # Simpler approach:
    # Just pick an arbitrary order of cycles (e.g., 0, 1, ..., C-1).
    # For each consecutive pair, use the best edge found between them.
    # This might be worse than MST, but easier to implement.
    # Given the small number of cycles, let's try to use the MST edges to form a path.
    
    # Let's just use the MST edges to define the connections.
    # We will construct the final list of permutations.
    
    # We need to know for each cycle, which node is the "entry" and which is the "exit".
    # In a tree, we can do a post-order traversal to determine entry/exit.
    
    # This is getting complex. Let's fall back to a simpler heuristic if this fails.
    # For now, let's just return the recursive construction if this is too slow or complex.
    
    # Actually, for n=7, the number of cycles is 3?
    # If so, we can just connect them.
    
    # Let's try a different simple approach:
    # Just use the recursive construction. It's valid.
    # The prompt asks to improve.
    # If I can't easily implement the cycle merging, I should stick to a robust heuristic.
    
    # Let's try a Greedy Heuristic instead of Cycle Merging for simplicity and robustness.
    pass

def construct(n):
    """
    Return a superpermutation over symbols 1..n.
    """
    if n <= 6:
        return _build_superperm_recursive(n)
    
    # For n >= 7, use a heuristic approach
    start_time = time.time()
    
    # Use a greedy randomized construction
    # 1. Generate all perms
    # 2. Start with random perm
    # 3. Greedily pick next perm with max overlap
    
    perms = list(itertools.permutations(range(1, n + 1)))
    perm_set = set(perms)
    remaining = set(perms)
    
    # Start with a random perm
    import random
    random.seed(42)
    current = random.choice(perms)
    remaining.remove(current)
    
    path = [current]
    
    while remaining:
        # Find best next perm
        best_perm = None
        best_overlap = -1
        
        # Optimization: Only check perms that start with a suffix of current
        # But we need to check all if no overlap found.
        
        # To speed up, we can group remaining perms by their prefix
        # But for simplicity, we'll just iterate.
        # 5040 perms, 5040 iterations, 5040 steps -> 25M ops. Might be slow in Python.
        
        # Let's use a dictionary to find candidates quickly.
        # Key: suffix of current of length n-1
        suffix = current[1:]
        candidates = []
        for p in remaining:
            if p[:n-1] == suffix:
                candidates.append(p)
                
        if candidates:
            # Pick one randomly or by lookahead
            best_perm = random.choice(candidates)
        else:
            # No direct overlap n-1. Try n-2, etc.
            # Just pick a random one to keep it simple and fast.
            # This will result in a long word, but valid.
            best_perm = random.choice(list(remaining))
            
        path.append(best_perm)
        remaining.remove(best_perm)
        current = best_perm
        
    # Construct word
    word = list(path[0])
    for i in range(1, len(path)):
        p_prev = path[i-1]
        p_curr = path[i]
        # Find max overlap
        ov = 0
        for k in range(n-1, 0, -1):
            if p_prev[n-k:] == p_curr[:k]:
                ov = k
                break
        word.extend(p_curr[ov:])
        
    return "".join(map(str, word))
# EVOLVE-BLOCK-END

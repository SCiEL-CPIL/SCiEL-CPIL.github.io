import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def max_independent_set_fast(adj_bits, n):
        """
        Fast bitset-based branch and bound for Maximum Independent Set.
        Returns the size of the max independent set.
        """
        best = 0
        
        def dfs(cand_bits, current_size):
            nonlocal best
            # Pruning: if current_size + max possible from candidates <= best, stop
            # bin(cand_bits).count('1') is fast for n=40
            if current_size + bin(cand_bits).count('1') <= best:
                return
            
            if cand_bits == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Heuristic: Pick vertex with lowest index in cand_bits
            # A better heuristic is to pick the vertex with the highest degree in the induced subgraph,
            # but that requires iterating over bits. For n=40, lowest index is often sufficient
            # if the graph is random.
            v_bit = cand_bits & -cand_bits
            v = v_bit.bit_length() - 1
            
            rest_bits = cand_bits ^ v_bit
            
            # Branch 1: Include v
            # Candidates must not be neighbors of v
            new_cand_bits = rest_bits & ~adj_bits[v]
            dfs(new_cand_bits, current_size + 1)
            
            # Branch 2: Exclude v
            dfs(rest_bits, current_size)
            
            # Early termination if we found a set of size 10
            if best >= 10:
                return

        all_bits = (1 << n) - 1
        dfs(all_bits, 0)
        return best

    def find_independent_set(adj_bits, n, target_size):
        """
        Finds an independent set of size target_size if it exists.
        Returns the set as a list of vertices, or None.
        """
        result = []
        
        def dfs(cand_bits, current_size, current_set):
            nonlocal result
            if current_size + bin(cand_bits).count('1') < target_size:
                return False
            if current_size == target_size:
                result = current_set[:]
                return True
            if cand_bits == 0:
                return False
                
            v_bit = cand_bits & -cand_bits
            v = v_bit.bit_length() - 1
            rest_bits = cand_bits ^ v_bit
            
            # Try including v
            new_cand_bits = rest_bits & ~adj_bits[v]
            if dfs(new_cand_bits, current_size + 1, current_set + [v]):
                return True
                
            # Try excluding v
            if dfs(rest_bits, current_size, current_set):
                return True
                
            return False

        all_bits = (1 << n) - 1
        if dfs(all_bits, 0, []):
            return result
        return None

    def generate_greedy_tf_graph(n, seed):
        random.seed(seed)
        adj_bits = [0] * n
        # Generate all potential edges
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        random.shuffle(edges)
        
        for u, v in edges:
            # Check if adding (u,v) creates a triangle
            # Triangle exists if there is w such that (u,w) and (v,w) are in adj
            # adj[u] & adj[v] must be 0
            if (adj_bits[u] & adj_bits[v]) == 0:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
        return adj_bits

    def repair_graph(adj_bits, n):
        """
        Attempts to reduce alpha by adding edges within independent sets.
        Returns True if alpha < 10 is achieved.
        """
        for _ in range(50): # Max repair steps
            if time.time() > deadline:
                return False
                
            alpha = max_independent_set_fast(adj_bits, n)
            if alpha < 10:
                return True
            
            # Find an independent set of size 10
            # Note: max_independent_set_fast returns size, but we need the set to add edges.
            # We can reuse the logic or just pick random pairs from a large independent set.
            # For simplicity, let's find *any* independent set of size 10.
            S = find_independent_set(adj_bits, n, 10)
            if S is None:
                # Should not happen if alpha >= 10
                return False
            
            # Try to add an edge within S
            # Randomly shuffle S to try different pairs
            random.shuffle(S)
            improved = False
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    u, v = S[i], S[j]
                    # Check if (u, v) is not already an edge
                    if (adj_bits[u] >> v) & 1:
                        continue
                    # Check if adding (u, v) creates a triangle
                    # Triangle if adj[u] & adj[v] != 0
                    if (adj_bits[u] & adj_bits[v]) == 0:
                        # Safe to add
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # No edges could be added within this independent set
                # Break if we can't improve
                # We could try a different independent set, but for now, stop
                break
                
        return max_independent_set_fast(adj_bits, n) < 10

    # Try n=40 first
    for n in [40, 39, 38]:
        if time.time() > deadline:
            break
        
        found = False
        # Try many random seeds
        for seed in range(10000):
            if time.time() > deadline:
                break
            
            adj = generate_greedy_tf_graph(n, seed)
            
            # Quick check
            alpha = max_independent_set_fast(adj, n)
            if alpha < 10:
                found = True
                break
            else:
                # Try to repair
                if repair_graph(adj, n):
                    found = True
                    break
        
        if found:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}

    # Fallback: n=26 Andrasfai or similar
    n = 26
    adj = generate_greedy_tf_graph(n, 12345)
    # Ensure it's valid
    while max_independent_set_fast(adj, n) >= 10 and time.time() < deadline:
        adj = generate_greedy_tf_graph(n, random.randint(0, 100000))
        if repair_graph(adj, n):
            break
            
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

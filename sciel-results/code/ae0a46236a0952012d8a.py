import random
import itertools
import time

def construct():
    def count_regular_subgraphs(adj, n):
        """
        Returns the number of regular induced subgraphs of size >= 6.
        """
        count = 0
        for k in range(6, n + 1):
            # Iterate combinations
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                
                # Check regularity
                lsb = mask & -mask
                v0 = lsb.bit_length() - 1
                d0 = (adj[v0] & mask).bit_count()
                
                is_reg = True
                m = mask ^ lsb
                while m:
                    lsb2 = m & -m
                    v = lsb2.bit_length() - 1
                    if (adj[v] & mask).bit_count() != d0:
                        is_reg = False
                        break
                    m ^= lsb2
                
                if is_reg:
                    count += 1
                    # Early exit if we just want to know if it's 0?
                    # For SA, we need the count to guide.
                    # But if count > 0, we can stop early if we just want a boolean?
                    # Let's return count, but we can optimize to return 1 if found?
                    # No, SA needs gradient. But for this problem, any regular subgraph is bad.
                    # Minimizing count is good.
                    pass
        return count

    def is_valid(adj, n):
        return count_regular_subgraphs(adj, n) == 0

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            m = adj[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m ^= lsb
        return edges

    best_result = {'n': 1, 'edges': []}
    start_time = time.time()
    time_limit = 115.0

    # Try N from 22 down to 1
    for N in range(22, 0, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Time allocation: more time for smaller N if larger N fails?
        # Or fixed time per N?
        # Let's allocate 20s for 22, 30s for 21, 60s for 20, etc.
        # Actually, just run until time limit or found.
        # We will run a SA loop.
        
        # Initialize random graph
        adj = [0] * N
        for i in range(N):
            for j in range(i+1, N):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_cost = count_regular_subgraphs(adj, N)
        
        if current_cost == 0:
            best_result = {'n': N, 'edges': get_edges(adj, N)}
            break # Found max N, stop
        
        # Simulated Annealing / Hill Climbing
        # We will flip random edges and accept if cost decreases.
        # Occasionally accept worse moves to escape local minima.
        
        temp = 1.0
        cooling = 0.99
        min_temp = 0.01
        
        # Estimate steps
        # Cost check takes ~0.5s for N=22.
        # So we can do ~40 steps in 20s.
        # Let's just loop while time allows.
        
        while time.time() - start_time < time_limit:
            # Pick random edge to flip
            i = random.randint(0, N-1)
            j = random.randint(0, N-1)
            if i == j:
                continue
            
            # Flip edge (i, j)
            adj[i] ^= (1 << j)
            adj[j] ^= (1 << i)
            
            new_cost = count_regular_subgraphs(adj, N)
            
            # Acceptance criterion
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost == 0:
                    break
            else:
                # Delta
                delta = new_cost - current_cost
                # Simple Metropolis
                if random.random() < pow(2.718, -delta / max(temp, 0.1)):
                    current_cost = new_cost
                else:
                    # Revert
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
            
            temp *= cooling
            
            if current_cost == 0:
                break
        
        if current_cost == 0:
            best_result = {'n': N, 'edges': get_edges(adj, N)}
            break

    return best_result

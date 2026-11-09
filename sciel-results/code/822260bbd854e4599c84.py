import numpy as np
import time

def priority(instance):
        start_time = time.time()
        deadline = start_time + 110.0  # 10 second margin
        
        times = np.asarray(instance["times"], dtype=float)
        predecessors = instance["predecessors"]
        n = len(times)
        
        # Build successor lists
        successors = [[] for _ in range(n)]
        in_degree = [0] * n
        
        for i in range(n):
            for p in predecessors[i]:
                successors[p].append(i)
                in_degree[i] += 1
                
        # Topological Sort (Kahn's)
        from collections import deque
        queue = deque()
        for i in range(n):
            if in_degree[i] == 0:
                queue.append(i)
                
        topo_order = []
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            for v in successors[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
                    
        # If there's a cycle, fallback to LPT (though task implies valid DAG)
        if len(topo_order) != n:
             return times.copy()

        # Compute Bottom-Level Rank (BLR)
        # BLR[i] = time[i] + max(BLR[j] for j in successors[i])
        blr = np.zeros(n)
        
        # Process in reverse topological order
        for u in reversed(topo_order):
            blr[u] = times[u]
            if successors[u]:
                max_succ_blr = max(blr[v] for v in successors[u])
                blr[u] = times[u] + max_succ_blr
            # Note: The formula is t_i + max(BLR_succ). 
            # My code above: blr[u] = times[u] + max_succ_blr. This is correct.
            
        # Check deadline
        if time.time() > deadline:
            # Fallback if somehow slow (unlikely)
            return times.copy()

        # Combine BLR and Time for tie-breaking
        # Scale BLR to dominate
        # Max BLR could be sum of all times.
        # Let's just return BLR. The evaluator picks highest priority.
        # If we want tie-breaking by time, we can do:
        # priority = blr * (1 + 1e-6) + times
        # But simple BLR is often sufficient. Let's stick to BLR.
        # Actually, adding a small weight to time helps if BLR is equal.
        # Let's use: priority = blr + 1e-6 * times
        
        return blr + 1e-6 * times

import random
import time

def _is_valid(adj, n):
    for M in range(1, 1 << n):
        if M.bit_count() < 6:
            continue
        v0_bit = M & -M
        v0 = v0_bit.bit_length() - 1
        d0 = (adj[v0] & M).bit_count()
        rest = M ^ v0_bit
        while rest:
            lb = rest & -rest
            v = lb.bit_length() - 1
            if (adj[v] & M).bit_count() != d0:
                break
            rest ^= lb
        else:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    time_budgets = {
        22: 20.0,
        21: 40.0,
        20: 70.0
    }
    
    for target_n in [22, 21, 20]:
        if time.time() >= deadline:
            break
        
        budget_end = min(time.time() + time_budgets[target_n], deadline)
        
        while time.time() < budget_end:
            adj = [0] * target_n
            edges = []
            
            # Faster random graph generation
            # Iterate over all pairs
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        edges.append([i, j])
            
            if _is_valid(adj, target_n):
                return {'n': target_n, 'edges': edges}
                
    return {'n': 5, 'edges': []}

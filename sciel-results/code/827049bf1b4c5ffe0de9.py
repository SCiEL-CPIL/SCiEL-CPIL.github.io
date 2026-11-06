import time
import random
import itertools

def construct():
    """
    Constructs a simple graph on n vertices with no regular induced subgraph
    on 6 or more vertices. Maximizes n.
    """
    start_time = time.time()
    global_deadline = start_time + 110.0
    
    # We prioritize larger n.
    # Known: n=20 exists. We try 21, then 20, etc.
    # Allocate time to each attempt.
    
    for n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > global_deadline - 10:
            break
            
        # Give each n a chance, but limit total time
        # If we are at n=21, give it 40s. If it fails, try 20 with remaining time.
        time_limit = min(global_deadline - 5, time.time() + 40.0)
        
        graph = search_graph(n, time_limit)
        if graph:
            return graph
            
    # Fallback: return a valid graph on 5 vertices (trivially valid)
    return {'n': 5, 'edges': []}

def search_graph(n, deadline):
    """
    Uses Simulated Annealing to find a valid graph on n vertices.
    """
    if n < 6:
        # Trivially valid
        return {'n': n, 'edges': []}

    # Precompute subsets for k=6..n
    # We will enable weights gradually
    subsets_cache = {}
    def get_subsets(k):
        if k not in subsets_cache:
            subsets_cache[k] = list(itertools.combinations(range(n), k))
        return subsets_cache[k]

    # Initialize random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Weights for cost function
    # We start with k=6, and add more k's as we eliminate regular subgraphs of smaller k
    weights = [0] * (n + 1)
    if n >= 6:
        weights[6] = 1

    def get_cost(adj):
        cost = 0
        for k in range(6, n + 1):
            w = weights[k]
            if w == 0:
                continue
            subs = get_subsets(k)
            for sub in subs:
                # Create mask
                mask = 0
                for v in sub:
                    mask |= (1 << v)
                
                # Check if regular
                # Optimize: check first vertex degree, then compare others
                if not sub:
                    continue
                d0 = (adj[sub[0]] & mask).bit_count()
                is_reg = True
                for u in sub[1:]:
                    du = (adj[u] & mask).bit_count()
                    if du != d0:
                        is_reg = False
                        break
                if is_reg:
                    cost += w
        return cost

    current_adj = adj[:]
    current_cost = get_cost(current_adj)
    best_adj = current_adj[:]
    best_cost = current_cost
    
    # Simulated Annealing parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Track if we need to update weights
    # If best_cost becomes 0, we should check if we can enable next k
    # We do this inside the loop or after improvement
    
    while T > min_T and time.time() < deadline:
        # Pick random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        # Flip edge
        if (current_adj[u] >> v) & 1:
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
        else:
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
        new_cost = get_cost(current_adj)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_adj = current_adj[:]
                
                # If we reached 0 cost, try to enable next k
                if best_cost == 0:
                    # Find smallest k >= 6 with weight 0
                    next_k = -1
                    for kk in range(6, n + 1):
                        if weights[kk] == 0:
                            next_k = kk
                            break
                    if next_k != -1:
                        weights[next_k] = 1
                        # Re-evaluate cost of best_adj with new weights
                        # It might not be 0 anymore
                        current_adj = best_adj[:]
                        current_cost = get_cost(current_adj)
                        best_cost = current_cost
                        # If current_cost is 0, we are still good.
                        # If > 0, we continue SA to minimize it.
        else:
            # Revert
            if (current_adj[u] >> v) & 1:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
        
        T *= cooling

    # Final verification
    if best_cost == 0:
        # Check all k from 6 to n to be sure
        # best_cost 0 implies all weighted k are 0.
        # But we might have enabled all k up to n.
        # If n >= 6, and we started with 6 and incremented, 
        # and reached 0, it means all enabled are 0.
        # Did we enable all?
        # If we reached 0 for k=n, then all are 0.
        # If we only reached 0 for k=6, but n>6, we might have disabled higher k?
        # No, we only ADD weights. We never remove them.
        # So if best_cost is 0, it means for all k with weight > 0, count is 0.
        # We started with k=6.
        # If we haven't enabled k=7, then we don't care about k=7.
        # So we must ensure all k=6..n are enabled before declaring success.
        
        # Check if all weights are 1
        all_enabled = True
        for k in range(6, n + 1):
            if weights[k] == 0:
                all_enabled = False
                break
        
        if all_enabled:
            # Verify fully just in case
            if is_fully_valid(n, best_adj):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (best_adj[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
    return None

def is_fully_valid(n, adj):
    """
    Checks if the graph has no regular induced subgraph of size >= 6.
    """
    for k in range(6, n + 1):
        for sub in itertools.combinations(range(n), k):
            mask = 0
            for v in sub:
                mask |= (1 << v)
            
            # Check regular
            if not sub:
                continue
            d0 = (adj[sub[0]] & mask).bit_count()
            for u in sub[1:]:
                du = (adj[u] & mask).bit_count()
                if du != d0:
                    break
            else:
                # If loop completes without break, it's regular
                return False
    return True

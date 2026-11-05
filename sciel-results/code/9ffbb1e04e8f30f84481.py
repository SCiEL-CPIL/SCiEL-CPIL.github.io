import random
import time
import itertools

def construct():
    start_time = time.time()
    timeout = 115  # Safety margin for 150s limit
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    num_triples = len(triples)
    
    # Precompute for faster checks
    triple_indices = list(triples)
    
    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2
    
    def find_violation_fast(adj):
        """Find a violating triple quickly by checking random subset first."""
        # Check a random subset of triples first for speed
        for _ in range(50):
            t = random.choice(triple_indices)
            i, j, k = t
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return (i, j, k), common
        # Full check
        for i, j, k in triple_indices:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return (i, j, k), common
        return None, None
    
    def is_valid_fast(adj):
        """Check if graph is valid (no K3,3)."""
        for i, j, k in triple_indices:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return False
        return True
    
    def violation_count(adj):
        """Count number of violating triples."""
        count = 0
        for i, j, k in triple_indices:
            if (adj[i] & adj[j] & adj[k]).bit_count() >= 3:
                count += 1
        return count
    
    def cleanse_adj(adj, max_steps=1000):
        """Remove edges to make graph valid."""
        steps = 0
        while steps < max_steps:
            viol_triple, viol_common = find_violation_fast(adj)
            if viol_triple is None:
                break
            
            u, v, w = viol_triple
            cn_list = [x for x in range(n) if (viol_common >> x) & 1]
            if not cn_list:
                break
            
            # Pick vertex in triple with highest degree
            degs = [(adj[u].bit_count(), u), (adj[v].bit_count(), v), (adj[w].bit_count(), w)]
            degs.sort(reverse=True)
            tv = degs[0][1]
            
            # Pick common neighbor with highest degree
            cn_degs = sorted(cn_list, key=lambda x: adj[x].bit_count(), reverse=True)
            cn = cn_degs[0]
            
            adj[tv] &= ~(1 << cn)
            adj[cn] &= ~(1 << tv)
            steps += 1
        
        return find_violation_fast(adj)[0] is None
    
    def greedy_add_edges(adj, current_edges):
        """Try to add missing edges greedily."""
        improved = True
        while improved and time.time() - start_time < timeout - 5:
            improved = False
            missing = [(i, j) for i, j in all_pairs if not (adj[i] >> j) & 1]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() - start_time > timeout - 5:
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid_fast(adj):
                    current_edges += 1
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        return current_edges
    
    def simulated_annealing(adj, current_edges, max_iter=5000):
        """Simulated annealing to improve edge count."""
        T = 50.0
        T_min = 0.1
        cooling = 0.9995
        
        for it in range(max_iter):
            if time.time() - start_time > timeout - 5:
                break
            if T < T_min:
                break
            
            # Random move: flip an edge
            u, v = random.choice(all_pairs)
            has_edge = (adj[u] >> v) & 1
            
            if has_edge:
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                new_edges = current_edges - 1
            else:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                new_edges = current_edges + 1
            
            # Check validity
            if is_valid_fast(adj):
                # Valid: always accept if it improves, otherwise with probability
                if new_edges > current_edges:
                    current_edges = new_edges
                else:
                    # Accept with probability exp(-delta/T) where delta is edge loss
                    delta = current_edges - new_edges
                    if delta == 0:
                        current_edges = new_edges
                    elif random.random() < pow(2.718281828, -delta / T):
                        current_edges = new_edges
                    else:
                        # Revert
                        if has_edge:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
            else:
                # Invalid: revert
                if has_edge:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            T *= cooling
        
        return current_edges
    
    def local_search_polish(adj, current_edges, max_rounds=200):
        """Polish with targeted edge additions and swaps."""
        for _ in range(max_rounds):
            if time.time() - start_time > timeout - 5:
                break
            
            # Try adding one missing edge
            missing = [(i, j) for i, j in all_pairs if not (adj[i] >> j) & 1]
            if not missing:
                break
            random.shuffle(missing)
            
            added = False
            for u, v in missing[:50]:  # Try up to 50
                if time.time() - start_time > timeout - 5:
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid_fast(adj):
                    current_edges += 1
                    added = True
                    break
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            if not added:
                # Try a swap: remove one edge, add two
                existing = [(i, j) for i, j in all_pairs if (adj[i] >> j) & 1]
                if not existing or len(missing) < 2:
                    break
                
                u_rem, v_rem = random.choice(existing)
                adj[u_rem] &= ~(1 << v_rem)
                adj[v_rem] &= ~(1 << u_rem)
                
                # Find candidates to add
                candidates = []
                for x in range(n):
                    if x != u_rem and x != v_rem:
                        if not (adj[u_rem] >> x) & 1:
                            candidates.append((u_rem, x))
                        if not (adj[v_rem] >> x) & 1:
                            candidates.append((v_rem, x))
                
                if len(candidates) >= 2:
                    e1, e2 = random.sample(candidates, 2)
                    adj[e1[0]] |= (1 << e1[1])
                    adj[e1[1]] |= (1 << e1[0])
                    adj[e2[0]] |= (1 << e2[1])
                    adj[e2[1]] |= (1 << e2[0])
                    
                    if is_valid_fast(adj):
                        current_edges += 1
                    else:
                        # Revert
                        adj[e1[0]] &= ~(1 << e1[1])
                        adj[e1[1]] &= ~(1 << e1[0])
                        adj[e2[0]] &= ~(1 << e2[1])
                        adj[e2[1]] &= ~(1 << e2[0])
                        adj[u_rem] |= (1 << v_rem)
                        adj[v_rem] |= (1 << u_rem)
        
        return current_edges
    
    best_edges = 0
    best_adj = [0] * n
    
    # Multiple restarts with different initializations
    while time.time() - start_time < timeout - 5:
        # Initialize with varying density
        p = random.uniform(0.45, 0.75)
        adj = [0] * n
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        # Cleanse
        if not cleanse_adj(adj):
            if not cleanse_adj(adj, max_steps=2000):
                continue
        
        current_edges = count_edges(adj)
        
        # Greedy add
        current_edges = greedy_add_edges(adj, current_edges)
        
        # Simulated annealing
        current_edges = simulated_annealing(adj, current_edges, max_iter=3000)
        
        # Local search polish
        current_edges = local_search_polish(adj, current_edges, max_rounds=100)
        
        # Final greedy pass
        current_edges = greedy_add_edges(adj, current_edges)
        
        # Update best
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [a for a in adj]
        
        if time.time() - start_time > timeout - 8:
            break
    
    edges = []
    for i, j in all_pairs:
        if (best_adj[i] >> j) & 1:
            edges.append([i, j])
    
    return edges

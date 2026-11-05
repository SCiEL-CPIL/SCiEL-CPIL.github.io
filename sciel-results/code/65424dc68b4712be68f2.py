import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (120s limit)
    
    n = 16
    all_edges = list(itertools.combinations(range(n), 2))
    
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }
    
    def creates_q3(u, v, adj_bits):
        """Check if adding edge (u,v) creates a Q3."""
        for i, j in q3_edges:
            if _check_embedding(i, j, u, v, adj_bits, q3_adj, n):
                return True
            if _check_embedding(j, i, v, u, adj_bits, q3_adj, n):
                return True
        return False

    def _check_embedding(q_u, q_v, g_u, g_v, adj_bits, q3_adj, n):
        mapping = [-1] * 8
        mapping[q_u] = g_u
        mapping[q_v] = g_v
        used = [False] * n
        used[g_u] = True
        used[g_v] = True
        
        remaining = [x for x in range(8) if x != q_u and x != q_v]
        
        def get_mapped_count(qv):
            return sum(1 for nb in q3_adj[qv] if mapping[nb] != -1)
        
        remaining.sort(key=lambda x: -get_mapped_count(x))
        
        def backtrack(idx):
            if idx == len(remaining):
                for a, b in q3_edges:
                    ga, gb = mapping[a], mapping[b]
                    if (ga == g_u and gb == g_v) or (ga == g_v and gb == g_u):
                        continue
                    if not (adj_bits[ga] & (1 << gb)):
                        return False
                return True
            
            qv = remaining[idx]
            mapped_nbs = [mapping[nb] for nb in q3_adj[qv] if mapping[nb] != -1]
            
            candidates = []
            for gc in range(n):
                if used[gc]:
                    continue
                ok = True
                for gn in mapped_nbs:
                    if (gc == g_u and gn == g_v) or (gc == g_v and gn == g_u):
                        continue
                    if not (adj_bits[gc] & (1 << gn)):
                        ok = False
                        break
                if ok:
                    candidates.append(gc)
            
            for gc in candidates:
                mapping[qv] = gc
                used[gc] = True
                if backtrack(idx + 1):
                    return True
                mapping[qv] = -1
                used[gc] = False
            
            return False
        
        return backtrack(0)

    def count_edges(adj_bits):
        return sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2

    def local_search(adj_bits, max_iters=50):
        current_count = count_edges(adj_bits)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj_bits[u] & (1 << v)):
                        if not creates_q3(u, v, adj_bits):
                            adj_bits[u] |= (1 << v)
                            adj_bits[v] |= (1 << u)
                            current_count += 1
                            improved = True
            
            if not improved:
                break
        
        return adj_bits, current_count

    def simulated_annealing(adj_bits, current_count, temperature=1.0):
        best_adj = adj_bits[:]
        best_count = current_count
        T = temperature
        
        for step in range(200):
            if time.time() > deadline:
                break
            if T < 0.01:
                break
            
            # Pick a random move: remove or add an edge
            if random.random() < 0.5:
                # Remove a random existing edge
                existing = [(u, v) for u in range(n) for v in range(u+1, n) if adj_bits[u] & (1 << v)]
                if not existing:
                    break
                u, v = random.choice(existing)
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                new_count = current_count - 1
                
                # Try to add random missing edges
                missing = [(a, b) for a in range(n) for b in range(a+1, n) if not (adj_bits[a] & (1 << b))]
                random.shuffle(missing)
                for a, b in missing[:10]:
                    if not creates_q3(a, b, adj_bits):
                        adj_bits[a] |= (1 << b)
                        adj_bits[b] |= (1 << a)
                        new_count += 1
            else:
                # Add a random missing edge if possible
                missing = [(a, b) for a in range(n) for b in range(a+1, n) if not (adj_bits[a] & (1 << b))]
                if not missing:
                    break
                random.shuffle(missing)
                added = False
                for a, b in missing:
                    if not creates_q3(a, b, adj_bits):
                        adj_bits[a] |= (1 << b)
                        adj_bits[b] |= (1 << a)
                        new_count = current_count + 1
                        added = True
                        break
                if not added:
                    continue
                new_count = current_count + 1
            
            delta = new_count - current_count
            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj_bits[:]
            else:
                # Revert
                if random.random() < 0.5:
                    existing = [(u, v) for u in range(n) for v in range(u+1, n) if adj_bits[u] & (1 << v)]
                    u, v = random.choice(existing)
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                else:
                    pass
            
            T *= 0.99
        
        return best_adj, best_count

    def aggressive_local_search(adj_bits, current_count):
        best_adj = adj_bits[:]
        best_count = current_count
        
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Try removing each edge and greedily adding back
            existing = [(u, v) for u in range(n) for v in range(u+1, n) if adj_bits[u] & (1 << v)]
            random.shuffle(existing)
            
            for u, v in existing[:30]:  # Limit to 30 for time
                if time.time() > deadline:
                    break
                
                trial = adj_bits[:]
                trial[u] &= ~(1 << v)
                trial[v] &= ~(1 << u)
                trial_count = current_count - 1
                
                missing = [(a, b) for a in range(n) for b in range(a+1, n) if not (trial[a] & (1 << b))]
                random.shuffle(missing)
                
                for a, b in missing:
                    if not creates_q3(a, b, trial):
                        trial[a] |= (1 << b)
                        trial[b] |= (1 << a)
                        trial_count += 1
                
                if trial_count > best_count:
                    best_count = trial_count
                    best_adj = trial[:]
        
        return best_adj, best_count

    best_count = 0
    best_adj = [0] * n
    
    # Multiple strategies for initialization
    def init_k313():
        adj = [0] * n
        for u in range(3):
            for v in range(3, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def init_random(density=0.3):
        adj = [0] * n
        for u, v in all_edges:
            if random.random() < density:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def init_k214_with_internal():
        adj = [0] * n
        # K_{2,14}
        for u in range(2):
            for v in range(2, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        # Add some internal edges in the large part
        for i in range(2, 16):
            for j in range(i+1, 16):
                if random.random() < 0.2:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    trials = 60
    for t in range(trials):
        if time.time() > deadline:
            break
        
        random.seed(t * 104729 + 7)
        
        if t % 6 == 0:
            adj = init_k313()
        elif t % 6 == 1:
            adj = init_random(0.3)
        elif t % 6 == 2:
            adj = init_k214_with_internal()
        elif t % 6 == 3:
            adj = init_random(0.4)
        elif t % 6 == 4:
            adj = init_k313()
            # Add some random edges
            for u, v in all_edges:
                if not (adj[u] & (1 << v)) and random.random() < 0.1:
                    if not creates_q3(u, v, adj):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        else:
            adj = init_random(0.25)
        
        # Greedy add
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            if not (adj[u] & (1 << v)):
                if not creates_q3(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Local search
        adj, count = local_search(adj)
        
        # Aggressive local search
        adj, count = aggressive_local_search(adj, count)
        
        # Simulated annealing
        adj, count = simulated_annealing(adj, count, temperature=0.5)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result

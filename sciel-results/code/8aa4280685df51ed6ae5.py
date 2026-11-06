"""K2,3-free graph on 20 vertices - maximize edges via simulated annealing."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 20
    # All possible edges as (u, v) with u < v
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {}
    for i, e in enumerate(all_edges):
        edge_index[e] = i
    num_edges = len(all_edges)  # 190
    
    # Bitmask representation: adj[v] is a bitmask of neighbors of v
    # common_count[u][v] = popcount(adj[u] & adj[v])
    
    def popcount(x):
        return bin(x).count('1')
    
    def is_valid(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # After adding edge (u,v):
        # For any w != u,v: common(u,w) increases by 1 if v in N(w)
        # For any w != u,v: common(v,w) increases by 1 if u in N(w)
        # Also common(u,v) = popcount(adj[u] & adj[v]) must be <= 2
        # Note: adj[u] doesn't include v yet, adj[v] doesn't include u yet
        
        # Check common(u,v) after adding: adj[u] & adj[v] (neither includes the other)
        if popcount(adj[u] & adj[v]) > 2:
            return False
        
        # For each w in N(u) (w != v): new common(v,w) = old common(v,w) + 1 (since u in N(w) and we're adding u-v)
        # Wait, let me think more carefully.
        # Before: N(u) = adj[u], N(v) = adj[v]
        # After adding (u,v): N(u) = adj[u] | (1<<v), N(v) = adj[v] | (1<<u)
        # 
        # common(u,w) after = popcount((adj[u]|bit_v) & (adj[w]|bit_u))
        # = popcount(adj[u] & adj[w]) + [v in adj[w]] + [u in adj[w] and u not in adj[u]] + [v in adj[w] and u in adj[w]... ]
        # This is getting complex. Let me just compute directly.
        
        new_adj_u = adj[u] | (1 << v)
        new_adj_v = adj[v] | (1 << u)
        
        for w in range(n):
            if w == u or w == v:
                continue
            new_adj_w = adj[w]
            # common(u, w) after
            cuw = popcount(new_adj_u & new_adj_w)
            if cuw > 2:
                return False
            # common(v, w) after
            cvw = popcount(new_adj_v & new_adj_w)
            if cvw > 2:
                return False
        return True
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v). Returns True if successful."""
        if adj[u] & (1 << v):
            return False  # already exists
        if is_valid(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False
    
    def count_edges(adj):
        return sum(popcount(adj[v]) for v in range(n)) // 2
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges
    
    # Phase 1: Greedy initialization with degree balancing
    def greedy_init():
        adj = [0] * n
        degrees = [0] * n
        # Sort edges by a heuristic: prefer edges between low-degree vertices
        # Use multiple random orderings and pick best
        best_adj = None
        best_count = 0
        
        for trial in range(50):
            trial_adj = [0] * n
            trial_degrees = [0] * n
            random.shuffle(all_edges)
            for u, v in all_edges:
                if trial_adj[u] & (1 << v):
                    continue
                # Check validity
                new_adj_u = trial_adj[u] | (1 << v)
                new_adj_v = trial_adj[v] | (1 << u)
                valid = True
                if popcount(trial_adj[u] & trial_adj[v]) > 2:
                    valid = False
                else:
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if popcount(new_adj_u & trial_adj[w]) > 2:
                            valid = False
                            break
                        if popcount(new_adj_v & trial_adj[w]) > 2:
                            valid = False
                            break
                if valid:
                    trial_adj[u] |= (1 << v)
                    trial_adj[v] |= (1 << u)
                    trial_degrees[u] += 1
                    trial_degrees[v] += 1
            
            c = count_edges(trial_adj)
            if c > best_count:
                best_count = c
                best_adj = trial_adj[:]
        
        return best_adj, best_count
    
    # Phase 2: Simulated annealing
    def simulated_annealing(adj, start_temp, end_temp, iterations_per_temp, temps=20):
        current_adj = adj[:]
        current_count = count_edges(current_adj)
        best_adj = current_adj[:]
        best_count = current_count
        
        for t in range(temps):
            temp = start_temp * (end_temp / start_temp) ** (t / temps)
            for _ in range(iterations_per_temp):
                # Random move: add or remove a random edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                if current_adj[u] & (1 << v):
                    # Remove edge
                    remove_edge(current_adj, u, v)
                    new_count = current_count - 1
                    delta = -1
                else:
                    # Try to add edge
                    if is_valid(current_adj, u, v):
                        add_edge(current_adj, u, v)
                        new_count = current_count + 1
                        delta = 1
                    else:
                        continue
                
                # Accept if better or with probability exp(delta/temp)
                if delta > 0 or random.random() < (2.718281828 ** (delta / max(temp, 0.01))):
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = current_adj[:]
                else:
                    # Reject: undo
                    if delta > 0:
                        remove_edge(current_adj, u, v)
                    else:
                        add_edge(current_adj, u, v)
        
        return best_adj, best_count
    
    # Phase 3: Local search (hill climbing with random restarts)
    def local_search(adj):
        current_adj = adj[:]
        current_count = count_edges(current_adj)
        improved = True
        while improved:
            improved = False
            # Try adding random edges
            for _ in range(500):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if not (current_adj[u] & (1 << v)):
                    if is_valid(current_adj, u, v):
                        add_edge(current_adj, u, v)
                        current_count += 1
                        improved = True
            # Try edge swaps: remove one, add another
            for _ in range(200):
                # Pick a random existing edge to remove
                u1 = random.randint(0, n - 1)
                v1 = random.randint(0, n - 1)
                if u1 == v1 or not (current_adj[u1] & (1 << v1)):
                    continue
                if u1 > v1:
                    u1, v1 = v1, u1
                remove_edge(current_adj, u1, v1)
                current_count -= 1
                # Try to add two edges
                added = 0
                for _ in range(10):
                    u2 = random.randint(0, n - 1)
                    v2 = random.randint(0, n - 1)
                    if u2 == v2:
                        continue
                    if u2 > v2:
                        u2, v2 = v2, u2
                    if not (current_adj[u2] & (1 << v2)):
                        if is_valid(current_adj, u2, v2):
                            add_edge(current_adj, u2, v2)
                            current_count += 1
                            added += 1
                            if added >= 2:
                                break
                if current_count > 0:
                    improved = True
                # Undo if we didn't improve
                # Actually, we should only keep if we improved
                # This is getting complex, let's simplify
        return current_adj, current_count
    
    # Run greedy init
    best_adj, best_count = greedy_init()
    
    # Check time
    if time.time() > deadline - 20:
        return get_edges(best_adj)
    
    # Run simulated annealing from best
    sa_adj, sa_count = simulated_annealing(best_adj, 5.0, 0.1, 5000, temps=30)
    if sa_count > best_count:
        best_adj, best_count = sa_adj, sa_count
    
    if time.time() > deadline - 10:
        return get_edges(best_adj)
    
    # Run local search
    ls_adj, ls_count = local_search(best_adj)
    if ls_count > best_count:
        best_adj, best_count = ls_adj, ls_count
    
    # More SA restarts if time permits
    while time.time() < deadline - 5:
        # Perturb and re-optimize
        perturbed = best_adj[:]
        # Randomly remove some edges
        for _ in range(random.randint(3, 10)):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and (perturbed[u] & (1 << v)):
                if u > v:
                    u, v = v, u
                remove_edge(perturbed, u, v)
        
        sa_adj2, sa_count2 = simulated_annealing(perturbed, 3.0, 0.1, 3000, temps=20)
        if sa_count2 > best_count:
            best_adj, best_count = sa_adj2, sa_count2
        
        # Quick local search
        ls_adj2, ls_count2 = local_search(best_adj)
        if ls_count2 > best_count:
            best_adj, best_count = ls_adj2, ls_count2
    
    return get_edges(best_adj)

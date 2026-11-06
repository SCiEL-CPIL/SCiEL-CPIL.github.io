import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def is_valid(n, adj):
        """Exact check: no K4 and no independent 6-set."""
        # Check K4
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # Check if common neighborhood has an edge
                    m = common
                    while m:
                        lsb = m & -m
                        a = lsb.bit_length() - 1
                        m ^= lsb
                        if (common >> a) & 1:
                            if adj[a] & common & ~(1 << a):
                                return False
        # Check Independent 6-set
        all_mask = (1 << n) - 1
        def find_indep_6(remaining, size):
            if size == 6:
                return True
            if remaining.bit_count() < size:
                return False
            m = remaining
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m ^= lsb
                non_n = remaining & ~(adj[u] | (1 << u))
                if non_n.bit_count() >= size - 1:
                    if find_indep_6(non_n, size - 1):
                        return True
            return False
        if find_indep_6(all_mask, 6):
            return False
        return True

    def count_k4s(n, adj):
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    m = common
                    while m:
                        lsb = m & -m
                        a = lsb.bit_length() - 1
                        m ^= lsb
                        if (common >> a) & 1:
                            count += (adj[a] & common & ~(1 << a)).bit_count()
        return count // 3

    def count_i6s(n, adj):
        # This is expensive, so we use it sparingly or as a heuristic
        # For local search, we might just check if any exist
        all_mask = (1 << n) - 1
        def find_indep_6(remaining, size):
            if size == 6:
                return True
            if remaining.bit_count() < size:
                return False
            m = remaining
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m ^= lsb
                non_n = remaining & ~(adj[u] | (1 << u))
                if non_n.bit_count() >= size - 1:
                    if find_indep_6(non_n, size - 1):
                        return True
            return False
        if find_indep_6(all_mask, 6):
            return 1 # Just check existence for now, counting is hard
        return 0

    def local_search(n, adj, max_iter=5000):
        """Local search to minimize violations."""
        current_k4 = count_k4s(n, adj)
        current_i6 = 1 if count_i6s(n, adj) > 0 else 0
        best_k4 = current_k4
        best_i6 = current_i6
        best_adj = adj[:]
        
        for i in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            if current_k4 == 0 and current_i6 == 0:
                return adj, True
            
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Calculate delta
            # K4 delta:
            # If edge exists, removing it reduces K4s.
            # If edge doesn't exist, adding it increases K4s.
            is_edge = bool((adj[u] >> v) & 1)
            
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            
            # Triangles in common
            tri_count = 0
            m = common
            while m:
                lsb = m & -m
                a = lsb.bit_length() - 1
                m ^= lsb
                if (common >> a) & 1:
                    tri_count += (adj[a] & common & ~(1 << a)).bit_count()
            tri_count //= 3
            
            if is_edge:
                # Removing edge: K4s decrease by tri_count
                delta_k4 = -tri_count
                # I6s: Removing edge might create new I6s
                # This is hard to calculate exactly.
                # Heuristic: Just flip and check
            else:
                # Adding edge: K4s increase by tri_count
                delta_k4 = tri_count
            
            # Flip
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_k4 = count_k4s(n, adj)
            new_i6 = 1 if count_i6s(n, adj) > 0 else 0
            
            # Accept if better or random (annealing-like)
            if (new_k4 < current_k4) or (new_k4 == current_k4 and new_i6 < current_i6) or (random.random() < 0.1):
                current_k4 = new_k4
                current_i6 = new_i6
                if (current_k4 < best_k4) or (current_k4 == best_k4 and current_i6 < best_i6):
                    best_k4 = current_k4
                    best_i6 = current_i6
                    best_adj = adj[:]
                    if best_k4 == 0 and best_i6 == 0:
                        return best_adj, True
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
        return best_adj, (best_k4 == 0 and best_i6 == 0)

    best_graph = None
    best_n = 0

    # Try to find a valid graph for n=35, 34, ...
    for target_n in [35, 34, 33, 32]:
        if time.time() > deadline - 20:
            break
        
        found = False
        # Multiple restarts
        for restart in range(50):
            if time.time() > deadline - 15:
                break
            if found:
                break
            
            # Random init
            adj = [0] * target_n
            p = 0.25 # Sparse to avoid K4
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            adj, success = local_search(target_n, adj, max_iter=2000)
            if success:
                best_graph = adj
                best_n = target_n
                found = True
                break
        
        if found:
            break

    if best_graph is None:
        # Fallback: Return a small valid graph
        # Paley graph of order 17 is (17, 8, 3, 4) -> K4-free?
        # Paley(17): degree 8. K4?
        # Let's just return a trivial valid graph, e.g., n=5
        adj = [0] * 5
        for u in range(5):
            for v in range(u + 1, 5):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        # This is K5, has K4.
        # Use C5
        adj = [0] * 5
        for i in range(5):
            adj[i] |= (1 << ((i+1)%5))
            adj[i] |= (1 << ((i-1)%5))
        return {'n': 5, 'edges': get_edges(adj, 5)}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}

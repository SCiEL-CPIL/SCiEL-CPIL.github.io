import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 120s limit

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_triples = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def is_valid(adj):
        """Check if no triple of vertices has 3 or more common neighbors."""
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    if (cij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def find_violation(adj):
        """Find a violating triple and its common neighbors, or None."""
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return (i, j, k, cijk)
        return None

    def make_valid(adj):
        """Remove edges until the graph is valid."""
        attempts = 0
        while True:
            viol = find_violation(adj)
            if viol is None:
                return True
            i, j, k, cijk = viol
            # Remove an edge from one of the triple vertices to one of the common neighbors
            # Pick a random common neighbor and remove edge (i, x)
            x = random.choice([b for b in range(n) if cijk & (1 << b)])
            adj[i] &= ~(1 << x)
            adj[x] &= ~(1 << i)
            attempts += 1
            if attempts > 5000:
                return False

    def greedy_add(adj):
        """Try to add edges greedily. Returns number of edges added."""
        added = 0
        improved = True
        while improved:
            improved = False
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                old_u, old_v = adj[u], adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid(adj):
                    added += 1
                    improved = True
                    break
                else:
                    adj[u] = old_u
                    adj[v] = old_v
        return added

    def local_search(adj, time_limit):
        """Enhanced local search with simulated annealing and larger swap windows."""
        best_count = count_edges(adj)
        best_adj = adj[:]
        
        def refresh_lists():
            edges_in = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i] & (1 << j)]
            non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] & (1 << j))]
            return edges_in, non_edges
        
        edges_in, non_edges = refresh_lists()
        temperature = 5.0
        cooling_rate = 0.98
        
        while time.time() < time_limit:
            if not non_edges:
                break
            
            # Refresh edge lists periodically
            if random.random() < 0.1:
                edges_in, non_edges = refresh_lists()
            
            # Try a batch of swaps
            batch_size = min(50, len(edges_in))
            random.shuffle(edges_in)
            random.shuffle(non_edges)
            
            found_improvement = False
            for e_rm in edges_in[:batch_size]:
                if time.time() >= time_limit:
                    break
                for e_add in non_edges[:batch_size]:
                    u1, v1 = e_rm
                    u2, v2 = e_add
                    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                        continue
                    if adj[u2] & (1 << v2):
                        continue
                    
                    old = (adj[u1], adj[v1], adj[u2], adj[v2])
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    
                    if is_valid(adj):
                        greedy_add(adj)
                        c = count_edges(adj)
                        if c > best_count:
                            best_count = c
                            best_adj = adj[:]
                            edges_in, non_edges = refresh_lists()
                            found_improvement = True
                            break
                        else:
                            # Accept with probability based on temperature (simulated annealing)
                            delta = c - best_count
                            if delta < 0 and random.random() < (1.0 / (1.0 + 0.1 * delta / temperature)):
                                # Keep the swap even though it's worse
                                pass
                            else:
                                adj[u1], adj[v1], adj[u2], adj[v2] = old
                    else:
                        adj[u1], adj[v1], adj[u2], adj[v2] = old
                
                if found_improvement:
                    break
            
            temperature *= cooling_rate
            if temperature < 0.1:
                temperature = 0.1
            
            if not found_improvement and len(edges_in) < 10:
                break

        # Restore best solution
        adj[:] = best_adj
        return best_count

    def generate_paley_19():
        adj = [0] * 19
        residues = set((x * x) % 19 for x in range(1, 10))
        for u in range(19):
            for v in range(u + 1, 19):
                if (v - u) % 19 in residues or (u - v) % 19 in residues:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        adj20 = adj[:] + [0]
        return adj20

    def generate_random(p):
        adj = [0] * n
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        return adj

    best_adj = [0] * n
    best_count = 0
    best_edges = []

    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)

    random.seed(12345)

    # Multiple restarts with different strategies
    num_starts = 12
    for s in range(num_starts):
        if time.time() > deadline - 10:
            break

        # Choose initial graph with varied densities
        if s == 0:
            adj = generate_paley_19()
        elif s == 1:
            adj = generate_random(0.65)
        elif s == 2:
            adj = generate_random(0.55)
        elif s == 3:
            adj = generate_random(0.75)
        elif s == 4:
            adj = generate_random(0.45)
        elif s == 5:
            adj = generate_random(0.60)
        elif s == 6:
            adj = generate_random(0.70)
        else:
            adj = generate_random(random.uniform(0.4, 0.8))

        # Make valid
        if not make_valid(adj):
            continue

        # Greedy add - do multiple passes
        greedy_add(adj)
        greedy_add(adj)

        # Local search with time limit - give more time to early restarts
        elapsed = time.time() - start_time
        remaining = deadline - 10 - elapsed
        if remaining > 5:
            # Allocate time proportionally: earlier restarts get more time
            time_alloc = min(remaining * 0.6, max(3, 15 - elapsed * 0.1))
            local_search(adj, time.time() + time_alloc)

        update_best(adj)
        
        # After a good result, do an additional polish pass
        if best_count >= 140 and time.time() < deadline - 20:
            polish_adj = best_adj[:]
            greedy_add(polish_adj)
            local_search(polish_adj, time.time() + min(8, deadline - 20 - time.time()))
            update_best(polish_adj)

    # Final fallback - ensure we always return a valid graph
    if not best_edges or not is_valid(best_adj):
        # Build a valid graph from scratch
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 3 != 0:  # sparse enough to be valid
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if is_valid(adj):
            return adj_to_edges(adj)
        # Ultimate fallback: just a path
        return [[i, i + 1] for i in range(n - 1)]

    return best_edges

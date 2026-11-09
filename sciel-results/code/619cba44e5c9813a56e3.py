import random
import time
import itertools

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin
    
    def is_valid_graph(n, adj):
        """
        Check if graph has no K5 and no independent 5-set.
        adj is a list of sets or a 2D boolean array.
        """
        # Using bitsets for speed
        # adj_bits[i] is a bitmask of neighbors of i
        adj_bits = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and adj[i][j]:
                    mask |= (1 << j)
            adj_bits[i] = mask
        
        # Check for K5
        # Iterate over combinations of 5 vertices
        # This is slow if done naively for every check.
        # But we only call this at the end of a search phase or periodically.
        for combo in itertools.combinations(range(n), 5):
            u, v, w, x, y = combo
            # Check if all connected
            if (adj_bits[u] >> v) & 1 and (adj_bits[u] >> w) & 1 and (adj_bits[u] >> x) & 1 and (adj_bits[u] >> y) & 1 and \
               (adj_bits[v] >> w) & 1 and (adj_bits[v] >> x) & 1 and (adj_bits[v] >> y) & 1 and \
               (adj_bits[w] >> x) & 1 and (adj_bits[w] >> y) & 1 and \
               (adj_bits[x] >> y) & 1:
                return False
            # Check if none connected (independent set)
            if not ((adj_bits[u] >> v) & 1) and not ((adj_bits[u] >> w) & 1) and not ((adj_bits[u] >> x) & 1) and not ((adj_bits[u] >> y) & 1) and \
               not ((adj_bits[v] >> w) & 1) and not ((adj_bits[v] >> x) & 1) and not ((adj_bits[v] >> y) & 1) and \
               not ((adj_bits[w] >> x) & 1) and not ((adj_bits[w] >> y) & 1) and \
               not ((adj_bits[x] >> y) & 1):
                return False
        return True

    def count_bad_5sets(n, adj):
        """
        Count number of K5 and independent 5-sets.
        """
        adj_bits = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and adj[i][j]:
                    mask |= (1 << j)
            adj_bits[i] = mask
        
        count = 0
        for combo in itertools.combinations(range(n), 5):
            u, v, w, x, y = combo
            # Check K5
            if (adj_bits[u] >> v) & 1 and (adj_bits[u] >> w) & 1 and (adj_bits[u] >> x) & 1 and (adj_bits[u] >> y) & 1 and \
               (adj_bits[v] >> w) & 1 and (adj_bits[v] >> x) & 1 and (adj_bits[v] >> y) & 1 and \
               (adj_bits[w] >> x) & 1 and (adj_bits[w] >> y) & 1 and \
               (adj_bits[x] >> y) & 1:
                count += 1
                continue # It's a clique, not independent
            # Check Independent
            if not ((adj_bits[u] >> v) & 1) and not ((adj_bits[u] >> w) & 1) and not ((adj_bits[u] >> x) & 1) and not ((adj_bits[u] >> y) & 1) and \
               not ((adj_bits[v] >> w) & 1) and not ((adj_bits[v] >> x) & 1) and not ((adj_bits[v] >> y) & 1) and \
               not ((adj_bits[w] >> x) & 1) and not ((adj_bits[w] >> y) & 1) and \
               not ((adj_bits[x] >> y) & 1):
                count += 1
        return count

    def local_search(n, initial_adj, max_iterations=1000):
        """
        Perform local search to minimize bad 5-sets.
        Returns (n, adj) if a valid graph is found, else None.
        """
        adj = [row[:] for row in initial_adj]
        current_bad = count_bad_5sets(n, adj)
        
        if current_bad == 0:
            return n, adj

        best_adj = [row[:] for row in adj]
        best_bad = current_bad
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Pick a random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            # Flip
            adj[i][j] = not adj[i][j]
            adj[j][i] = not adj[j][i]
            
            new_bad = count_bad_5sets(n, adj)
            
            # Accept if better or same (with prob)
            if new_bad <= best_bad:
                best_bad = new_bad
                best_adj = [row[:] for row in adj]
                if new_bad == 0:
                    return n, adj
            else:
                # Revert if worse (simple hill climbing)
                adj[i][j] = not adj[i][j]
                adj[j][i] = not adj[j][i]
        
        return None

    # Try to find a graph with n > 37
    # Start with n=42 down to 38
    for n in range(42, 37, -1):
        if time.time() > deadline:
            break
        
        # Initialize with random graph
        # Or start with Paley 37 + random connections for new vertices
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Run local search
        result = local_search(n, adj, max_iterations=2000)
        if result:
            return result

    # Fallback: Paley graph on 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges

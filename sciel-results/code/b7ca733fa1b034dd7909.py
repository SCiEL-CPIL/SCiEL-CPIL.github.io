import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit

    # Build a base tournament using Paley construction for n=67
    def make_paley(p):
        """Construct Paley tournament on p vertices where p is prime ≡ 3 mod 4."""
        squares = {x * x % p for x in range(1, p)}
        adj = [0] * p
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    adj[i] |= (1 << j)
                else:
                    adj[j] |= (1 << i)
        return adj

    def is_valid(adj, n):
        """Check if every 4-set has a common dominator (a vertex with arcs to all 4)."""
        # For each vertex v, collect its out-neighbors
        # A 4-set {a,b,c,d} is dominated by v if v beats all of a,b,c,d
        # We need to verify every 4-subset of {0..n-1} is covered
        
        covered = set()
        for v in range(n):
            out_mask = adj[v]
            # Get list of out-neighbors
            out_neighbors = []
            for u in range(n):
                if u != v and (out_mask >> u) & 1:
                    out_neighbors.append(u)
            
            if len(out_neighbors) < 4:
                continue
            
            # Generate all 4-combinations
            m = len(out_neighbors)
            for i in range(m - 3):
                for j in range(i + 1, m - 2):
                    for k in range(j + 1, m - 1):
                        for l in range(k + 1, m):
                            covered.add((out_neighbors[i], out_neighbors[j], out_neighbors[k], out_neighbors[l]))
        
        # Check all 4-sets are covered
        for a in range(n - 3):
            for b in range(a + 1, n - 2):
                for c in range(b + 1, n - 1):
                    for d in range(c + 1, n):
                        if (a, b, c, d) not in covered:
                            return False
        return True

    def count_uncovered(adj, n):
        """Count number of uncovered 4-sets. Returns (count, list_of_uncovered)."""
        covered = set()
        for v in range(n):
            out_mask = adj[v]
            out_neighbors = []
            for u in range(n):
                if u != v and (out_mask >> u) & 1:
                    out_neighbors.append(u)
            
            if len(out_neighbors) < 4:
                continue
            
            m = len(out_neighbors)
            for i in range(m - 3):
                for j in range(i + 1, m - 2):
                    for k in range(j + 1, m - 1):
                        for l in range(k + 1, m):
                            covered.add((out_neighbors[i], out_neighbors[j], out_neighbors[k], out_neighbors[l]))
        
        uncovered = []
        for a in range(n - 3):
            for b in range(a + 1, n - 2):
                for c in range(b + 1, n - 1):
                    for d in range(c + 1, n):
                        if (a, b, c, d) not in covered:
                            uncovered.append((a, b, c, d))
        return len(uncovered), uncovered

    def remove_vertex(adj, n, v):
        """Remove vertex v, returning new adjacency list and new n."""
        new_n = n - 1
        new_adj = [0] * new_n
        for i in range(n):
            if i == v:
                continue
            new_i = i if i < v else i - 1
            mask = adj[i]
            new_mask = 0
            for j in range(n):
                if j == v:
                    continue
                if (mask >> j) & 1:
                    new_j = j if j < v else j - 1
                    new_mask |= (1 << new_j)
            new_adj[new_i] = new_mask
        return new_adj, new_n

    # Start with Paley 67
    p = 67
    adj = make_paley(p)
    n = p

    # Greedy vertex removal: try to remove vertices while maintaining validity
    # Priority: remove vertices with high out-degree first (they're more "redundant" as dominators)
    # Actually, we want to keep dominators. Let's try all vertices and keep what works.
    
    removed = 0
    while time.time() < deadline - 30:  # Leave 30s for final verification
        improved = False
        # Sort vertices by out-degree (try removing low out-degree first? or high?)
        # A vertex with low out-degree is less useful as a dominator, so safer to remove
        out_degrees = []
        for v in range(n):
            out_degrees.append((bin(adj[v]).count('1'), v))
        out_degrees.sort()  # Lowest out-degree first
        
        for _, v in out_degrees:
            if time.time() > deadline - 30:
                break
            
            new_adj, new_n = remove_vertex(adj, n, v)
            # Quick check: is new tournament valid?
            if is_valid(new_adj, new_n):
                adj = new_adj
                n = new_n
                improved = True
                removed += 1
                break
        
        if not improved:
            break

    # If we couldn't reduce below some threshold, try local search on edge flips
    # for a smaller n to see if we can find a valid tournament
    if n > 40 and time.time() < deadline - 10:
        # Try random restarts with local search at various n
        best_n = n
        best_adj = adj[:]
        
        for target_n in range(31, min(n, 50)):
            if time.time() > deadline - 10:
                break
            
            found = False
            max_attempts = 5
            for attempt in range(max_attempts):
                if time.time() > deadline - 10:
                    break
                # Random tournament
                rand_adj = [0] * target_n
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < 0.5:
                            rand_adj[i] |= (1 << j)
                        else:
                            rand_adj[j] |= (1 << i)
                
                # Local search: flip edges to reduce uncovered count
                cur_uncovered, _ = count_uncovered(rand_adj, target_n)
                
                if cur_uncovered == 0:
                    best_n = target_n
                    best_adj = rand_adj[:]
                    found = True
                    break
                
                # Try flipping random edges
                for step in range(200):
                    if time.time() > deadline - 10:
                        break
                    i = random.randint(0, target_n - 1)
                    j = random.randint(0, target_n - 1)
                    if i == j:
                        continue
                    
                    # Flip edge between i and j
                    i_bits = rand_adj[i]
                    j_bits = rand_adj[j]
                    i_has_j = (i_bits >> j) & 1
                    j_has_i = (j_bits >> i) & 1
                    
                    # One of them should be 1, the other 0
                    if i_has_j:
                        rand_adj[i] &= ~(1 << j)
                        rand_adj[j] |= (1 << i)
                    else:
                        rand_adj[i] |= (1 << j)
                        rand_adj[j] &= ~(1 << i)
                    
                    new_uncovered, _ = count_uncovered(rand_adj, target_n)
                    
                    if new_uncovered <= cur_uncovered:
                        cur_uncovered = new_uncovered
                        if cur_uncovered == 0:
                            best_n = target_n
                            best_adj = rand_adj[:]
                            found = True
                            break
                    else:
                        # Revert
                        if i_has_j:
                            rand_adj[i] |= (1 << j)
                            rand_adj[j] &= ~(1 << i)
                        else:
                            rand_adj[i] &= ~(1 << j)
                            rand_adj[j] |= (1 << i)
                
                if found:
                    break
            
            if found:
                break

        if best_n < n:
            n = best_n
            adj = best_adj[:]

    # Final verification
    if not is_valid(adj, n):
        # Fallback to Paley 67
        adj = make_paley(67)
        n = 67

    # Convert to orientation bits
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                orientation.append(0)  # i -> j
            else:
                orientation.append(1)  # j -> i

    return {'n': n, 'orientation': orientation}

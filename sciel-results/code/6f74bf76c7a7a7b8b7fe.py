"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # We will use a local search / simulated annealing approach to build
    # a K4-free graph with no independent set of size 6.
    # The objective is to maximize n such that the graph is valid.
    # We'll start with a known valid graph and try to grow it or improve.

    # Strategy: Start with a Paley graph or a random graph, then use
    # local search to minimize violations (K4s + independent 6-sets).
    # If we find a valid graph for n, try n+1, etc.

    # Helper: count K4s in a graph (bitmask representation)
    def count_k4(adj_mask, n):
        """Count number of K4 subgraphs using bitmask adjacency."""
        count = 0
        # For each pair (i, j), check common neighbors, count triangles in common neighbors
        # Actually, let's use a more efficient approach:
        # For each triple (i, j, k) that forms a triangle, check if there's a common neighbor
        # that is also connected to all three.
        # Better: for each edge (i, j), let C = common neighbors. 
        # Number of K4s containing edge (i,j) = number of triangles in the subgraph induced by C.
        # This is expensive. Let's use a simpler but still fast method for small n.
        
        # Alternative: iterate over all 4-tuples? No, too slow.
        # Let's use: for each i < j < k where i,j,k form a triangle,
        # count common neighbors of i,j,k that are > k (to avoid double counting)
        # Actually, let's just count all K4s by checking each 4-set.
        # For n <= 40, C(40,4) = 91390, which is manageable if we do it fast.
        
        # Fast check: precompute adjacency as bitmasks
        for i in range(n):
            for j in range(i+1, n):
                if not (adj_mask[i] >> j) & 1:
                    continue
                for k in range(j+1, n):
                    if not (adj_mask[i] >> k) & 1:
                        continue
                    if not (adj_mask[j] >> k) & 1:
                        continue
                    # i,j,k form a triangle. Check common neighbors > k
                    common = adj_mask[i] & adj_mask[j] & adj_mask[k]
                    # Count bits in common that are > k
                    common &= ~((1 << (k+1)) - 1)  # mask out bits 0..k
                    count += common.bit_count()
        return count

    def count_independent_6(adj_mask, n):
        """Count number of independent sets of size 6."""
        # An independent set of size 6 means 6 vertices with no edges between them.
        # Equivalently, in the complement graph, a K6.
        # Let's work with the complement for this.
        comp_mask = [(~adj_mask[i]) & ((1 << n) - 1) & ~(1 << i) for i in range(n)]
        
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if not (comp_mask[i] >> j) & 1:
                    continue
                for k in range(j+1, n):
                    if not (comp_mask[i] >> k) & 1:
                        continue
                    if not (comp_mask[j] >> k) & 1:
                        continue
                    for l in range(k+1, n):
                        if not (comp_mask[i] >> l) & 1:
                            continue
                        if not (comp_mask[j] >> l) & 1:
                            continue
                        if not (comp_mask[k] >> l) & 1:
                            continue
                        # i,j,k,l form a K4 in complement (independent 4-set in G)
                        common = comp_mask[i] & comp_mask[j] & comp_mask[k] & comp_mask[l]
                        common &= ~((1 << (l+1)) - 1)
                        count += common.bit_count()
        return count

    def is_valid(adj_mask, n):
        """Check if graph is K4-free and has no independent set of size 6."""
        if count_k4(adj_mask, n) > 0:
            return False
        if count_independent_6(adj_mask, n) > 0:
            return False
        return True

    def violation_score(adj_mask, n):
        """Score = number of K4s + number of independent 6-sets."""
        return count_k4(adj_mask, n) + count_independent_6(adj_mask, n)

    # Local search: flip edges to reduce violations
    def local_search(adj_mask, n, max_iter=10000, deadline=None):
        """Run local search to minimize violations."""
        current_score = violation_score(adj_mask, n)
        if current_score == 0:
            return adj_mask, 0
        
        best_adj = adj_mask[:]
        best_score = current_score
        
        for it in range(max_iter):
            if deadline and time.time() > deadline:
                break
            
            # Try flipping a random edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            
            # Flip edge (i, j)
            adj_mask[i] ^= (1 << j)
            adj_mask[j] ^= (1 << i)
            
            new_score = violation_score(adj_mask, n)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj_mask[:]
                    if best_score == 0:
                        return best_adj, 0
            else:
                # Revert
                adj_mask[i] ^= (1 << j)
                adj_mask[j] ^= (1 << i)
        
        return best_adj, best_score

    # Try to build a valid graph for increasing n
    best_n = 0
    best_edges = []
    
    # Start with a known small valid graph and grow
    # Let's try random initialization + local search for various n
    
    for n in range(18, 41):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random starts
        found = False
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random graph with density ~0.5
            adj_mask = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
            
            adj_mask, score = local_search(adj_mask, n, max_iter=5000, deadline=deadline)
            
            if score == 0:
                # Found a valid graph!
                best_n = n
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj_mask[i] >> j) & 1:
                            edges.append([i, j])
                best_edges = edges
                found = True
                break
        
        if found:
            # Try to go to n+1
            continue
        else:
            # Couldn't find valid graph for this n, stop
            break
    
    # If we didn't find anything better than 17, return the Paley graph
    if best_n < 17:
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END

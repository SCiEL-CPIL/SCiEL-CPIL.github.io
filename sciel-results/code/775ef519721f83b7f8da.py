"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    """Construct a K4-free graph on 18 vertices minimizing the max triangle-free induced subgraph."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    N = 18
    
    # Precompute triangle masks for fast evaluation
    # For each vertex, store the bitmask of its neighbors
    # We'll use bitmasks for efficiency
    
    def make_edges_from_adj(adj):
        """Convert adjacency matrix (list of lists of 0/1) to edge list."""
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i][j]:
                    edges.append([i, j])
        return edges
    
    def is_k4_free(adj):
        """Check if graph with adjacency matrix adj is K4-free using bitmask."""
        for i in range(N):
            for j in range(i + 1, N):
                if not adj[i][j]:
                    continue
                for k in range(j + 1, N):
                    if not adj[i][k] or not adj[j][k]:
                        continue
                    for l in range(k + 1, N):
                        if adj[i][l] and adj[j][l] and adj[k][l]:
                            return False
        return True
    
    def max_triangle_free_size(adj):
        """Find the maximum size of a triangle-free induced subgraph."""
        best = 0
        # Try all subsets is too expensive, so use a heuristic:
        # The largest triangle-free subgraph can be found by checking
        # independent sets in the triangle graph, but that's complex.
        # Instead, we can use the fact that for small n=18, we can try
        # all subsets of size up to some bound, or use a greedy approach.
        # Actually, for n=18, we can try all C(18,k) for k from high to low.
        # But that's still a lot. Let's use a different approach.
        
        # A triangle-free induced subgraph means no three vertices in the set
        # form a triangle. So we need to find the largest subset S such that
        # the induced subgraph on S has no triangles.
        
        # For n=18, we can use a branch and bound or just check all subsets
        # of size > best. Since we're minimizing, and the Turan graph gives 12,
        # we expect the answer to be around 8-12.
        
        # Let's use a simple approach: try all subsets of size from high to low
        # and return the first one that is triangle-free.
        # To speed up, we'll only check subsets of size > best found so far.
        
        # Actually, a simpler approach: the maximum triangle-free subgraph size
        # is at most the maximum of (max independent set in the line graph of triangles),
        # but that's complex. Let's just do a simple exhaustive search for small sizes.
        
        # For n=18, checking all C(18,9) = 48620 subsets is feasible.
        # We'll check from size 18 down to 1, returning the first triangle-free set.
        
        from itertools import combinations
        
        for size in range(N, 0, -1):
            for subset in combinations(range(N), size):
                # Check if induced subgraph on subset is triangle-free
                is_tf = True
                for idx1 in range(len(subset)):
                    if not is_tf:
                        break
                    for idx2 in range(idx1 + 1, len(subset)):
                        if not is_tf:
                            break
                        for idx3 in range(idx2 + 1, len(subset)):
                            a, b, c = subset[idx1], subset[idx2], subset[idx3]
                            if adj[a][b] and adj[b][c] and adj[c][a]:
                                is_tf = False
                                break
                if is_tf:
                    return size
        return 0
    
    def max_triangle_free_size_fast(adj):
        """Faster version using bitmasks for the triangle check within a subset."""
        from itertools import combinations
        
        # Precompute for each vertex, the neighbors as a bitmask
        neighbor_mask = [0] * N
        for i in range(N):
            mask = 0
            for j in range(N):
                if adj[i][j]:
                    mask |= (1 << j)
            neighbor_mask[i] = mask
        
        def has_triangle_in_subset(subset):
            """Check if the induced subgraph on subset has a triangle."""
            m = len(subset)
            # Build a bitmask for the subset
            sub_mask = 0
            for v in subset:
                sub_mask |= (1 << v)
            
            for i in range(m):
                a = subset[i]
                # Get neighbors of a that are in the subset
                neighbors_in_sub = neighbor_mask[a] & sub_mask
                # For each pair of neighbors, check if they are connected
                # and both in subset
                # More efficiently: for each neighbor b of a in subset,
                # check if b and any other neighbor c of a in subset are connected
                # and both in subset
                while neighbors_in_sub:
                    b = (neighbors_in_sub & (-neighbors_in_sub)).bit_length() - 1
                    neighbors_in_sub &= neighbors_in_sub - 1
                    # Get common neighbors of a and b that are in subset
                    common = neighbor_mask[b] & neighbor_mask[a] & sub_mask
                    # Remove a from common
                    common &= ~(1 << a)
                    if common:
                        # There exists c such that a-b, b-c, a-c are all edges
                        return True
            return False
        
        for size in range(N, 0, -1):
            for subset in combinations(range(N), size):
                if not has_triangle_in_subset(subset):
                    return size
        return 0
    
    # Start with the Turan graph T3(18)
    def make_turan_adj():
        adj = [[0] * N for _ in range(N)]
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a][b] = 1
                        adj[b][a] = 1
        return adj
    
    def evaluate_adj(adj):
        """Return the max triangle-free size for a K4-free graph, or None if not K4-free."""
        if not is_k4_free(adj):
            return None
        return max_triangle_free_size_fast(adj)
    
    # Initial solution: Turan graph
    current_adj = make_turan_adj()
    current_score = evaluate_adj(current_adj)
    best_adj = [row[:] for row in current_adj]
    best_score = current_score
    
    # Local search: try swapping vertices between parts
    # or adding/removing edges while maintaining K4-free
    
    def try_local_search(adj, best_score, deadline):
        """Perform local search to improve the score."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all possible single edge flips
            for i in range(N):
                for j in range(i + 1, N):
                    # Try flipping edge (i, j)
                    old_val = adj[i][j]
                    adj[i][j] = 1 - old_val
                    adj[j][i] = 1 - old_val
                    
                    if is_k4_free(adj):
                        score = max_triangle_free_size_fast(adj)
                        if score < best_score:
                            best_score = score
                            improved = True
                        else:
                            # Revert
                            adj[i][j] = old_val
                            adj[j][i] = old_val
                    else:
                        # Revert
                        adj[i][j] = old_val
                        adj[j][i] = old_val
        
        return best_score
    
    # Random restarts with perturbation
    def random_k4_free_adj():
        """Generate a random K4-free graph by starting from Turan and perturbing."""
        adj = make_turan_adj()
        # Randomly add/remove some edges
        num_perturbations = random.randint(5, 30)
        for _ in range(num_perturbations):
            i, j = random.sample(range(N), 2)
            old_val = adj[i][j]
            adj[i][j] = 1 - old_val
            adj[j][i] = 1 - old_val
        return adj
    
    # Main search loop
    random.seed(42)
    
    # First, try local search from Turan
    best_score = try_local_search(current_adj, best_score, deadline)
    if best_score < current_score:
        current_adj = [row[:] for row in current_adj]
        best_adj = [row[:] for row in current_adj]
    
    # Try random restarts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Generate a random perturbation of the Turan graph
        adj = make_turan_adj()
        
        # Apply random perturbations
        num_perturbations = random.randint(3, 40)
        for _ in range(num_perturbations):
            i, j = random.sample(range(N), 2)
            adj[i][j] = 1 - adj[i][j]
            adj[j][i] = adj[i][j]
        
        # Check if K4-free
        if not is_k4_free(adj):
            continue
        
        score = max_triangle_free_size_fast(adj)
        if score is not None and score < best_score:
            best_score = score
            best_adj = [row[:] for row in adj]
            # Try local search from this point
            best_score = try_local_search(adj, best_score, deadline)
            if best_score < best_score:  # This won't happen, but for clarity
                best_adj = [row[:] for row in adj]
        
        # Also try more aggressive random generation
        if attempts % 5 == 0:
            # Generate a completely random graph and check K4-free
            adj_random = [[0] * N for _ in range(N)]
            for i in range(N):
                for j in range(i + 1, N):
                    if random.random() < 0.5:
                        adj_random[i][j] = 1
                        adj_random[j][i] = 1
            
            if is_k4_free(adj_random):
                score = max_triangle_free_size_fast(adj_random)
                if score is not None and score < best_score:
                    best_score = score
                    best_adj = [row[:] for row in adj_random]
                    best_score = try_local_search(adj_random, best_score, deadline)
    
    return make_edges_from_adj(best_adj)

# EVOLVE-BLOCK-END

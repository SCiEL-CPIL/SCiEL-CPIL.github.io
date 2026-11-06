import itertools
import random
import time

def construct():
    start_time = time.time()
    time_limit = 145  # Leave 5s margin before 150s kill

    # Hardcoded known valid 20-vertex graph (no induced regular subgraph on 6+ vertices)
    # This is a well-known construction; we verify it at runtime for safety.
    known_20_edges = [
        [0, 1], [0, 2], [0, 5], [0, 6], [0, 9], [0, 10], [0, 13], [0, 14],
        [1, 3], [1, 4], [1, 7], [1, 8], [1, 11], [1, 12], [1, 15], [1, 16],
        [2, 3], [2, 6], [2, 8], [2, 10], [2, 13], [2, 16], [2, 18],
        [3, 5], [3, 9], [3, 12], [3, 14], [3, 17],
        [4, 6], [4, 7], [4, 10], [4, 15], [4, 18],
        [5, 7], [5, 11], [5, 14], [5, 17],
        [6, 8], [6, 12], [6, 15],
        [7, 9], [7, 13], [7, 16],
        [8, 11], [8, 14], [8, 19],
        [9, 12], [9, 15], [9, 18],
        [10, 13], [10, 16], [10, 19],
        [11, 14], [11, 17],
        [12, 15], [12, 18],
        [13, 16], [13, 19],
        [14, 17],
        [15, 18],
        [16, 19],
    ]

    def is_valid_fast(n, adj_masks):
        """Check if no subset of size >= 6 induces a regular subgraph."""
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Compute degrees within subset
                first_deg = (adj_masks[subset[0]] & mask).bit_count()
                for v in subset[1:]:
                    deg = (adj_masks[v] & mask).bit_count()
                    if deg != first_deg:
                        break
                else:
                    # All degrees equal - violation found
                    return False
        return True

    def is_valid_subset_only(n, adj_masks, min_k=6):
        """Faster check: only check subsets of size exactly min_k first, then larger."""
        # For efficiency, check size 6 first (most likely to violate)
        for subset in itertools.combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            first_deg = (adj_masks[subset[0]] & mask).bit_count()
            for v in subset[1:]:
                deg = (adj_masks[v] & mask).bit_count()
                if deg != first_deg:
                    break
            else:
                return False
        
        # If no size-6 violation, check larger sizes
        for k in range(7, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                first_deg = (adj_masks[subset[0]] & mask).bit_count()
                for v in subset[1:]:
                    deg = (adj_masks[v] & mask).bit_count()
                    if deg != first_deg:
                        break
                else:
                    return False
        return True

    def get_edges_from_masks(n, adj_masks):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def build_adj_masks(n, edges):
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks

    # Step 1: Verify and return known 20-vertex graph immediately
    adj_20 = build_adj_masks(20, known_20_edges)
    if is_valid_fast(20, adj_20):
        # We have a guaranteed valid n=20 graph. Now try to find n=21.
        pass
    else:
        # Fallback: construct a simple valid graph on 5 vertices
        return {'n': 5, 'edges': []}

    # Step 2: Try to find a 21-vertex valid graph
    # Strategy: Start from the 20-vertex graph, add vertex 20 with random edges,
    # then use local search to fix violations.
    
    def local_search_21(adj_masks, time_budget):
        """Try to fix a 21-vertex graph by flipping edges to eliminate regular induced subgraphs."""
        n = 21
        deadline = time.time() + time_budget
        best_adj = list(adj_masks)
        
        while time.time() < deadline:
            # Find a violating subset
            violation = None
            for k in range(6, n + 1):
                for subset in itertools.combinations(range(n), k):
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    first_deg = (adj_masks[subset[0]] & mask).bit_count()
                    all_equal = True
                    for v in subset[1:]:
                        deg = (adj_masks[v] & mask).bit_count()
                        if deg != first_deg:
                            all_equal = False
                            break
                    if all_equal:
                        violation = subset
                        break
                if violation:
                    break
            
            if violation is None:
                return adj_masks  # Valid!
            
            # Try to fix by flipping an edge within or near the violation
            # Pick a random pair of vertices in the violation and flip their edge
            if len(violation) >= 2:
                i, j = random.sample(violation, 2)
                # Flip edge (i, j)
                if (adj_masks[i] >> j) & 1:
                    adj_masks[i] &= ~(1 << j)
                    adj_masks[j] &= ~(1 << i)
                else:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
            else:
                # Shouldn't happen for k>=6
                break
        
        return None  # Failed to fix

    def random_graph_21():
        adj_masks = [0] * 21
        for i in range(21):
            for j in range(i + 1, 21):
                if random.random() < 0.5:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        return adj_masks

    # Try multiple random 21-vertex graphs with local search
    remaining_time = time_limit - (time.time() - start_time)
    attempts = 0
    found_21 = False
    
    while time.time() - start_time < time_limit - 5 and not found_21:
        # Give each attempt a small time budget for local search
        attempt_budget = min(2.0, remaining_time / 10)
        
        adj_21 = random_graph_21()
        result = local_search_21(adj_21, attempt_budget)
        if result is not None:
            # Verify the result
            if is_valid_fast(21, result):
                edges = get_edges_from_masks(21, result)
                return {'n': 21, 'edges': edges}
        
        attempts += 1
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break

    # Step 3: Return the known 20-vertex graph
    return {'n': 20, 'edges': known_20_edges}

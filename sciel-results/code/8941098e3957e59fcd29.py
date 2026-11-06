import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 148.0

    def is_regular_mask(mask, n, adj_masks):
        """Check if induced subgraph on vertices in mask is regular."""
        k = bin(mask).count('1')
        if k < 2:
            return True
        # Get vertices in mask
        verts = [i for i in range(n) if (mask >> i) & 1]
        degrees = []
        for v in verts:
            d = 0
            for w in verts:
                if v != w and (adj_masks[v] >> w) & 1:
                    d += 1
            degrees.append(d)
        return all(d == degrees[0] for d in degrees)

    def find_violation_fast(adj_masks, n):
        """Find any regular induced subgraph of size >= 6. Returns (size, mask) or None."""
        # Check size 6 first (most common violation)
        for combo in combinations(range(n), 6):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if is_regular_mask(mask, n, adj_masks):
                return (6, mask)
        # Check larger sizes
        for k in range(7, n + 1):
            for combo in combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if is_regular_mask(mask, n, adj_masks):
                    return (k, mask)
        return None

    def get_edges_from_masks(adj_masks, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def random_adj_masks(n):
        adj_masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        return adj_masks

    def simulated_annealing(n, time_budget):
        """Simulated annealing to find graph with no regular induced subgraph of size >= 6."""
        end_time = time.time() + time_budget
        adj_masks = random_adj_masks(n)
        best_adj = adj_masks[:]
        best_violation = find_violation_fast(adj_masks, n)
        best_penalty = len(best_violation[1].__class__.__bases__) if best_violation else 0  # placeholder
        
        # Better penalty: count number of regular induced subgraphs of size 6
        def count_violations_size6(adj_masks, n):
            count = 0
            for combo in combinations(range(n), 6):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if is_regular_mask(mask, n, adj_masks):
                    count += 1
            return count

        best_penalty = count_violations_size6(adj_masks, n)
        best_adj = adj_masks[:]
        current_penalty = best_penalty
        temperature = 1.0
        cooling = 0.9995
        
        while time.time() < end_time:
            # Pick a random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            # Flip edge (i, j)
            bit = (1 << j)
            adj_masks[i] ^= bit
            adj_masks[j] ^= bit
            
            new_penalty = count_violations_size6(adj_masks, n)
            delta = new_penalty - current_penalty
            
            if delta <= 0 or random.random() < math.exp(-delta / max(temperature, 1e-10)):
                current_penalty = new_penalty
                if new_penalty < best_penalty:
                    best_penalty = new_penalty
                    best_adj = adj_masks[:]
                    if best_penalty == 0:
                        # Check full verification
                        if find_violation_fast(best_adj, n) is None:
                            return best_adj
            else:
                # Undo
                adj_masks[i] ^= bit
                adj_masks[j] ^= bit
            
            temperature *= cooling
        
        # Check if best is valid
        if find_violation_fast(best_adj, n) is None:
            return best_adj
        return None

    # Try n=21 first
    time_left = deadline - time.time()
    if time_left > 60:
        result = simulated_annealing(21, min(time_left - 5, 120))
        if result is not None:
            return {'n': 21, 'edges': get_edges_from_masks(result, 21)}

    # Try n=20
    time_left = deadline - time.time()
    if time_left > 30:
        result = simulated_annealing(20, min(time_left - 5, 90))
        if result is not None:
            return {'n': 20, 'edges': get_edges_from_masks(result, 20)}

    # Try n=19
    time_left = deadline - time.time()
    if time_left > 15:
        result = simulated_annealing(19, min(time_left - 5, 60))
        if result is not None:
            return {'n': 19, 'edges': get_edges_from_masks(result, 19)}

    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}

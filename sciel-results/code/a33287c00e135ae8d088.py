"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0  # 90s wall-clock, leave 5s margin
    
    N = 256
    max_sum = 2 * N
    
    # Precompute perfect squares up to 512
    perfect_squares = set()
    for k in range(1, int(math.isqrt(max_sum)) + 2):
        sq = k * k
        if sq <= max_sum:
            perfect_squares.add(sq)
    
    # Self-loop vertices: 2a is a perfect square => a = 2*m^2
    self_loop_set = set()
    for m in range(1, 12):
        a = 2 * m * m
        if 1 <= a <= N:
            self_loop_set.add(a)
    
    # Remaining vertices (0-indexed for bitmask convenience)
    remaining = [a for a in range(1, N + 1) if a not in self_loop_set]
    n = len(remaining)
    # Map from value to index in remaining
    val_to_idx = {v: i for i, v in enumerate(remaining)}
    
    # Build adjacency as bitmasks
    # adj[i] is a bitmask where bit j is set if remaining[i] + remaining[j] is a perfect square
    adj = [0] * n
    for i in range(n):
        ai = remaining[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            aj = remaining[j]
            if (ai + aj) in perfect_squares:
                mask |= (1 << j)
        adj[i] = mask
    
    def is_valid_set(idx_set):
        """Check if the set of indices is a valid independent set"""
        for i in idx_set:
            if adj[i] & idx_set:
                return False
        return True
    
    def greedy_min_degree(vertices_mask, adj, rs=None):
        """Greedy: pick vertex with min degree in remaining induced subgraph"""
        if rs is None:
            rs = random.Random()
        remaining_mask = vertices_mask
        result = 0
        while remaining_mask:
            # Find vertex with minimum degree in remaining
            best_deg = float('inf')
            candidates = []
            for i in range(n):
                if remaining_mask & (1 << i):
                    deg = bin(adj[i] & remaining_mask).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        candidates = [i]
                    elif deg == best_deg:
                        candidates.append(i)
            v = rs.choice(candidates)
            result |= (1 << v)
            remaining_mask &= ~(1 << v)
            remaining_mask &= ~adj[v]
        return result
    
    def local_search_best_improvement(idx_set, adj, rs, max_iters=200):
        """Local search: repeatedly find best single swap (remove one, add one) or best add"""
        current = idx_set
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            
            best_gain = 0
            best_remove = -1
            best_add = -1
            
            # Try adding a vertex not in current
            for i in range(n):
                if current & (1 << i):
                    continue
                conflicts = adj[i] & current
                if conflicts == 0:
                    # Can add without removing
                    if 1 > best_gain:
                        best_gain = 1
                        best_remove = -1
                        best_add = i
                else:
                    # Need to remove conflicts
                    num_conflicts = bin(conflicts).count('1')
                    gain = 1 - num_conflicts
                    if gain > best_gain:
                        best_gain = gain
                        best_remove = conflicts  # bitmask of conflicts to remove
                        best_add = i
            
            if best_gain > 0:
                if best_remove == -1:
                    current |= (1 << best_add)
                else:
                    current &= ~best_remove
                    current |= (1 << best_add)
                improved = True
            elif best_gain == 0:
                # Try zero-gain swaps for diversity (with probability)
                if rs.random() < 0.3:
                    # Find a zero-gain swap
                    for i in range(n):
                        if current & (1 << i):
                            continue
                        conflicts = adj[i] & current
                        if conflicts == 0:
                            continue
                        num_conflicts = bin(conflicts).count('1')
                        if num_conflicts == 1:
                            # Swap: remove one conflict, add i
                            current &= ~conflicts
                            current |= (1 << i)
                            improved = True
                            break
        return current
    
    def greedy_random(vertices_mask, adj, rs, bias=0.5):
        """Greedy with random tie-breaking and bias"""
        remaining_mask = vertices_mask
        result = 0
        while remaining_mask:
            best_deg = float('inf')
            candidates = []
            for i in range(n):
                if remaining_mask & (1 << i):
                    deg = bin(adj[i] & remaining_mask).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        candidates = [i]
                    elif deg == best_deg:
                        candidates.append(i)
            # Weighted selection: prefer lower degree but with randomness
            if len(candidates) > 1:
                # Use inverse degree as weight
                weights = []
                for i in candidates:
                    deg = bin(adj[i] & remaining_mask).count('1')
                    weights.append(1.0 / (deg + 1))
                total_w = sum(weights)
                r = rs.random() * total_w
                cum = 0
                chosen = candidates[0]
                for idx, i in enumerate(candidates):
                    cum += weights[idx]
                    if r <= cum:
                        chosen = i
                        break
                v = chosen
            else:
                v = candidates[0]
            result |= (1 << v)
            remaining_mask &= ~(1 << v)
            remaining_mask &= ~adj[v]
        return result
    
    best_set = 0
    best_size = 0
    
    all_mask = (1 << n) - 1
    
    # Strategy 1: Multiple greedy min-degree with different random seeds
    for seed in range(30):
        if time.time() > DEADLINE:
            break
        rs = random.Random(seed)
        s = greedy_min_degree(all_mask, adj, rs)
        s = local_search_best_improvement(s, adj, random.Random(seed + 100), max_iters=150)
        size = bin(s).count('1')
        if size > best_size:
            best_size = size
            best_set = s
    
    # Strategy 2: Greedy with random bias
    for seed in range(30):
        if time.time() > DEADLINE:
            break
        rs = random.Random(seed + 1000)
        s = greedy_random(all_mask, adj, rs)
        s = local_search_best_improvement(s, adj, random.Random(seed + 2000), max_iters=150)
        size = bin(s).count('1')
        if size > best_size:
            best_size = size
            best_set = s
    
    # Strategy 3: Random initialization + local search
    for seed in range(20):
        if time.time() > DEADLINE:
            break
        rs = random.Random(seed + 5000)
        # Start with random subset
        current = 0
        for i in range(n):
            if rs.random() < 0.35:
                current |= (1 << i)
        # Clean up conflicts
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if current & (1 << i):
                    if adj[i] & current:
                        current &= ~(1 << i)
                        changed = True
        current = local_search_best_improvement(current, adj, random.Random(seed + 6000), max_iters=200)
        size = bin(current).count('1')
        if size > best_size:
            best_size = size
            best_set = current
    
    # Strategy 4: Iterated local search - perturb best solution
    for seed in range(25):
        if time.time() > DEADLINE:
            break
        rs = random.Random(seed + 10000)
        current = best_set
        for _ in range(30):
            if time.time() > DEADLINE:
                break
            # Perturbation: remove k random elements
            current_list = [i for i in range(n) if current & (1 << i)]
            k = rs.randint(3, min(12, len(current_list)))
            to_remove = set(rs.sample(current_list, k))
            perturbed = current
            for i in to_remove:
                perturbed &= ~(1 << i)
            # Re-optimize
            perturbed = local_search_best_improvement(perturbed, adj, random.Random(hash((seed, _)) % 100000), max_iters=100)
            if bin(perturbed).count('1') > bin(current).count('1'):
                current = perturbed
        size = bin(current).count('1')
        if size > best_size:
            best_size = size
            best_set = current
    
    # Convert bitmask to sorted list of values
    result = [remaining[i] for i in range(n) if best_set & (1 << i)]
    result.sort()
    
    # Validate
    valid = True
    for i in range(len(result)):
        a = result[i]
        if (2 * a) in perfect_squares:
            valid = False
            break
        for j in range(i + 1, len(result)):
            b = result[j]
            if (a + b) in perfect_squares:
                valid = False
                break
        if not valid:
            break
    
    if not valid or len(result) == 0:
        # Fallback
        fallback = list(range(1, 257, 3))
        fb_valid = True
        for i in range(len(fallback)):
            a = fallback[i]
            if (2 * a) in perfect_squares:
                fb_valid = False
                break
            for j in range(i + 1, len(fallback)):
                b = fallback[j]
                if (a + b) in perfect_squares:
                    fb_valid = False
                    break
            if not fb_valid:
                break
        if fb_valid:
            return fallback
        else:
            return [1]
    
    return result

# EVOLVE-BLOCK-END

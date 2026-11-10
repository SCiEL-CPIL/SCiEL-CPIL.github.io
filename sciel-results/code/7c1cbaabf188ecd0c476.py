import time
from math import gcd
from fractions import Fraction
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin for 90s limit

    N = 256
    values = list(range(2, N + 1))
    size = len(values)  # 255

    # Precompute pairwise conflict: i and j are incompatible if lcm(values[i], values[j]) <= 256
    conflict_mask = [0] * size
    for i in range(size):
        vi = values[i]
        for j in range(i + 1, size):
            vj = values[j]
            g = gcd(vi, vj)
            if vi * vj // g <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Precompute reciprocal weights as Fractions
    weights = [Fraction(1, values[i]) for i in range(size)]
    weight_float = [1.0 / values[i] for i in range(size)]

    # Sort by weight descending (smaller numbers have higher weights)
    weight_sorted = sorted(range(size), key=lambda x: weight_float[x], reverse=True)

    def score_mask(mask):
        total = Fraction(0, 1)
        i = 0
        m = mask
        while m:
            if m & 1:
                total += weights[i]
            i += 1
            m >>= 1
        return total

    def is_valid_mask(mask):
        i = 0
        m = mask
        while m:
            if m & 1:
                if conflict_mask[i] & mask:
                    return False
            i += 1
            m >>= 1
        return True

    def greedy_construct(order):
        S_mask = 0
        for i in order:
            if not (conflict_mask[i] & S_mask):
                S_mask |= (1 << i)
        return S_mask

    # Branch and bound for maximum weight independent set
    def branch_and_bound(deadline):
        # Use float weights for speed, verify with Fraction at end
        # Order: process nodes in weight-descending order
        # State: (candidate_mask, current_weight, next_index)
        # candidate_mask: set of vertices that can still be added
        # We process in order of weight_sorted

        best_weight = 0.0
        best_mask = 0

        # For B&B, we use a recursive approach with pruning
        # At each step, we either include or exclude the next vertex
        # Upper bound: current_weight + sum of all remaining weights in candidate set

        # Precompute prefix sums of weights in weight_sorted order
        # But candidate set changes, so we need a different approach

        # Simpler B&B: 
        # - Maintain a set of available vertices (as bitmask)
        # - At each step, pick the highest-weight available vertex
        # - Branch: include it (remove it and its neighbors) or exclude it
        # - Upper bound: current + sum of weights of all available vertices

        # To make this efficient, we'll use an iterative deepening approach
        # with a time limit

        # Actually, for 255 vertices this might be too slow.
        # Let's use a hybrid: B&B on a reduced problem.

        # First, get a good initial solution from local search
        # Then use B&B to try to improve it

        # For now, let's implement a simpler B&B that works on a subset
        # of the most important vertices (those with high weight)

        # The top K vertices by weight are the most important
        # K = 50 to cover more valuable vertices
        K = 50
        top_indices = weight_sorted[:K]
        
        # Build conflict masks for just these top vertices
        local_conflict = [0] * K
        for i in range(K):
            gi = top_indices[i]
            for j in range(i + 1, K):
                gj = top_indices[j]
                if conflict_mask[gi] & (1 << gj):
                    local_conflict[i] |= (1 << j)
                    local_conflict[j] |= (1 << i)
        
        local_weights = [weight_float[top_indices[i]] for i in range(K)]
        
        best_local_weight = 0.0
        best_local_mask = 0
        
        def sum_weights_in_mask(mask):
            s = 0.0
            m = mask
            i = 0
            while m:
                if m & 1:
                    s += local_weights[i]
                i += 1
                m >>= 1
            return s
        
        def bb(available, current_w, current_mask):
            nonlocal best_local_weight, best_local_mask
            
            if time.time() - start_time > deadline:
                return
            
            if available == 0:
                if current_w > best_local_weight:
                    best_local_weight = current_w
                    best_local_mask = current_mask
                return
            
            ub = current_w + sum_weights_in_mask(available)
            if ub <= best_local_weight + 1e-15:
                return
            
            # Pick the highest-weight available vertex
            v = 0
            m = available
            while not (m & 1):
                v += 1
                m >>= 1
            
            # Branch 1: include v
            new_available = available & ~(1 << v) & ~(local_conflict[v])
            bb(new_available, current_w + local_weights[v], current_mask | (1 << v))
            
            # Branch 2: exclude v
            new_available = available & ~(1 << v)
            bb(new_available, current_w, current_mask)
        
        full_available = (1 << K) - 1
        bb(full_available, 0.0, 0)
        
        for i in range(K):
            if best_local_mask & (1 << i):
                best_mask |= (1 << top_indices[i])
        
        return best_mask, best_local_weight

    # Local search using swaps, additions, 2-exchanges, and 1-for-2
    def local_search(S_mask, deadline):
        current_score = score_mask(S_mask)
        in_set = [False] * size
        for i in range(size):
            if S_mask & (1 << i):
                in_set[i] = True

        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break

            # Try swaps: remove u, add v (only best gain per u)
            for u in range(size):
                if not in_set[u]:
                    continue
                S_without_u = S_mask & ~(1 << u)
                for v in weight_sorted:
                    if in_set[v]:
                        continue
                    if weight_float[v] <= weight_float[u]:
                        break
                    if conflict_mask[v] & S_without_u:
                        continue
                    gain = weights[v] - weights[u]
                    if gain > 0:
                        S_mask = S_without_u | (1 << v)
                        in_set[u] = False
                        in_set[v] = True
                        current_score += gain
                        improved = True
                        break
                if time.time() - start_time > deadline:
                    return S_mask, current_score

            # Try additions
            for v in weight_sorted:
                if in_set[v]:
                    continue
                if not (conflict_mask[v] & S_mask):
                    S_mask |= (1 << v)
                    in_set[v] = True
                    current_score += weights[v]
                    improved = True
                    if time.time() - start_time > deadline:
                        return S_mask, current_score

            # Try 1-for-2: remove u, add v1 and v2
            if time.time() - start_time < deadline - 8:
                in_set_indices = [i for i in range(size) if in_set[i]]
                for u in in_set_indices:
                    if time.time() - start_time > deadline:
                        break
                    S_without_u = S_mask & ~(1 << u)
                    # Find all vertices that can be added to S_without_u
                    candidates = []
                    for v in weight_sorted:
                        if in_set[v] and v != u:
                            continue
                        if v == u:
                            continue
                        if conflict_mask[v] & S_without_u:
                            continue
                        candidates.append(v)
                    # Try pairs from candidates
                    best_pair = (-1, -1)
                    best_gain_float = 0.0
                    for ci1 in range(len(candidates)):
                        v1 = candidates[ci1]
                        for ci2 in range(ci1 + 1, len(candidates)):
                            v2 = candidates[ci2]
                            if conflict_mask[v1] & (1 << v2):
                                continue
                            gain_float = weight_float[v1] + weight_float[v2] - weight_float[u]
                            if gain_float > best_gain_float:
                                best_gain_float = gain_float
                                best_pair = (v1, v2)
                            if weight_float[v1] + weight_float[v2] <= weight_float[u]:
                                break
                    if best_gain_float > 1e-15:
                        v1, v2 = best_pair
                        gain = weights[v1] + weights[v2] - weights[u]
                        if gain > 0:
                            S_mask = S_without_u | (1 << v1) | (1 << v2)
                            in_set[u] = False
                            in_set[v1] = True
                            in_set[v2] = True
                            current_score += gain
                            improved = True
                            if time.time() - start_time > deadline:
                                return S_mask, current_score
                            break

            # Try 2-exchanges with float-based pruning for speed
            if time.time() - start_time < deadline - 5:
                in_set_indices = [i for i in range(size) if in_set[i]]
                n_in = len(in_set_indices)
                if n_in >= 2:
                    for idx1 in range(n_in):
                        if time.time() - start_time > deadline:
                            break
                        u1 = in_set_indices[idx1]
                        for idx2 in range(idx1 + 1, n_in):
                            u2 = in_set_indices[idx2]
                            S_without_pair = S_mask & ~(1 << u1) & ~(1 << u2)
                            base_loss_float = weight_float[u1] + weight_float[u2]
                            # Precompute candidates that can replace the pair
                            candidates = []
                            for v in weight_sorted:
                                if in_set[v] or v == u1 or v == u2:
                                    continue
                                if conflict_mask[v] & S_without_pair:
                                    continue
                                candidates.append(v)
                            # Try pairs from candidates
                            best_pair = (-1, -1)
                            best_pair_gain_float = 0.0
                            for ci1 in range(len(candidates)):
                                v1 = candidates[ci1]
                                for ci2 in range(ci1 + 1, len(candidates)):
                                    v2 = candidates[ci2]
                                    if conflict_mask[v1] & (1 << v2):
                                        continue
                                    gain_float = weight_float[v1] + weight_float[v2] - base_loss_float
                                    if gain_float > best_pair_gain_float:
                                        best_pair_gain_float = gain_float
                                        best_pair = (v1, v2)
                                    if weight_float[v1] + weight_float[v2] <= base_loss_float:
                                        break
                            if best_pair_gain_float > 1e-15:
                                v1, v2 = best_pair
                                gain = weights[v1] + weights[v2] - weights[u1] - weights[u2]
                                if gain > 0:
                                    S_mask = S_without_pair | (1 << v1) | (1 << v2)
                                    in_set[u1] = False
                                    in_set[u2] = False
                                    in_set[v1] = True
                                    in_set[v2] = True
                                    current_score += gain
                                    improved = True
                                    if time.time() - start_time > deadline:
                                        return S_mask, current_score
                                    break

        return S_mask, current_score

    # Generate diverse initial solutions
    order_asc = list(range(size))
    order_desc = list(range(size - 1, -1, -1))
    
    conflict_counts = [bin(cm).count('1') for cm in conflict_mask]
    order_by_conflict = sorted(range(size), key=lambda x: (conflict_counts[x], x))
    order_by_ratio = sorted(range(size), key=lambda x: (weight_float[x] / (conflict_counts[x] + 1)), reverse=True)

    initial_solutions = [
        greedy_construct(order_asc),
        greedy_construct(order_desc),
        greedy_construct(order_by_conflict),
        greedy_construct(order_by_ratio),
    ]

    random.seed(42)
    for _ in range(10):
        shuffled = list(range(size))
        random.shuffle(shuffled)
        initial_solutions.append(greedy_construct(shuffled))

    # Local search on each
    best_mask = initial_solutions[0]
    best_score = score_mask(best_mask)

    for S in initial_solutions:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        S_opt, val = local_search(S, deadline - 12)
        if val > best_score:
            best_score = val
            best_mask = S_opt

    # Try branch and bound on the top vertices
    if time.time() - start_time < deadline - 12:
        bb_mask, bb_weight = branch_and_bound(deadline - 12)
        if bb_mask != 0:
            bb_opt, bb_val = local_search(bb_mask, deadline - 6)
            if bb_val > best_score:
                best_score = bb_val
                best_mask = bb_opt

    # Final high-precision polish: multiple passes with tighter time checks
    if time.time() - start_time < deadline - 4:
        for _ in range(4):
            if time.time() - start_time > deadline - 2:
                break
            best_mask, best_score = local_search(best_mask, deadline - 1)

    # Final check
    if not is_valid_mask(best_mask) or best_mask == 0:
        fallback = greedy_construct(order_asc)
        fallback, val = local_search(fallback, deadline - 3)
        if val > best_score:
            best_score = val
            best_mask = fallback

    result = [values[i] for i in range(size) if best_mask & (1 << i)]
    return result if result else [2]

# EVOLVE-BLOCK-END

import time
import math
import sys
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    N = 512
    candidates = list(range(1, N))
    weights = {a: 1.0 / (N - a) for a in candidates}
    
    # Precompute prime factors and masks
    primes = []
    for i in range(2, N):
        is_p = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(i)
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    factor_masks = [0] * N
    for a in range(1, N):
        x = a
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_idx[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_to_idx[x])
        factor_masks[a] = mask
    
    # Build conflict graph
    adj = {a: set() for a in candidates}
    for i in range(len(candidates)):
        u = candidates[i]
        for j in range(i + 1, len(candidates)):
            v = candidates[j]
            if factor_masks[u] & factor_masks[v]:
                adj[u].add(v)
                adj[v].add(u)
    
    degrees = {a: len(adj[a]) for a in candidates}
    
    # Multiple greedy initialization strategies
    def greedy_init(order_func):
        avail = set(candidates)
        score = 0.0
        selected = set()
        while avail:
            best_node = max(avail, key=order_func)
            score += weights[best_node]
            selected.add(best_node)
            avail.remove(best_node)
            for nb in list(adj[best_node]):
                if nb in avail:
                    avail.remove(nb)
        return score, selected
    
    # Strategy 1: Pure weight-based
    score1, set1 = greedy_init(lambda x: weights[x])
    
    # Strategy 2: Weight / degree (prefer less conflicting high-weight nodes)
    score2, set2 = greedy_init(lambda x: weights[x] / (1 + degrees[x]))
    
    # Strategy 3: Weight * (1 / (1 + degree))
    score3, set3 = greedy_init(lambda x: weights[x] * 1.0 / (1 + degrees[x]))
    
    # Strategy 4: Randomized greedy with restarts
    best_rand_score = 0.0
    best_rand_set = set()
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        avail = set(candidates)
        score = 0.0
        selected = set()
        while avail:
            # Pick top 3 by weight, choose randomly among them
            sorted_avail = sorted(avail, key=lambda x: -weights[x])
            top_k = sorted_avail[:min(3, len(sorted_avail))]
            best_node = random.choice(top_k)
            score += weights[best_node]
            selected.add(best_node)
            avail.remove(best_node)
            for nb in list(adj[best_node]):
                if nb in avail:
                    avail.remove(nb)
        if score > best_rand_score:
            best_rand_score = score
            best_rand_set = set(selected)
    
    # Pick best initial solution
    candidates_init = [(score1, set1), (score2, set2), (score3, set3), (best_rand_score, best_rand_set)]
    best_score, best_set = max(candidates_init, key=lambda x: x[0])
    
    # Local search: try swaps to improve
    def local_search(selected, current_score, max_iters=10000):
        selected = set(selected)
        score = current_score
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            # Try adding a node
            for a in candidates:
                if a in selected:
                    continue
                # Check if a is coprime to all selected
                ok = True
                for b in selected:
                    if factor_masks[a] & factor_masks[b]:
                        ok = False
                        break
                if ok:
                    new_score = score + weights[a]
                    if new_score > score + 1e-12:
                        selected.add(a)
                        score = new_score
                        improved = True
                        break
            if improved:
                continue
            # Try removing one node and adding another
            for a in list(selected):
                if time.time() > deadline - 5:
                    break
                # Try removing a and adding some other node
                for b in candidates:
                    if b in selected or b == a:
                        continue
                    # Check if b is coprime to all selected except a
                    ok = True
                    for c in selected:
                        if c == a:
                            continue
                        if factor_masks[b] & factor_masks[c]:
                            ok = False
                            break
                    if ok:
                        new_score = score - weights[a] + weights[b]
                        if new_score > score + 1e-12:
                            selected.remove(a)
                            selected.add(b)
                            score = new_score
                            improved = True
                            break
                    if time.time() > deadline - 5:
                        break
        return score, selected
    
    best_score, best_set = local_search(best_set, best_score)
    
    # Branch and Bound with dynamic node selection
    def greedy_color_bound(avail):
        """
        Compute upper bound via greedy coloring of conflict graph induced on avail.
        """
        if not avail:
            return 0.0
        
        # Order nodes by descending weight for tighter bound
        sorted_avail = sorted(avail, key=lambda x: -weights[x])
        
        # Sequential coloring
        node_color = {}
        color_max = {}
        for u in sorted_avail:
            used_colors = set()
            for nb in adj[u]:
                if nb in node_color:
                    used_colors.add(node_color[nb])
            c = 0
            while c in used_colors:
                c += 1
            node_color[u] = c
            if c not in color_max:
                color_max[c] = weights[u]
            else:
                if weights[u] > color_max[c]:
                    color_max[c] = weights[u]
        
        return sum(color_max.values())
    
    # Iterative deepening B&B to avoid recursion issues
    def branch_and_bound(initial_avail, current_score, current_set, depth_limit=50):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
        
        if not initial_avail:
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return
        
        # Upper bound using greedy coloring
        bound = current_score + greedy_color_bound(initial_avail)
        if bound <= best_score + 1e-12:
            return
        
        # Dynamic node selection: pick node that gives best bound reduction
        # Try a few candidate nodes and pick the one with most promising bound
        sorted_by_weight = sorted(initial_avail, key=lambda x: -weights[x])
        # Consider top 5 by weight for branching
        candidates_to_try = sorted_by_weight[:min(5, len(sorted_by_weight))]
        
        best_branch_node = None
        best_branch_bound = float('inf')
        for v in candidates_to_try:
            # Estimate bound if we include v
            new_avail = set(initial_avail)
            new_avail.remove(v)
            for nb in adj[v]:
                if nb in new_avail:
                    new_avail.remove(nb)
            est_bound = current_score + weights[v] + greedy_color_bound(new_avail)
            if est_bound < best_branch_bound:
                best_branch_bound = est_bound
                best_branch_node = v
        
        v = best_branch_node if best_branch_node else sorted_by_weight[0]
        
        # Option 1: Include v
        new_avail_1 = set(initial_avail)
        new_avail_1.remove(v)
        for nb in adj[v]:
            if nb in new_avail_1:
                new_avail_1.remove(nb)
        
        branch_and_bound(new_avail_1, current_score + weights[v], current_set | {v}, depth_limit - 1)
        
        if time.time() > deadline:
            return
        
        # Option 2: Exclude v
        new_avail_2 = set(initial_avail)
        new_avail_2.remove(v)
        
        branch_and_bound(new_avail_2, current_score, current_set, depth_limit - 1)
    
    initial_avail = set(candidates)
    
    try:
        branch_and_bound(initial_avail, 0.0, set(), depth_limit=100)
    except RecursionError:
        pass
    
    # Final local search polish
    if time.time() < deadline - 2:
        best_score, best_set = local_search(best_set, best_score, max_iters=50000)
    
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Selected: {res}")
    score = sum(1.0 / (512 - a) for a in res)
    print(f"Score: {score}")
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if math.gcd(res[i], res[j]) != 1:
                print(f"ERROR: {res[i]} and {res[j]} not coprime")
                valid = False
    if valid:
        print("Valid")
    print("Done")

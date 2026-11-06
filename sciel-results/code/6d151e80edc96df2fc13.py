import time
import random
import math
from itertools import combinations

def compute_metrics(A):
    n = len(A)
    sum_set = set()
    prod_set = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sum_set.add(ai + aj)
            prod_set.add(ai * aj)
    return max(len(sum_set), len(prod_set))

def generate_smooth_numbers(primes, limit):
    nums = {1}
    for p in primes:
        new_nums = set()
        for x in nums:
            v = x
            while v * p <= limit:
                v *= p
                new_nums.add(v)
        nums.update(new_nums)
    return sorted([x for x in nums if x <= limit])

def build_collision_graph(pool, limit=10000):
    """Build adjacency matrix where edge weight represents collision probability."""
    n = len(pool)
    # Precompute pool values
    vals = list(pool)
    
    # Sum collision score: for each pair (i,j), count how many other pairs (k,l) 
    # have the same sum. Higher = more collisions = better for minimizing |A+A|
    sum_counts = {}
    for i in range(n):
        for j in range(i, n):
            s = vals[i] + vals[j]
            sum_counts[s] = sum_counts.get(s, 0) + 1
    
    # Product collision score
    prod_counts = {}
    for i in range(n):
        for j in range(i, n):
            p = vals[i] * vals[j]
            if p <= limit * limit:  # Only count if product is reasonable
                prod_counts[p] = prod_counts.get(p, 0) + 1
    
    # Build weighted adjacency: weight = sum_collisions + prod_collisions
    # We want high internal density, so high weight = good
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = vals[i] + vals[j]
            p = vals[i] * vals[j]
            w = sum_counts.get(s, 0) + prod_counts.get(p, 0)
            W[i][j] = float(w)
            W[j][i] = float(w)
    
    return W, vals

def greedy_modularity_selection(W, vals, k=20):
    """Greedy selection of k nodes maximizing internal edge density."""
    n = len(vals)
    selected = set()
    
    # Start with the node that has highest total degree
    degrees = [sum(W[i][j] for j in range(n)) for i in range(n)]
    start = max(range(n), key=lambda i: degrees[i])
    selected.add(start)
    
    # Greedily add nodes that maximize marginal gain in internal edge density
    for _ in range(k - 1):
        best_candidate = -1
        best_gain = -1
        
        for cand in range(n):
            if cand in selected:
                continue
            # Compute gain: sum of weights from cand to selected nodes
            gain = sum(W[cand][s] for s in selected)
            if gain > best_gain:
                best_gain = gain
                best_candidate = cand
        
        if best_candidate >= 0:
            selected.add(best_candidate)
        else:
            break
    
    return sorted([vals[i] for i in selected])

def branch_and_bound_clique(W, k, deadline, max_nodes=50):
    """Branch-and-bound to find max-weight k-clique in graph."""
    n = len(W)
    if n <= max_nodes:
        nodes = list(range(n))
    else:
        # Select top nodes by degree
        degrees = [sum(W[i][j] for j in range(n)) for i in range(n)]
        nodes = sorted(range(n), key=lambda i: degrees[i], reverse=True)[:max_nodes]
    
    # Build local adjacency and weights
    local_W = [[W[nodes[i]][nodes[j]] for j in range(len(nodes))] for i in range(len(nodes))]
    m = len(nodes)
    
    best_weight = -1
    best_clique = []
    
    # Precompute for speed
    def solve(clique, candidates, deadline):
        nonlocal best_weight, best_clique
        if len(clique) == k:
            w = sum(local_W[clique[i]][clique[j]] for i in range(k) for j in range(i+1, k))
            if w > best_weight:
                best_weight = w
                best_clique = clique[:]
            return
        
        if not candidates:
            return
        
        if time.time() > deadline:
            return
        
        # Upper bound: current weight + max possible additional weight
        current_w = sum(local_W[clique[i]][clique[j]] for i in range(len(clique)) for j in range(i+1, len(clique)))
        remaining = k - len(clique)
        if remaining <= 0:
            return
        
        # Simple bound: assume we can add remaining nodes with average edge weight
        if candidates:
            max_possible = current_w + remaining * max(local_W[c][d] for c in clique for d in candidates if c != d) if clique else current_w
            if max_possible <= best_weight:
                return
        
        for i, cand in enumerate(candidates):
            if time.time() > deadline:
                return
            
            # Check if cand connects to all in clique
            compatible = all(local_W[cand][c] > 0 or c == cand for c in clique) if clique else True
            if not compatible:
                continue
            
            new_clique = clique + [cand]
            new_candidates = [c for c in candidates[i+1:] if all(local_W[c][cc] > 0 or c == cc for cc in new_clique)]
            
            solve(new_clique, new_candidates, deadline)
    
    # Start with empty clique, all nodes as candidates
    solve([], list(range(m)), deadline)
    
    if best_clique:
        return [nodes[i] for i in best_clique]
    return []

def local_search_polish(A, pool, deadline):
    """Local search to improve the given set using swap moves."""
    current_A = sorted(set(A))
    if len(current_A) != 20:
        return A
    
    current_score = compute_metrics(current_A)
    best_score = current_score
    best_A = current_A[:]
    
    pool_set = set(pool)
    pool_list = sorted(pool_set)
    pool_set_list = list(pool_set)
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
        
        improved = False
        # Try systematic swaps: for each position, try top candidates by collision
        for idx in range(20):
            if time.time() >= deadline:
                break
            
            old_val = current_A[idx]
            # Try a few random candidates
            for _ in range(5):
                if time.time() >= deadline:
                    break
                new_val = pool_set_list[random.randint(0, len(pool_set_list) - 1)]
                if new_val == old_val or new_val in current_A:
                    continue
                
                trial_A = current_A[:]
                trial_A[idx] = new_val
                trial_A = sorted(set(trial_A))
                
                if len(trial_A) != 20:
                    continue
                
                trial_score = compute_metrics(trial_A)
                if trial_score < current_score:
                    current_A = trial_A
                    current_score = trial_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_A = current_A[:]
                    no_improve = 0
                    break
            if time.time() >= deadline:
                break
        
        if not improved:
            no_improve += 1
            # Perturbation: swap two random elements
            if no_improve % 3 == 0:
                idx1 = random.randint(0, 19)
                idx2 = random.randint(0, 19)
                if idx1 != idx2:
                    new_val1 = pool_set_list[random.randint(0, len(pool_set_list) - 1)]
                    new_val2 = pool_set_list[random.randint(0, len(pool_set_list) - 1)]
                    trial_A = current_A[:]
                    trial_A[idx1] = new_val1
                    trial_A[idx2] = new_val2
                    trial_A = sorted(set(trial_A))
                    if len(trial_A) == 20:
                        trial_score = compute_metrics(trial_A)
                        if trial_score < current_score:
                            current_A = trial_A
                            current_score = trial_score
                            if current_score < best_score:
                                best_score = current_score
                                best_A = current_A[:]
                            no_improve = 0
    
    return best_A

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Generate candidate pool - focus on smooth numbers for high collision density
    pool = set(range(1, 51))
    pool.update(generate_smooth_numbers([2, 3, 5], 10000))
    pool.update(generate_smooth_numbers([2, 3, 5, 7], 10000))
    for p in [2, 3, 5, 7, 11]:
        v = 1
        while v <= 10000:
            pool.add(v)
            v *= p
    
    pool = sorted([x for x in pool if 1 <= x <= 10000])
    
    # Build collision graph
    W, vals = build_collision_graph(pool)
    
    # Get initial set via greedy modularity
    initial_set = greedy_modularity_selection(W, vals, k=20)
    
    # Also try branch-and-bound clique search
    random.seed(42)
    other_initials = [
        list(range(1, 21)),
        generate_smooth_numbers([2, 3, 5], 10000)[:20],
        generate_smooth_numbers([2, 3, 5, 7], 10000)[:20],
        sorted(random.sample(pool, 20))
    ]
    
    # Add branch-and-bound result
    bnb_deadline = start_time + 30.0
    bnb_result = branch_and_bound_clique(W, 20, bnb_deadline, max_nodes=40)
    if len(bnb_result) == 20:
        other_initials.append(sorted([vals[i] for i in bnb_result]))
    
    best_score = float('inf')
    best_A = initial_set
    
    # Evaluate all initial sets
    candidates = [initial_set] + other_initials
    for s in candidates:
        s = sorted(set(s))
        if len(s) == 20:
            score = compute_metrics(s)
            if score < best_score:
                best_score = score
                best_A = s[:]
    
    # Polish with local search
    if time.time() < deadline - 5:
        best_A = local_search_polish(best_A, pool, deadline - 3)
    
    # Final check
    final_score = compute_metrics(best_A)
    if final_score < best_score:
        best_score = final_score
    
    return best_A

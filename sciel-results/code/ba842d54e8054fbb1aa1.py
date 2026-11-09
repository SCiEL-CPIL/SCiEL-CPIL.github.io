import math
import random
import time
from itertools import combinations

def construct():
    N = 255
    deadline = time.time() + 85  # 5s safety margin
    
    # Candidates must share a factor with 255 (primes: 3, 5, 17)
    candidates = [x for x in range(1, N + 1) if math.gcd(x, N) > 1]
    n_cand = len(candidates)
    cand_to_idx = {val: idx for idx, val in enumerate(candidates)}
    idx_255 = cand_to_idx[255]
    
    # Precompute adjacency: adj[i] = set of j where gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # For fast compatibility check: compat_mask[i] is a set of indices compatible with i
    # We'll use a list of sets for O(1) membership tests
    
    def greedy_clique(start_node=None, random_order=False, order=None):
        if order is None:
            if random_order:
                nodes = list(range(n_cand))
                random.shuffle(nodes)
            else:
                nodes = sorted(range(n_cand), key=lambda x: len(adj[x]), reverse=True)
        else:
            nodes = list(order)
        
        clique = []
        if start_node is not None:
            if start_node in nodes:
                nodes.remove(start_node)
            nodes.insert(0, start_node)
            clique.append(start_node)
            candidates_for_next = set(adj[start_node])
        else:
            candidates_for_next = set(nodes)
        
        for node in nodes:
            if node in candidates_for_next:
                clique.append(node)
                candidates_for_next &= adj[node]
        
        return clique
    
    def validate_set(S):
        S = set(S)
        if 255 not in S:
            return False
        if len(S) < 1:
            return False
        for a, b in combinations(sorted(S), 2):
            if math.gcd(a, b) == 1:
                return False
        return True
    
    def local_search(S, max_rounds=50, time_limit=None):
        """Swap-based local search: try replacing one element with another to increase size."""
        S = set(S)
        for _ in range(max_rounds):
            if time_limit is not None and time.time() > time_limit:
                break
            improved = False
            S_list = sorted(S)
            for i, remove in enumerate(S_list):
                if time_limit is not None and time.time() > time_limit:
                    break
                S_minus = S - {remove}
                # Find a candidate not in S_minus that is compatible with all of S_minus
                for cand in candidates:
                    if cand in S_minus or cand == remove:
                        continue
                    # Check if cand is compatible with all elements in S_minus
                    compatible = True
                    for a in S_minus:
                        if math.gcd(a, cand) == 1:
                            compatible = False
                            break
                    if compatible:
                        S = S_minus | {cand}
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return S
    
    best_set = None
    best_size = 0
    
    # Strategy 1: Greedy clique with degree ordering
    clique_indices = greedy_clique(start_node=idx_255, random_order=False)
    clique_vals = [candidates[i] for i in clique_indices]
    if 255 not in clique_vals:
        clique_vals.append(255)
    if len(clique_vals) > best_size:
        best_set = set(clique_vals)
        best_size = len(best_set)
    
    # Strategy 2: Random restarts of greedy clique
    random.seed(42)
    restart_count = 0
    while time.time() < deadline - 20:
        clique_indices = greedy_clique(start_node=idx_255, random_order=True)
        clique_vals = [candidates[i] for i in clique_indices]
        if 255 not in clique_vals:
            clique_vals.append(255)
        if len(clique_vals) > best_size:
            best_set = set(clique_vals)
            best_size = len(best_set)
        restart_count += 1
    
    # Strategy 3: Union of multiples of prime subsets
    prime_sets_to_try = [
        [3],
        [5],
        [17],
        [2, 3],
        [2, 5],
        [3, 5],
        [2, 3, 5],
        [2, 3, 7],
        [2, 5, 7],
        [3, 5, 7],
        [2, 3, 11],
        [2, 5, 11],
        [3, 5, 11],
        [2, 3, 17],
        [3, 5, 17],
        [2, 5, 17],
        [2, 3, 5, 7],
        [2, 3, 5, 11],
        [2, 3, 5, 13],
    ]
    
    for primes in prime_sets_to_try:
        if time.time() > deadline - 10:
            break
        S = set()
        for p in primes:
            for k in range(p, N + 1, p):
                S.add(k)
        S.add(255)
        # Greedy complete: add any number that's compatible with all in S
        added = True
        while added and time.time() < deadline - 5:
            added = False
            for cand in range(1, N + 1):
                if cand in S:
                    continue
                compatible = True
                for a in S:
                    if math.gcd(a, cand) == 1:
                        compatible = False
                        break
                if compatible:
                    S.add(cand)
                    added = True
        # Local search polish
        S = local_search(S, max_rounds=10, time_limit=deadline - 5)
        if len(S) > best_size:
            best_set = S
            best_size = len(best_set)
    
    # Strategy 4: Final local search on the best result
    if best_set is None:
        best_set = set(x for x in range(1, N + 1) if x % 3 == 0)
    best_set = local_search(best_set, max_rounds=30, time_limit=deadline - 2)
    if len(best_set) > best_size:
        best_size = len(best_set)
    
    # Final validation and fallback
    if not validate_set(best_set):
        best_set = set(x for x in range(1, N + 1) if x % 3 == 0)
    
    return sorted(best_set)

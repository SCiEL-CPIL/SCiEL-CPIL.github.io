import time
from math import gcd
from itertools import combinations
import random

def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    
    Approach: Minimum hitting set on bad triples.
    1. Enumerate all bad triples (a,b,c) with gcd(a,b)=gcd(a,c)=gcd(b,c).
    2. Find minimum set of elements to remove so every bad triple is hit.
    3. Use branch-and-bound with good heuristics (most-constrained-first).
    """
    DEADLINE = time.time() + 80.0  # safety margin before 90s kill
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]
    
    # Precompute all bad triples
    bad_triples = []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            for k in range(j + 1, N + 1):
                if gcd_mat[i][j] == gcd_mat[i][k] == gcd_mat[j][k]:
                    bad_triples.append((i, j, k))
    
    # For each element, list of bad triple indices it belongs to
    elem_to_triples = [[] for _ in range(N + 1)]
    for ti, (a, b, c) in enumerate(bad_triples):
        elem_to_triples[a].append(ti)
        elem_to_triples[b].append(ti)
        elem_to_triples[c].append(ti)
    
    # Count how many bad triples each element is in
    triple_count = [len(elem_to_triples[x]) for x in range(N + 1)]
    
    num_triples = len(bad_triples)
    
    # For each element x, precompute the list of (u,v) pairs that conflict with x.
    # conflict_pairs[x] = list of (u,v) with u<v, u!=x, v!=x, such that gcd(x,u)==gcd(x,v)==gcd(u,v)
    
    conflict_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for u in range(1, N + 1):
            if u == x:
                continue
            for v in range(u + 1, N + 1):
                if v == x:
                    continue
                if gcd_mat[x][u] == gcd_mat[x][v] == gcd_mat[u][v]:
                    conflict_pairs[x].append((u, v))
    
    # Build bitmask representation for fast conflict checking
    # For each element x, create a bitmask of all elements that form a conflict pair with x
    # Actually, we need: for each x, a set of pairs (u,v) such that {x,u,v} is bad.
    # Better approach: for each x, precompute a 2D structure or use a different method.
    
    # Alternative: for each x, store a list of (u, v) pairs. To check if any pair is fully in S,
    # we can precompute for each x a list of pairs and check membership.
    
    # For faster checking, let's precompute for each x a list of pairs as tuples,
    # and also create a reverse index: for each pair (u,v), which elements x conflict with it.
    
    # pair_to_elements[(u,v)] = list of x such that {x,u,v} is a bad triple
    pair_to_elements = {}
    for x in range(1, N + 1):
        for (u, v) in conflict_pairs[x]:
            key = (u, v)
            if key not in pair_to_elements:
                pair_to_elements[key] = []
            pair_to_elements[key].append(x)
    
    def can_add_fast(x, S_set):
        """Check if x can be added to S without creating a bad triple.
        S_set should be a set for O(1) lookup."""
        for (u, v) in conflict_pairs[x]:
            if u in S_set and v in S_set:
                return False
        return True
    
    def count_conflicts_in_set(x, S_set):
        """Count how many conflict pairs of x are fully contained in S_set."""
        cnt = 0
        for (u, v) in conflict_pairs[x]:
            if u in S_set and v in S_set:
                cnt += 1
        return cnt
    
    def greedy_delete():
        """Greedy: repeatedly remove the element in the most bad triples within S."""
        S = set(range(1, N + 1))
        removed = []
        
        while True:
            if time.time() > DEADLINE:
                break
            
            # Find the element with the most active conflicts
            max_conflicts = -1
            to_remove = None
            for x in S:
                cnt = count_conflicts_in_set(x, S)
                if cnt > max_conflicts:
                    max_conflicts = cnt
                    to_remove = x
            
            if to_remove is None or max_conflicts == 0:
                break
            
            S.discard(to_remove)
            removed.append(to_remove)
        
        return S, removed
    
    def local_search(S, removed, max_rounds=100):
        """Try to re-add removed elements and swap to improve."""
        S = set(S)
        removed = list(removed)
        
        for round_num in range(max_rounds):
            if time.time() > DEADLINE:
                break
            improved = False
            
            # Try adding each removed element (sorted by fewest conflicts)
            def conflict_score(y):
                return count_conflicts_in_set(y, S)
            
            removed_sorted = sorted(removed, key=conflict_score)
            for x in removed_sorted:
                if time.time() > DEADLINE:
                    break
                if x in S:
                    continue
                if can_add_fast(x, S):
                    S.add(x)
                    removed.remove(x)
                    improved = True
            
            # Try swaps: remove one from S, add one from removed
            if not improved:
                # Sort removed by how "easy" they are to add
                for y in removed_sorted:
                    if time.time() > DEADLINE:
                        break
                    if y in S:
                        continue
                    
                    # Find the element x in S whose removal allows y to be added
                    # and whose removal loses the fewest "value"
                    best_swap_x = None
                    best_swap_loss = float('inf')
                    
                    # For each x in S, check if removing x allows y to be added
                    # and compute the "loss" of removing x
                    for x in list(S):
                        if time.time() > DEADLINE:
                            break
                        temp_S = S - {x}
                        if can_add_fast(y, temp_S):
                            # Compute loss: how many conflicts does x have in S?
                            loss = count_conflicts_in_set(x, S)
                            # We want to minimize loss (remove the element with fewest conflicts)
                            if loss < best_swap_loss:
                                best_swap_loss = loss
                                best_swap_x = x
                    
                    if best_swap_x is not None:
                        S.discard(best_swap_x)
                        S.add(y)
                        removed.remove(y)
                        removed.append(best_swap_x)
                        improved = True
                        break
            
            if not improved:
                break
        
        return S, removed
    
    # Phase 1: Greedy deletion
    S, removed = greedy_delete()
    best_S = set(S)
    best_size = len(S)
    
    # Phase 2: Local search
    S, removed = local_search(S, removed, max_rounds=100)
    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)
    
    # Phase 3: Multiple randomized greedy starts
    random.seed(42)
    
    for trial in range(30):
        if time.time() > DEADLINE:
            break
        
        # Randomized greedy
        S = set(range(1, N + 1))
        removed = []
        
        while True:
            if time.time() > DEADLINE:
                break
            
            # Find all elements with their conflict counts
            conflict_counts = {}
            for x in S:
                cnt = count_conflicts_in_set(x, S)
                conflict_counts[x] = cnt
            
            if not conflict_counts or max(conflict_counts.values()) == 0:
                break
            
            max_count = max(conflict_counts.values())
            candidates = [x for x, c in conflict_counts.items() if c == max_count]
            to_remove = random.choice(candidates)
            S.discard(to_remove)
            removed.append(to_remove)
        
        # Local search
        S, removed = local_search(S, removed, max_rounds=80)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Phase 4: More aggressive local search on best
    S, removed = local_search(best_S, [x for x in range(1, N + 1) if x not in best_S], max_rounds=150)
    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)
    
    # Phase 5: Additional multi-start with different strategies
    for trial in range(20):
        if time.time() > DEADLINE:
            break
        
        # Strategy: start with a subset, then try to grow
        # Randomly select a starting subset of size ~best_size
        start_size = max(1, best_size - random.randint(0, 5))
        start_set = set(random.sample(range(1, N + 1), start_size))
        
        # Verify start_set is valid
        valid = True
        for x in start_set:
            if not can_add_fast(x, start_set - {x}):
                valid = False
                break
        
        if not valid:
            # Fix by removing conflicting elements
            S = set(start_set)
            removed = [x for x in range(1, N + 1) if x not in S]
            # Greedy fix
            while True:
                if time.time() > DEADLINE:
                    break
                conflict_counts = {}
                for x in S:
                    cnt = count_conflicts_in_set(x, S)
                    conflict_counts[x] = cnt
                if not conflict_counts or max(conflict_counts.values()) == 0:
                    break
                max_count = max(conflict_counts.values())
                candidates = [x for x, c in conflict_counts.items() if c == max_count]
                to_remove = random.choice(candidates)
                S.discard(to_remove)
                removed.append(to_remove)
        else:
            S = set(start_set)
            removed = [x for x in range(1, N + 1) if x not in S]
        
        # Local search to grow
        S, removed = local_search(S, removed, max_rounds=100)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Phase 6: Final polish - try all possible single additions
    final_removed = [x for x in range(1, N + 1) if x not in best_S]
    for x in final_removed:
        if can_add_fast(x, best_S):
            best_S.add(x)
            final_removed.remove(x)
    
    if not best_S:
        best_S = {1}
    
    return sorted(best_S)

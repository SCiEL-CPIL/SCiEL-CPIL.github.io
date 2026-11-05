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
    
    # Precompute for each pair (a,b), the set of elements c that form a bad triple with a,b
    # bad_pair[a][b] = list of c such that (min(a,b,c), mid, max) is a bad triple
    # Actually, let's precompute: for each pair (u,v), which w would make (u,v,w) a bad triple?
    # gcd(u,v) == gcd(u,w) == gcd(v,w)
    # So w must satisfy: gcd(u,w) == gcd(u,v) AND gcd(v,w) == gcd(u,v)
    
    # For each element x, precompute which other elements y have gcd(x,y) = d for each d
    # This is expensive. Instead, let's precompute for each pair (u,v), the list of w
    # that would create a bad triple.
    
    # For efficiency, precompute: for each pair (u,v), the set of w in [1,N] such that
    # gcd(u,v) == gcd(u,w) == gcd(v,w)
    # We'll store this as a set for quick lookup.
    
    # Actually, let's just precompute a 3D-like structure:
    # For each (u,v) pair, store the list of w that form a bad triple.
    # But that's O(N^2 * N) = O(N^3) which is fine for N=128.
    
    # bad_with_pair[u][v] = list of w such that {u,v,w} is a bad triple
    bad_with_pair = {}
    for (a, b, c) in bad_triples:
        for pair in [(a,b), (a,c), (b,c)]:
            u, v = pair
            key = (min(u,v), max(u,v))
            if key not in bad_with_pair:
                bad_with_pair[key] = []
            w = a ^ b ^ c  # the third element
            bad_with_pair[key].append(w)
    
    # Now for the "can_add" check: given set S and element x,
    # x can be added iff for all pairs (u,v) in S, x is NOT in bad_with_pair[(u,v)]
    # Equivalently, for all pairs (u,v) in S, gcd(x,u) != gcd(u,v) OR gcd(x,v) != gcd(u,v)
    
    # Let's use a different approach for speed:
    # For each element x, precompute for each other element y, the gcd value.
    # Then "can_add" check: for all pairs (u,v) in S, check if gcd(x,u)==gcd(u,v) and gcd(x,v)==gcd(u,v)
    
    # Better: for each x, precompute the set of "conflict pairs" - pairs (u,v) such that
    # adding x would create a bad triple. This is: pairs (u,v) where gcd(x,u)==gcd(x,v)==gcd(u,v)
    
    # Let's precompute for each x, the list of (u,v) pairs that conflict with x.
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
    
    # Now can_add check: x can be added to S iff no pair (u,v) in conflict_pairs[x] has both u,v in S
    # We can check this by: for each (u,v) in conflict_pairs[x], check if u in S and v in S
    
    def can_add_fast(x, S):
        """Check if x can be added to S without creating a bad triple."""
        for (u, v) in conflict_pairs[x]:
            if u in S and v in S:
                return False
        return True
    
    def greedy_delete():
        """Greedy: repeatedly remove the element in the most bad triples."""
        S = set(range(1, N + 1))
        removed = []
        
        # Count active bad triples per element
        # An active bad triple is one where all 3 elements are in S
        # We'll maintain a count for each element
        
        # Initial count: triple_count[x] for each x
        active_count = [0] * (N + 1)
        for x in range(1, N + 1):
            active_count[x] = triple_count[x]
        
        # But we need to update when we remove elements.
        # When we remove element r, for each triple containing r, the other two elements
        # lose one active triple count.
        
        # Actually, let's just recompute or use a simpler approach.
        # Given the small N, let's use the conflict_pairs approach for checking.
        
        while True:
            if time.time() > DEADLINE:
                break
            
            # Find the element with the most conflicts (in most bad triples within S)
            # Count for each x in S, how many pairs (u,v) in S are in conflict_pairs[x]
            max_conflicts = -1
            to_remove = None
            for x in S:
                cnt = 0
                for (u, v) in conflict_pairs[x]:
                    if u in S and v in S:
                        cnt += 1
                if cnt > max_conflicts:
                    max_conflicts = cnt
                    to_remove = x
            
            if to_remove is None or max_conflicts == 0:
                break
            
            S.discard(to_remove)
            removed.append(to_remove)
        
        return S, removed
    
    def local_search(S, removed, max_rounds=50):
        """Try to re-add removed elements and swap to improve."""
        S = set(S)
        removed = list(removed)
        
        for round_num in range(max_rounds):
            if time.time() > DEADLINE:
                break
            improved = False
            
            # Try adding each removed element
            for x in removed[:]:
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
                # Sort removed by how "easy" they are to add (fewer conflicts)
                removed_sorted = sorted(removed, key=lambda y: len([
                    (u,v) for (u,v) in conflict_pairs[y] if u in S or v in S
                ]))
                
                for y in removed_sorted:
                    if time.time() > DEADLINE:
                        break
                    if y in S:
                        continue
                    # Try removing each x from S and adding y
                    for x in list(S):
                        if time.time() > DEADLINE:
                            break
                        temp_S = S - {x}
                        if can_add_fast(y, temp_S):
                            S = temp_S
                            S.add(y)
                            removed.remove(y)
                            removed.append(x)
                            improved = True
                            break
                    if improved:
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
    
    for trial in range(20):
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
                cnt = 0
                for (u, v) in conflict_pairs[x]:
                    if u in S and v in S:
                        cnt += 1
                conflict_counts[x] = cnt
            
            if not conflict_counts or max(conflict_counts.values()) == 0:
                break
            
            max_count = max(conflict_counts.values())
            candidates = [x for x, c in conflict_counts.items() if c == max_count]
            to_remove = random.choice(candidates)
            S.discard(to_remove)
            removed.append(to_remove)
        
        # Local search
        S, removed = local_search(S, removed, max_rounds=50)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Phase 4: More aggressive local search on best
    S, removed = local_search(best_S, [x for x in range(1, N + 1) if x not in best_S], max_rounds=200)
    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)
    
    # Phase 5: Simulated annealing
    if time.time() < DEADLINE - 10:
        T = 5.0
        cooling = 0.995
        S = set(best_S)
        removed = [x for x in range(1, N + 1) if x not in S]
        
        for sa_iter in range(5000):
            if time.time() > DEADLINE - 5:
                break
            
            # Random move: either add a removed element or swap
            move_type = random.random()
            
            if move_type < 0.3 and removed:
                # Try to add a random removed element
                y = random.choice(removed)
                if can_add_fast(y, S):
                    S.add(y)
                    removed.remove(y)
                else:
                    # Check how many conflicts
                    conflicts = 0
                    for (u, v) in conflict_pairs[y]:
                        if u in S and v in S:
                            conflicts += 1
                    # Accept with probability based on T
                    if random.random() < (0.1 / (conflicts + 1)):
                        # Force add - but this creates invalid state, so skip
                        pass
            
            elif move_type < 0.7 and removed and S:
                # Swap: remove random x from S, add random y from removed
                x = random.choice(list(S))
                y = random.choice(removed)
                temp_S = S - {x}
                if can_add_fast(y, temp_S):
                    S = temp_S
                    S.add(y)
                    removed.remove(y)
                    removed.append(x)
            
            elif S and len(S) > 1:
                # Remove a random element (to explore)
                x = random.choice(list(S))
                S.discard(x)
                removed.append(x)
                # With some probability, immediately try to re-add
                if random.random() < 0.5:
                    if can_add_fast(x, S):
                        S.add(x)
                        removed.remove(x)
            
            T *= cooling
        
        # Final local search after SA
        S, removed = local_search(S, removed, max_rounds=50)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    if not best_S:
        best_S = {1}
    
    return sorted(best_S)

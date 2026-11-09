import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = 180
    
    # Precompute triples (a,b,c) such that 1/a = 1/b + 1/c
    # a < b < c
    # (b-a)(c-a) = a^2
    triples = []
    for a in range(1, n + 1):
        a2 = a * a
        # Find divisors d of a2
        # d and e such that d*e = a2
        # b = a + d, c = a + e
        # We need b < c => d < e
        # b, c <= n => a + d <= n, a + e <= n
        # So d <= n - a, e <= n - a
        d = 1
        while d * d <= a2:
            if a2 % d == 0:
                e = a2 // d
                if d < e:
                    b = a + d
                    c = a + e
                    if c <= n:
                        triples.append((a, b, c))
            d += 1
            
    # Build conflict graph or list of pairs for each vertex
    # Incidence list: for each index i (0 to 179, value i+1),
    # store list of tuples (j, k) such that (i+1, j+1, k+1) is a triple.
    # Note: if i is the 'a', pairs are (b-1, c-1).
    # If i is the 'b', pairs are (a-1, c-1).
    # If i is the 'c', pairs are (a-1, b-1).
    
    conflicts = [[] for _ in range(n)]
    for (a, b, c) in triples:
        # a is smallest
        conflicts[a - 1].append((b - 1, c - 1))
        # b is middle
        conflicts[b - 1].append((a - 1, c - 1))
        # c is largest
        conflicts[c - 1].append((a - 1, b - 1))
        
    def is_valid(S):
        # S is a boolean list or set of indices
        # Check all triples
        for (a, b, c) in triples:
            if S[a - 1] and S[b - 1] and S[c - 1]:
                return False
        return True

    best_set = None
    best_size = 0
    
    def try_add(S, a):
        """Try to add element a (0-indexed) to set S. Returns True if successful."""
        for (u, v) in conflicts[a]:
            if S[u] and S[v]:
                return False
        S[a] = True
        return True

    def local_search(S, max_iters=5000):
        """Improve set S using greedy local search: try adding any element, then remove conflicting ones."""
        cur_size = sum(S)
        improved = True
        while improved and cur_size > 0:
            improved = False
            # Try to add each missing element in random order
            missing = [i for i in range(n) if not S[i]]
            random.shuffle(missing)
            for a in missing:
                if try_add(S, a):
                    cur_size += 1
                    improved = True
            # Try to remove and re-add elements to escape local optima
            present = [i for i in range(n) if S[i]]
            random.shuffle(present)
            for a in present:
                # Try removing a and see if we can add more
                S[a] = False
                cur_size -= 1
                # Try to add all missing elements
                missing2 = [i for i in range(n) if not S[i]]
                random.shuffle(missing2)
                added = 0
                for b in missing2:
                    if try_add(S, b):
                        added += 1
                # If we added more than we removed, keep the change
                if added > 1:  # net gain
                    cur_size += added
                    improved = True
                else:
                    # Restore a
                    try_add(S, a)
                    cur_size += 1
        return cur_size

    # Strategy 1: Suffix + Greedy Add
    # Try different starting K
    for K in range(1, n + 1):
        if time.time() - t0 > DEADLINE - 15:
            break
            
        # Initialize S with {K, ..., n}
        S = [False] * n
        for i in range(K - 1, n):
            S[i] = True
            
        # Check if this suffix is valid on its own
        valid_suffix = True
        for (a, b, c) in triples:
            if a >= K and b >= K and c >= K:
                valid_suffix = False
                break
        
        if not valid_suffix:
            continue
            
        cur_size = n - K + 1
        if cur_size > best_size:
            best_size = cur_size
            best_set = S[:]
            
        # Greedy add smaller numbers
        # Iterate a from K-1 down to 1
        for a in range(K - 1, 0, -1):
            if S[a - 1]:
                continue
            # Check if adding a creates a violation
            can_add = True
            for (u, v) in conflicts[a - 1]:
                if S[u] and S[v]:
                    can_add = False
                    break
                    
            if can_add:
                S[a - 1] = True
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = S[:]
                    
    # Strategy 2: Local search from best set found
    if best_set is not None and time.time() - t0 < DEADLINE - 5:
        S2 = best_set[:]
        cur_size = local_search(S2)
        if cur_size > best_size:
            best_size = cur_size
            best_set = S2[:]

    # Strategy 3: Simulated Annealing
    # Formulate as Max-SAT: maximize number of satisfied clauses
    # Each clause is "not (all three of triple are present)"
    # Objective: maximize sum over triples of [1 - prod(S[a],S[b],S[c])]
    # Equivalently, minimize number of violated triples
    # We use SA to optimize this landscape
    
    def simulate_annealing(S_init, t_max, t_min, cooling_rate, max_steps):
        """Run simulated annealing starting from S_init."""
        S = S_init[:]
        cur_size = sum(S)
        
        # Compute current number of violated triples
        def count_violations(S):
            cnt = 0
            for (a, b, c) in triples:
                if S[a-1] and S[b-1] and S[c-1]:
                    cnt += 1
            return cnt
        
        cur_viol = count_violations(S)
        best_local_viol = cur_viol
        best_local_set = S[:]
        
        T = t_max
        step = 0
        while T > t_min and step < max_steps and time.time() - t0 < DEADLINE - 3:
            # Pick a random element to flip
            i = random.randint(0, n - 1)
            
            # Compute delta in violations if we flip element i
            if S[i]:
                # Removing i: violations decrease by number of triples containing i where all three are present
                delta_viol = 0
                for (u, v) in conflicts[i]:
                    if S[u] and S[v]:
                        delta_viol -= 1  # violation removed
            else:
                # Adding i: violations increase by number of triples containing i where the other two are present
                delta_viol = 0
                for (u, v) in conflicts[i]:
                    if S[u] and S[v]:
                        delta_viol += 1  # violation added
            
            # Acceptance probability
            if delta_viol <= 0:
                # Always accept improvements or neutral moves
                S[i] = not S[i]
                cur_viol += delta_viol
                cur_size += 1 if delta_viol == 0 and not S[i] else (-1 if S[i] else 1)
                # Actually, let's recompute size properly
                cur_size = sum(S)
                
                if cur_viol < best_local_viol:
                    best_local_viol = cur_viol
                    best_local_set = S[:]
            else:
                # Worse move: accept with probability exp(-delta_viol / T)
                if random.random() < math.exp(-delta_viol / T):
                    S[i] = not S[i]
                    cur_viol += delta_viol
                    cur_size = sum(S)
                    
                    if cur_viol < best_local_viol:
                        best_local_viol = cur_viol
                        best_local_set = S[:]
            
            T *= cooling_rate
            step += 1
        
        return best_local_set, best_local_viol
    
    if time.time() - t0 < DEADLINE - 10:
        # Run SA from best set and from several random restarts
        sa_configs = [
            (10.0, 0.01, 0.995, 5000),
            (5.0, 0.01, 0.99, 3000),
            (20.0, 0.05, 0.997, 8000),
        ]
        
        for restart in range(15):
            if time.time() - t0 > DEADLINE - 3:
                break
            
            if restart == 0:
                S_init = best_set[:]
            else:
                # Random starting point
                K = random.randint(30, 120)
                S_init = [False] * n
                for i in range(K - 1, n):
                    S_init[i] = True
                # Check validity
                valid = True
                for (a, b, c) in triples:
                    if a >= K and b >= K and c >= K:
                        valid = False
                        break
                if not valid:
                    continue
                # Greedy add
                for a in range(K - 1, 0, -1):
                    if S_init[a - 1]:
                        continue
                    can_add = True
                    for (u, v) in conflicts[a - 1]:
                        if S_init[u] and S_init[v]:
                            can_add = False
                            break
                    if can_add:
                        S_init[a - 1] = True
            
            # Run SA with different configs
            for t_max, t_min, cooling, max_steps in sa_configs:
                if time.time() - t0 > DEADLINE - 3:
                    break
                
                S_sa, viol = simulate_annealing(S_init, t_max, t_min, cooling, max_steps)
                
                # If SA found a valid set (0 violations), check its size
                if viol == 0:
                    sa_size = sum(S_sa)
                    if sa_size > best_size:
                        best_size = sa_size
                        best_set = S_sa[:]
                else:
                    # SA didn't find valid set, but we can try to repair
                    # by removing elements involved in violations
                    S_repair = S_sa[:]
                    improved = True
                    while improved and time.time() - t0 < DEADLINE - 3:
                        improved = False
                        for (a, b, c) in triples:
                            if S_repair[a-1] and S_repair[b-1] and S_repair[c-1]:
                                # Remove the element with most conflicts
                                # Simple heuristic: remove 'a' (smallest)
                                S_repair[a-1] = False
                                improved = True
                                break
                    
                    if is_valid(S_repair):
                        repair_size = sum(S_repair)
                        if repair_size > best_size:
                            best_size = repair_size
                            best_set = S_repair[:]

    if best_set is None:
        best_set = [False] * n
        best_set[n-1] = True # [180]
        best_size = 1

    # Final local search polish
    if time.time() - t0 < DEADLINE - 2:
        S_final = best_set[:]
        final_size = local_search(S_final)
        if final_size > best_size:
            best_size = final_size
            best_set = S_final[:]

    # Convert to list of values
    out = [i + 1 for i in range(n) if best_set[i]]
    
    return out

import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    def gen_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def solve_instance(elements, seed_off, budget):
        t_start = time.time()
        V = list(elements)
        V_set = set(V)
        n = len(V)
        
        # Build Hyperedges: list of tuples (a, b, c) such that a+b=c
        # We only care about distinct elements? No, a=b is allowed.
        # So (a, a, 2a) is a hyperedge if 2a in V.
        # (a, b, a+b) is a hyperedge if a!=b and a+b in V.
        
        hyperedges = []
        V_list = list(V_set)
        for i, a in enumerate(V_list):
            # a + a = 2a
            if 2 * a in V_set:
                hyperedges.append((a, a, 2 * a))
            for j in range(i + 1, len(V_list)):
                b = V_list[j]
                c = a + b
                if c in V_set:
                    hyperedges.append((a, b, c))
        
        # Build adjacency: adj[v] = list of (u, w) such that {v, u, w} is a hyperedge.
        # For (a, a, 2a):
        #   v=a: (a, 2a)
        #   v=2a: (a, a)
        # For (a, b, c) with a!=b:
        #   v=a: (b, c)
        #   v=b: (a, c)
        #   v=c: (a, b)
        
        adj = {v: [] for v in V_set}
        for (a, b, c) in hyperedges:
            if a == b:
                adj[a].append((a, 2 * a))
                adj[2 * a].append((a, a))
            else:
                adj[a].append((b, c))
                adj[b].append((a, c))
                adj[c].append((a, b))
        
        # Precompute degree (number of hyperedges containing v)
        deg = {v: 0 for v in V_set}
        for (a, b, c) in hyperedges:
            deg[a] += 1
            deg[b] += 1
            deg[c] += 1

        def is_sum_free(S):
            """Check if S is sum-free. O(|S|^2) but fast for small S."""
            S_set = set(S)
            for a in S_set:
                for b in S_set:
                    if a + b in S_set:
                        return False
            return True

        def get_conflicts(v, S):
            """
            Return list of conflicts for adding v to S.
            Each conflict is a pair (u, w) such that {v, u, w} is a hyperedge and u, w in S.
            If u == v, then only w needs to be in S (and it is, by construction of conflict).
            """
            conflicts = []
            for (u, w) in adj[v]:
                if u == v:
                    # v + v = w. Need w in S.
                    if w in S:
                        conflicts.append((w, w))
                else:
                    # v + u = w or u + v = w. Need u in S and w in S.
                    if u in S and w in S:
                        conflicts.append((u, w))
            return conflicts

        def try_add(v, S, rng):
            """
            Try to add v to S.
            Returns (new_S, gain) if successful, else (None, 0).
            """
            if v in S:
                return S, 0
            
            conflicts = get_conflicts(v, S)
            
            if not conflicts:
                S.add(v)
                return S, 1
            
            # We have conflicts.
            # We need to remove a set R from S such that for every (u, w) in conflicts:
            # if u == w: u in R.
            # if u != w: u in R or w in R.
            # We want to minimize |R|.
            
            # Count frequency of each vertex in conflicts
            freq = {}
            for (u, w) in conflicts:
                if u == w:
                    freq[u] = freq.get(u, 0) + 1
                else:
                    freq[u] = freq.get(u, 0) + 1
                    freq[w] = freq.get(w, 0) + 1
            
            if not freq:
                return None, 0
                
            max_freq = max(freq.values())
            candidates = [k for k, v in freq.items() if v == max_freq]
            
            # Try removing each candidate
            best_S = None
            best_gain = 0
            
            for rem in candidates:
                trial_S = set(S)
                trial_S.discard(rem)
                trial_S.add(v)
                
                # Check if trial_S is valid (no conflicts remaining)
                valid = True
                for (u, w) in conflicts:
                    if u == w:
                        if u in trial_S:
                            valid = False
                            break
                    else:
                        if u in trial_S and w in trial_S:
                            valid = False
                            break
                
                if valid:
                    gain = 1 - 1 # Added v, removed rem. Net 0.
                    # We only accept if gain > 0 OR (gain == 0 and random accept).
                    # For now, let's only accept gain > 0.
                    # If gain == 0, we might cycle.
                    
                    # Actually, if we can add v and remove rem, and the new set has FEWER total conflicts,
                    # it might be better. But we don't track that easily.
                    
                    # Let's look for Gain > 0 cases first.
                    # Gain > 0 implies we added v and removed 0 vertices.
                    # That means conflicts was empty.
                    # So this branch (conflicts not empty) always has gain <= 0.
                    
                    # So we should only enter this branch if we are willing to do 0-gain moves?
                    # Or if we can remove 1 vertex and add 2? (2-opt)
                    
                    # Let's stick to 1-opt (Add 1, Remove 0) for positive gain.
                    # And 1-swap (Add 1, Remove 1) for neutral moves (with probability).
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_S = trial_S
            
            # If best_gain > 0, return it.
            # If best_gain == 0, return with probability p.
            
            return best_S, best_gain

        def local_search(S, V_set, budget, rng):
            t_start = time.time()
            S = set(S)
            best_S = set(S)
            best_size = len(S)
            
            # Precompute adj for this instance
            # (Already done in outer scope? No, adj depends on V_set)
            # I need to build adj inside solve_instance.
            
            while time.time() - t_start < budget:
                # Pick random v not in S
                remaining = [x for x in V_set if x not in S]
                if not remaining:
                    break
                v = rng.choice(remaining)
                
                # Get conflicts
                conflicts = get_conflicts(v, S)
                
                if not conflicts:
                    S.add(v)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    continue
                
                # Try to resolve by removing one vertex
                # Find vertex in S that is in the most conflicts
                freq = {}
                for (u, w) in conflicts:
                    if u == w:
                        freq[u] = freq.get(u, 0) + 1
                    else:
                        freq[u] = freq.get(u, 0) + 1
                        freq[w] = freq.get(w, 0) + 1
                
                if not freq:
                    continue
                    
                max_f = max(freq.values())
                candidates = [k for k, c in freq.items() if c == max_f]
                rem = rng.choice(candidates)
                
                # Try removing rem
                trial_S = set(S)
                trial_S.discard(rem)
                trial_S.add(v)
                
                # Check if valid
                valid = True
                for (u, w) in conflicts:
                    if u == w:
                        if u in trial_S:
                            valid = False
                            break
                    else:
                        if u in trial_S and w in trial_S:
                            valid = False
                            break
                
                if valid:
                    # Size is same (added 1, removed 1)
                    # Accept with probability 0.2 to explore
                    if rng.random() < 0.2:
                        S = trial_S
                    # Update best if size is larger (not possible here)
                    
            return best_S

        # Initial Solutions
        best_S = set()
        best_size = 0
        rng = random.Random(seed_off)
        
        # 1. Odds
        S_odds = set(x for x in V_set if x % 2 != 0)
        if len(S_odds) > best_size:
            best_size = len(S_odds)
            best_S = S_odds
            
        # 2. Greedy by Degree (fewest conflicts first)
        # Conflict count: number of hyperedges containing v
        S_greedy = set()
        for v in sorted(V_set, key=lambda x: (deg[x], abs(x))):
            # Check if v can be added
            can_add = True
            for (u, w) in adj[v]:
                if u == v:
                    if w in S_greedy:
                        can_add = False
                        break
                else:
                    if u in S_greedy and w in S_greedy:
                        can_add = False
                        break
            if can_add:
                S_greedy.add(v)
                
        if len(S_greedy) > best_size:
            best_size = len(S_greedy)
            best_S = S_greedy
            
        # 3. Local Search from best_S
        budget_ls = budget * 0.8
        best_S = local_search(best_S, V_set, budget_ls, rng)
        
        return list(best_S)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]

    results = []
    budgets = [12, 30, 60]
    for i, elems in enumerate(instances):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 5
        if remaining < 4:
            res = [x for x in elems if x % 2 != 0]
            results.append(res)
            continue
        b = min(budgets[i], max(4, remaining))
        res = solve_instance(elems, seed_off=801 + i * 7, budget=b)
        results.append(res)

    return results

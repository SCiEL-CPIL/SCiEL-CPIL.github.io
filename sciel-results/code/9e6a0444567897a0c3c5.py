import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Set recursion limit high just in case
    sys.setrecursionlimit(10000)

    def is_sum_free(n, S):
        """
        Check if S is sum-free modulo n.
        S is a set of integers.
        """
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if (a + b) % n in S_set:
                    return False
        return True

    def greedy_alpha(adj, n):
        """
        Compute the size of a greedy maximal independent set.
        Returns the size.
        """
        candidates = (1 << n) - 1
        size = 0
        while candidates:
            # Pick vertex with min degree in candidates
            best_v = -1
            min_deg = n + 1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
            
            size += 1
            candidates &= ~adj[best_v] & ~(1 << best_v)
        
        return size

    def has_independent_set_10(adj, n):
        """
        Exact check for independent set of size 10.
        """
        mask_all = (1 << n) - 1
        
        # Quick greedy lower bound
        if greedy_alpha(adj, n) >= 10:
            return True

        def backtrack(candidates, size):
            if size == 10:
                return True
            if candidates == 0:
                return False
            
            remaining = candidates.bit_count()
            if size + remaining < 10:
                return False
            
            # Pick vertex with max degree in candidates to prune effectively
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Branch 1: Include best_v
            if backtrack(candidates & ~adj[best_v] & ~(1 << best_v), size + 1):
                return True
            
            # Branch 2: Exclude best_v
            if backtrack(candidates & ~(1 << best_v), size):
                return True
            
            return False

        return backtrack(mask_all, 0)

    def build_adj(n, S):
        """
        Build adjacency bitmasks for circulant graph with difference set S.
        """
        adj = [0] * n
        S_set = set(S)
        for i in range(n):
            mask = 0
            for d in S_set:
                j = (i + d) % n
                if i != j:
                    mask |= (1 << j)
            adj[i] = mask
        return adj

    def find_circulant_graph(n):
        """
        Search for a circulant graph on n vertices that is triangle-free and has alpha < 10.
        Uses local search on the difference set S.
        """
        if time.time() > deadline:
            return None
        
        max_d = n // 2
        base_candidates = list(range(1, max_d + 1))
        
        # We will try a few random initial sets and then local search
        for attempt in range(100):
            if time.time() > deadline:
                return None
            
            # 1. Generate an initial random maximal sum-free set
            random.shuffle(base_candidates)
            D = set()
            for d in base_candidates:
                new_D = set(D)
                new_D.add(d)
                # Check sum-free for new_D (and n-d if needed)
                # For simplicity in local search, we work with the full set S
                # But D is the "half" set. Let's work with full S directly for clarity in local search?
                # No, working with D is more compact.
                # S = D U {n-d for d in D}
                
                # Check if adding d to D keeps S sum-free
                S_old = set(D)
                for x in D:
                    if x != n - x:
                        S_old.add(n - x)
                
                S_new = set(S_old)
                S_new.add(d)
                if d != n - d:
                    S_new.add(n - d)
                
                if is_sum_free(n, S_new):
                    D = new_D
            
            # Construct S from D
            S = set(D)
            for d in D:
                if d != n - d:
                    S.add(n - d)
            
            if not S:
                continue

            adj = build_adj(n, S)
            
            # Check greedy alpha
            g_alpha = greedy_alpha(adj, n)
            
            if g_alpha < 10:
                # Check exact alpha
                if not has_independent_set_10(adj, n):
                    # Success
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
            
            # 2. Local Search: Try to improve S to reduce greedy alpha
            # We will try random moves: Add, Remove, Swap
            # We maintain S as a set of differences in 1..n-1
            # But we must maintain symmetry: if d in S, n-d in S.
            # So we operate on the "half" set D.
            
            current_D = set(D)
            current_g_alpha = g_alpha
            no_improvement_count = 0
            
            for step in range(200): # Limit local search steps
                if time.time() > deadline:
                    return None
                
                if no_improvement_count > 50:
                    break
                    
                # Generate a random move
                move_type = random.choice(['add', 'remove', 'swap'])
                new_D = set(current_D)
                valid_move = False
                
                if move_type == 'add':
                    # Pick d not in current_D
                    available = [d for d in base_candidates if d not in current_D]
                    if not available:
                        continue
                    d = random.choice(available)
                    new_D.add(d)
                    
                elif move_type == 'remove':
                    if not current_D:
                        continue
                    d = random.choice(list(current_D))
                    new_D.discard(d)
                    
                elif move_type == 'swap':
                    if not current_D:
                        continue
                    d_in = random.choice(list(current_D))
                    available = [d for d in base_candidates if d not in current_D]
                    if not available:
                        continue
                    d_out = random.choice(available)
                    new_D.discard(d_in)
                    new_D.add(d_out)
                
                # Check if new_D is sum-free
                # Construct S from new_D
                S_new = set(new_D)
                for d in new_D:
                    if d != n - d:
                        S_new.add(n - d)
                
                if is_sum_free(n, S_new):
                    valid_move = True
                    adj_new = build_adj(n, S_new)
                    g_alpha_new = greedy_alpha(adj_new, n)
                    
                    if g_alpha_new < current_g_alpha:
                        # Accept improvement
                        current_D = new_D
                        current_g_alpha = g_alpha_new
                        no_improvement_count = 0
                        
                        # Check if we found a solution
                        if current_g_alpha < 10:
                            if not has_independent_set_10(adj_new, n):
                                edges = []
                                for i in range(n):
                                    for j in range(i + 1, n):
                                        if (adj_new[i] >> j) & 1:
                                            edges.append([i, j])
                                return {'n': n, 'edges': edges}
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1

        return None

    # Iterate n downwards
    for n in range(40, 24, -1):
        if time.time() > deadline:
            break
            
        result = find_circulant_graph(n)
        if result is not None:
            return result
            
    # Fallback
    n = 18
    edges = []
    for i in range(9):
        for j in range(9, 18):
            edges.append([i, j])
    
    return {'n': n, 'edges': edges}

if __name__ == "__main__":
    res = construct()
    print(f"Result: n={res['n']}, edges={len(res['edges'])}")

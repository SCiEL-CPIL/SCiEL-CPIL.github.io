import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def has_independent_set_10(adj, n):
        """Check if graph has independent set of size 10. Returns True if it does."""
        mask_all = (1 << n) - 1
        
        # Quick greedy lower bound
        candidates = mask_all
        size = 0
        while candidates and size < 10:
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
        
        if size >= 10:
            return True
        
        memo = {}
        
        def backtrack(candidates, size):
            key = (candidates, size)
            if key in memo:
                return memo[key]
            
            if size == 10:
                memo[key] = True
                return True
            if candidates == 0:
                memo[key] = False
                return False
            remaining = candidates.bit_count()
            if size + remaining < 10:
                memo[key] = False
                return False
            
            # Pick vertex with max degree in candidates
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
                memo[key] = False
                return False
            
            result = backtrack(candidates & ~adj[best_v] & ~(1 << best_v), size + 1)
            if result:
                memo[key] = True
                return True
            result = backtrack(candidates & ~(1 << best_v), size)
            memo[key] = result
            return result
        
        return backtrack(mask_all, 0)

    def is_triangle_free(adj, n):
        """Check if graph is triangle-free."""
        for u in range(n):
            neighbors = adj[u]
            temp = neighbors
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True

    def build_edge_list(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            temp = adj[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                temp &= temp - 1
        return edges

    def make_circulant(n, S):
        """Create circulant graph: vertex i connected to i+s mod n for s in S."""
        adj = [0] * n
        for s in S:
            s = s % n
            if s == 0:
                continue
            for i in range(n):
                j = (i + s) % n
                if j > i:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def is_sum_free(S, n):
        """Check if S is sum-free mod n: no a,b,c in S with a+b≡c mod n."""
        S_set = set(s % n for s in S if s % n != 0)
        for a in S_set:
            for b in S_set:
                c = (a + b) % n
                if c in S_set:
                    return False
        return True

    def make_andrasfai(k):
        """Andrásfai graph And(k): vertices Z_{3k}, i~j iff (j-i) mod 3k in {1, 3k-1}."""
        n = 3 * k
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 1 or d == n - 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, n

    def add_chords_to_andrasfai(adj, n, extra_offsets):
        """Add edges at extra_offsets to Andrásfai graph, checking triangle-freeness."""
        adj = list(adj)
        for offset in extra_offsets:
            offset = offset % n
            if offset == 0:
                continue
            # Check if adding all edges at this offset preserves triangle-freeness
            valid = True
            for i in range(n):
                j = (i + offset) % n
                if j <= i:
                    continue
                # Check common neighbors
                if adj[i] & adj[j]:
                    valid = False
                    break
            if valid:
                for i in range(n):
                    j = (i + offset) % n
                    if j > i:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        return adj

    best_result = None

    # 1. Try known constructions: Andrásfai graphs with optional chords
    for k in range(10, 6, -1):
        adj_k, n_k = make_andrasfai(k)
        # Try adding various extra offsets
        best_adj_k = None
        extra_patterns = [[], [2], [3], [2,3], [4], [2,4], [3,4], [2,3,4], [5], [2,5], [3,5],
                         [6], [2,6], [3,6], [4,6], [5,6], [2,3,6], [4,5,6], [2,4,6], [3,5,6],
                         [7], [2,7], [3,7], [4,7], [5,7], [6,7], [2,3,7], [4,5,7], [2,4,7], [3,5,7]]
        for extra in extra_patterns:
            if time.time() > deadline - 15:
                break
            candidate_adj = add_chords_to_andrasfai(adj_k, n_k, extra)
            if not has_independent_set_10(candidate_adj, n_k):
                if best_adj_k is None or len(build_edge_list(candidate_adj, n_k)) > len(build_edge_list(best_adj_k, n_k)):
                    best_adj_k = candidate_adj
        if best_adj_k is not None:
            best_result = {'n': n_k, 'edges': build_edge_list(best_adj_k, n_k)}
            break
        elif not has_independent_set_10(adj_k, n_k):
            best_result = {'n': n_k, 'edges': build_edge_list(adj_k, n_k)}
            break

    # 2. Systematic search over circulant graphs for n=40, 39, 38...
    for target_n in range(40, (best_result['n'] if best_result else 27), -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # Try structured sum-free sets first (arithmetic progressions)
        half = target_n // 2
        for start in range(1, min(6, half)):
            for step in range(1, 4):
                S_half = set()
                s = start
                while s <= half:
                    S_half.add(s)
                    s += step
                if len(S_half) < 3:
                    continue
                S = set()
                for s in S_half:
                    S.add(s)
                    S.add(target_n - s)
                if not is_sum_free(S, target_n):
                    continue
                adj = make_circulant(target_n, S)
                avg_deg = 2 * len(S_half)
                if avg_deg < target_n / 12:
                    continue
                if not has_independent_set_10(adj, target_n):
                    found = True
                    best_result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                    break
            if found:
                break
        
        # Random search with wider density range
        if not found:
            max_attempts = 80000
            attempts = 0
            while attempts < max_attempts and time.time() < deadline - 5:
                attempts += 1
                
                half = target_n // 2
                # Use structured random: pick a base density and perturb
                base_density = random.uniform(0.15, 0.45)
                S_half = set()
                for s in range(1, half + 1):
                    # Add some structure: favor certain residue classes
                    residue_bonus = 0.05 * ((s % 3 == 1) - (s % 3 == 2))
                    if random.random() < base_density + residue_bonus:
                        S_half.add(s)
                
                S = set()
                for s in S_half:
                    S.add(s)
                    S.add(target_n - s)
                
                if len(S) < 4:
                    continue
                
                if not is_sum_free(S, target_n):
                    continue
                
                adj = make_circulant(target_n, S)
                
                avg_deg = 2 * len(S_half)
                if avg_deg < target_n / 12:
                    continue
                
                if not has_independent_set_10(adj, target_n):
                    found = True
                    best_result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                    break
        
        if found:
            break
    
    # 3. If still not found large enough, try random triangle-free search
    if best_result is None or best_result['n'] < 39:
        def random_triangle_free_greedy(n, p_base, p_var):
            """Generate triangle-free graph with degree-biased edge addition."""
            adj = [0] * n
            degrees = [0] * n
            # Process vertices in random order, adding edges to lower-degree vertices first
            order = list(range(n))
            random.shuffle(order)
            for u in order:
                candidates = []
                for v in range(n):
                    if v != u and (adj[u] & (1 << v)) == 0 and (adj[v] & (1 << u)) == 0:
                        if (adj[u] & adj[v]) == 0:
                            candidates.append(v)
                random.shuffle(candidates)
                # Add edges with probability that favors lower-degree vertices
                for v in candidates:
                    p = p_base + p_var * (1 - degrees[v] / max(1, n - 1))
                    if random.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
            return adj
        
        min_target = (best_result['n'] + 1) if best_result else 30
        for target_n in range(40, min_target - 1, -1):
            if time.time() > deadline - 5:
                break
            for p_base in [0.3, 0.4, 0.5, 0.6]:
                for p_var in [0.2, 0.4, 0.6]:
                    for _ in range(500):
                        if time.time() > deadline - 5:
                            break
                        adj = random_triangle_free_greedy(target_n, p_base, p_var)
                        if not has_independent_set_10(adj, target_n):
                            best_result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                            break
                    if best_result and best_result['n'] == target_n:
                        break
                if best_result and best_result['n'] == target_n:
                    break
            if best_result and best_result['n'] == target_n:
                break
    
    # Fallback
    if best_result is None:
        adj9, n9 = make_andrasfai(9)
        best_result = {'n': n9, 'edges': build_edge_list(adj9, n9)}

    # Final verification
    final_adj = [0] * best_result['n']
    for u, v in best_result['edges']:
        final_adj[u] |= (1 << v)
        final_adj[v] |= (1 << u)
    
    assert is_triangle_free(final_adj, best_result['n']), "Result is not triangle-free!"
    assert not has_independent_set_10(final_adj, best_result['n']), "Result has independent set of size 10!"
    
    return best_result

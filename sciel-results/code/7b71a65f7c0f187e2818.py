import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def has_independent_set_10(adj, n):
        mask_all = (1 << n) - 1
        
        # Heuristic: If average degree is very low, alpha is likely >= 10
        # But we rely on B&B.
        
        def bb(candidates, size):
            if size >= 10:
                return True
            if candidates == 0:
                return False
            
            # Pruning
            if size + candidates.bit_count() < 10:
                return False
            
            # Select vertex with max degree in candidates
            # This helps find independent sets faster if they exist
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Branch 1: Include best_v
            # Remove best_v and its neighbors
            new_cands = candidates & ~(1 << best_v) & ~adj[best_v]
            if bb(new_cands, size + 1):
                return True
            
            # Branch 2: Exclude best_v
            new_cands = candidates & ~(1 << best_v)
            if bb(new_cands, size):
                return True
            
            return False
        
        return bb(mask_all, 0)

    def build_edge_list(adj, n):
        edges = []
        for i in range(n):
            temp = adj[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                temp ^= lsb
        return edges

    def make_circulant(n, S):
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
        S_set = set(s % n for s in S if s % n != 0)
        # Check a+b=c
        # Optimization: sort S_set
        S_list = sorted(S_set)
        for a in S_list:
            for b in S_list:
                c = (a + b) % n
                if c in S_set:
                    return False
        return True

    # Fallback: Andrásfai graph A_9 (n=27)
    # Construction of A_k:
    # Vertices 0..3k-1.
    # Edge (i, j) if (j-i) mod 3k is in {1, 3k-1} ? No.
    # Correct A_k construction:
    # It is a 3-partite graph.
    # But we can just use the known valid graph from the current program's logic if we can't find better.
    # The current program's make_andrasfai(9) uses diffs {1, k, 2k-1, 2k, 3k-1}.
    # Let's verify if that's sum-free.
    # k=9, n=27. S = {1, 9, 17, 18, 26}.
    # 1+1=2 (not in S). 1+9=10 (not). 1+17=18 (IN S!). So it's NOT sum-free?
    # If 1+17=18, then (0,1), (0,17), (1,18) is a triangle?
    # 0-1 (diff 1), 1-18 (diff 17), 0-18 (diff 18).
    # Yes, triangle. So the current program's A_9 construction might be flawed or I am misinterpreting "sum-free" vs "triangle-free" for non-circulant or specific circulant definitions.
    # Wait, A_k is NOT a circulant graph in the simple sense for k>3? Or is it?
    # Actually, A_k is a circulant graph.
    # Let's check A_3 (n=9). S={1, 3, 5, 6, 8}.
    # 1+5=6 (in S). Triangle 0-1-6-0?
    # 0-1 (1), 1-6 (5), 0-6 (6). Yes.
    # So the "make_andrasfai" in the current program likely produces triangles.
    # The current program checks `is_triangle_free`. If it fails, it falls back to A_6, then A_3, then C5.
    # So the current program might actually be returning C5 (n=5) or A_6 if A_6 is valid.
    # A_6 (n=18). S={1, 6, 11, 12, 17}.
    # 1+11=12 (in S). Triangle.
    # So the current program is likely returning a very small graph if the A_k construction is always triangle-free?
    # Wait, if `is_triangle_free` returns False, it goes to the next fallback.
    # If A_9, A_6, A_3 all have triangles, it returns C5 (n=5).
    # Score 1.1538 implies n is around 5? Or maybe the random search found something?
    # If seed is n=5, score 1.15 means n ~ 5.75? No, score is value/seed.
    # If seed value is 5, score 1.15 -> value 5.75? Impossible.
    # Maybe seed is 40? No.
    # Let's assume the current program's A_k construction is actually valid and I'm miscalculating sum-free.
    # In a circulant graph, a triangle exists if $a+b=c$ or $a+b=n-c$ etc.
    # My sum-free check `a+b=c` is only part of it.
    # For triangle-free circulant, we need no $a,b,c \in S$ such that $a+b \equiv c$, $a+b \equiv -c$, etc.
    # Basically, $S$ must not contain solutions to $x+y=z$ in $\mathbb{Z}_n$.
    # This is exactly the sum-free condition.
    # So if 1+17=18, and 1,17,18 are in S, there is a triangle.
    # So the current program's A_k is likely invalid.
    # Therefore, the current program likely relies on the random search or returns C5.
    
    # My new program will use correct sum-free check.
    
    best_result = None
    
    # 1. Try Circulant Search for n=40 down to 28
    for target_n in range(40, 27, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        attempts = 0
        max_attempts = 50000
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Generate random symmetric set
            half = target_n // 2
            # Try different densities
            # For n=40, we need alpha < 10.
            # Let's try density 0.3 to 0.5
            density = random.uniform(0.2, 0.5)
            S_half = set()
            for s in range(1, half + 1):
                if random.random() < density:
                    S_half.add(s)
            
            S = set()
            for s in S_half:
                S.add(s)
                S.add(target_n - s)
            
            if len(S) < 4:
                continue
                
            # Check sum-free
            if not is_sum_free(S, target_n):
                continue
                
            # Build graph
            adj = make_circulant(target_n, S)
            
            # Check independence
            if not has_independent_set_10(adj, target_n):
                found = True
                best_result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                break
        
        if found:
            break
            
    # 2. Fallback if no circulant found
    if best_result is None:
        # Try a known valid small graph
        # C5 is triangle free, alpha=2.
        n = 5
        adj = [0]*5
        for i in range(5):
            adj[i] |= (1 << ((i+1)%5)) | (1 << ((i+4)%5))
            adj[(i+1)%5] |= (1 << i)
            adj[(i+4)%5] |= (1 << i)
        best_result = {'n': n, 'edges': build_edge_list(adj, n)}
        
        # Try random triangle-free for n=39..28 if time permits
        # This is less likely to succeed than circulant search but worth a shot
        if time.time() < deadline - 10:
            for target_n in range(39, 27, -1):
                if time.time() > deadline - 5:
                    break
                for _ in range(1000):
                    if time.time() > deadline - 5:
                        break
                    adj_try = [0] * target_n
                    edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
                    random.shuffle(edges)
                    for u, v in edges:
                        if not (adj_try[u] & adj_try[v]):
                            # Add with some probability to make it dense enough?
                            # Or just add all possible to make it maximal triangle-free?
                            # Maximal triangle-free graphs often have small alpha?
                            # Let's just add all valid edges.
                            adj_try[u] |= (1 << v)
                            adj_try[v] |= (1 << u)
                    
                    if not has_independent_set_10(adj_try, target_n):
                        best_result = {'n': target_n, 'edges': build_edge_list(adj_try, target_n)}
                        break
                if best_result['n'] == target_n:
                    break

    return best_result

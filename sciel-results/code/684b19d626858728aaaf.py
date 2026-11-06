import time
import random

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def verify(n, colors):
        """Verify no color class contains a C6."""
        for c in range(4):
            adj = [[0] * n for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == c:
                        adj[i][j] = 1
                        adj[j][i] = 1
                    idx += 1
            
            # Check for C6: for each edge (u,v) in color c, check if there's a simple path of length 4 from u to v
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u][v] == 0:
                        continue
                    # Check if there's a simple path u-a-b-d-v of length 4 in color c
                    # a in N(u)\{v}, b in N(a)\{u,v}, d in N(b)\{u,a,v} and d in N(v)
                    for a in range(n):
                        if a == u or a == v or not adj[u][a]:
                            continue
                        for b in range(n):
                            if b == u or b == v or b == a or not adj[a][b]:
                                continue
                            for d in range(n):
                                if d == u or d == v or d == a or d == b:
                                    continue
                                if adj[b][d] and adj[d][v]:
                                    return False
        return True

    def attempt(n, time_budget):
        """
        Backtracking with forward checking.
        """
        local_deadline = time.time() + time_budget
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        
        # Adjacency bitsets: adj[c][i] is bitmask of neighbors of i in color c
        adj = [[0] * n for _ in range(4)]
        
        def is_valid(c, u, v):
            """Check if assigning color c to edge (u,v) creates a C6."""
            # Would create C6 iff there's a simple path u-a-b-d-v of length 4 in color c
            mask_u = adj[c][u]
            mask_v = adj[c][v]
            
            # a in N(u) \ {v}
            a_mask = mask_u & ~(1 << v)
            while a_mask:
                lsb_a = a_mask & -a_mask
                a = lsb_a.bit_length() - 1
                # b in N(a) \ {u, v}
                b_mask = adj[c][a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                while b_mask:
                    lsb_b = b_mask & -b_mask
                    b = lsb_b.bit_length() - 1
                    # d in N(b) \ {u, a, v} and d in N(v)
                    d_mask = adj[c][b] & mask_v & ~(1 << u) & ~(1 << a) & ~(1 << b)
                    if d_mask:
                        return False  # Invalid: would create C6
                    b_mask &= b_mask - 1
                a_mask &= a_mask - 1
            return True  # Valid
        
        # Backtracking with lexicographic edge order
        colors = [-1] * num_edges
        
        def backtrack(idx):
            if time.time() > local_deadline:
                return False
            if idx == num_edges:
                return True
            
            u, v = edges[idx]
            # Try colors 0,1,2,3
            for c in range(4):
                if is_valid(c, u, v):
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Undo
                    colors[idx] = -1
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
            
            return False
        
        # Try with different color orderings to increase diversity
        rng = random.Random(12345)
        
        for trial in range(50):
            if time.time() > local_deadline:
                break
            
            # Reset
            for c in range(4):
                for i in range(n):
                    adj[c][i] = 0
            for i in range(num_edges):
                colors[i] = -1
            
            # Shuffle color order
            color_order = list(range(4))
            rng.shuffle(color_order)
            
            def backtrack_shuffled(idx):
                if time.time() > local_deadline:
                    return False
                if idx == num_edges:
                    return True
                
                u, v = edges[idx]
                for c in color_order:
                    if is_valid(c, u, v):
                        colors[idx] = c
                        adj[c][u] |= (1 << v)
                        adj[c][v] |= (1 << u)
                        
                        if backtrack_shuffled(idx + 1):
                            return True
                        
                        colors[idx] = -1
                        adj[c][u] &= ~(1 << v)
                        adj[c][v] &= ~(1 << u)
                
                return False
            
            if backtrack_shuffled(0):
                return colors[:]
        
        return None

    # Try from n=17 down to n=10
    for n in range(17, 9, -1):
        if time.time() > hard_deadline - 5:
            break
        
        # Allocate time based on n
        remaining = hard_deadline - time.time()
        if n >= 16:
            budget = min(20, remaining * 0.4)
        elif n >= 14:
            budget = min(15, remaining * 0.3)
        else:
            budget = min(10, remaining * 0.2)
        
        budget = max(2, budget)
        res = attempt(n, budget)
        if res is not None:
            if verify(n, res):
                return {'n': n, 'colors': res}
    
    # Fallback
    for n in range(10, 7, -1):
        if time.time() > hard_deadline - 2:
            break
        res = attempt(n, 2)
        if res is not None:
            if verify(n, res):
                return {'n': n, 'colors': res}
    
    return {'n': 1, 'colors': []}

import time
import sys
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140.0
    
    adj = [0] * n
    
    def is_valid_add(i, j):
        mask_j = adj[j]
        mask_i = adj[i]
        k = mask_j
        while k:
            lsb = k & -k
            k_idx = lsb.bit_length() - 1
            k -= lsb
            cnt = (adj[i] & adj[k_idx]).bit_count()
            if cnt >= 2:
                return False
        k = mask_i
        while k:
            lsb = k & -k
            k_idx = lsb.bit_length() - 1
            k -= lsb
            cnt = (adj[j] & adj[k_idx]).bit_count()
            if cnt >= 2:
                return False
        return True

    def add_edge(i, j):
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
        
    def remove_edge(i, j):
        adj[i] &= ~(1 << j)
        adj[j] &= ~(1 << i)
    
    def extract_edges():
        edges = []
        for u in range(n):
            mask = adj[u]
            k = mask
            while k:
                lsb = k & -k
                v = lsb.bit_length() - 1
                k -= lsb
                if u < v:
                    edges.append([u, v])
        return edges

    def count_edges():
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def tight_upper_bound(current_count, next_i):
        """Tighter upper bound considering current constraints."""
        ub = current_count
        for i in range(next_i, n):
            # Count how many j < i can actually be added to i
            possible = 0
            for j in range(i):
                if is_valid_add(i, j):
                    possible += 1
            ub += possible
        return ub

    def greedy(order):
        nonlocal adj
        adj = [0] * n
        for i in order:
            for j in range(i):
                if is_valid_add(i, j):
                    add_edge(i, j)
        return count_edges()

    def dfs(i, current_count, j):
        nonlocal best_edges, best_count
        
        if time.time() > deadline:
            return
            
        if tight_upper_bound(current_count, i) <= best_count:
            return

        if i == n:
            if current_count > best_count:
                best_count = current_count
                best_edges = extract_edges()
            return

        if j < 0:
            dfs(i + 1, current_count, i)
            return

        if is_valid_add(i, j):
            add_edge(i, j)
            dfs(i, current_count + 1, j - 1)
            remove_edge(i, j)
            
        dfs(i, current_count, j - 1)

    best_count = 0
    best_edges = []
    
    # Multiple greedy restarts
    rng = random.Random(42)
    base_order = list(range(n))
    
    cnt = greedy(base_order)
    if cnt > best_count:
        best_count = cnt
        best_edges = extract_edges()
    
    while time.time() < deadline - 30.0:
        order = base_order.copy()
        rng.shuffle(order)
        cnt = greedy(order)
        if cnt > best_count:
            best_count = cnt
            best_edges = extract_edges()
    
    # Backtracking refinement if time permits
    if time.time() < deadline - 5.0:
        adj = [0] * n
        dfs(0, 0, -1)
    
    if not best_edges:
        greedy(base_order)
        best_edges = extract_edges()
        
    return best_edges

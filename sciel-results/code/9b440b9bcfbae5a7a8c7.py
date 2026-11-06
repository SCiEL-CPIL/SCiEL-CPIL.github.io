import time
import random
import math

def construct():
    start_time = time.time()
    timeout = 85.0

    n = 256
    adj = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)

    def greedy_mis(adj, n, order=None):
        if order is None:
            order = sorted(range(1, n + 1), key=lambda x: len(adj[x]))
        S = set()
        for v in order:
            if not (adj[v] & S):
                S.add(v)
        return S

    def local_search_swap(S, adj, n, time_budget):
        S = set(S)
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            
            # 1. Try to add any valid vertex
            for v in range(1, n + 1):
                if v not in S and not (adj[v] & S):
                    S.add(v)
                    improved = True
            
            # 2. Try random swaps if no single addition
            if not improved:
                S_list = list(S)
                not_S = [v for v in range(1, n + 1) if v not in S]
                
                if S_list and not_S:
                    # Try a few random swaps
                    for _ in range(10):
                        if time.time() - start_time > time_budget:
                            break
                        u = random.choice(S_list)
                        v = random.choice(not_S)
                        
                        S_temp = S - {u}
                        
                        # Check if v can be added
                        if not (adj[v] & S_temp):
                            S_temp.add(v)
                            
                            # Greedy fill: add any valid neighbors of u or v
                            # Actually, just add any valid vertex
                            # This is O(N)
                            for w in range(1, n + 1):
                                if w not in S_temp and not (adj[w] & S_temp):
                                    S_temp.add(w)
                            
                            if len(S_temp) > len(S):
                                S = S_temp
                                improved = True
                                break # Restart the while loop to try more additions
                            elif len(S_temp) == len(S) and random.random() < 0.1:
                                S = S_temp
                                improved = True
                                break

        return S

    best_sol = set()
    best_size = 0

    # Initial 1: Odds
    odds = set(range(1, n + 1, 2))
    best_sol = odds
    best_size = len(best_sol)

    # Initial 2: Greedy by degree
    g_sol = greedy_mis(adj, n)
    if len(g_sol) > best_size:
        best_size = len(g_sol)
        best_sol = g_sol

    # Refine best
    ls_sol = local_search_swap(best_sol, adj, n, timeout - 5)
    if len(ls_sol) > best_size:
        best_size = len(ls_sol)
        best_sol = ls_sol

    # Random restarts
    while time.time() - start_time < timeout - 2:
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = greedy_mis(adj, n, order)
        
        # Refine
        S = local_search_swap(S, adj, n, min(10, timeout - (time.time() - start_time) - 2))
        
        if len(S) > best_size:
            best_size = len(S)
            best_sol = S

    return list(best_sol)

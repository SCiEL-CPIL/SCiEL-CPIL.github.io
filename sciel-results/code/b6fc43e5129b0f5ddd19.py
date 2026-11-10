import math
import time
import sys
from itertools import combinations

def construct():
    N = 255
    deadline = time.time() + 55.0
    
    # Candidate pool: all numbers in [1,255] divisible by 3, 5, or 17
    candidates = [i for i in range(1, N + 1) if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    n_cand = len(candidates)
    
    # Build adjacency: adj[i] = set of j != i with gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Convert to sorted lists for potential use
    adj_lists = [sorted(s) for s in adj]
    
    # Greedy initialization: multiple strategies
    def greedy_clique(start_order=None):
        if start_order is None:
            start_order = list(range(n_cand))
        clique = []
        P = set(range(n_cand))
        order = list(start_order)
        while P:
            # Pick vertex with max degree in P
            best_v = None
            best_deg = -1
            for v in order:
                if v not in P:
                    continue
                deg = len(adj[v] & P)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v is None:
                break
            clique.append(best_v)
            P = P & adj[best_v]
        return clique
    
    # Try several greedy orders
    best_clique = []
    
    # Order by degree descending
    deg_order = sorted(range(n_cand), key=lambda x: len(adj[x]), reverse=True)
    c1 = greedy_clique(deg_order)
    if len(c1) > len(best_clique):
        best_clique = c1
    
    # Order by value (smaller first)
    val_order = sorted(range(n_cand), key=lambda x: candidates[x])
    c2 = greedy_clique(val_order)
    if len(c2) > len(best_clique):
        best_clique = c2
    
    # Order by degree ascending (sometimes helps)
    deg_asc = sorted(range(n_cand), key=lambda x: len(adj[x]))
    c3 = greedy_clique(deg_asc)
    if len(c3) > len(best_clique):
        best_clique = c3
    
    best_size = len(best_clique)
    
    # Tomita's maximum clique algorithm with greedy coloring
    # P: candidate set, X: excluded set, clique: current clique
    # We use a coloring-based bound for pruning
    
    def color_sort(P):
        """Color the subgraph induced by P greedily, return (ordered_list, color_upper_bounds)"""
        U = list(P)
        colors = {}
        color_num = 0
        ordered = []
        while U:
            color_num += 1
            S = []
            for v in list(U):
                if all(v not in adj[u] for u in S):
                    S.append(v)
            for v in S:
                U.remove(v)
                colors[v] = color_num
                ordered.append(v)
        return ordered, colors
    
    def tomita(P, X, clique):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        
        if not P:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)
            return
        
        if len(clique) + len(P) <= best_size:
            return
        
        # Color sort P
        ordered, colors = color_sort(P)
        
        # Process in reverse color order (highest color first for better pruning)
        for i in range(len(ordered) - 1, -1, -1):
            v = ordered[i]
            if v not in P:
                continue
            # Upper bound: clique size + color of v
            if len(clique) + colors[v] <= best_size:
                return
            
            tomita(P & adj[v], X | {v}, clique + [v])
            P.remove(v)
            X.add(v)
    
    tomita(set(range(n_cand)), set(), [])
    
    result = [candidates[i] for i in best_clique]
    
    # Ensure 255 is in result
    if 255 not in result:
        result.append(255)
    
    return result

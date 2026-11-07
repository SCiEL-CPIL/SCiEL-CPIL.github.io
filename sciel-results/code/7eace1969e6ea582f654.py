"""Rewritten program using Greedy Hitting Set on 3-uniform hypergraph of bad triples."""
# EVOLVE-BLOCK-START
import time
import heapq
import random
from math import gcd

def construct():
    start_time = time.time()
    timeout = 55.0  # 5s margin before 60s limit
    
    n = 256
    elements = list(range(2, n + 1))
    N = len(elements)
    
    # 1. Identify all "bad" triples (a, b, c) distinct such that a | b*c
    bad_triples = []
    
    for i, a in enumerate(elements):
        for j, b in enumerate(elements):
            if i == j:
                continue
            if b % a == 0:
                for k, c in enumerate(elements):
                    if k != i and k != j:
                        bad_triples.append((i, j, k))
                continue
            g = gcd(a, b)
            req = a // g
            for k, c in enumerate(elements):
                if k == i or k == j:
                    continue
                if c % req == 0:
                    bad_triples.append((i, j, k))
    
    # Build adjacency
    node_triples = [[] for _ in range(N)]
    for tid, (i, j, k) in enumerate(bad_triples):
        node_triples[i].append(tid)
        node_triples[j].append(tid)
        node_triples[k].append(tid)
    
    best_size = 0
    best_set = []
    
    # Randomized Greedy Search
    # We run multiple iterations with random tie-breaking
    # Time budget for iterations
    max_iterations = 20
    time_per_iter = (timeout - (time.time() - start_time)) / max_iterations
    
    for _ in range(max_iterations):
        if time.time() - start_time > timeout:
            break
            
        active = [True] * N
        triple_active = [True] * len(bad_triples)
        deg = [len(nt) for nt in node_triples]
        
        # Max-heap of (-deg, node, random_key)
        # We use random key to break ties
        heap = [(-deg[i], i, random.random()) for i in range(N)]
        heapq.heapify(heap)
        
        current_set_size = N
        
        while True:
            if time.time() - start_time > timeout:
                break
                
            found = None
            while heap:
                neg_d, node, r = heapq.heappop(heap)
                if not active[node]:
                    continue
                if -neg_d == deg[node]:
                    found = (node, -neg_d, r)
                    break
                else:
                    heapq.heappush(heap, (-deg[node], node, random.random()))
            
            if found is None or found[1] == 0:
                break
                
            node, d, r = found
            active[node] = False
            current_set_size -= 1
            
            for tid in node_triples[node]:
                if not triple_active[tid]:
                    continue
                triple_active[tid] = False
                i, j, k = bad_triples[tid]
                for neighbor in (i, j, k):
                    if neighbor != node:
                        deg[neighbor] -= 1
                        if active[neighbor]:
                            heapq.heappush(heap, (-deg[neighbor], neighbor, random.random()))
        
        if current_set_size > best_size:
            best_size = current_set_size
            best_set = [elements[i] for i in range(N) if active[i]]
            
    # If we didn't find anything better, fallback to a simple heuristic or the first run
    if not best_set:
        # Fallback: Primes
        def is_prime(x):
            if x < 2: return False
            for i in range(2, int(x**0.5)+1):
                if x % i == 0: return False
            return True
        best_set = [x for x in elements if is_prime(x)]
        
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END

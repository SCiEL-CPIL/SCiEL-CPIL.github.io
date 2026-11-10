import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    n = 20
    vertices = list(range(n))
    
    # Precompute all triples
    all_triples = list(combinations(vertices, 3))
    
    # Precompute which triples are affected by each vertex
    # affected_triples[u] is a list of triples (as indices in all_triples) that contain u
    affected_triples = [[] for _ in range(n)]
    triple_indices = {}
    for i, t in enumerate(all_triples):
        triple_indices[t] = i
        for v in t:
            affected_triples[v].append(i)
    
    # Function to check if adding edge (u, v) is valid
    # We only need to check triples that contain u or v
    def can_add(u, v, adj):
        bit_u = 1 << u
        bit_v = 1 << v
        
        # Get unique affected triples
        # Union of affected_triples[u] and affected_triples[v]
        # To avoid creating a set (slow), we can just iterate both and use a small set or check duplicates?
        # n=20, list size ~170. Union size ~320.
        # A set is fine.
        affected_set = set(affected_triples[u])
        affected_set.update(affected_triples[v])
        
        for idx in affected_set:
            a, b, c = all_triples[idx]
            
            # Get masks
            ma = adj[a]
            mb = adj[b]
            mc = adj[c]
            
            # Apply hypothetical edge (u,v)
            if a == u:
                ma |= bit_v
            elif a == v:
                ma |= bit_u
            
            if b == u:
                mb |= bit_v
            elif b == v:
                mb |= bit_u
            
            if c == u:
                mc |= bit_v
            elif c == v:
                mc |= bit_u
            
            common = ma & mb & mc
            if common.bit_count() >= 3:
                return False
        return True

    def edge_count(adj):
        return sum(m.bit_count() for m in adj) // 2

    def mask_to_edges(adj):
        edges = []
        for i in range(n):
            m = adj[i]
            for j in range(i + 1, n):
                if m & (1 << j):
                    edges.append([i, j])
        return edges

    rng = random.Random(42)
    
    # Initialization: Greedy Random
    # Start with empty graph
    adj = [0] * n
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_pairs)
    
    # Try to add edges greedily
    for u, v in all_pairs:
        if can_add(u, v, adj):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    best_adj = list(adj)
    best_score = edge_count(best_adj)
    current_adj = list(adj)
    current_score = best_score
    
    # Simulated Annealing Parameters
    initial_temp = 50.0
    cooling_rate = 0.995
    min_temp = 0.1
    max_steps = 200000  # We can do more because can_add is faster
    
    current_temp = initial_temp
    steps = 0
    
    # Precompute missing and existing edges lists? 
    # It's better to generate them on the fly or maintain them.
    # For n=20, generating missing edges is 190 checks. Fast.
    
    while current_temp > min_temp and steps < max_steps:
        if time.time() > deadline:
            break
            
        # Pick a random move
        # Move types:
        # 1. Add edge
        # 2. Remove edge
        # 3. Swap edge
        
        move_type = rng.random()
        cand_adj = list(current_adj)
        
        if move_type < 0.4:
            # Try to add an edge
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (current_adj[i] & (1 << j))]
            if not missing:
                current_temp *= cooling_rate
                steps += 1
                continue
            u, v = rng.choice(missing)
            if can_add(u, v, current_adj):
                cand_adj[u] |= (1 << v)
                cand_adj[v] |= (1 << u)
            else:
                # Invalid, skip
                current_temp *= cooling_rate
                steps += 1
                continue
                
        elif move_type < 0.7:
            # Try to remove an edge
            existing = [(i, j) for i in range(n) for j in range(i + 1, n) if current_adj[i] & (1 << j)]
            if not existing:
                current_temp *= cooling_rate
                steps += 1
                continue
            u, v = rng.choice(existing)
            cand_adj[u] &= ~(1 << v)
            cand_adj[v] &= ~(1 << u)
            
        else:
            # Swap: Remove one, add another
            existing = [(i, j) for i in range(n) for j in range(i + 1, n) if current_adj[i] & (1 << j)]
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (current_adj[i] & (1 << j))]
            if not existing or not missing:
                current_temp *= cooling_rate
                steps += 1
                continue
            u, v = rng.choice(existing)
            a, b = rng.choice(missing)
            
            # Remove u-v
            cand_adj[u] &= ~(1 << v)
            cand_adj[v] &= ~(1 << u)
            
            # Check if we can add a-b to the *modified* graph
            if can_add(a, b, cand_adj):
                cand_adj[a] |= (1 << b)
                cand_adj[b] |= (1 << a)
            else:
                # If swap fails validity, treat as removal only? 
                # Or just skip the move?
                # Let's treat it as removal to allow escape from local minima.
                pass
        
        cand_score = edge_count(cand_adj)
        delta = cand_score - current_score
        
        # Acceptance probability
        if delta > 0:
            accept = True
        else:
            if current_temp < 1e-6:
                accept = False
            else:
                # Use log for stability if delta is large negative? 
                # delta is small (-1, 0).
                ratio = np.exp(delta / current_temp)
                accept = rng.random() < ratio
        
        if accept:
            current_adj = cand_adj
            current_score = cand_score
            
            if current_score > best_score:
                best_score = current_score
                best_adj = list(current_adj)
                
        current_temp *= cooling_rate
        steps += 1
        
    # Final polish: Greedy add on best_adj
    # Try to add any remaining missing edges
    for i in range(n):
        for j in range(i + 1, n):
            if not (best_adj[i] & (1 << j)):
                if can_add(i, j, best_adj):
                    best_adj[i] |= (1 << j)
                    best_adj[j] |= (1 << i)
                    
    edges = mask_to_edges(best_adj)
    return edges

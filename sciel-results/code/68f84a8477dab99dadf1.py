"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
BASE = set()
BASE_ADJ = [0] * N

for e in P['edges']:
    u, v = e
    if u > v: u, v = v, u
    BASE.add((u, v))
    BASE_ADJ[u] |= (1 << v)
    BASE_ADJ[v] |= (1 << u)

# Candidates: edges not in base, length >= 3
CANDIDATES = []
for u in range(N):
    for v in range(u + 3, N):
        CANDIDATES.append((u, v))

def get_diameter(adj):
    max_dist = 0
    for start in range(N):
        dist = [-1] * N
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            neighbors = adj[u]
            v_bits = neighbors
            while v_bits:
                lsb = v_bits & -v_bits
                v = lsb.bit_length() - 1
                v_bits ^= lsb
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_dist:
                        max_dist = dist[v]
                    q.append(v)
    return max_dist

def is_triangle_free(adj):
    for u in range(N):
        v_bits = adj[u]
        while v_bits:
            lsb = v_bits & -v_bits
            v = lsb.bit_length() - 1
            v_bits ^= lsb
            if v > u:
                if adj[u] & adj[v]:
                    return False
    return True

def evaluate(added_edges, adj):
    # Returns (cost, valid)
    # cost = num_edges * 10000 + diameter * 100 + (0 if triangle_free else 1000000)
    if not is_triangle_free(adj):
        return (10**9, False)
    
    d = get_diameter(adj)
    if d > 4:
        return (len(added_edges) * 10000 + d * 100, False)
    else:
        return (len(added_edges), True)

def solve():
    deadline = time.time() + 85
    random.seed(42)
    
    best_cost = float('inf')
    best_solution = None
    
    # Greedy Init
    adj = BASE_ADJ[:]
    added = []
    # Sort candidates by potential benefit (descending length)
    cands = sorted(CANDIDATES, key=lambda x: x[1]-x[0], reverse=True)
    
    for e in cands:
        if time.time() > deadline: break
        u, v = e
        if (adj[u] >> v) & 1: continue
        # Check triangle
        if adj[u] & adj[v]: continue
        
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        added.append(e)
        
        if is_triangle_free(adj):
             d = get_diameter(adj)
             if d <= 4:
                 best_cost = len(added)
                 best_solution = list(added)
                 break
        else:
             # Revert if triangle? 
             # Actually, we checked triangle before adding. 
             # But adding might create triangle with OTHER edges? 
             # No, we checked common neighbors. 
             # So it should be triangle free if it was before.
             pass

    if best_solution is None:
        # Try a few random inits
        for _ in range(10):
             if time.time() > deadline: break
             adj = BASE_ADJ[:]
             added = []
             random.shuffle(cands)
             for e in cands:
                 u, v = e
                 if (adj[u] >> v) & 1: continue
                 if adj[u] & adj[v]: continue
                 adj[u] |= (1 << v)
                 adj[v] |= (1 << u)
                 added.append(e)
                 if is_triangle_free(adj):
                     d = get_diameter(adj)
                     if d <= 4:
                         if len(added) < best_cost:
                             best_cost = len(added)
                             best_solution = list(added)
                         break
                 else:
                     # Revert
                     adj[u] &= ~(1 << v)
                     adj[v] &= ~(1 << u)
                     added.pop()
                     # Continue trying other edges? 
                     # If triangle formed, it means my check was wrong or 
                     # the edge created a triangle with existing ones.
                     # My check `adj[u] & adj[v]` ensures no common neighbor.
                     # So it shouldn't form a triangle.
                     pass

    if best_solution is None:
        return []

    # Local Search
    cur_adj = BASE_ADJ[:]
    for e in best_solution:
        u, v = e
        cur_adj[u] |= (1 << v)
        cur_adj[v] |= (1 << u)
    cur_added = list(best_solution)
    cur_cost, cur_valid = evaluate(cur_added, cur_adj)
    
    for _ in range(5000):
        if time.time() > deadline: break
        
        # Random move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        trial_adj = cur_adj[:]
        trial_added = cur_added[:]
        
        if move_type == 'add':
            # Pick random candidate not in added
            candidates_not_in = [e for e in CANDIDATES if e not in set(cur_added)]
            if not candidates_not_in: continue
            e = random.choice(candidates_not_in)
            u, v = e
            if (trial_adj[u] >> v) & 1: continue
            # Check triangle
            if trial_adj[u] & trial_adj[v]: continue
            trial_adj[u] |= (1 << v)
            trial_adj[v] |= (1 << u)
            trial_added.append(e)
            
        elif move_type == 'remove':
            if not cur_added: continue
            idx = random.randint(0, len(cur_added)-1)
            e = cur_added[idx]
            u, v = e
            trial_adj[u] &= ~(1 << v)
            trial_adj[v] &= ~(1 << u)
            trial_added.pop(idx)
            
        else: # swap
            if not cur_added: continue
            idx = random.randint(0, len(cur_added)-1)
            e_old = cur_added[idx]
            u, v = e_old
            trial_adj[u] &= ~(1 << v)
            trial_adj[v] &= ~(1 << u)
            trial_added.pop(idx)
            
            candidates_not_in = [e for e in CANDIDATES if e not in set(trial_added)]
            if not candidates_not_in: continue
            e_new = random.choice(candidates_not_in)
            u2, v2 = e_new
            if (trial_adj[u2] >> v2) & 1: continue
            if trial_adj[u2] & trial_adj[v2]: continue
            trial_adj[u2] |= (1 << v2)
            trial_adj[v2] |= (1 << u2)
            trial_added.append(e_new)
            
        trial_cost, trial_valid = evaluate(trial_added, trial_adj)
        
        # Accept if better
        if trial_cost < cur_cost:
            cur_adj = trial_adj
            cur_added = trial_added
            cur_cost = trial_cost
            if trial_valid and len(cur_added) < best_cost:
                best_cost = len(cur_added)
                best_solution = list(cur_added)

    return [[int(u), int(v)] for (u, v) in best_solution]

def construct():
    result = solve()
    return result
# EVOLVE-BLOCK-END

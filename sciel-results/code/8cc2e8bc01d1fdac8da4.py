import numpy as np
import time
import itertools
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Precomputation
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    
    # Build adjacency list for conflict graph (intersection >= 3)
    adj = [[] for _ in range(N_S)]
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                adj[i].append(j)
                adj[j].append(i)

    # Maximum flow approach:
    # Left side: coordinates 0..10 (11 nodes)
    # Right side: supports 0..329 (330 nodes)
    # Edge from coord c to support s if c in support s
    # Capacity 2 on each coordinate node
    # We want to maximize number of supports "covered" (matched)
    
    # Actually, let me think about this differently.
    # The constraint is: any two selected supports intersect in at most 2 coordinates.
    # This is equivalent to: for each coordinate c, at most 2 selected supports contain c.
    # Wait, that's not quite right. If support A contains coords {1,2,3,4} and support B contains {1,2,5,6},
    # they intersect in {1,2} which is size 2. OK.
    # But if support C contains {1,2,3,7}, it intersects A in {1,2,3} (size 3) and B in {1,2} (size 2).
    # So C conflicts with A.
    # The constraint "at most 2 supports contain each coordinate" is NOT sufficient.
    # Consider: coord 1 in supports S1={1,2,3,4}, S2={1,5,6,7}. Coord 2 in S1, S3={2,8,9,10}.
    # S1 and S3 intersect in {2} only. OK.
    # But what about S1={1,2,3,4} and S2={1,2,5,6}? They intersect in {1,2} (size 2). OK.
    # The issue is: we need pairwise intersection <= 2, not just per-coordinate count <= 2.
    
    # Let me use a different formulation. Since the problem is small (330 nodes),
    # I'll use a better local search with simulated annealing.
    
    # Simulated Annealing for Maximum Independent Set
    rng = np.random.RandomState(42)
    
    def random_greedy_init():
        x = np.zeros(N_S, dtype=bool)
        order = rng.permutation(N_S)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not x[v] and not blocked[v]:
                x[v] = True
                for nb in adj[v]:
                    blocked[nb] = True
        return x

    def local_search_sa(x, max_iter=2000):
        current_x = x.copy()
        current_score = int(np.sum(current_x))
        best_x = current_x.copy()
        best_score = current_score
        
        # Precompute which nodes are in current set
        in_set = set(np.where(current_x)[0].tolist())
        
        # Blocked nodes: those that conflict with at least one node in current set
        blocked = np.zeros(N_S, dtype=bool)
        for v in in_set:
            for nb in adj[v]:
                blocked[nb] = True
        
        T_start = 2.0
        T_end = 0.01
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            T = T_start * (T_end / T_start) ** (it / max_iter)
            
            r = rng.random()
            if r < 0.35:
                # Try to add a node
                candidates = [v for v in range(N_S) if not current_x[v] and not blocked[v]]
                if candidates:
                    v = int(rng.choice(candidates))
                    new_score = current_score + 1
                    # Always accept improvement
                    current_x[v] = True
                    in_set.add(v)
                    for nb in adj[v]:
                        blocked[nb] = True
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_x = current_x.copy()
                        
            elif r < 0.65:
                # Try to remove a node (to escape local optima)
                if in_set:
                    v = int(rng.choice(list(in_set)))
                    # Check if removing v allows adding more nodes later
                    # For now, just remove with Metropolis acceptance
                    new_score = current_score - 1
                    delta = new_score - current_score
                    if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                        current_x[v] = False
                        in_set.discard(v)
                        # Recompute blocked
                        blocked = np.zeros(N_S, dtype=bool)
                        for u in in_set:
                            for nb in adj[u]:
                                blocked[nb] = True
                        current_score = new_score
                        
            else:
                # Swap: remove u, add v
                if in_set:
                    u = int(rng.choice(list(in_set)))
                    # Temporarily remove u
                    temp_in_set = in_set.copy()
                    temp_in_set.discard(u)
                    # Compute blocked without u
                    temp_blocked = np.zeros(N_S, dtype=bool)
                    for w in temp_in_set:
                        for nb in adj[w]:
                            temp_blocked[nb] = True
                    # Candidates: not in set, not blocked by temp set
                    candidates = [v for v in range(N_S) if not current_x[v] and not temp_blocked[v]]
                    if candidates:
                        v = int(rng.choice(candidates))
                        new_score = current_score  # Size unchanged
                        # Accept neutral moves with some probability
                        if rng.random() < 0.5:
                            current_x[u] = False
                            in_set.discard(u)
                            current_x[v] = True
                            in_set.add(v)
                            # Recompute blocked
                            blocked = np.zeros(N_S, dtype=bool)
                            for w in in_set:
                                for nb in adj[w]:
                                    blocked[nb] = True
                            if new_score > best_score:
                                best_score = new_score
                                best_x = current_x.copy()
                                
        return best_x

    best_support_set = set()
    best_size = 0
    
    # Run multiple restarts with varying parameters
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        x0 = random_greedy_init()
        x_best = local_search_sa(x0, max_iter=1500)
        size = int(np.sum(x_best))
        
        if size > best_size:
            best_size = size
            best_support_set = set(np.where(x_best)[0].tolist())
            
            # Try to improve further with more iterations
            if time.time() < deadline - 20.0:
                x_improved = local_search_sa(x_best, max_iter=3000)
                size2 = int(np.sum(x_improved))
                if size2 > best_size:
                    best_size = size2
                    best_support_set = set(np.where(x_improved)[0].tolist())

    # Construct vectors
    vectors = []
    
    # Type A: ±2e_i for i in range(11)
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors.append(v2)
        
    # Type D: All 16 sign vectors for each selected support
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors

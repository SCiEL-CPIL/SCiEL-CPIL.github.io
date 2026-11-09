import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 148
    
    n = 16
    adj = [[False]*n for _ in range(n)]
    edge_set = set()
    
    def add_edge(u, v):
        u, v = min(u,v), max(u,v)
        if u != v and (u,v) not in edge_set:
            edge_set.add((u,v))
            adj[u][v] = adj[v][u] = True
    
    def remove_edge(u, v):
        u, v = min(u,v), max(u,v)
        if (u,v) in edge_set:
            edge_set.remove((u,v))
            adj[u][v] = adj[v][u] = False
    
    def compute_score():
        deg = [sum(adj[i]) for i in range(n)]
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    def compute_triangle_sums():
        """Return list of degree sums for all triangles"""
        deg = [sum(adj[i]) for i in range(n)]
        tri_sums = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            tri_sums.append(deg[i] + deg[j] + deg[k])
        return tri_sums
    
    def compute_softmax_penalty(beta=2.0):
        """Smooth surrogate: soft-max of triangle degree sums via log-sum-exp"""
        tri_sums = compute_triangle_sums()
        if not tri_sums:
            return 0.0
        m = max(tri_sums)
        # log-sum-exp for numerical stability
        log_sum_exp = m + math.log(sum(math.exp(s - m) for s in tri_sums))
        # soft-max approximation: (1/beta) * log(sum(exp(beta*s)))
        soft_max = log_sum_exp / beta
        # Also add a small penalty for variance to flatten distribution
        mean_s = sum(tri_sums) / len(tri_sums)
        variance = sum((s - mean_s)**2 for s in tri_sums) / len(tri_sums)
        return soft_max + 0.05 * variance
    
    def compute_penalty_score():
        """Penalty-based objective: sum of squares of triangle degree sums, plus a large penalty for max > threshold"""
        deg = [sum(adj[i]) for i in range(n)]
        tri_sums = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            tri_sums.append(deg[i] + deg[j] + deg[k])
        if not tri_sums:
            return 0.0
        max_sum = max(tri_sums)
        sum_sq = sum(s*s for s in tri_sums)
        penalty = max_sum * max_sum * 10 + sum_sq
        return penalty
    
    all_edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    
    best_score = float('inf')
    best_edge_set = None
    best_penalty = float('inf')
    
    def try_local_search(init_edge_set, max_iterations=50000, mode='softmax'):
        """
        mode: 'softmax' uses smooth surrogate, 'penalty' uses original penalty, 'score' uses raw max
        """
        nonlocal best_score, best_edge_set, best_penalty
        edge_set.clear()
        for i in range(n):
            adj[i] = [False]*n
        for e in init_edge_set:
            add_edge(*e)
        
        current_score = compute_score()
        
        if mode == 'softmax':
            current_obj = compute_softmax_penalty()
        elif mode == 'penalty':
            current_obj = compute_penalty_score()
        else:
            current_obj = current_score
        
        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)
        
        if mode != 'score' and current_obj < best_penalty:
            best_penalty = current_obj
        
        no_improve = 0
        for it in range(max_iterations):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try all possible swaps
            for e_remove in list(edge_set):
                if time.time() > deadline - 3:
                    break
                for e_add in all_edges:
                    if e_add in edge_set or e_add == e_remove:
                        continue
                    
                    remove_edge(*e_remove)
                    add_edge(*e_add)
                    
                    if mode == 'softmax':
                        new_obj = compute_softmax_penalty()
                    elif mode == 'penalty':
                        new_obj = compute_penalty_score()
                    else:
                        new_obj = compute_score()
                    
                    if new_obj < current_obj:
                        current_obj = new_obj
                        improved = True
                        new_score = compute_score()
                        if new_score < best_score:
                            best_score = new_score
                            best_edge_set = set(edge_set)
                        if mode != 'score' and new_obj < best_penalty:
                            best_penalty = new_obj
                        break
                    else:
                        remove_edge(*e_add)
                        add_edge(*e_remove)
                
                if improved:
                    break
            
            if not improved:
                no_improve += 1
                if no_improve > 30:
                    break
    
    def random_init():
        """Generate a random 65-edge graph"""
        all_e = list(all_edges)
        random.shuffle(all_e)
        return set(all_e[:65])
    
    def balanced_bipartite_init():
        """K_{8,8} plus a random intra-part edge"""
        base = set()
        for u in range(8):
            for v in range(8, 16):
                base.add((u, v))
        # Add random intra-part edges to get to 65
        intra0 = [(u,v) for u in range(8) for v in range(u+1, 8)]
        intra1 = [(u+8,v+8) for u in range(8) for v in range(u+1, 8)]
        random.shuffle(intra0)
        random.shuffle(intra1)
        edges = list(base)
        for e in intra0 + intra1:
            if len(edges) >= 65:
                break
            edges.append(e)
        return set(edges[:65])
    
    # Try multiple initialization strategies
    random.seed(42)
    
    # Strategy 1: K_{8,8} + specific intra edges
    base_edges = set()
    for u in range(8):
        for v in range(8, 16):
            base_edges.add((u, v))
    
    # Try a few specific intra-part edges
    for part in [0, 1]:
        for u in range(8):
            for v in range(u+1, 8):
                if part == 0:
                    edge = (u, v)
                else:
                    edge = (u+8, v+8)
                
                init_edges = base_edges | {edge}
                try_local_search(init_edges, max_iterations=15000, mode='softmax')
                
                if time.time() > deadline - 15:
                    break
            if time.time() > deadline - 15:
                break
        if time.time() > deadline - 15:
            break
    
    # Strategy 2: Random restarts with softmax objective
    if time.time() < deadline - 10:
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            init = random_init()
            try_local_search(init, max_iterations=10000, mode='softmax')
    
    # Strategy 3: Balanced bipartite with random extras
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            init = balanced_bipartite_init()
            try_local_search(init, max_iterations=10000, mode='softmax')
    
    # Final polish: pure score-based local search from best found
    if best_edge_set is not None and time.time() < deadline - 15:
        try_local_search(best_edge_set, max_iterations=30000, mode='score')
    
    # Second polish with softmax for fine-tuning
    if best_edge_set is not None and time.time() < deadline - 8:
        try_local_search(best_edge_set, max_iterations=15000, mode='softmax')
    
    # Final exact score check
    if best_edge_set is not None and time.time() < deadline - 5:
        try_local_search(best_edge_set, max_iterations=10000, mode='score')
    
    if best_edge_set is None:
        edge_set.clear()
        for i in range(n):
            adj[i] = [False]*n
        for u in range(8):
            for v in range(8, 16):
                add_edge(u, v)
        add_edge(0, 1)
        best_edge_set = set(edge_set)
    
    edge_list = [[u, v] for (u, v) in best_edge_set]
    return {'n': n, 'edges': edge_list}

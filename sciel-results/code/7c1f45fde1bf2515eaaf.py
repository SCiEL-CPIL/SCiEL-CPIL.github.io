import time
import random
from itertools import combinations

DEADLINE = 145

def construct():
    start_time = time.time()
    n = 16
    
    def elapsed():
        return time.time() - start_time
    
    def fast_has_q3(adj_mask):
        for L in combinations(range(n), 4):
            L_set = set(L)
            L_list = list(L)
            missing_count = {u: 0 for u in L_list}
            for v in range(n):
                if v in L_set:
                    continue
                missing = None
                count = 0
                for u in L_list:
                    if (adj_mask[v] >> u) & 1:
                        count += 1
                    else:
                        missing = u
                if count == 3:
                    missing_count[missing] += 1
            if all(missing_count[u] >= 1 for u in L_list):
                return True
        return False
    
    def greedy_construct(seed):
        random.seed(seed)
        adj_mask = [0] * n
        edges = []
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if elapsed() > DEADLINE:
                break
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            if fast_has_q3(adj_mask):
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            else:
                edges.append([u, v])
        return edges
    
    best_edges = []
    best_count = 0
    
    for seed in range(200):
        if elapsed() > DEADLINE - 10:
            break
        edges = greedy_construct(seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            if elapsed() < DEADLINE - 15:
                adj_mask = [0] * n
                for u, v in best_edges:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
                random.shuffle(all_edges)
                for u, v in all_edges:
                    if elapsed() > DEADLINE - 5:
                        break
                    if (adj_mask[u] >> v) & 1:
                        continue
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    if fast_has_q3(adj_mask):
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                    else:
                        best_edges.append([u, v])
                        best_count = len(best_edges)
    
    return best_edges

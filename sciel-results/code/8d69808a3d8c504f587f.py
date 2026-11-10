import itertools
import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115

    best_edges = None
    best_count = -1

    all_pairs = list(itertools.combinations(range(n), 2))
    triples = list(itertools.combinations(range(n), 3))
    
    # Precompute bit masks for vertices
    # mask[i] = 1 << i

    def run_greedy():
        # Initialize complete graph
        adj_mask = [(1 << n) - 2 for _ in range(n)]
        for i in range(n):
            adj_mask[i] &= ~(1 << i)
            
        edges = set(all_pairs)
        
        # Initialize counts
        cnt = {e: 0 for e in edges}
        
        for B in triples:
            b1, b2, b3 = B
            common = adj_mask[b1] & adj_mask[b2] & adj_mask[b3]
            A_B = [i for i in range(n) if (common >> i) & 1]
            k = len(A_B)
            if k >= 3:
                add_val = (k - 1) * (k - 2) // 2
                for u in A_B:
                    for b in B:
                        e = (min(u, b), max(u, b))
                        if e in cnt:
                            cnt[e] += add_val

        while True:
            max_cnt = -1
            candidates = []
            for e, c in cnt.items():
                if c > max_cnt:
                    max_cnt = c
                    candidates = [e]
                elif c == max_cnt:
                    candidates.append(e)
            
            if max_cnt == 0:
                break
                
            e = random.choice(candidates)
            u, v = e
            
            # Remove (u, v)
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            edges.remove(e)
            
            # Update counts
            others = [x for x in range(n) if x != u and x != v]
            
            # Case 1: v in B, u in A
            for b2, b3 in itertools.combinations(others, 2):
                if (adj_mask[u] >> b2) & 1 and (adj_mask[u] >> b3) & 1:
                    common = adj_mask[v] & adj_mask[b2] & adj_mask[b3]
                    # u is excluded from common because adj_mask[v] no longer has u
                    S = [x for x in range(n) if (common >> x) & 1]
                    if len(S) >= 2:
                        for a2, a3 in itertools.combinations(S, 2):
                            # Edges to decrement
                            e1 = (min(u, b2), max(u, b2))
                            e2 = (min(u, b3), max(u, b3))
                            e3 = (min(a2, v), max(a2, v))
                            e4 = (min(a2, b2), max(a2, b2))
                            e5 = (min(a2, b3), max(a2, b3))
                            e6 = (min(a3, v), max(a3, v))
                            e7 = (min(a3, b2), max(a3, b2))
                            e8 = (min(a3, b3), max(a3, b3))
                            
                            for ee in (e1, e2, e3, e4, e5, e6, e7, e8):
                                if ee in cnt:
                                    cnt[ee] -= 1

            # Case 2: u in B, v in A
            for b2, b3 in itertools.combinations(others, 2):
                if (adj_mask[v] >> b2) & 1 and (adj_mask[v] >> b3) & 1:
                    common = adj_mask[u] & adj_mask[b2] & adj_mask[b3]
                    S = [x for x in range(n) if (common >> x) & 1]
                    if len(S) >= 2:
                        for a2, a3 in itertools.combinations(S, 2):
                            e1 = (min(v, b2), max(v, b2))
                            e2 = (min(v, b3), max(v, b3))
                            e3 = (min(a2, u), max(a2, u))
                            e4 = (min(a2, b2), max(a2, b2))
                            e5 = (min(a2, b3), max(a2, b3))
                            e6 = (min(a3, u), max(a3, u))
                            e7 = (min(a3, b2), max(a3, b2))
                            e8 = (min(a3, b3), max(a3, b3))
                            
                            for ee in (e1, e2, e3, e4, e5, e6, e7, e8):
                                if ee in cnt:
                                    cnt[ee] -= 1

            del cnt[e]
            
            if time.time() > deadline:
                return list(edges)

        return list(edges)

    while time.time() < deadline:
        res = run_greedy()
        if len(res) > best_count:
            best_count = len(res)
            best_edges = res

    if best_edges is None:
        return []
    return [[u, v] for u, v in best_edges]

import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def is_valid(adj):
        degs = [len(adj[i]) for i in range(n)]
        groups = {}
        for i in range(n):
            d = degs[i]
            if d not in groups:
                groups[d] = []
            groups[d].append(i)
        
        for d, group in groups.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                s = group[i]
                for j in range(i+1, len(group)):
                    t = group[j]
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[t]:
                            if v == s:
                                continue
                            if u == v:
                                continue
                            if v in adj[u]:
                                return False
        return True

    def find_violation(adj):
        degs = [len(adj[i]) for i in range(n)]
        groups = {}
        for i in range(n):
            d = degs[i]
            if d not in groups:
                groups[d] = []
            groups[d].append(i)
        
        for d, group in groups.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                s = group[i]
                for j in range(i+1, len(group)):
                    t = group[j]
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[t]:
                            if v == s:
                                continue
                            if u == v:
                                continue
                            if v in adj[u]:
                                return (s, u, v, t)
        return None

    def run_trial(seed_val):
        random.seed(seed_val)
        adj = [set() for _ in range(n)]
        for u, v in all_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        removed_edges = []
        max_iterations = 5000
        iter_count = 0
        while not is_valid(adj):
            if time.time() > deadline - 5:
                break
            viol = find_violation(adj)
            if viol is None:
                break
            s, u, v, t = viol
            choice = random.randint(0, 2)
            if choice == 0:
                edge = (s, u)
            elif choice == 1:
                edge = (u, v)
            else:
                edge = (v, t)
            
            a, b = edge
            adj[a].discard(b)
            adj[b].discard(a)
            removed_edges.append(edge)
            iter_count += 1
            if iter_count > max_iterations:
                break

        random.shuffle(removed_edges)
        for edge in removed_edges:
            if time.time() > deadline - 5:
                break
            u, v = edge
            adj[u].add(v)
            adj[v].add(u)
            if not is_valid(adj):
                adj[u].discard(v)
                adj[v].discard(u)

        missing = [e for e in all_edges if e[1] not in adj[e[0]]]
        random.shuffle(missing)
        for edge in missing:
            if time.time() > deadline - 5:
                break
            u, v = edge
            adj[u].add(v)
            adj[v].add(u)
            if not is_valid(adj):
                adj[u].discard(v)
                adj[v].discard(u)

        return len(removed_edges) # Actually we want edge count
        # Edge count = 105 - len(removed_edges_final)
        # But removed_edges list might have duplicates if we added back and removed again? No, we only add back.
        # So final edge count = 105 - (len(removed_edges) - added_back_count)
        # Easier: sum(len(adj[i]) for i in range(n)) // 2

    best_edges = 0
    best_adj = [set() for _ in range(n)]
    
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        adj_t = [set() for _ in range(n)]
        for u, v in all_edges:
            adj_t[u].add(v)
            adj_t[v].add(u)
        
        removed_edges = []
        max_iterations = 5000
        iter_count = 0
        while not is_valid(adj_t):
            if time.time() > deadline - 5:
                break
            viol = find_violation(adj_t)
            if viol is None:
                break
            s, u, v, t = viol
            choice = random.randint(0, 2)
            if choice == 0:
                edge = (s, u)
            elif choice == 1:
                edge = (u, v)
            else:
                edge = (v, t)
            
            a, b = edge
            adj_t[a].discard(b)
            adj_t[b].discard(a)
            removed_edges.append(edge)
            iter_count += 1
            if iter_count > max_iterations:
                break

        random.shuffle(removed_edges)
        for edge in removed_edges:
            if time.time() > deadline - 5:
                break
            u, v = edge
            adj_t[u].add(v)
            adj_t[v].add(u)
            if not is_valid(adj_t):
                adj_t[u].discard(v)
                adj_t[v].discard(u)

        missing = [e for e in all_edges if e[1] not in adj_t[e[0]]]
        random.shuffle(missing)
        for edge in missing:
            if time.time() > deadline - 5:
                break
            u, v = edge
            adj_t[u].add(v)
            adj_t[v].add(u)
            if not is_valid(adj_t):
                adj_t[u].discard(v)
                adj_t[v].discard(u)

        cnt = sum(len(adj_t[i]) for i in range(n)) // 2
        if cnt > best_edges:
            best_edges = cnt
            best_adj = [set(adj_t[i]) for i in range(n)]
            random.seed(trial) # Reset seed for next trial? No, just let it run.

    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([int(u), int(v)])
    
    return edges

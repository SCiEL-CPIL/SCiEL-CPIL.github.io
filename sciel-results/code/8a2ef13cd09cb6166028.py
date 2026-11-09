"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

DEADLINE = 85.0  # safety margin before 90s kill

def _is_valid(edges, n):
    """Check if no 3-edge simple path has endpoints of equal degree."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    deg = [len(adj[i]) for i in range(n)]
    # For every edge (u,v) and every s in adj[u] (s!=v) and t in adj[v] (t!=u),
    # if s != t then deg[s] must != deg[t]
    for u, v in edges:
        for s in adj[u]:
            if s == v:
                continue
            ds = deg[s]
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
    return True

def _count_violations(edges, n):
    """Count number of violating (s,t) endpoint pairs across all 3-edge paths."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    deg = [len(adj[i]) for i in range(n)]
    count = 0
    for u, v in edges:
        for s in adj[u]:
            if s == v:
                continue
            ds = deg[s]
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    count += 1
    return count

def _get_edges_set(edges):
    return set((min(u,v), max(u,v)) for u,v in edges)

def _all_pairs(n):
    return [(i,j) for i in range(n) for j in range(i+1, n)]

def construct():
    start = time.time()
    n = 15
    all_edges = _all_pairs(n)
    
    best_edges = None
    best_count = 0
    
    # Phase 1: Greedy construction - try to add edges one by one
    # Start from a star (valid baseline)
    edges = [(0, i) for i in range(1, n)]
    edge_set = _get_edges_set(edges)
    
    # Greedy: try adding each missing edge if it keeps validity
    changed = True
    while changed and (time.time() - start) < DEADLINE - 20:
        changed = False
        for e in all_edges:
            if e in edge_set:
                continue
            new_edges = edges + [e]
            if _is_valid(new_edges, n):
                edges = new_edges
                edge_set = edge_set | {e}
                changed = True
        # Try adding edges in different orders for better greedy
        if (time.time() - start) < DEADLINE - 15:
            random.shuffle(all_edges)
    
    best_edges = list(edges)
    best_count = len(edges)
    
    # Phase 2: Local search - try adding single edges, removing single edges, 1-swap
    # Use violation count as objective (minimize violations, then maximize edges among valid)
    
    # First, try to get a valid solution with more edges via local search
    def local_search(edges, edge_set, max_time):
        """Local search to maximize valid edge count."""
        n = 15
        all_e = _all_pairs(n)
        best = list(edges)
        best_set = set(edge_set)
        best_len = len(best)
        
        while (time.time() - start) < max_time:
            improved = False
            
            # Try adding a single edge
            for e in all_e:
                if e in best_set:
                    continue
                new_edges = best + [e]
                if _is_valid(new_edges, n):
                    best = new_edges
                    best_set = best_set | {e}
                    best_len += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-swap: remove one, add one
            for rem in list(best_set):
                for add in all_e:
                    if add in best_set:
                        continue
                    new_set = best_set - {rem} | {add}
                    new_edges = [e for e in best if e != rem] + [add]
                    if _is_valid(new_edges, n):
                        best = new_edges
                        best_set = new_set
                        best_len = len(best)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try 2-swap: remove two, add two
                found = False
                rem_list = list(best_set)
                random.shuffle(rem_list)
                for i in range(min(len(rem_list), 30)):
                    for j in range(i+1, min(len(rem_list), 30)):
                        rem1, rem2 = rem_list[i], rem_list[j]
                        for add1 in all_e:
                            if add1 in best_set:
                                continue
                            for add2 in all_e:
                                if add2 in best_set or add2 == add1:
                                    continue
                                new_set = best_set - {rem1, rem2} | {add1, add2}
                                new_edges = [e for e in best if e != rem1 and e != rem2] + [add1, add2]
                                if _is_valid(new_edges, n):
                                    best = new_edges
                                    best_set = new_set
                                    best_len = len(best)
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    improved = True
            
            if not improved:
                break
        
        return best, best_set, best_len
    
    # Run local search with time budget
    remaining = DEADLINE - (time.time() - start)
    if remaining > 5:
        best_edges, best_set, best_count = local_search(
            best_edges, _get_edges_set(best_edges), time.time() + remaining - 3
        )
    
    # Phase 3: Try multiple random restarts with different greedy orders
    remaining = DEADLINE - (time.time() - start)
    if remaining > 10:
        for attempt in range(50):
            if (time.time() - start) > DEADLINE - 5:
                break
            # Random greedy
            random.shuffle(all_edges)
            rand_edges = []
            rand_set = set()
            for e in all_edges:
                if e in rand_set:
                    continue
                test = rand_edges + [e]
                if _is_valid(test, n):
                    rand_edges = test
                    rand_set = rand_set | {e}
            
            # Local search on this
            if (time.time() - start) > DEADLINE - 8:
                break
            le, ls, lc = local_search(rand_edges, rand_set, time.time() + 5)
            if lc > best_count:
                best_edges = le
                best_count = lc
    
    # Ensure return type is list of lists with Python ints
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return result

# EVOLVE-BLOCK-END

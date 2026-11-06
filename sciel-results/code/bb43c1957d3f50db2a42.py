import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    neighbors = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            s = i + j
            r = int(math.isqrt(s))
            if r * r == s:
                neighbors[i].add(j)
                if i != j:
                    neighbors[j].add(i)
    
    def can_add(node, S):
        for n in neighbors[node]:
            if n in S:
                return False
        return True

    def greedy_init():
        S = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for c in candidates:
            if can_add(c, S):
                S.add(c)
        return S

    def local_search(S):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add a single node
            candidates = [i for i in range(1, N + 1) if i not in S]
            random.shuffle(candidates)
            for c in candidates:
                if can_add(c, S):
                    S.add(c)
                    improved = True
                    break
            
            if improved:
                continue

            # 2. Systematic "remove 1, add 2"
            # For each u in S, check if we can remove u and add two nodes v, w
            # that are not in S and compatible with S\{u} and each other.
            
            # To speed up, we can precompute for each u, the list of nodes that become "free" or "available"
            # But "available" depends on S\{u}.
            
            # Let's iterate u in S.
            # For efficiency, we can break early if we find a move.
            
            # We need to find v, w not in S such that:
            # 1. v, w not in neighbors of any node in S\{u}
            # 2. v not in neighbors[w]
            
            # Let's collect candidates not in S.
            # This list is the same for all u in this iteration.
            candidates_not_in_S = [i for i in range(1, N + 1) if i not in S]
            
            if len(candidates_not_in_S) < 2:
                break # Can't add 2 if less than 2 candidates

            # For each u, we check if there's a pair (v,w) from candidates_not_in_S
            # that is valid w.r.t S\{u}.
            
            # Optimization:
            # A node x is compatible with S\{u} if it has no neighbors in S\{u}.
            # This is equivalent to: all neighbors of x that are in S must be u.
            # i.e., (neighbors[x] & S) is a subset of {u}.
            # Since u is in S, neighbors[x] & S can be empty or {u}.
            
            # So for a fixed u, the valid v's are those in candidates_not_in_S
            # such that (neighbors[v] & S) <= {u}.
            
            # Let's compute this for each u.
            
            # To avoid O(|S| * |C| * |S|) for intersection, we can precompute 
            # for each x, the set of neighbors in S.
            # But S changes.
            
            # Let's just do the check.
            # |S| ~ 100, |C| ~ 150.
            # For each u (100), for each v (150), check if neighbors[v] & S is subset of {u}.
            # neighbors[v] size ~ 10.
            # Check: all(n in neighbors[v] if n in S are n == u).
            # This is fast.
            
            found_move = False
            # We can shuffle u to diversify
            u_list = list(S)
            random.shuffle(u_list)
            
            for u in u_list:
                # Find v, w candidates compatible with S\{u}
                valid_vs = []
                for v in candidates_not_in_S:
                    # Check if v is compatible with S\{u}
                    # i.e., no neighbor of v is in S\{u}
                    # Equivalent to: if n in neighbors[v] and n in S, then n must be u.
                    ok = True
                    for n in neighbors[v]:
                        if n in S and n != u:
                            ok = False
                            break
                    if ok:
                        valid_vs.append(v)
                
                if len(valid_vs) < 2:
                    continue
                
                # Now find a pair (v, w) in valid_vs such that v, w not neighbors.
                # We can iterate pairs.
                # If valid_vs is large, this is O(|valid_vs|^2).
                # But valid_vs is likely small because removing one node u only frees up nodes that were ONLY blocked by u.
                # Most nodes are blocked by multiple nodes in S.
                
                # Let's try to find such a pair.
                # We can shuffle valid_vs to find a pair quickly.
                random.shuffle(valid_vs)
                pair_found = False
                for i in range(len(valid_vs)):
                    v = valid_vs[i]
                    for j in range(i + 1, len(valid_vs)):
                        w = valid_vs[j]
                        if w not in neighbors[v]:
                            # Found a valid pair
                            S.remove(u)
                            S.add(v)
                            S.add(w)
                            improved = True
                            found_move = True
                            pair_found = True
                            break
                    if pair_found:
                        break
                
                if found_move:
                    break

            if found_move:
                continue

            # 3. Perturbation
            if len(S) > 0:
                to_remove = random.sample(list(S), min(5, len(S)))
                for r in to_remove:
                    S.remove(r)
                improved = True

    best_S = set()
    best_size = 0
    
    while time.time() < deadline:
        S = greedy_init()
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()

    return sorted(list(best_S))

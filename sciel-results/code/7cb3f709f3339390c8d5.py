import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin

    n_max = 180
    all_elements = list(range(1, n_max + 1))

    def _build_edges(n_max=180):
        """Build all edges (triples) a,b,c distinct in [1,n_max] with 1/a=1/b+1/c."""
        edges = []
        for a in range(1, n_max + 1):
            a_sq = a * a
            for d in range(1, int(a_sq**0.5) + 1):
                if a_sq % d == 0:
                    candidates = [d]
                    other_d = a_sq // d
                    if other_d != d:
                        candidates.append(other_d)
                    for d_val in candidates:
                        if d_val == a:
                            continue
                        b = a + d_val
                        c = a + a_sq // d_val
                        if b <= n_max and c <= n_max:
                            edge = (min(a,b,c), min(max(a,b), max(a,c)), max(a,b,c))
                            edges.append(edge)
        unique_edges = list(set(edges))
        return unique_edges

    edges = _build_edges(n_max)
    m = len(edges)
    
    # Precompute: for each vertex, which edges it's in, and for each edge, which vertices
    vertex_to_edges = defaultdict(list)
    for idx, (a, b, c) in enumerate(edges):
        vertex_to_edges[a].append(idx)
        vertex_to_edges[b].append(idx)
        vertex_to_edges[c].append(idx)
    
    # Precompute: for each vertex, the set of vertices that form a conflict with it
    # (i.e., if x is in S, then no pair (u,v) from its edges can both be in S)
    # Actually, for fast checking: if we add x, we need to check all edges containing x
    # and see if both other vertices are in S.

    def check_conflict(x, S):
        """Check if adding x to set S creates any forbidden triple."""
        for e_idx in vertex_to_edges[x]:
            a, b, c = edges[e_idx]
            others = [v for v in (a, b, c) if v != x]
            if others[0] in S and others[1] in S:
                return True
        return False

    def count_conflicts(S):
        """Count number of forbidden triples within S."""
        count = 0
        for (a, b, c) in edges:
            if a in S and b in S and c in S:
                count += 1
        return count

    best_set = set(range(91, n_max + 1))
    best_size = len(best_set)

    def greedy_build():
        """Build a set by starting from [91,180] and adding smaller numbers greedily."""
        S = set(range(91, n_max + 1))
        # Try adding numbers from 90 down to 1
        for v in range(90, 0, -1):
            if not check_conflict(v, S):
                S.add(v)
        return S

    def local_search(S, max_iter=2000):
        """Local search with 1-swap and 2-swap moves."""
        S = set(S)
        current_size = len(S)
        removed_list = list(set(all_elements) - S)
        random.shuffle(removed_list)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try 1-swap: remove one, add one
            for v_remove in list(S):
                if time.time() > deadline:
                    break
                S.remove(v_remove)
                added = False
                for v_add in removed_list:
                    if v_add in S:
                        continue
                    if not check_conflict(v_add, S):
                        S.add(v_add)
                        removed_list.remove(v_add)
                        added = True
                        break
                if not added:
                    S.add(v_remove)
                    continue
                if len(S) > current_size:
                    current_size = len(S)
                    improved = True
                    break
                else:
                    # Revert if no size improvement
                    S.discard(v_add)
                    S.add(v_remove)
                    removed_list.append(v_remove)
            
            if improved:
                continue
            
            # Try 2-swap: remove two, add two (or remove one, add two)
            s_list = list(S)
            random.shuffle(s_list)
            for i in range(min(20, len(s_list))):
                if time.time() > deadline:
                    break
                v1 = s_list[i]
                S.remove(v1)
                added = []
                for v_add in removed_list:
                    if v_add in S:
                        continue
                    if not check_conflict(v_add, S):
                        S.add(v_add)
                        added.append(v_add)
                        removed_list.remove(v_add)
                    if len(added) >= 2:
                        break
                if len(added) > 1:
                    current_size = len(S)
                    improved = True
                    break
                elif len(added) == 1:
                    S.discard(added[0])
                    S.add(v1)
                    removed_list.append(v1)
                else:
                    S.add(v1)
            
            if not improved:
                # Try removing one and adding one more carefully
                found_improvement = False
                for v_remove in s_list[:10]:
                    if v_remove not in S:
                        continue
                    S.remove(v_remove)
                    best_add = None
                    for v_add in removed_list:
                        if v_add in S:
                            continue
                        if not check_conflict(v_add, S):
                            best_add = v_add
                            break
                    if best_add is not None:
                        S.add(best_add)
                        removed_list.remove(best_add)
                        if len(S) > current_size:
                            current_size = len(S)
                            found_improvement = True
                            break
                    else:
                        S.add(v_remove)
                        removed_list.append(v_remove)
                
                if not found_improvement:
                    break
        
        return S

    # Phase 1: Greedy build
    S = greedy_build()
    if len(S) > best_size:
        best_size = len(S)
        best_set = set(S)
    
    # Phase 2: Multiple local searches from different starting points
    num_runs = 50
    for run in range(num_runs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        if run == 0:
            S = greedy_build()
        elif run == 1:
            # Start from [1,180] and remove greedily
            S = set(all_elements)
            # Remove vertices with highest conflict count
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                max_conf = 0
                best_v = None
                for v in S:
                    conf = 0
                    for e_idx in vertex_to_edges[v]:
                        a, b, c = edges[e_idx]
                        others = [x for x in (a, b, c) if x != v]
                        if others[0] in S and others[1] in S:
                            conf += 1
                    if conf > max_conf:
                        max_conf = conf
                        best_v = v
                if best_v is None or max_conf == 0:
                    break
                S.remove(best_v)
        else:
            # Random perturbation of best_set
            S = set(best_set)
            # Remove k random elements
            k = random.randint(2, 5)
            to_remove = random.sample(list(S), min(k, len(S)))
            for v in to_remove:
                S.remove(v)
            # Try to add back
            for v in to_remove:
                if not check_conflict(v, S):
                    S.add(v)
            # Also try adding other elements
            for v in all_elements:
                if v not in S and not check_conflict(v, S):
                    S.add(v)
        
        # Local search
        S = local_search(S, max_iter=1000)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Final polish: try to add any remaining safe elements
    for v in all_elements:
        if v not in best_set and not check_conflict(v, best_set):
            best_set.add(v)
    
    if not best_set:
        best_set = set(range(91, n_max + 1))
    
    return sorted(best_set)

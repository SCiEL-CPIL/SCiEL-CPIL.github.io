import random
from itertools import combinations
from math import gcd
import time

N = 128

def generate_bad_triples():
    """
    Generate all triples (a, b, c) with a < b < c such that
    gcd(a, b) == gcd(b, c) == gcd(c, a).
    """
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    triples.append((a, b, c))
    return triples

def heuristic_solve():
    """
    Solve using Max Degree Removal heuristic.
    """
    bad_triples = generate_bad_triples()
    
    # Map vertex to list of indices of triples it is in
    vertex_to_triples = {i: [] for i in range(1, N + 1)}
    for i, t in enumerate(bad_triples):
        for v in t:
            vertex_to_triples[v].append(i)
    
    # Count of active triples for each vertex
    conflict_count = {i: len(vertex_to_triples[i]) for i in range(1, N + 1)}
    
    # Active flag for triples
    active = [True] * len(bad_triples)
    
    current_set = set(range(1, N + 1))
    
    # Priority queue or just scan for max?
    # N is small, scanning is fine.
    # We iterate until no active triples remain.
    
    # To handle ties or randomization, we can shuffle the vertex order initially
    # or just pick the first max.
    
    while True:
        # Find vertex with max conflict count among those in current_set
        max_count = -1
        max_vertex = -1
        for v in current_set:
            if conflict_count[v] > max_count:
                max_count = conflict_count[v]
                max_vertex = v
        
        if max_count == 0:
            break
        
        # Remove max_vertex
        current_set.remove(max_vertex)
        
        # Update counts for all triples involving max_vertex
        for t_idx in vertex_to_triples[max_vertex]:
            if active[t_idx]:
                a, b, c = bad_triples[t_idx]
                # Mark as inactive
                active[t_idx] = False
                # Decrease count for the other two vertices
                # Note: we already removed max_vertex, so we don't update its count
                for v in (a, b, c):
                    if v != max_vertex:
                        conflict_count[v] -= 1

    return current_set

def local_search(s):
    """
    Try to improve the set s by adding elements or swapping.
    """
    improved = True
    while improved:
        improved = False
        # 1. Try to add any single element
        candidates = [x for x in range(1, N + 1) if x not in s]
        random.shuffle(candidates)
        for x in candidates:
            # Check if x can be added
            # We need to check if adding x creates a bad triple.
            # A bad triple involving x is {x, u, v} where u, v in s.
            # This is equivalent to checking if there exist u, v in s
            # such that gcd(x, u) == gcd(u, v) == gcd(v, x).
            
            # Efficient check:
            # For each u in s, calculate g = gcd(x, u).
            # We need another v in s such that gcd(x, v) == g and gcd(u, v) == g.
            # This is O(|s|^2) per x. |s| <= 128. 128^2 * 128 ~ 2M ops.
            # This might be slow if done for many x.
            
            # Precomputation approach for check:
            # For a fixed x, group elements in s by gcd(x, u).
            # If any group has size >= 2, and within that group, there is a pair u, v
            # with gcd(u, v) == g, then it's a conflict.
            
            # Let's just do the direct check for simplicity, N is small.
            can_add = True
            # Optimization: only check pairs (u, v) in s
            # But we don't have the list of bad triples for x precomputed in this function easily.
            # Let's use the global bad triples list?
            # Or just re-calculate.
            
            # Re-calculate is safer.
            # s is a set.
            list_s = list(s)
            for i in range(len(list_s)):
                u = list_s[i]
                g_xu = gcd(x, u)
                for j in range(i + 1, len(list_s)):
                    v = list_s[j]
                    if gcd(x, v) == g_xu and gcd(u, v) == g_xu:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                s.add(x)
                improved = True
                break

        if improved:
            continue

        # 2. Try swaps
        list_s = list(s)
        random.shuffle(list_s)
        
        for u in list_s:
            s_without_u = s - {u}
            new_s = set(s_without_u)
            
            # Try to add elements from complement (excluding u)
            addable = [x for x in range(1, N + 1) if x not in s and x != u]
            random.shuffle(addable)
            
            # Try to add as many as possible? Or just one?
            # If we add one, size is same. We need to add > 1?
            # No, if we remove 1 and add 1, size is same.
            # We want size to increase.
            # So we need to add at least 2 elements?
            # Or remove 1 and add 2?
            # Or remove 2 and add 3?
            
            # Let's try: Remove u. Try to add any x not in s.
            # If we can add x, we have same size.
            # Then try to add y?
            
            # Simple swap: Remove u, try to add x.
            # If we can add x, and then we can add y?
            
            # Let's just try to add one element x.
            # If successful, we have same size.
            # Then we run the "add" loop again?
            # The "add" loop above already tries to add any element.
            # If it couldn't add any, it means no single element can be added to s.
            # So we must swap.
            
            # Remove u.
            # Now check if we can add x.
            # If yes, add x. Size is same.
            # Then check if we can add y.
            # If yes, add y. Size is +1.
            
            # Let's try to add multiple.
            added_count = 0
            for x in addable:
                # Check if x can be added to new_s
                list_new_s = list(new_s)
                can_add = True
                for i in range(len(list_new_s)):
                    u2 = list_new_s[i]
                    g_xu = gcd(x, u2)
                    for j in range(i + 1, len(list_new_s)):
                        v2 = list_new_s[j]
                        if gcd(x, v2) == g_xu and gcd(u2, v2) == g_xu:
                            can_add = False
                            break
                    if not can_add:
                        break
                
                if can_add:
                    new_s.add(x)
                    added_count += 1
            
            if len(new_s) > len(s):
                s = new_s
                improved = True
                break

    return s

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    best_set = set()
    
    # 1. Heuristic Solve
    s = heuristic_solve()
    s = local_search(s)
    if len(s) > len(best_set):
        best_set = s.copy()
        
    # 2. Randomized Removal
    # We can randomize the tie-breaking in heuristic_solve
    # Or just run local_search on different subsets
    
    # Let's try to run heuristic_solve with some randomization
    # By shuffling the vertex order when counts are equal?
    # Or just running local_search multiple times on the result.
    
    while time.time() < deadline:
        # Random restart of local search?
        # Or perturb the set?
        
        # Let's try a different greedy:
        # Sort by number of conflicts, pick the one with least conflicts?
        # This is a different heuristic.
        
        # Let's stick to improving the current best.
        # Try swapping in local_search with different random seeds?
        # random module is global.
        
        s = best_set.copy()
        s = local_search(s)
        
        if len(s) > len(best_set):
            best_set = s.copy()
            
    return list(best_set)

# Test locally?
# if __name__ == "__main__":
#     res = construct()
#     print(len(res))

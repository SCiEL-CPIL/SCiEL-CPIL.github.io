import time
import math
import random

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    N = 256
    
    # Precompute conflicts: for each vertex, list of vertices it conflicts with
    # Edge (a,b) exists if a+b is a perfect square (a <= b, including a=b)
    # We store conflicts as sets for fast lookup
    conflicts = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            s = a + b
            if _is_square(s):
                conflicts[a].add(b)
                if b != a:
                    conflicts[b].add(a)
    
    # Greedy construction: try to add vertices in an order that minimizes conflicts
    # Try multiple orderings and keep the best
    best_set = set()
    
    def greedy_build(order):
        """Build independent set greedily by trying vertices in given order."""
        S = set()
        for v in order:
            # Check if v conflicts with any vertex in S
            ok = True
            for u in S:
                if v in conflicts[u]:
                    ok = False
                    break
            if ok:
                S.add(v)
        return S
    
    # Try several orderings
    base_order = list(range(1, N + 1))
    
    # Ordering 1: natural
    candidates = [greedy_build(base_order)]
    
    # Ordering 2: by number of conflicts (fewest first)
    by_conflicts = sorted(base_order, key=lambda v: len(conflicts[v]))
    candidates.append(greedy_build(by_conflicts))
    
    # Ordering 3: reverse natural
    candidates.append(greedy_build(list(reversed(base_order))))
    
    # Ordering 4: random permutations
    rng = random.Random(42)
    for _ in range(20):
        order = base_order[:]
        rng.shuffle(order)
        candidates.append(greedy_build(order))
    
    # Pick best greedy result
    best_set = max(candidates, key=len)
    
    # Local search with 1-1 swaps
    def local_search(S, deadline):
        """Improve independent set S using 1-1 swaps."""
        S = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # For each vertex in S, try swapping with a vertex not in S
            for v in list(S):
                # Try removing v and adding some u not in S
                S_without_v = S - {v}
                # Find candidates u that can be added
                # u must not conflict with any vertex in S_without_v
                # and u must not be v itself
                # We want to find u such that S_without_v ∪ {u} is independent
                # and |S_without_v ∪ {u}| > |S| (which means we need to add without removing, 
                # but that's just addition. For 1-1 swap, size stays same but we might 
                # enable future additions)
                # Actually for 1-1 swap: remove v, add u. Size stays same.
                # But we can also try: just add u if possible (pure addition)
                
                # First try pure addition
                for u in range(1, N + 1):
                    if u in S:
                        continue
                    # Check if u can be added to S
                    ok = True
                    for w in S:
                        if u in conflicts[w]:
                            ok = False
                            break
                    if ok:
                        S.add(u)
                        improved = True
                        break
                
                if improved:
                    break
                
                if time.time() >= deadline:
                    break
                
                # Try 1-1 swap: remove v, add u
                # This helps when we're stuck - removing v frees up space for u
                # which might then allow adding more
                best_swap = None
                best_swap_gain = 0
                for u in range(1, N + 1):
                    if u in S:
                        continue
                    # Check if u conflicts with any vertex in S \ {v}
                    ok = True
                    for w in S:
                        if w == v:
                            continue
                        if u in conflicts[w]:
                            ok = False
                            break
                    if not ok:
                        continue
                    # S \ {v} ∪ {u} is valid. Now check if we can add more after this swap.
                    # For now, just do the swap and let next iteration find additions.
                    # But to be smarter, estimate: how many new vertices can we add?
                    # Simple heuristic: just do the swap
                    S_new = (S - {v}) | {u}
                    # Count how many additional vertices can be added to S_new
                    add_count = 0
                    for w in range(1, N + 1):
                        if w in S_new:
                            continue
                        ok2 = True
                        for x in S_new:
                            if w in conflicts[x]:
                                ok2 = False
                                break
                        if ok2:
                            add_count += 1
                    if add_count > best_swap_gain:
                        best_swap_gain = add_count
                        best_swap = u
                
                if best_swap is not None:
                    S.discard(v)
                    S.add(best_swap)
                    improved = True
                    break
                
                if time.time() >= deadline:
                    break
        
        return S
    
    # Run local search
    best_set = local_search(best_set, deadline)
    
    # Try multiple restarts with different initial sets
    rng2 = random.Random(123)
    for restart in range(50):
        if time.time() >= deadline:
            break
        
        # Create a random initial set by greedy with random order
        order = list(range(1, N + 1))
        rng2.shuffle(order)
        S_init = greedy_build(order)
        
        # Local search
        S_new = local_search(S_init, deadline)
        
        if len(S_new) > len(best_set):
            best_set = S_new
    
    # Final polish: try to add any remaining vertices
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for u in range(1, N + 1):
            if u in best_set:
                continue
            ok = True
            for w in best_set:
                if u in conflicts[w]:
                    ok = False
                    break
            if ok:
                best_set.add(u)
                changed = True
    
    return sorted(best_set)

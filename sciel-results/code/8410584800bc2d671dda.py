import time
import random

def construct():
    N = 256
    deadline = time.time() + 55.0  # 5s margin
    
    # 1. Precompute conflicts
    # conflicts[v] contains list of pairs (a, b) such that {a, b, v} is a 3-AP
    # a, b are distinct, a != v, b != v.
    # We can store them as frozensets or just tuples.
    # For fast check, we need to see if S contains a and b.
    
    conflicts = [[] for _ in range(N + 1)]
    
    for v in range(1, N + 1):
        # v is middle: a + c = 2v. a < v < c.
        # Pairs (a, c)
        for a in range(1, v):
            c = 2 * v - a
            if c > v and c <= N:
                conflicts[v].append((a, c))
        
        # v is first: v + c = 2b. b = (v+c)/2.
        # c must be such that b is integer and in range.
        # v < b < c.
        # c = 2b - v.
        # b must be > v and <= N.
        # c must be > b and <= N.
        for b in range(v + 1, N + 1):
            c = 2 * b - v
            if c > b and c <= N:
                # AP is (v, b, c). Pairs are (v, b), (v, c), (b, c)?
                # The constraint is that the set {v, b, c} cannot be in S.
                # If v is in S, we can't have both b and c.
                # So if we are checking if v can be added to S,
                # we check if S contains any pair from the AP.
                # The pairs in the AP {v, b, c} are:
                # (v, b), (v, c), (b, c).
                # If S contains (v, b), v is already in S? No, v is candidate.
                # We check if S contains (b, c) or (v, b) or (v, c).
                # Since v is NOT in S, we only check pairs not involving v.
                # So we check (b, c).
                # Wait, if we add v, we need to ensure v doesn't form AP with any TWO elements in S.
                # So we check all pairs (x, y) in S such that {x, y, v} is AP.
                # So for a candidate v, we need to know all pairs (x, y) that form AP with v.
                # My `conflicts[v]` list should contain ALL pairs (x, y) with x != v, y != v, x != y
                # such that {x, y, v} is a 3-AP.
                
                # So for v as first term: AP is (v, b, c).
                # The pair in S that would conflict with v is (b, c).
                conflicts[v].append((b, c))
                
                # v as last term: AP is (a, b, v).
                # Pair in S is (a, b).
                # a + v = 2b. b = (a+v)/2.
                # a < b < v.
                # a must be such that b is integer.
                # b in [v/2, v-1].
                for a in range(1, v):
                    if (a + v) % 2 == 0:
                        b = (a + v) // 2
                        if a < b < v:
                            conflicts[v].append((a, b))

    # 2. Greedy Initialization
    S = set()
    # We can speed up validity check during greedy by just checking against current S
    # But for simplicity, let's just run the standard greedy, it's fast enough for N=256.
    for x in range(1, N + 1):
        valid = True
        # Check if x forms AP with any pair in S
        # We can use the conflicts[x] list? No, conflicts[x] lists pairs (a,b) such that {a,b,x} is AP.
        # So x is valid if S does not contain any pair from conflicts[x].
        for a, b in conflicts[x]:
            if a in S and b in S:
                valid = False
                break
        if valid:
            S.add(x)
    
    current_weight = sum(1.0 / x for x in S)
    
    # 3. Local Search (Swaps)
    # We will try to improve the weight by removing one element and adding a better one.
    # To be efficient, we prioritize removing larger elements (lower weight) and adding smaller ones (higher weight).
    
    # Convert S to list for random selection
    # We'll use a loop structure
    
    best_S = S
    best_weight = current_weight
    
    # Helper to check if v can be added to a set S_candidate
    def can_add(v, S_candidate):
        for a, b in conflicts[v]:
            if a in S_candidate and b in S_candidate:
                return False
        return True

    # Helper to get best candidate v not in S_candidate
    # We iterate 1..N and return first valid one
    def get_best_candidate(S_candidate, exclude_set):
        for v in range(1, N + 1):
            if v in S_candidate or v in exclude_set:
                continue
            if can_add(v, S_candidate):
                return v
        return -1

    # Main loop
    # We use a time limit.
    # We try random swaps.
    # To avoid getting stuck in local minima, we might accept non-improving moves (Simulated Annealing)
    # but given the time constraint and discrete nature, pure hill climbing with random restarts or
    # just random swaps that improve is a good start.
    
    # Let's stick to improving moves first.
    
    # We'll maintain S as a set for O(1) lookups.
    # We'll keep a list of elements in S for random removal.
    
    S_list = list(S)
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Pick a random element to remove
        if not S_list:
            break
        idx = random.randint(0, len(S_list) - 1)
        u = S_list[idx]
        
        # Remove u
        S.remove(u)
        S_list.pop(idx)
        
        # Find best candidate to add
        # We want the smallest v not in S such that v is valid.
        # Note: u is not in S, so it's a candidate. But we are replacing u, so we shouldn't just add u back.
        # Actually, if we add u back, weight is same. So exclude u?
        # If we add u, weight doesn't change. So we look for v != u.
        
        best_v = -1
        # We can iterate from 1 to N.
        # Optimization: if we know the current max index in S, we can limit search?
        # No, small v is best.
        
        for v in range(1, N + 1):
            if v == u:
                continue
            if v in S:
                continue
            if can_add(v, S):
                best_v = v
                break
        
        if best_v != -1:
            # Check if swap improves weight
            # Gain = 1/best_v - 1/u
            if (1.0 / best_v) > (1.0 / u):
                # Accept swap
                S.add(best_v)
                S_list.append(best_v)
                current_weight += (1.0 / best_v) - (1.0 / u)
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_S = set(S)
            else:
                # Revert
                S.add(u)
                S_list.append(u)
        else:
            # No valid replacement, revert
            S.add(u)
            S_list.append(u)
        
        # If we hit time limit, break
        if time.time() > deadline - 1.0:
            break

    # Return the best set found
    # Convert to sorted list
    return sorted(best_S)


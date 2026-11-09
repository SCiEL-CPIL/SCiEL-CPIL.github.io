import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit
    n = 180
    
    # 1. Precompute forbidden triples and conflict lists
    # Triple (a, b, c) with 1 <= a < b < c <= n and (b-a)(c-a) = a^2
    # 0-indexed: a' = a-1, b' = b-1, c' = c-1
    # Condition: (b'-a')*(c'-a') = (a'+1)^2
    # Let's just iterate a from 1 to n
    conflicts = [[] for _ in range(n)]
    
    for a in range(1, n + 1):
        a_sq = a * a
        # b = a + u, c = a + v
        # u * v = a_sq
        # u < v
        # b < c => u < v
        # c <= n => a + v <= n => v <= n - a
        # b <= n => a + u <= n => u <= n - a (implied by u < v)
        
        # Iterate divisors u of a_sq
        # u must be < sqrt(a_sq)
        # v = a_sq // u
        # Check bounds
        for u in range(1, int(math.isqrt(a_sq)) + 1):
            if a_sq % u == 0:
                v = a_sq // u
                if u < v:
                    b = a + u
                    c = a + v
                    if b <= n and c <= n:
                        # Valid triple (a, b, c)
                        ai = a - 1
                        bi = b - 1
                        ci = c - 1
                        # Add to conflicts
                        # For ai, the other two are bi, ci
                        # For bi, the other two are ai, ci
                        # For ci, the other two are ai, bi
                        conflicts[ai].append((bi, ci))
                        conflicts[bi].append((ai, ci))
                        conflicts[ci].append((ai, bi))

    # 2. Initial Solution
    # Start with [91, 180] (0-indexed: 90 to 179)
    # These elements cannot be 'a' in any triple because a >= 91 => b > 91, c > b => c > 182 > 180.
    # So they only appear as b or c.
    # Actually, if we include all of them, we just need to ensure no triple is fully contained.
    # Since a < 91 is not in the set, no triple can be formed entirely within [91, 180].
    # So [91, 180] is a valid independent set.
    
    S = [0] * n
    for i in range(90, n):
        S[i] = 1
        
    cur_size = sum(S)
    cur_viol = 0
    
    # Greedy Addition of smaller elements
    # Try to add elements 1..90 if they don't create a violation
    # We process in increasing order? Or random?
    # Increasing order might be good as small 'a's are in many triples.
    # But adding a small 'a' might block many others.
    # Let's just try all and add if possible.
    
    for i in range(0, 90):
        # Check if adding i creates any violation
        # A violation is created if there exists (u, v) in conflicts[i] such that S[u] and S[v] are both 1
        can_add = True
        for u, v in conflicts[i]:
            if S[u] and S[v]:
                can_add = False
                break
        if can_add:
            S[i] = 1
            cur_size += 1
            # Note: Adding i might create new violations?
            # No, we checked all pairs. If S[u] and S[v] were 1, we wouldn't add.
            # So cur_viol remains 0.

    best_S = S[:]
    best_size = cur_size
    best_viol = cur_viol

    # 3. Simulated Annealing
    random.seed(42)
    
    T_start = 10.0
    T_end = 0.01
    max_iter = 500000  # Adjust based on speed
    
    # Penalty for violations
    # We want to maximize size, so penalty should be large enough to prefer feasible solutions
    # But we also want to explore infeasible states to escape local optima.
    # Let's use a dynamic penalty or a large fixed one.
    # If we use a large penalty, we stay feasible. If we allow infeasible, we can jump over barriers.
    # Let's use a penalty of 100.
    
    def get_obj(size, viol):
        return size - 100 * viol

    cur_obj = get_obj(cur_size, cur_viol)
    
    # We will run SA for a fixed number of steps or until time limit
    # We need to track best feasible solution
    
    # To speed up, we can pre-compute the number of conflicts for each node?
    # No, the conflict check is O(degree). Degree is small.
    
    for it in range(max_iter):
        if time.time() - t0 > DEADLINE - 2:
            break
            
        x = random.randint(0, n - 1)
        
        if S[x]:
            # Remove x
            delta_size = -1
            delta_viol = 0
            # Removing x removes any violation where x is part of a triple with two other selected nodes
            for u, v in conflicts[x]:
                if S[u] and S[v]:
                    delta_viol -= 1
        else:
            # Add x
            delta_size = 1
            delta_viol = 0
            # Adding x adds a violation if there is a pair (u, v) in conflicts[x] where S[u] and S[v] are selected
            for u, v in conflicts[x]:
                if S[u] and S[v]:
                    delta_viol += 1
                    
        new_size = cur_size + delta_size
        new_viol = cur_viol + delta_viol
        new_obj = get_obj(new_size, new_viol)
        delta_obj = new_obj - cur_obj
        
        # Temperature
        progress = it / max_iter
        T_cur = T_start * (T_end / T_start) ** progress
        
        if delta_obj >= 0 or random.random() < math.exp(delta_obj / T_cur):
            S[x] = 1 - S[x]
            cur_size = new_size
            cur_viol = new_viol
            cur_obj = new_obj
            
            # Update best if feasible
            if cur_viol == 0 and cur_size > best_size:
                best_size = cur_size
                best_S = S[:]

    # 4. Final Greedy Polish
    # Try to add any remaining element
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if not best_S[i]:
                can_add = True
                for u, v in conflicts[i]:
                    if best_S[u] and best_S[v]:
                        can_add = False
                        break
                if can_add:
                    best_S[i] = 1
                    best_size += 1
                    improved = True
                    
    # Convert to 1-indexed list
    out = [i + 1 for i in range(n) if best_S[i]]
    if not out:
        out = [180]
    return out

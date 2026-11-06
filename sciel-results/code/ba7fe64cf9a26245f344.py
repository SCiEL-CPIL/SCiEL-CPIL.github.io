import time
import random
import sys
import math

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    n = 192
    max_k = 3

    # Precompute nothing heavy, just use efficient checks
    # is_valid(S) returns True if S is valid.
    # violation_count(S) returns the number of violating (d, triple) pairs.
    # For speed, we can just count how many d in S have at least one violating triple?
    # Or count total violating triples?
    # Let's count the number of d in S that are "bad" (have at least one violating triple).
    # This is O(K^4) in worst case if not careful, but K is small.
    
    def get_bad_elements(S):
        """Returns the set of elements in S that violate the condition."""
        bad = set()
        S_list = list(S)
        K = len(S_list)
        if K < 4:
            return bad
        
        for i, d in enumerate(S_list):
            others = [x for j, x in enumerate(S_list) if j != i]
            L = len(others)
            if L < 3:
                continue
            
            # Check if any triple in 'others' sums to a multiple of d
            found_violation = False
            # Optimize: use residues
            counts = {}
            for val in others:
                r = val % d
                counts[r] = counts.get(r, 0) + 1
            
            # Iterate pairs in others
            for u_idx in range(L):
                u = others[u_idx]
                ru = u % d
                for v_idx in range(u_idx + 1, L):
                    v = others[v_idx]
                    rv = v % d
                    target_r = (- (ru + rv)) % d
                    
                    cnt = counts.get(target_r, 0)
                    if cnt == 0:
                        continue
                    
                    # Check if we can pick w distinct from u, v
                    # w must have residue target_r
                    # If target_r != ru and target_r != rv, any w with that residue works (cnt >= 1)
                    # If target_r == ru (and ru != rv), we need another u (cnt_ru >= 2) -> but u is one of them. 
                    # Wait, 'counts' includes u and v.
                    # We need w such that w != u and w != v.
                    
                    if target_r != ru and target_r != rv:
                        # w can be any element with residue target_r.
                        # Since target_r is distinct from ru and rv, w is distinct from u and v automatically?
                        # Yes, if residues are distinct, values are distinct mod d, so values are distinct.
                        # Actually, values are distinct in S. So if r_w != r_u, w != u.
                        if cnt >= 1:
                            found_violation = True
                            break
                    elif target_r == ru and target_r != rv:
                        # w must have residue ru.
                        # We already have u with residue ru.
                        # We need another element with residue ru, distinct from u.
                        # counts[ru] includes u. So we need counts[ru] >= 2.
                        if cnt >= 2:
                            found_violation = True
                            break
                    elif target_r != ru and target_r == rv:
                        # w must have residue rv.
                        # We already have v with residue rv.
                        # We need another element with residue rv, distinct from v.
                        # counts[rv] includes v. So we need counts[rv] >= 2.
                        if cnt >= 2:
                            found_violation = True
                            break
                    elif target_r == ru and target_r == rv:
                        # w must have residue ru (which is rv).
                        # We have u and v with this residue.
                        # We need a third distinct element with this residue.
                        # counts[ru] includes u and v. So we need counts[ru] >= 3.
                        if cnt >= 3:
                            found_violation = True
                            break
                
                if found_violation:
                    break
            
            if found_violation:
                bad.add(d)
        
        return bad

    def is_valid(S):
        return len(get_bad_elements(S)) == 0

    def penalty(S):
        return len(get_bad_elements(S))

    def greedy_construct(order):
        S = set()
        for x in order:
            if time.time() - start_time > time_limit * 0.85:
                break
            # Fast check for adding x
            # 1. Does x divide sum of 3 in S?
            S_list = list(S)
            K = len(S_list)
            if K >= 3:
                valid_add = True
                for i in range(K):
                    u = S_list[i]
                    for j in range(i + 1, K):
                        v = S_list[j]
                        for k in range(j + 1, K):
                            w = S_list[k]
                            if (u + v + w) % x == 0:
                                valid_add = False
                                break
                            if not valid_add: break
                        if not valid_add: break
                    if not valid_add: break
                if not valid_add:
                    continue
            
            # 2. Does any y in S divide sum of triple including x?
            # Triple is (x, u, v) where u,v in S, u!=v, u!=x, v!=x.
            # Sum = x + u + v.
            # Check if (x + u + v) % y == 0 for any y in S.
            # Note: y can be x? No, y is in S, x is not in S yet.
            # But the condition is "No selected integer may divide...".
            # When we add x, x becomes selected.
            # So we must check:
            # A) x divides sum of 3 OTHERS (already checked above)
            # B) For each y in S (now S U {x}), y divides sum of 3 OTHERS in S U {x}.
            #    The "OTHERS" for y must not include y.
            #    The sum must use 3 distinct elements from S U {x} \ {y}.
            #    Since x is new, the triples involving x are (x, u, v) with u,v in S.
            #    Triples NOT involving x are already valid in S.
            #    So we only need to check triples involving x.
            #    For a fixed y in S, we check if (x + u + v) % y == 0 for any u,v in S \ {y}.
            
            valid_add = True
            for y in S_list:
                # u, v from S \ {y}
                # We need to check if there exist u, v in S \ {y} such that (x + u + v) % y == 0.
                # This is equivalent to (u + v) % y == (-x) % y.
                # Let target = (-x) % y.
                # We need two distinct elements u, v in S \ {y} such that (u+v) % y == target.
                
                # Build counts of residues for S \ {y}
                residues = {}
                for val in S_list:
                    if val == y: continue
                    r = val % y
                    residues[r] = residues.get(r, 0) + 1
                
                found = False
                # Iterate pairs of residues? Or just iterate u, v?
                # K is small, so O(K^2) is fine.
                for i in range(K):
                    u = S_list[i]
                    if u == y: continue
                    ru = u % y
                    for j in range(i + 1, K):
                        v = S_list[j]
                        if v == y: continue
                        rv = v % y
                        if (ru + rv) % y == target:
                            found = True
                            break
                    if found: break
                
                if found:
                    valid_add = False
                    break
            
            if valid_add:
                S.add(x)
        
        return S

    def simulated_annealing(initial_S, max_time=40.0):
        S = set(initial_S)
        best_S = set(initial_S)
        best_size = len(S)
        
        # Calculate initial penalty
        current_pen = penalty(S)
        
        # Objective: Maximize Size - Lambda * Penalty
        # Lambda should be large enough to prefer valid sets, but allow jumps.
        # Let's say Lambda = 100. Size is max 48. Penalty is max 48.
        # So a valid set of size 40 has score 40.
        # An invalid set of size 45 with penalty 1 has score 45 - 100 = -55.
        # This strongly prefers valid sets.
        # But if we are stuck at 40 valid, and there is a 45 invalid with penalty 1,
        # we might jump to it if T is high enough?
        # Score diff: -55 - 40 = -95. exp(-95/T).
        # If T=100, exp(-0.95) ~ 0.38. Good chance to jump.
        
        Lambda = 100.0
        T = 100.0
        T_min = 0.1
        cooling_rate = 0.995
        
        start_sa_time = time.time()
        
        while time.time() - start_sa_time < max_time and time.time() - start_time < time_limit * 0.9:
            if T < T_min:
                break
                
            # Generate a move
            # 1. Add a random element not in S
            # 2. Remove a random element from S
            # 3. Swap: Remove one, Add one
            
            move_type = random.random()
            new_S = set(S)
            
            if move_type < 0.4:
                # Add
                candidates = [x for x in range(1, n + 1) if x not in S]
                if candidates:
                    x = random.choice(candidates)
                    new_S.add(x)
            elif move_type < 0.7:
                # Remove
                if S:
                    x = random.choice(list(S))
                    new_S.remove(x)
            else:
                # Swap
                if S:
                    x_out = random.choice(list(S))
                    candidates = [x for x in range(1, n + 1) if x not in S]
                    if candidates:
                        x_in = random.choice(candidates)
                        new_S.remove(x_out)
                        new_S.add(x_in)
            
            # Evaluate
            new_pen = penalty(new_S)
            new_size = len(new_S)
            
            old_score = len(S) - Lambda * current_pen
            new_score = new_size - Lambda * new_pen
            
            delta_score = new_score - old_score
            
            accept = False
            if delta_score >= 0:
                accept = True
            else:
                # Accept with probability exp(delta / T)
                # Avoid underflow
                if delta_score > -1000 * T:
                    prob = math.exp(delta_score / T)
                    if random.random() < prob:
                        accept = True
            
            if accept:
                S = new_S
                current_pen = new_pen
                
                if current_pen == 0 and len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
            
            T *= cooling_rate
        
        return best_S

    # Phase 1: Greedy Seeds
    best_S = set()
    best_size = 0
    
    # Greedy from high to low
    S1 = greedy_construct(range(n, 0, -1))
    if len(S1) > best_size:
        best_S = S1
        best_size = len(S1)
    
    # Greedy from low to high
    S2 = greedy_construct(range(1, n + 1))
    if len(S2) > best_size:
        best_S = S2
        best_size = len(S2)
    
    # Randomized greedy
    for _ in range(5):
        if time.time() - start_time > time_limit * 0.2:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_rand = greedy_construct(order)
        if len(S_rand) > best_size:
            best_S = S_rand
            best_size = len(S_rand)
    
    # Phase 2: Simulated Annealing
    # Run SA from best seed
    if time.time() - start_time < time_limit * 0.8:
        improved_S = simulated_annealing(best_S, max_time=30.0)
        if len(improved_S) > best_size:
            best_S = improved_S
            best_size = len(improved_S)
            
    # Phase 3: More SA iterations if time permits
    while time.time() - start_time < time_limit * 0.9:
        # Try SA from current best
        improved_S = simulated_annealing(best_S, max_time=10.0)
        if len(improved_S) > best_size:
            best_S = improved_S
            best_size = len(improved_S)
        else:
            # If no improvement, maybe try a fresh random seed SA
            if random.random() < 0.1:
                order = list(range(1, n + 1))
                random.shuffle(order)
                S_new = greedy_construct(order)
                S_new = simulated_annealing(S_new, max_time=5.0)
                if len(S_new) > best_size:
                    best_S = S_new
                    best_size = len(S_new)
            else:
                # Break if stuck for a while?
                # Just continue to use time, maybe T cooling restarts help?
                # Let's just run another short SA
                improved_S = simulated_annealing(best_S, max_time=5.0)
                if len(improved_S) > best_size:
                    best_S = improved_S
                    best_size = len(improved_S)

    return list(best_S)

if __name__ == "__main__":
    res = construct()
    print(len(res))

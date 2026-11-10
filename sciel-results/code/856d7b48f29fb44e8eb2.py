"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin before 90s kill
    n = 128
    
    # U is fixed to {65, ..., 128}
    U = list(range(65, n + 1))
    U_set = set(U)
    candidates_L = list(range(1, 65))
    
    def is_valid_L(L_list):
        L_set = set(L_list)
        # Check Condition 4: Three in L
        # For distinct a,b,c in L, NOT ( (a+b in S) AND (a+c in S) AND (b+c in S) )
        # s in S iff s >= 65 or s in L_set
        for i in range(len(L_list)):
            a = L_list[i]
            for j in range(i + 1, len(L_list)):
                b = L_list[j]
                ab = a + b
                ab_in_S = (ab >= 65) or (ab in L_set)
                if not ab_in_S:
                    continue # One sum not in S, so triple (a,b,?) cannot be forbidden via this pair sum? 
                    # Wait, the condition is on the triple (a,b,c).
                    # If ab not in S, then for any c, (a,b,c) is safe?
                    # Yes, because we need ALL three sums to be in S.
                    # So if ab not in S, we don't need to check c for this pair?
                    # No, we need to check if there exists a c such that ac in S and bc in S.
                    # If ab not in S, then for ANY c, the triple (a,b,c) is safe.
                    # So we can skip checking c for this pair?
                    # Yes!
                # If ab in S, we need to check if there is a c in L (c != a, b)
                # such that ac in S and bc in S.
                for k in range(j + 1, len(L_list)):
                    c = L_list[k]
                    ac = a + c
                    ac_in_S = (ac >= 65) or (ac in L_set)
                    if not ac_in_S:
                        continue
                    bc = b + c
                    bc_in_S = (bc >= 65) or (bc in L_set)
                    if bc_in_S:
                        return False
        # Check Condition 3: One in U, two in L
        # For u in U, b,c in L (distinct),
        # Forbidden iff: (u+b <= 128) AND (u+c <= 128) AND ((b+c >= 65) OR (b+c in L_set))
        # Note: u+b in S iff u+b <= 128 (since u+b >= 66, if <=128 it's in U)
        # We can optimize: 
        # For a pair (b,c) in L, let max_u_allowed = 128 - max(b,c).
        # If there exists u in U such that u <= 128-b and u <= 128-c,
        # i.e., u <= 128 - max(b,c).
        # Since U = {65..128}, such a u exists iff 128 - max(b,c) >= 65.
        # i.e., max(b,c) <= 63.
        # If max(b,c) >= 64, then 128 - max(b,c) <= 64, so no u in U satisfies u <= 128 - max(b,c).
        # So if max(b,c) >= 64, the condition (u+b<=128 AND u+c<=128) is never true for any u in U.
        # Thus, pairs with max(b,c) >= 64 are safe from Condition 3.
        # If max(b,c) <= 63, there IS a u (e.g., u=65 if 65 <= 128-max(b,c)).
        # Actually, we just need to know if ANY u in U works.
        # The condition is: EXISTS u in U such that (u+b <= 128) AND (u+c <= 128) AND ((b+c >= 65) OR (b+c in L_set)).
        # The first part is equivalent to: u <= 128 - max(b,c).
        # Let M = max(b,c).
        # If M > 63, no u in U works. Safe.
        # If M <= 63, there are u's in U that work (specifically u in [65, 128-M]).
        # So for M <= 63, the condition reduces to: ((b+c >= 65) OR (b+c in L_set)).
        # If this is true, then there EXISTS a u (any u in the valid range) that forms a forbidden triple.
        # So we must ensure that for all pairs (b,c) in L with max(b,c) <= 63:
        # NOT ((b+c >= 65) OR (b+c in L_set))
        # i.e., b+c < 65 AND b+c not in L_set.
        
        for i in range(len(L_list)):
            b = L_list[i]
            for j in range(i + 1, len(L_list)):
                c = L_list[j]
                M = max(b, c)
                if M <= 63:
                    bc = b + c
                    # Condition to be FORBIDDEN is: (bc >= 65) OR (bc in L_set)
                    # We must AVOID this.
                    # So we require: bc < 65 AND bc not in L_set.
                    if (bc >= 65) or (bc in L_set):
                        return False
        return True

    def can_add_to_L(L_list, L_set, x):
        # Check if adding x to L creates a violation.
        # We only need to check triples involving x.
        # 1. Triple (x, b, c) with b,c in L.
        #    Check Condition 4 (all in L) for (x,b,c).
        #    Check Condition 3 (one in U) for (x,b) and (x,c) and (b,c)?
        #    Condition 3 involves one U and two L.
        #    If we add x, new pairs in L are (x,b) for all b in L.
        #    We need to check if (x,b) is forbidden with some u in U.
        #    And if (b,c) was already safe, it remains safe?
        #    Wait, Condition 3 depends on L_set. If we add x, L_set changes.
        #    Does adding x make an existing pair (b,c) forbidden?
        #    Condition 3 for (b,c): Forbidden if max(b,c)<=63 AND ((b+c>=65) OR (b+c in L_set)).
        #    If we add x, L_set grows.
        #    So if b+c == x, then b+c is now in L_set.
        #    If max(b,c) <= 63, then (b,c) becomes forbidden!
        #    So we must ensure: for all pairs (b,c) in L, if max(b,c) <= 63, then b+c != x.
        #    Also, for the new pairs (x,b):
        #    If max(x,b) <= 63, we need NOT ((x+b >= 65) OR (x+b in L_set_new)).
        #    x+b in L_set_new means x+b in L_set OR x+b == x (impossible) OR x+b == b (impossible).
        #    So x+b in L_set.
        
        # Check existing pairs (b,c) in L:
        for i in range(len(L_list)):
            b = L_list[i]
            for j in range(i + 1, len(L_list)):
                c = L_list[j]
                M = max(b, c)
                if M <= 63:
                    bc = b + c
                    # If bc == x, then bc is now in L_set.
                    # If (bc >= 65) was false (i.e., bc < 65), then before it was safe (assuming bc not in L_set).
                    # Now bc in L_set, so it becomes forbidden.
                    if bc == x:
                        return False
                    # If bc >= 65, it was already forbidden, so L would have been invalid already.
        
        # Check new pairs (x, b) for b in L:
        for b in L_list:
            # Condition 3 for pair (x, b)
            M = max(x, b)
            if M <= 63:
                s = x + b
                # Forbidden if (s >= 65) OR (s in L_set)
                # Note: s cannot be x or b.
                # s in L_set means s in L_list.
                if (s >= 65) or (s in L_set):
                    return False
            
            # Condition 4 for triple (x, b, c) with c in L, c != b
            # We need to check if there exists c in L such that (x,b,c) is forbidden in Condition 4.
            # Forbidden if: (x+b in S) AND (x+c in S) AND (b+c in S)
            # x+b in S: (x+b >= 65) OR (x+b in L_set)
            # Let's precompute x+b status
            xb = x + b
            xb_in_S = (xb >= 65) or (xb in L_set)
            if not xb_in_S:
                continue # If x+b not in S, no c can make it forbidden.
            
            for c in L_list:
                if c == b:
                    continue
                xc = x + c
                xc_in_S = (xc >= 65) or (xc in L_set)
                if not xc_in_S:
                    continue
                bc = b + c
                bc_in_S = (bc >= 65) or (bc in L_set)
                if bc_in_S:
                    return False
        return True

    best_L = []
    best_L_set = set()
    
    # Greedy construction of L
    # Try to add elements in random order
    random.seed(42)
    orders = [
        list(range(1, 65)),
        list(range(64, 0, -1)),
    ]
    for _ in range(20):
        o = list(range(1, 65))
        random.shuffle(o)
        orders.append(o)
        
    for order in orders:
        if time.time() > deadline:
            break
        L_list = []
        L_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add_to_L(L_list, L_set, x):
                L_list.append(x)
                L_set.add(x)
        if len(L_list) > len(best_L):
            best_L = L_list
            best_L_set = L_set
            # Try to improve locally
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for x in range(1, 65):
                    if x in best_L_set:
                        continue
                    if can_add_to_L(best_L, best_L_set, x):
                        best_L.append(x)
                        best_L_set.add(x)
                        improved = True

    # Final Result
    S = U + best_L
    # Verify
    S_set = set(S)
    valid = True
    # Simple verification for safety
    # We trust the logic above, but a quick check on the best_L is good.
    if not is_valid_L(best_L):
        # Fallback to empty L
        best_L = []
    
    return [int(x) for x in S]

# EVOLVE-BLOCK-END

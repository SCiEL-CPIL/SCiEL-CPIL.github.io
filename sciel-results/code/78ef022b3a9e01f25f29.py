"""Rewritten program using greedy heuristic with incremental conflict checking."""

def construct():
    S = []
    S_3_sums = set()
    
    # Iterate from largest to smallest to prioritize "safer" numbers
    for x in range(192, 0, -1):
        valid = True
        
        # Check 1: Does x divide any existing sum of 3 distinct elements in S?
        # If so, x cannot be added because x would divide the sum of 3 others (the ones in S).
        for s in S_3_sums:
            if s % x == 0:
                valid = False
                break
        
        if not valid:
            continue
            
        # Check 2: For each existing a in S, does a divide any NEW sum involving x?
        # New sums are of the form x + u + v where u, v are in S and u != v.
        # The triple {x, u, v} must consist of distinct elements other than a.
        # So u and v must be in S \ {a}.
        
        for a in S:
            # We need to check if there exist u, v in S \ {a} such that (x + u + v) % a == 0
            found_conflict = False
            
            # Optimization: if a is very large, x+u+v might be < a
            # Max x+u+v is 192+191+190 = 573.
            # If a > 573, no conflict. But a <= 192, so this doesn't help much.
            
            # Iterate through pairs in S
            # To avoid checking u,v and v,u, we can use indices
            # But we must skip a.
            
            # Create a list of elements in S excluding a?
            # S is small, so just check in loop.
            
            for i in range(len(S)):
                u = S[i]
                if u == a:
                    continue
                for j in range(i + 1, len(S)):
                    v = S[j]
                    if v == a:
                        continue
                    # u and v are distinct and not a
                    if (x + u + v) % a == 0:
                        found_conflict = True
                        break
                if found_conflict:
                    break
            
            if found_conflict:
                valid = False
                break
        
        if valid:
            # Add x to S
            S.append(x)
            
            # Update S_3_sums with new sums involving x
            # New sums are x + u + v for all u, v in S_old (which is S before append)
            # We need to use the state of S before adding x.
            # Since we appended x, S[-1] is x.
            # We need pairs from S[0:-1].
            
            old_S = S[:-1]
            for i in range(len(old_S)):
                u = old_S[i]
                for j in range(i + 1, len(old_S)):
                    v = old_S[j]
                    S_3_sums.add(x + u + v)
                    
    return S

if __name__ == "__main__":
    res = construct()
    print(len(res))
    # print(res)

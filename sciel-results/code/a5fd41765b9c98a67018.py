import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # 60s limit, leave margin for overhead
    
    n = 192
    max_size = 48
    
    S = []
    
    # Candidate integers from largest to smallest
    # Larger numbers are less likely to divide sums of others
    candidates = list(range(n, 0, -1))
    
    # We want to maximize cardinality.
    # Greedy approach: add x if it doesn't violate constraints with existing S.
    
    for x in candidates:
        if time.time() > deadline:
            break
            
        if len(S) >= max_size:
            break
            
        # Check 1: x does not divide sum of any 3 distinct elements in S
        valid = True
        if len(S) >= 3:
            # Optimization: If x is large, few sums will be divisible by it.
            # Iterate triples
            for triple in itertools.combinations(S, 3):
                s = sum(triple)
                if s % x == 0:
                    valid = False
                    break
        
        if not valid:
            continue
            
        # Check 2: For each a in S, a does not divide x + b + c for any distinct b, c in S \ {a}
        # i.e., there are no b, c in S \ {a} such that (x + b + c) % a == 0
        # <=> b + c == k*a - x for some integer k
        # <=> b + c == -x (mod a)
        
        if S:
            for a in S:
                target_res = (-x) % a
                # We need to find b, c in S \ {a} such that (b + c) % a == target_res
                # and b != c.
                
                # Iterate b in S \ {a}
                found = False
                for b in S:
                    if b == a:
                        continue
                    # We need c in S \ {a, b} such that c % a == (target_res - b) % a
                    req_res = (target_res - b) % a
                    # Check if any c in S \ {a, b} has residue req_res mod a
                    for c in S:
                        if c == a or c == b:
                            continue
                        if c % a == req_res:
                            found = True
                            break
                    if found:
                        break
                if found:
                    valid = False
                    break
            
        if valid:
            S.append(x)
            
    return S

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)

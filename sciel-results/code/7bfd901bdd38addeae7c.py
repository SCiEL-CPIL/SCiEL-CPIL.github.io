import time
import random

N = 128
DEADLINE = 55.0

def _can_add(S, v):
    """
    Check if adding v to S creates a violation.
    Violation: exist distinct a,b,c in S_new such that a+b, a+c, b+c in S_new.
    Since S_old is valid, any new violation must involve v.
    So v must be one of a, b, or c.
    WLOG, let v = a.
    Then we need b, c in S_old such that b != c, b != v, c != v,
    and v+b in S_old, v+c in S_old, b+c in S_old.
    """
    S_set = set(S)
    # v is in S_new, so sums involving v must be in S_new.
    # The sums are v+b, v+c, b+c.
    # b+c is in S_old (since b,c in S_old and S_old is valid? No, S_old is valid, but b+c might not be in S_old).
    # Wait, if b+c is NOT in S_old, then b+c is NOT in S_new (unless b+c == v, but b,c >= 1, v >= 1, b+c > v usually? Not necessarily).
    # If b+c == v, then b+c is in S_new.
    # But b,c are from S_old.
    # So we check:
    # For all b, c in S_old with b < c:
    #   if b+c == v:
    #       # b+c is in S_new
    #       # need v+b and v+c in S_new
    #       # v+b = b+b+c = 2b+c. Is it in S_old? Or is it v? No.
    #       # This is getting complicated.
    # Let's just check the condition directly on S_new?
    # No, S_new is large.
    # Let's stick to: v is one of the "base" numbers.
    # So v, b, c are distinct.
    # Sums: v+b, v+c, b+c.
    # All must be in S_new.
    # Since S_old is valid, b+c is in S_old (if it were not, it wouldn't be in S_new unless b+c==v).
    # If b+c != v, then b+c must be in S_old.
    # If b+c == v, then b+c is in S_new.
    
    # So:
    # Iterate b, c in S_old, b < c.
    # s1 = v + b
    # s2 = v + c
    # s3 = b + c
    # If s1 in S_new and s2 in S_new and s3 in S_new:
    #    return False (violation)
    
    # S_new = S_old U {v}
    # s1 in S_new iff s1 in S_old or s1 == v
    # s2 in S_new iff s2 in S_old or s2 == v
    # s3 in S_new iff s3 in S_old or s3 == v
    
    for i, b in enumerate(S):
        for c in S[i+1:]:
            if b == v or c == v:
                continue
            s1 = v + b
            s2 = v + c
            s3 = b + c
            
            in1 = (s1 in S_set) or (s1 == v)
            in2 = (s2 in S_set) or (s2 == v)
            in3 = (s3 in S_set) or (s3 == v)
            
            if in1 and in2 and in3:
                return False
    return True

def construct():
    start = time.time()
    deadline = start + DEADLINE
    best_set = set(range(65, 129))
    best_size = len(best_set)
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        S = list(range(65, 129))
        S_set = set(S)
        
        candidates = list(range(1, 65))
        rng.shuffle(candidates)
        
        for v in candidates:
            if time.time() > deadline:
                break
            if _can_add(S, v):
                S.append(v)
                S_set.add(v)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Optional: Try to improve by removing/adding?
        # For now, just restart.
        
    return [int(x) for x in best_set]

import random
import time

def construct():
    N = 180
    vertices = list(range(1, N + 1))
    
    # Precompute forbidden triples
    # a = k*u*v, b = k*u*(u+v), c = k*v*(u+v)
    # Or just brute force check 1/a == 1/b + 1/c
    # 180^3 is ~5.8M, which is fine for precomputation.
    
    # To be safe and fast, we use the parametric form or direct check.
    # Direct check:
    # for a in 1..N:
    #   for b in 1..N:
    #     if b == a: continue
    #     # 1/c = 1/a - 1/b = (b-a)/(ab)
    #     # c = ab / (b-a)
    #     if b > a:
    #       num = a*b
    #       den = b-a
    #       if num % den == 0:
    #         c = num // den
    #         if 1 <= c <= N and c != a and c != b:
    #           # Found triple {a, b, c}
    #           pass
    #     # Note: 1/a = 1/b + 1/c implies b > a and c > a?
    #     # 1/b + 1/c = 1/a. Since b,c >= 1, 1/b <= 1, 1/c <= 1.
    #     # If b <= a, 1/b >= 1/a. Then 1/c = 1/a - 1/b <= 0, impossible.
    #     # So b > a and c > a.
    #     # Thus we only need to check b > a and c > a.
    
    pairs = {i: [] for i in vertices}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c <= N and c != a and c != b:
                    # Triple (a, b, c) is forbidden
                    # Update pairs for a, b, c
                    # For a, the other two are b, c
                    pairs[a].append((b, c))
                    # For b, the other two are a, c
                    pairs[b].append((a, c))
                    # For c, the other two are a, b
                    pairs[c].append((a, b))
    
    best_set = set()
    best_size = 0
    
    # Random Restart Greedy
    # Time limit ~ 50s
    start_time = time.time()
    attempts = 0
    
    while time.time() - start_time < 50:
        random.shuffle(vertices)
        current_set = set()
        
        for v in vertices:
            can_add = True
            for (u, w) in pairs[v]:
                if u in current_set and w in current_set:
                    can_add = False
                    break
            
            if can_add:
                current_set.add(v)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
        
        attempts += 1
        
        # Optional: Local Search
        # Try to improve current_set by swapping
        # If we have time, we can do a few local search steps.
        # But random restarts are often sufficient for this size.
        # Let's add a simple local search: try to add any vertex not in set.
        # If none, try to remove one vertex and see if we can add more.
        # This is complex. Let's stick to restarts.
        
    return sorted(best_set)

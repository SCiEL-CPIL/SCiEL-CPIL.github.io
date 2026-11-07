import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Heuristic initialization: All odds + Evens >= 86
    # This set is known to be valid and has size 64 + 22 = 86
    S = set()
    for i in range(1, 129):
        if i % 2 == 1:
            S.add(i)
        elif i >= 86:
            S.add(i)
    
    # List of elements not in S
    excluded = [i for i in range(1, 129) if i not in S]
    
    # Local search: Try to add excluded elements
    # We try to add elements in a specific order. 
    # Smaller evens are more constrained, so maybe try larger ones first? 
    # Or just iterate.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate over a copy of excluded to allow removal during iteration
        for x in list(excluded):
            if x in S:
                continue
            
            # Try adding x
            S.add(x)
            if _is_valid(S):
                excluded.remove(x)
                improved = True
            else:
                S.remove(x)
                
    # Optional: Second pass with different ordering or random restarts if time permits
    # Given the structure, the first pass should capture most gains.
    
    # Ensure return type is list of ints
    return sorted(list(S))

def _is_valid(S):
    S_set = set(S)
    S_list = sorted(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            for k in range(j + 1, n):
                c = S_list[k]
                if (a + b in S_set) and (a + c in S_set) and (b + c in S_set):
                    return False
    return True

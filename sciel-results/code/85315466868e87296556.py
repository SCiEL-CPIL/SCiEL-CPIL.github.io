def construct():
    # B is fixed to the upper half to ensure no internal B-triples
    B = list(range(65, 129))
    B_set = set(B)
    
    # Start with all numbers in the lower half
    A = list(range(1, 65))
    A_set = set(A)
    
    # Define the full set S
    S_set = A_set | B_set
    
    def has_forbidden_triple(A_list):
        A_set_local = set(A_list)
        S_local = A_set_local | B_set
        n = len(A_list)
        # Optimization: only check triples if n > 2
        for i in range(n):
            a = A_list[i]
            for j in range(i+1, n):
                b = A_list[j]
                for k in range(j+1, n):
                    c = A_list[k]
                    # Check if sums are in S
                    if (a + b in S_local) and (a + c in S_local) and (b + c in S_local):
                        return (a, b, c)
        return None

    # Greedy removal
    # To speed up, we can be smarter, but 64 elements is small.
    # However, checking all triples is O(N^3). 64^3 = 262k. 
    # If we do this 64 times, it's ~16M ops. Might be slow in Python if not optimized.
    
    # Better: Precompute conflicts?
    # Or just use a heuristic: 
    # Remove the element involved in the most conflicts.
    
    # Let's try a simpler approach: 
    # We know that if a+b <= 64, it's in A (if not removed). 
    # If a+b > 64, it's in B.
    
    # Let's just run the greedy loop.
    changed = True
    while changed:
        changed = False
        triple = has_forbidden_triple(A)
        if triple:
            a, b, c = triple
            # Remove the one with the largest value? Or smallest?
            # Usually removing larger numbers helps keep the set dense at the bottom?
            # Or remove the one that appears in most other potential triples?
            # For now, remove 'a' (the first one found)
            A.remove(a)
            changed = True
            
    # Ensure types are correct
    result = A + B
    return [int(x) for x in result]

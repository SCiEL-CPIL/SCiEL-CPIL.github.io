for M in candidates_for_M:
    new_elements = []
    if M == 1:
        new_elements = [1]
    else:
        new_elements = buckets.get(M, [])
    
    for x in new_elements:
        S.add(x)
    
    # Check: for each new element x, is N-x in S?
    # This covers both:
    #   - N-x was already in S (old element)
    #   - N-x is also a new element (just added)
    for x in new_elements:
        b = N - x
        if b >= 1 and b <= N and b in S:
            return (x, b)

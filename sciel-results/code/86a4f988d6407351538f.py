import itertools

def construct(n):
    if n <= 1:
        return '1' * n
    
    # Generate permutations in the "recursive insertion" order
    # This order is known to allow maximal overlap (n-1) in the standard construction
    # P_1 = [ (1,) ]
    # P_n: for each p in P_{n-1}, insert n at each position 0..n-1
    
    perms = [(1,)]
    for k in range(2, n + 1):
        new_perms = []
        for p in perms:
            for i in range(k):
                # Insert k at index i
                new_p = p[:i] + (k,) + p[i:]
                new_perms.append(new_p)
        perms = new_perms
        
    # Now construct the superpermutation
    if not perms:
        return ""
        
    word = list(perms[0])
    
    for i in range(1, len(perms)):
        p = perms[i]
        # Find max overlap
        max_overlap = 0
        # Max possible overlap is n-1
        for k in range(n - 1, 0, -1):
            if word[-k:] == list(p[:k]):
                max_overlap = k
                break
        
        # Append non-overlapping part
        word.extend(p[max_overlap:])
        
    return ''.join(map(str, word))

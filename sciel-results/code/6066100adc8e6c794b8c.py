def standard_superpermutation(n):
    if n == 1:
        return [1]
    prev = standard_superpermutation(n-1)
    # The standard construction inserts n into the permutations
    # Actually, the recursive definition is:
    # S_n = S_{n-1} with n inserted...
    # Simpler:
    # S_1 = 1
    # S_2 = 1 2 1
    # S_3 = 1 2 3 1 2 1 3 2 1
    # S_4 = ...
    # The length is sum_{k=1}^n k! - (n-1) + 1? No.
    # L_n = L_{n-1} + n! + (n-1)(n-2)
    # Let's just implement the recursive insertion.
    
    # A common way to generate S_n from S_{n-1}:
    # S_{n-1} contains all perms of 1..n-1.
    # We need to insert n.
    # The standard algorithm:
    # Start with S_{n-1}.
    # For each permutation p in S_{n-1} (in order of appearance?), insert n.
    # Actually, the recursive step is:
    # S_n = (S_{n-1} with n appended to the first occurrence of each prefix?)
    
    # Let's use the explicit formula for the recursive construction.
    # S_1 = [1]
    # S_{k} is formed by taking S_{k-1} and for each permutation of 1..k-1 in the order they appear in S_{k-1}, 
    # we insert k.
    
    # Actually, a simpler iterative construction:
    word = [1]
    for m in range(2, n+1):
        # Get all unique windows of length m-1 from word
        # This is what the seed does for n>7.
        # The seed's fallback for n>7 IS the standard construction!
        # `if n > 7: ... word = [1] ...`
        # So the seed *already* has the standard construction logic for n>7.
        # I can just use that logic for n=7 as well, if my backtracking fails.
        
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

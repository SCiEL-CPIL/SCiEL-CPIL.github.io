def construct(n):
    # Classic recursive construction
    # Length: 1! + 2! + ... + n!
    # For n=7, length is 5913.
    
    if n == 1:
        return "1"
        
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all (m-1)-permutations in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # Block is p + [m] + p
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap with the end of 'new'
            # 'new' ends with some sequence, 'block' starts with some sequence.
            # We want to match new[-o:] with block[:o]
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            
            new += block[overlap:]
        word = new
        
    return "".join(map(str, word))

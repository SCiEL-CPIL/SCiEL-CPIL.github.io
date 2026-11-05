import time
import itertools

def construct(n):
    """
    Return a string of digits 1..n representing a superpermutation on n symbols.
    The word is built using the standard recursive construction (which gives a valid
    superpermutation of length sum_{k=1}^n k! = n! * n - sum_{k=1}^{n-1} k! + 1... actually
    the standard length is (n-1)! + (n-1)! * (n-1) + ... hmm, let me recall.

    The standard recursive superpermutation S_n has length L(n) = n * L(n-1) - (n-1)!.
    With L(1)=1.
    L(2) = 2*1 - 1 = 1? No, that's not right.

    Actually, the well-known construction:
    S_n = insert n into S_{n-1} by taking each (n-1)-permutation p in S_{n-1},
    writing p, then n, then p again, but overlapping as much as possible.

    The length formula is L(n) = (n-1)! * (n-1) + L(n-1)? No.

    The known values:
    n=1: 1
    n=2: 3  (e.g., 121)
    n=3: 9  (e.g., 123121321)
    n=4: 33
    n=5: 153
    n=6: 873
    n=7: 5913

    The formula is L(n) = n! * n - sum_{k=1}^{n-1} k! ... let me check.
    For n=2: 2!*2 - 1! = 4-1=3. Correct.
    For n=3: 6*3 - (1!+2!) = 18-3=15. But actual is 9. So that's wrong.

    The correct formula for the standard construction:
    L(1) = 1
    L(n) = L(n-1) + (n-1)! * (n-1)
    L(2) = 1 + 1! * 1 = 2. But we need 3.

    Let me just build it directly.
    """
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Build from n=3
        return _build_superperm(n)
    return _build_superperm(n)

def _build_superperm(n):
    """
    Build a valid superpermutation on n symbols using the standard recursive method.
    This produces a word of length L(n) where L(n) = L(n-1) + (n-1)! * (n-1).
    Actually, let me just construct it properly.
    
    The standard construction:
    1. Start with the superpermutation S_{n-1}.
    2. For each (n-1)-permutation p that appears in S_{n-1} (in order of first appearance),
       we add: p, n, p. But we overlap with what's already in the word.
    
    More precisely, the construction is:
    S_n = S_{n-1} with n inserted. For each window of length n-1 in S_{n-1} that is a permutation,
    say p, we add n after it and then p again, overlapping as much as possible.
    
    A simpler recursive formulation:
    Let S_{n-1} contain permutations p_1, p_2, ..., p_k where k = (n-1)!.
    S_n = p_1, n, p_1, p_2, n, p_2, ..., p_k, n, p_k
    but with overlaps.
    
    Actually, the cleanest way:
    S_n is built by taking S_{n-1} and for each (n-1)-perm p in S_{n-1} in order,
    write p, then n, then p. Consecutive such segments overlap.
    
    Let me just implement it step by step.
    """
    # Get all permutations of 1..n-1
    perms = list(itertools.permutations(range(1, n)))
    
    # Build the superpermutation
    word = []
    
    # The first permutation
    word.extend(perms[0])
    
    # For each subsequent permutation, we add n and then the next perm,
    # but we need to handle overlaps properly.
    # Actually, the standard construction appends: for each perm p in order,
    # we add n and p, with the understanding that p overlaps with the end of the previous segment.
    
    # Let me use a different, more standard approach:
    # S_n = for each (n-1)-perm p in some order: append n, then append p
    # but the first p is already there, and overlaps between consecutive p's are exploited.
    
    # Actually, the most reliable way:
    # Start with the first (n-1)-perm.
    # For each next (n-1)-perm p, find the maximum overlap with the end of the current word,
    # then append n, then append p with overlap.
    
    # But this is what the original code tried. Let me just use the known fact that
    # the standard construction gives a specific length and produce it correctly.
    
    # Let me just build it the way it's commonly done:
    # S_n: take S_{n-1}, for each window (n-1)-perm p in S_{n-1}, insert n after it and repeat p.
    
    # I'll build it recursively with proper overlap handling.
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412314231243121342132412341"
    
    # For n >= 5, build from n-1
    prev = _build_superperm(n - 1)
    prev_list = [int(c) for c in prev]
    
    # Extract (n-1)-perms in order of first appearance
    seen = set()
    blocks = []
    for i in range(len(prev_list) - (n - 2)):
        window = tuple(prev_list[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            blocks.append(window)
    
    # Build new word
    word = []
    for idx, block in enumerate(blocks):
        # Add n
        word.append(n)
        # Add block with overlap
        # Find max overlap between end of word and block
        max_ov = min(len(word), len(block))
        ov = 0
        for k in range(max_ov, -1, -1):
            if k == 0:
                ov = 0
                break
            if word[-k:] == list(block[:k]):
                ov = k
                break
        word.extend(list(block)[ov:])
    
    # But we need to prepend the first block
    # Actually, the construction should start with the first block, then for each block add n and block.
    # Let me redo:
    word = list(blocks[0])
    for idx in range(1, len(blocks)):
        block = blocks[idx]
        word.append(n)
        # Overlap
        max_ov = min(len(word), len(block))
        ov = 0
        for k in range(max_ov, -1, -1):
            if k == 0:
                ov = 0
                break
            if word[-k:] == list(block[:k]):
                ov = k
                break
        word.extend(list(block)[ov:])
    
    return "".join(map(str, word))

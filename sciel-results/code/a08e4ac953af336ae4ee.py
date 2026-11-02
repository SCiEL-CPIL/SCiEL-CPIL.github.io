import itertools
import time

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # For n=7, we need a fast construction that is valid
    # The optimal length is 5905, but we need to finish in time
    # Let's use a known good construction or a fast heuristic
    
    start_time = time.time()
    timeout = 110  # Leave 10s for overhead
    
    # Precompute all permutations
    perms = list(itertools.permutations(range(1, n+1)))
    perm_set = set(perms)
    
    # Fast greedy construction with better heuristics
    remaining = set(perms)
    word = []
    
    # Precompute prefix sets for faster extendability checks
    # prefix_counts[k][prefix] = set of perms starting with prefix
    # But this is memory intensive. Instead, we'll use a different approach.
    
    # For n=7, let's try a different strategy:
    # Use the known structure of superpermutations
    # The shortest superpermutation for n=7 has length 5905
    # We can try to construct it using a systematic approach
    
    # Actually, let's use a simpler but faster approach:
    # Build the word greedily, but with a much faster extendability check
    # Precompute for each suffix length k, a set of all prefixes of length k
    # that appear in remaining perms
    
    # Better: just use the standard recursive construction which is guaranteed valid
    # and fast, even if not optimal
    
    return standard_construction(n)

def standard_construction(n):
    # Standard recursive construction for superpermutations
    # This is O(n * n!) which is fine for n=7
    # S_1 = "1"
    # S_n is built from S_{n-1} by inserting n appropriately
    
    # The correct recursive construction:
    # S_1 = 1
    # S_n = S_{n-1} with each permutation extended by n, then concatenated with overlaps
    
    # Actually, the simplest correct construction is:
    # List all permutations of 1..n, and concatenate them with maximum overlap
    
    perms = list(itertools.permutations(range(1, n+1)))
    word = list(perms[0])
    
    for p in perms[1:]:
        # Find maximum overlap between end of word and start of p
        max_overlap = min(len(word), n-1)
        overlap = 0
        for o in range(max_overlap, 0, -1):
            if word[-o:] == list(p[:o]):
                overlap = o
                break
        word.extend(p[overlap:])
    
    return "".join(map(str, word))

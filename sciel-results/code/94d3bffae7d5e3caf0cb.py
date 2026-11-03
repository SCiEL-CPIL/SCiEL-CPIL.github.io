"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(n):
    if n == 1:
        return "1"
    
    # Start with the superpermutation for n=1
    word = [1]
    
    # Iteratively add symbols from 2 to n
    for m in range(2, n + 1):
        # Extract all unique (m-1)-permutations from the current word
        # in the order they appear
        perms, seen = [], set()
        # The window size is m-1
        window_size = m - 1
        for i in range(len(word) - window_size + 1):
            window = tuple(word[i:i + window_size])
            # Check if it's a valid permutation of 1..m-1
            # Since the previous word was a superpermutation of 1..m-1,
            # any window of size m-1 that consists of distinct elements 
            # is a permutation.
            if len(set(window)) == window_size and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Construct the new word
        new_word = []
        for p in perms:
            # The block to insert is p + [m] + p
            block = list(p) + [m] + list(p)
            
            # Find the maximum overlap between the end of new_word and the start of block
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            # Append the non-overlapping part of the block
            new_word += block[overlap:]
        
        word = new_word
        
    return "".join(map(str, word))
# EVOLVE-BLOCK-END

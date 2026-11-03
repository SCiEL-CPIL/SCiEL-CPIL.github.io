import time

def construct(n):
    start_time = time.time()
    
    # Classic recursive construction
    word = [1]
    for m in range(2, n + 1):
        # Extract unique permutations of length m-1 from word
        perms = []
        seen = set()
        # The windows of length m-1 in word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation of 1..m-1
            # Since word is a superpermutation, all windows of length m-1
            # that are permutations of 1..m-1 are the ones we need.
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between end of new_word and start of block
            overlap = 0
            # Max possible overlap is min(len(new_word), len(block))
            # But we can limit to m-1? No, it can be larger.
            # However, the structure of superpermutations implies overlaps are bounded.
            max_ov = min(len(new_word), len(block))
            for o in range(max_ov, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        
        word = new_word
        
        # Deadline check
        if time.time() - start_time > 100:
            break
            
    return "".join(map(str, word))

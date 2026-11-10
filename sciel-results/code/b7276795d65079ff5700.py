import time

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # 10 seconds margin
    
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
        perms, seen = [], set()
        # Extract unique permutations of length m-1
        # Optimization: Use a set to track seen, and list for order
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
                
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            # Max possible overlap is min(len(new), len(block)) but practically limited by block size
            # block size is 2*(m-1)+1
            max_overlap = min(len(new), len(block))
            overlap = 0
            # Check from largest to smallest
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

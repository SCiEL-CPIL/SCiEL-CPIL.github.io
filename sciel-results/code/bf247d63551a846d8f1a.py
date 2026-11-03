import time

def construct(n):
    start_time = time.time()
    deadline = 110.0
    
    word = "1"
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            return word
            
        perms = []
        seen = set()
        
        # Find all unique permutations of length m-1 in word
        # Step 1 is fine
        for i in range(len(word) - m + 2):
            window = word[i:i + m - 1]
            # Check if distinct
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = ""
        
        for p in perms:
            block = p + str(m) + p
            
            # Find max overlap
            # The overlap is a suffix of new_word and prefix of block
            # Max possible overlap is min(len(new_word), len(block))
            # But usually it's small.
            # We can check from min(len(new_word), len(block)) down to 0
            # Optimization: The overlap is at most m-1 + 1? 
            # Actually, the structure ensures some overlap.
            
            max_ov = min(len(new_word), len(block))
            # In practice, overlap is often m-1 or m-2.
            # We can start from m-1 if we know the structure?
            # No, just iterate down.
            
            overlap = 0
            for o in range(max_ov, -1, -1):
                if o == 0:
                    break
                if new_word.endswith(block[:o]):
                    overlap = o
                    break
            
            new_word += block[overlap:]
        
        word = new_word
        
    return word

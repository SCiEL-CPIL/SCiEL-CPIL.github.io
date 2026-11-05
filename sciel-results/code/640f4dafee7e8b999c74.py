import time
import random
import itertools

def construct(q, n, radius):
    # The task specifies q=3, n=8, radius=1.
    # We assume these values, but handle general inputs if possible (though task is specific).
    # The prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # So we should use q, n, radius.
    
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters, though unlikely to be tested with different ones
        # For now, just return a trivial valid code if possible or empty
        # But the task is specific. Let's assume q=3, n=8, radius=1.
        # If not, we can try a greedy construction.
        pass

    # 1. Generate initial code: Direct sum of tetracode and all words on remaining n-4 coords
    # Tetracode: 9 words of length 4, covering radius 1.
    # T = [[a, b, (a+b)%3, (a+2b)%3] for a in range(3) for b in range(3)]
    # Code = T x {0,1,2}^(n-4)
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Generate all tails
    tails = list(itertools.product(range(q), repeat=n - 4))
    
    initial_codewords = []
    for h in tetracode:
        for tail in tails:
            initial_codewords.append(h + list(tail))
            
    # Convert to tuples for hashing and set operations
    codewords = set(tuple(c) for c in initial_codewords)
    
    # Helper to convert tuple to int and vice versa for coverage array indexing
    # Base q representation
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val
        
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return tuple(word)
        
    # Precompute neighbors for all possible words? 
    # 3^8 = 6561 words. Neighbors for each: 17. Total 111k entries.
    # This is feasible and speeds up the loop significantly.
    all_neighbors = {}
    for w in range(q ** n):
        word = to_word(w)
        neigh = [w] # Include self
        for i in range(n):
            orig = word[i]
            for d in range(q):
                if d == orig: continue
                new_word = list(word)
                new_word[i] = d
                neigh.append(to_int(tuple(new_word)))
        all_neighbors[w] = neigh
        
    # Initialize coverage counts
    cov = [0] * (q ** n)
    for c in codewords:
        c_int = to_int(c)
        for nb in all_neighbors[c_int]:
            cov[nb] += 1
            
    # List of active codewords (as ints for speed)
    active = [to_int(c) for c in codewords]
    active_set = set(active)
    
    # Greedy Removal
    # We need to find a codeword c such that min(cov[nb] for nb in all_neighbors[c]) > 1
    # If found, remove it.
    
    # To speed up, we can maintain a list of candidates.
    # But since removal reduces coverage, it might invalidate other candidates.
    # A simple approach: Loop until no more removals or deadline.
    # In each iteration, scan active codewords.
    
    # Optimization: Keep a list of "potentially removable" codewords.
    # Initially, all are candidates.
    # When we remove c, we update cov.
    # Any codeword c' that shares a neighbor with c might have its min-coverage change.
    # Specifically, if cov[nb] becomes 1, then any c' with nb in N(c') is NOT removable.
    # If cov[nb] drops from 3 to 2, it doesn't change removability (still >1).
    # So we only need to worry about neighbors that drop to 1.
    
    # Let's just do a full scan in each "phase" or just iterate.
    # Given the small size, a full scan of ~700 items is cheap.
    
    while active:
        if time.time() > deadline:
            break
            
        removed_any = False
        # Shuffle to avoid bias? Or just pick the first one found.
        # Randomization might help escape local minima?
        # For now, deterministic scan.
        
        # We need to check candidates.
        # Let's iterate through a copy of active to allow removal.
        for c in active[:]:
            if c not in active_set: continue # In case of concurrent mod? No, single thread.
            
            # Check if removable
            removable = True
            for nb in all_neighbors[c]:
                if cov[nb] <= 1:
                    removable = False
                    break
            
            if removable:
                # Remove c
                active_set.remove(c)
                active.remove(c)
                
                for nb in all_neighbors[c]:
                    cov[nb] -= 1
                removed_any = True
                break # Restart scan? Or continue?
                # Breaking and restarting ensures we always pick a valid one from the start of the list.
                # It might be slower but safer.
                # Actually, we can just continue and check `if c not in active_set` at the start of next iter?
                # No, `active` is a list. Removing from list is O(N).
                # Let's use a set for active_set and a list for iteration order?
                # Better: Just maintain a list and filter out removed ones?
                # Or just use a set and iterate over a list of keys?
                
                # Let's stick to:
                # active is a list.
                # If we remove, we break and restart the loop.
                # This is O(N^2) in worst case, but N is small.
                pass
        
        if not removed_any:
            # No more simple removals.
            # Try a random restart or swap?
            # For now, stop.
            break
            
    return [list(to_word(c)) for c in active]

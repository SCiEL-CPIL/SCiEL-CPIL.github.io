import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    # Global variables for backtracking
    best_seq = []
    best_len = 0
    time_exceeded = [False]
    
    def check_time():
        if time.time() - start_time > DEADLINE:
            time_exceeded[0] = True
            return True
        return False

    def backtrack(seq, prod_set, sufs, last_val):
        nonlocal best_seq, best_len, time_exceeded
        
        if time_exceeded[0]:
            return
            
        # Pruning: Max possible length
        # We can add at most (128 - last_val) more elements.
        if len(seq) + (128 - last_val) <= best_len:
            return
            
        # Update best if current is longer
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        # Try to add next elements
        for x in range(last_val + 1, 129):
            if check_time():
                return
                
            # Generate new suffix products
            # new_sufs = [x] + [x * p for p in sufs]
            # Check if any new product is in prod_set
            valid = True
            new_sufs = []
            
            # x itself
            if x in prod_set:
                valid = False
            else:
                new_sufs.append(x)
                
            # x * p for p in sufs
            if valid:
                for p in sufs:
                    np = x * p
                    if np in prod_set:
                        valid = False
                        break
                    new_sufs.append(np)
                    
            if not valid:
                continue
                
            # If valid, proceed
            # Update state
            new_prod_set = set(prod_set)
            for p in new_sufs:
                new_prod_set.add(p)
                
            # Recurse
            backtrack(seq + [x], new_prod_set, new_sufs, x)
            
            if time_exceeded[0]:
                return

    # Initial Greedy to set a good lower bound
    # This helps the pruning in backtracking significantly
    seq = []
    prod_set = set()
    sufs = []
    last_val = 1
    
    for x in range(2, 129):
        valid = True
        new_sufs = []
        
        if x in prod_set:
            valid = False
        else:
            new_sufs.append(x)
            
        if valid:
            for p in sufs:
                np = x * p
                if np in prod_set:
                    valid = False
                    break
                new_sufs.append(np)
                
        if valid:
            seq.append(x)
            prod_set.update(new_sufs)
            sufs = new_sufs
            last_val = x
            
    best_seq = seq[:]
    best_len = len(seq)
    
    # Now use backtracking to try to find a longer sequence
    backtrack([], set(), [], 1)
    
    return best_seq

# Test locally
if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Length: {len(res)}")
    print(f"Time: {end - start:.2f}s")
    print(res[:10])
    
    # Verify
    def verify(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True
    print("Valid:", verify(res))

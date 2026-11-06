import time
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s limit (evaluator kills at 90s, but we want to be safe)
    
    # We want to find a strictly increasing sequence of integers in [1, 256]
    # such that sums of all nonempty consecutive blocks are distinct.
    # Maximize length.
    
    # Let's use a backtracking search with strong pruning.
    # State:
    #   seq: current sequence (list)
    #   sums_mask: bitset of all consecutive block sums seen so far.
    #   last_val: last element in seq.
    #   suffix_sums: list of suffix sums of seq. suffix_sums[0] = seq[-1], suffix_sums[1] = seq[-1]+seq[-2], etc.
    #               This helps in computing new sums quickly.
    
    best_seq = []
    
    def check_time():
        return time.time() - start_time > time_limit
    
    def update_best(seq):
        nonlocal best_seq
        if len(seq) > len(best_seq):
            best_seq = seq.copy()
    
    # Greedy initialization to get a decent starting point
    def greedy_init():
        vals = []
        sums = set()
        suffix_sums = []
        for x in range(1, 257):
            # New sums: x, x+suffix_sums[0], x+suffix_sums[1], ...
            # Actually, new sums are x, x+seq[-1], x+seq[-1]+seq[-2], ...
            # Let's compute them.
            new_sums = [x]
            total = x
            for s in suffix_sums:
                total += s
                new_sums.append(total)
            if not sums.intersection(new_sums):
                vals.append(x)
                sums.update(new_sums)
                # Update suffix_sums
                # Old suffix_sums: [a_n, a_n+a_{n-1}, ...]
                # New suffix_sums: [x, x+a_n, x+a_n+a_{n-1}, ...]
                new_suffix = [x]
                total = x
                for s in suffix_sums:
                    total += s
                    new_suffix.append(total)
                suffix_sums = new_suffix
        return vals, sums, suffix_sums
    
    init_vals, init_sums, init_suffix = greedy_init()
    if init_vals:
        best_seq = init_vals.copy()
    
    # Convert sums to mask
    def sums_to_mask(sums):
        mask = 0
        for s in sums:
            mask |= (1 << s)
        return mask
    
    init_mask = sums_to_mask(init_sums)
    
    # DFS from the greedy result?
    # Or DFS from scratch?
    # DFS from scratch is more likely to find the global optimum, but might be slower.
    # Given the small size, let's try DFS from scratch but with good heuristics.
    # However, the branching factor is high.
    # Let's try to extend the greedy result first to see if we can beat it.
    # Then, if we have time, try other branches.
    
    # Actually, a pure DFS from scratch with bitset pruning is very effective for this kind of problem.
    # The number of valid sequences is small.
    
    # Let's implement a DFS that explores candidates in a smart order.
    # Candidates: last_val + 1 to 256.
    # Heuristic: Try smaller candidates first? Or larger?
    # Smaller candidates keep the total sum lower, which might allow more elements.
    # But larger candidates might avoid collisions with existing sums more easily?
    # Let's try smaller first.
    
    def dfs(seq, sums_mask, last_val, suffix_sums):
        nonlocal best_seq
        nonlocal start_time
        
        if len(seq) > len(best_seq):
            best_seq = seq.copy()
            
        if check_time():
            return
            
        # Pruning:
        # If len(seq) + (256 - last_val) <= len(best_seq), we can't beat best_seq.
        if len(seq) + (256 - last_val) <= len(best_seq):
            return
            
        # Try extending
        for x in range(last_val + 1, 257):
            # Compute new sums
            # New sums: x, x+suffix_sums[0], x+suffix_sums[1], ...
            new_sums_mask = 1 << x
            total = x
            for s in suffix_sums:
                total += s
                new_sums_mask |= (1 << total)
                
            if sums_mask & new_sums_mask:
                continue
                
            # Update suffix_sums for next recursion
            # New suffix_sums: [x, x+suffix_sums[0], x+suffix_sums[1], ...]
            new_suffix = [x]
            total = x
            for s in suffix_sums:
                total += s
                new_suffix.append(total)
                
            dfs(seq + [x], sums_mask | new_sums_mask, x, new_suffix)
            
            if check_time():
                return
            
            # Optional: If we found a longer sequence, and we are iterating x in increasing order,
            # maybe we can break? No, larger x might lead to even longer sequences later?
            # Unlikely, but possible.
            
    # Start DFS
    # We can start from empty, or from the greedy result.
    # Starting from empty explores more.
    # Let's start from empty.
    dfs([], 0, 0, [])
    
    return best_seq

# Test
if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
    print(f"Sequence: {res}")
    
    # Verify
    sums = set()
    n = len(res)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += res[j]
            if s in sums:
                print("Collision!")
                break
            sums.add(s)
    print("Valid")

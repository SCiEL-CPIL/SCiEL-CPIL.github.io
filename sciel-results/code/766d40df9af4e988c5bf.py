import time
import sys

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin for 60s limit
    
    best_seq = []
    best_len = 0
    
    # seen_sums: set of all consecutive sums
    # suffix_sums: list of suffix sums of the current sequence
    # seq: current sequence
    
    def dfs(seq, suffix_sums, seen_sums, last_val):
        nonlocal best_seq, best_len, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)
            
        # Pruning: Can we possibly beat best_len?
        # Max elements we can add is at most 256 - last_val (if we pick consecutive integers)
        # But distinct sums constraint is much stronger.
        # Simple bound: 256 - last_val
        if len(seq) + (256 - last_val) <= best_len:
            return
            
        # Try adding next value x
        for x in range(last_val + 1, 257):
            if time.time() - start_time > time_limit:
                return
                
            # Calculate new sums
            # New sums are x + s for s in suffix_sums, plus x itself (which is x + 0)
            # suffix_sums currently contains sums of seq[i:] for i in 0..n-1
            # When we append x, the new suffix sums are:
            # x + suffix_sums[i] for i in 0..n-1, and x.
            
            new_sums = []
            valid = True
            for s in suffix_sums:
                ns = x + s
                if ns in seen_sums:
                    valid = False
                    break
                new_sums.append(ns)
            
            if not valid:
                continue
                
            # Check x itself
            if x in seen_sums:
                continue
            new_sums.append(x)
            
            # All new sums are distinct from each other?
            # Since seq is strictly increasing, suffix sums are strictly increasing?
            # No, suffix sums are sums of positive integers.
            # seq: a1 < a2 < ... < an
            # suffix_sums: S1 = a1+...+an, S2 = a2+...+an, ... Sn = an
            # S1 > S2 > ... > Sn.
            # New sums: x+S1, x+S2, ..., x+Sn, x.
            # Since S1 > S2 > ... > Sn > 0, then x+S1 > x+S2 > ... > x+Sn > x.
            # So new_sums are distinct from each other.
            
            # Proceed with recursion
            # Update state
            # new_suffix_sums = [x + s for s in suffix_sums] + [x]
            new_suffix_sums = [x + s for s in suffix_sums] + [x]
            
            # Add new sums to seen_sums
            for ns in new_sums:
                seen_sums.add(ns)
            
            seq.append(x)
            dfs(seq, new_suffix_sums, seen_sums, x)
            seq.pop()
            
            # Backtrack
            for ns in new_sums:
                seen_sums.discard(ns)
                
            # Optimization: If we found a solution that uses the maximum possible length 
            # given the current last_val, we might not need to try larger x?
            # Not necessarily, a larger x might allow a different structure?
            # But since we want to MAXIMIZE LENGTH, and we are trying x in increasing order,
            # if we find a valid x, we recurse. If that branch doesn't yield a better length,
            # we try the next x.
            # We can't stop early unless we know no future x can beat best_len.
            
    # Initial state
    # seq = []
    # suffix_sums = []
    # seen_sums = set()
    # last_val = 0
    
    dfs([], [], set(), 0)
    
    # Fallback if no sequence found (should not happen)
    if not best_seq:
        best_seq = [1]
        
    return best_seq

# Test locally?
# if __name__ == "__main__":
#     seq = construct()
#     print(f"Length: {len(seq)}")
#     print(f"Sequence: {seq}")

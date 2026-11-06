import sys
import time

def construct():
    # Greedy initialization to set a good lower bound
    def greedy():
        seq = []
        prods = set()
        for x in range(2, 129):
            # Check if x can be added
            new_prods = set(prods)
            valid = True
            if x in new_prods:
                valid = False
            else:
                new_prods.add(x)
            
            if valid:
                suffix_prod = 1
                for i in range(len(seq)-1, -1, -1):
                    suffix_prod *= seq[i]
                    p = x * suffix_prod
                    if p in new_prods:
                        valid = False
                        break
                    else:
                        new_prods.add(p)
            
            if valid:
                seq.append(x)
                prods = new_prods
        return seq

    best_seq = greedy()
    
    # If greedy is already very long, we might not need to search much
    # But let's run DFS to ensure optimality.
    
    start_time = time.time()
    timeout = 55.0
    
    # Use DFS with Branch and Bound
    # State: seq (tuple), prods (set), next_x (int)
    
    def dfs(seq, prods, next_x):
        nonlocal best_seq
        
        # Time check
        if time.time() - start_time > timeout:
            return

        # Pruning
        # Max possible length: len(seq) + (128 - next_x)
        # Note: next_x is the smallest number we can pick.
        # We can pick at most one number per integer value, so max additional is 128 - next_x + 1?
        # No, range is [next_x, 128]. Count is 128 - next_x + 1.
        # But we might skip some.
        # Upper bound: len(seq) + (128 - next_x + 1)
        if len(seq) + (128 - next_x + 1) <= len(best_seq):
            return

        # Update best if current is better
        if len(seq) > len(best_seq):
            best_seq = list(seq)
            # Optional: If we reached a very high length, we can tighten timeout?
            # No, just keep going.

        # Try to pick each x
        for x in range(next_x, 129):
            # Check validity
            new_prods = set(prods)
            valid = True
            
            # Check x
            if x in new_prods:
                valid = False
            else:
                new_prods.add(x)
            
            if valid:
                suffix_prod = 1
                for i in range(len(seq)-1, -1, -1):
                    suffix_prod *= seq[i]
                    p = x * suffix_prod
                    if p in new_prods:
                        valid = False
                        break
                    else:
                        new_prods.add(p)
            
            if not valid:
                continue
            
            # Recurse
            dfs(seq + (x,), new_prods, x + 1)
            
            # If we just found a sequence of length 127 (max possible), we can stop?
            # Max possible length is bounded by 128.
            # If len(best_seq) >= 120, it's very good.
            # We can break early if we hit a theoretical max?
            # Theoretical max is not known, but 128 is the hard limit.
            
            if time.time() - start_time > timeout:
                return

    # Start DFS from empty sequence?
    # We can start from the greedy sequence's prefix?
    # No, DFS needs to explore all possibilities.
    # But we can start the DFS from the empty state.
    
    dfs((), set(), 2)
    
    return best_seq

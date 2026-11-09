import time

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin
    
    # S[k] is a set of sums of k-element subsets
    # S[0] = {0}
    S = [set() for _ in range(61)]
    S[0].add(0)
    current_size = 0
    result = []
    
    # Try to add elements from 1 to 256
    for x in range(1, 257):
        if time.time() > deadline:
            break
            
        valid = True
        
        # Check if adding x is valid
        # For each k, new sums are S[k] + x
        # These must not intersect S[i] for any i != k+1
        
        for k in range(current_size + 1):
            if not valid:
                break
            # Calculate new sums for cardinality k+1
            T = {s + x for s in S[k]}
            
            # Check against all existing S[i] where i != k+1
            for i in range(current_size + 1):
                if i == k + 1:
                    continue
                # Check intersection
                if T.intersection(S[i]):
                    valid = False
                    break
        
        if valid:
            # Update S
            # Update in reverse order to avoid using updated values in the same iteration
            for k in range(current_size, -1, -1):
                S[k+1] = S[k+1].union({s + x for s in S[k]})
            
            current_size += 1
            result.append(x)

    return list(result)

import random

N = 256

def construct():
    S = []
    Sigma = set()
    unblocked = set(range(1, N + 1))
    
    while unblocked:
        best_x = None
        best_gain = -1
        
        # Pre-compute compatibility and potential blocks for all x in unblocked
        # To optimize, we iterate and calculate gain efficiently.
        
        candidates = []
        for x in unblocked:
            # Check compatibility
            # 1. 2x not in Sigma
            if 2 * x in Sigma:
                continue
            # 2. x+s not in Sigma for any s in S
            compat = True
            for s in S:
                if x + s in Sigma:
                    compat = False
                    break
            if not compat:
                continue
            
            # Calculate gain: how many unblocked z != x become blocked?
            # z is blocked if:
            # (2z == 2x) -> z=x (skip)
            # (2z == x+s) for s in S -> z = (x+s)/2
            # (z+x == 2x) -> z=x (skip)
            # (z+x == x+s) for s in S -> z=s (skip, z not in S)
            # (z+s == 2x) for s in S -> z = 2x-s
            # (z+s == x+s') for s, s' in S -> z = x + s' - s
            
            blocked_by_x = set()
            
            # Type 1: 2z = x+s
            for s in S:
                val = x + s
                if val % 2 == 0:
                    z = val // 2
                    if 1 <= z <= N and z != x and z in unblocked:
                        blocked_by_x.add(z)
            
            # Type 2: z = 2x - s
            for s in S:
                z = 2 * x - s
                if 1 <= z <= N and z != x and z in unblocked:
                    blocked_by_x.add(z)
            
            # Type 3: z = x + s' - s
            # This can generate many z.
            # We can just check if z is in unblocked.
            for s in S:
                for sp in S:
                    z = x + sp - s
                    if 1 <= z <= N and z != x and z in unblocked:
                        blocked_by_x.add(z)
            
            gain = len(blocked_by_x)
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None:
            # No compatible element found?
            # This means S is maximal.
            break
        
        # Add best_x
        S.append(best_x)
        Sigma.update([best_x + s for s in S[:-1]] + [2 * best_x])
        unblocked.remove(best_x)
        
        # Update unblocked: remove elements that are now blocked
        # An element z is blocked if 2z in Sigma or exists s in S such that z+s in Sigma
        new_unblocked = set()
        for z in unblocked:
            blocked = False
            if 2 * z in Sigma:
                blocked = True
            else:
                for s in S:
                    if z + s in Sigma:
                        blocked = True
                        break
            if not blocked:
                new_unblocked.add(z)
        unblocked = new_unblocked
        
    return S

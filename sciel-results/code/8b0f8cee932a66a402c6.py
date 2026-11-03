import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though the task specifies 7 matches ternary
        # The prompt says "Football pool problem for 7 matches... ternary covering code K_3(7,1)"
        # But the interface is generic. We should handle the specific case efficiently 
        # and fallback for others.
        if q == 3 and n == 7 and radius == 1:
            pass # Handled below
        else:
            # Simple fallback: return all words if small, else greedy
            if q**n < 10000:
                return [list(t) for t in itertools.product(range(q), repeat=n)]
            else:
                # Generic greedy fallback
                # This might be slow, so we limit it
                pass

    # Specific optimization for q=3, n=7, r=1
    if q == 3 and n == 7 and radius == 1:
        space_size = 3**n
        # Map word tuple to int
        def tuple_to_int(t):
            val = 0
            for i in range(n):
                val = val * 3 + t[i]
            return val
        
        def int_to_tuple(val):
            t = [0]*n
            for i in range(n-1, -1, -1):
                t[i] = val % 3
                val //= 3
            return t

        # Precompute spheres
        # sphere[i] is a list of ints
        spheres = []
        for i in range(space_size):
            t = int_to_tuple(i)
            s = [i]
            for pos in range(n):
                orig = t[pos]
                for new_val in range(3):
                    if new_val != orig:
                        t[pos] = new_val
                        s.append(tuple_to_int(t))
                t[pos] = orig
            spheres.append(s)

        cov = [0] * space_size
        code = []
        uncovered = space_size
        
        # Greedy Phase
        while uncovered > 0:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_c = -1
            # Iterate all candidates
            for c in range(space_size):
                gain = 0
                for w in spheres[c]:
                    if cov[w] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_gain == 0:
                break # Should not happen if uncovered > 0
            
            code.append(best_c)
            for w in spheres[best_c]:
                if cov[w] == 0:
                    uncovered -= 1
                cov[w] += 1
            
            # Early exit if we have too many
            if len(code) > 240: 
                 # We are worse than seed, stop and try removal? 
                 # Actually if greedy is bad, removal won't fix it. 
                 # But greedy is usually good.
                 pass

        # Removal Phase
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            # Iterate backwards to remove
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                is_essential = False
                for w in spheres[c]:
                    if cov[w] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    code.pop(i)
                    for w in spheres[c]:
                        cov[w] -= 1
                    changed = True
        
        # Convert back to lists of ints
        result = [int_to_tuple(c) for c in code]
        return result

    # Fallback for other cases (simple greedy or brute force)
    # For the sake of this task, the main case is 3,7,1.
    # We return a trivial solution if not the main case to avoid timeout
    if q**n <= 100:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Generic Greedy for other cases
    # This is a placeholder to ensure the function doesn't crash
    # In a real scenario, we'd implement a generic version.
    return [list(t) for t in itertools.product(range(q), repeat=n)]

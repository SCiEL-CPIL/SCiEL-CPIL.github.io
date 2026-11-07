import time
import sys

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAXV = 1000000
N = len(DIVISORS)

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin

    # We search for the minimum span S
    # Start from S=1
    # For each S, we search for a valid window [L, L+S]
    
    # To speed up L search, we can note that the configuration of candidates
    # only changes when L crosses a multiple of any divisor.
    # However, for small S, simple scanning is fast.
    
    max_S_to_try = 500  # A reasonable upper bound for S
    max_L_to_try = 5000  # A reasonable upper bound for L for small S
    
    for S in range(1, max_S_to_try + 1):
        if time.time() > deadline:
            break
            
        # Iterate L
        # We can step L by 1, but we can also be smarter.
        # For now, simple scan.
        for L in range(1, max_L_to_try + 1):
            if time.time() > deadline:
                break
                
            # Check if L+S exceeds MAXV
            if L + S > MAXV:
                # We could still have a solution if we pick smaller numbers?
                # No, L is the min. If L+S > MAXV, it's invalid for this L.
                # But maybe a larger L? No, L is increasing.
                # If L+S > MAXV, then for all larger L, L+S > MAXV.
                # So we can break the L loop.
                break
            
            # Generate candidates for each divisor
            candidates = []
            valid = True
            for d in DIVISORS:
                # Smallest multiple of d >= L
                # ceil(L/d) * d
                k = (L + d - 1) // d
                m = k * d
                cands = []
                while m <= L + S:
                    cands.append(m)
                    m += d
                if not cands:
                    valid = False
                    break
                candidates.append(cands)
            
            if not valid:
                continue
                
            # Now we have candidates for each divisor.
            # We need to pick one from each such that all are distinct.
            # This is a matching problem.
            # Since N=10 and candidates are small, backtracking is fine.
            
            assignment = [None] * N
            used = set()
            
            # Sort divisors by number of candidates (ascending) to prune early
            # indices sorted by len(candidates[i])
            indices = sorted(range(N), key=lambda i: len(candidates[i]))
            
            def backtrack(idx):
                if idx == len(indices):
                    return True
                i = indices[idx]
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        assignment[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assignment[i] = None
                return False
            
            if backtrack(0):
                # Found a solution
                # Convert to list of ints in original order
                result = [int(assignment[i]) for i in range(N)]
                return result

    # Fallback if no solution found in the bounds
    # This should not happen, but we need to return something valid
    # Just return the divisors themselves? They are distinct and <= 1000000.
    # Span is 39-6=33.
    # But we might have missed a better one.
    # Let's do a quick greedy check for a few more S if time permits?
    # For now, return a valid set.
    return DIVISORS[:]
